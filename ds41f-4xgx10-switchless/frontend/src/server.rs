//! OpenAI-compatible HTTP server: /v1/models, /v1/chat/completions (streaming and not), /health.
//!
//! Requests are rendered with the publisher's chat encoding the way the kit's SGLang does it (tools
//! hosted on a leading system message, effort tiers from chat_template_kwargs or reasoning_effort),
//! matched to a session for prefix reuse, sent to rank 0, and the streamed ids are decoded, checked
//! for stop strings and loops, and parsed into reasoning, content and tool calls.

use crate::encoding::{api_tool_name, encode_messages, Delta, EncodeOptions, Encoding, StreamParser, ThinkingMode};
use crate::engine::{EngineClient, Event as EngineEvent};
use crate::gate::{Gate, Pass, Refused, RETRY_AFTER_S};
use crate::guard::{LoopGuard, StopMatcher, LINE_WINDOW_TOKENS};
use crate::protocol::{DoneReason, Submit, FLAG_KEEP, FLAG_SEED};
use crate::pyjson::truthy;
use crate::session::SessionTable;
use crate::switch::{Family, Switcher, Ticket};
use crate::tokenizer::{StreamDecoder, Tokenizer};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Map, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model_name: String,
    pub api_key: Option<String>,
    /// Default and cap for max_tokens (the kit's DSV41_MAX_NEW_TOKENS).
    pub max_tokens_cap: u32,
    /// Reasoning effort budget when the request names none.
    pub default_effort: i64,
    /// The chat encoding: the weights' own encoding.py (V4.1, or 1017's).
    pub encoding: Encoding,
    pub default_thinking: bool,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub loop_guard: Option<LoopGuard>,
    pub tokenize_threads: usize,
    /// Write each request, its prompt and its ids here, and at its end the output's ids (debugging and
    /// bench traces; holds conversation text).
    pub dump_dir: Option<std::path::PathBuf>,
    /// How long a request waits while the engine is down (the ring lent, or a restart) before it is
    /// answered 503 with Retry-After.
    pub hold: Duration,
    /// vision: the image encoder's address (host:port); None: image input is refused as before
    pub vision_addr: Option<String>,
    /// The model name whose requests run on the engine's weight variant (docs/variants.md); None: one model
    pub variant_model: Option<String>,
    /// The GLM-5.3 backend (gr8, docs/glm.md): its model names go there; None: DeepSeek only
    pub glm: Option<crate::glm::GlmConfig>,
    /// Model names that run on whichever engine family is loaded (docs/glm.md): stock, and uncensored
    /// (DeepSeek's variant, GLM with the steering profile). None: no alias
    pub active_alias: Option<String>,
    pub active_alias_uncensored: Option<String>,
}

impl ServerConfig {
    /// The weight variant a request's model name runs on: 1 for variant_model, else 0.
    pub fn variant_of(&self, model: Option<&str>) -> u32 {
        (model.is_some() && model == self.variant_model.as_deref()) as u32
    }
    /// The model name an active-engine alias stands for on family `f`.
    pub fn alias_target(&self, f: Family, uncensored: bool) -> &str {
        match (f, &self.glm) {
            (Family::Glm, Some(g)) => if uncensored { &g.steered_model } else { &g.model },
            _ if uncensored => self.variant_model.as_deref().unwrap_or(&self.model_name),
            _ => &self.model_name,
        }
    }
    /// Some(uncensored) when `model` is an active-engine alias. With GLM beside DeepSeek, a name neither
    /// engine serves counts as the plain alias: Claude Code's own model names (T3 writes a thread's titles
    /// and commit messages with claude-haiku-*) run on the loaded engine instead of switching to DeepSeek.
    pub fn alias_of(&self, model: &str) -> Option<bool> {
        if Some(model) == self.active_alias.as_deref() {
            Some(false)
        } else if Some(model) == self.active_alias_uncensored.as_deref() {
            Some(true)
        } else if self.glm.is_some() && !self.serves(model) {
            Some(false)
        } else {
            None
        }
    }
    /// A model name one of the engines serves: DeepSeek's, its variant's, or a GLM name.
    pub fn serves(&self, model: &str) -> bool {
        model == self.model_name
            || Some(model) == self.variant_model.as_deref()
            || self.glm.as_ref().is_some_and(|g| model == g.model || model == g.steered_model)
    }
    /// The model name of weight variant v.
    pub fn name_of(&self, v: u32) -> &str {
        match (&self.variant_model, v) {
            (Some(m), 1) => m,
            _ => &self.model_name,
        }
    }
}

pub struct App {
    pub cfg: ServerConfig,
    pub tok: Arc<Tokenizer>,
    pub engine: EngineClient,
    pub gate: Arc<Gate>,
    sessions: Mutex<SessionTable>,
    think_end_id: Option<u32>,
    counter: AtomicU64,
    /// The dump files' numbers go on after the highest a dump directory already holds (dump_base):
    /// the request ids start at 1 with every start.
    dump_base: u64,
    /// engine switching (--switch-cmd, switch.rs); unset: the DeepSeek engine only, as before
    switcher: std::sync::OnceLock<Arc<Switcher>>,
}

/// The highest number of the `NNNNNN.json` and `NNNNNN.out.json` files in dir (0: none).
pub(crate) fn dump_base(dir: &std::path::Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.filter_map(|e| {
        let name = e.ok()?.file_name().into_string().ok()?;
        let n = name.strip_suffix(".out.json").or_else(|| name.strip_suffix(".json"))?;
        n.bytes().all(|b| b.is_ascii_digit()).then(|| n.parse::<u64>().ok()).flatten()
    })
    .max()
    .unwrap_or(0)
}

impl App {
    pub fn new(cfg: ServerConfig, tok: Arc<Tokenizer>, engine: EngineClient) -> Arc<App> {
        let think_end_id = tok.token_id("</think>");
        let gate = Gate::new(cfg.hold);
        let dump_base = cfg.dump_dir.as_deref().map_or(0, dump_base);
        Arc::new(App { cfg, tok, engine, gate, sessions: Mutex::new(SessionTable::new(256, 16 << 20)), think_end_id, counter: AtomicU64::new(0), dump_base, switcher: Default::default() })
    }

    /// Engine switching from now on (set once, before serving).
    pub fn set_switcher(&self, sw: Arc<Switcher>) {
        let _ = self.switcher.set(sw);
    }

    pub fn switcher(&self) -> Option<&Arc<Switcher>> {
        self.switcher.get()
    }

    /// The session table an earlier run saved (session.rs `load`), for the next turns of its
    /// conversations: what happened, for the log.
    pub fn load_sessions(&self, path: &std::path::Path, max_age_s: u64) -> String {
        match self.sessions.lock().unwrap().load(path, &self.tok, max_age_s) {
            Ok(n) => format!("session table: {n} sessions from {}", path.display()),
            Err(e) if e.is_empty() => format!("session table: none saved at {}", path.display()),
            Err(e) => format!("session table: none taken ({e})"),
        }
    }

    /// The session table for the next start (session.rs `save`): what happened, for the log.
    pub fn save_sessions(&self, path: &std::path::Path, max_age_s: u64) -> String {
        match self.sessions.lock().unwrap().save(path, max_age_s) {
            Ok(n) => format!("session table: {n} sessions saved to {}", path.display()),
            Err(e) => format!("session table: NOT saved: {e}"),
        }
    }

    pub(crate) fn random_hex(&self, n: usize) -> String {
        // splitmix64 over time and a counter: ids only need to be unique
        let mut x = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
            ^ self.counter.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E3779B97F4A7C15);
        let mut s = String::new();
        while s.len() < n {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            s.push_str(&format!("{z:016x}"));
        }
        s.truncate(n);
        s
    }
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/messages", post(crate::anthropic::messages))
        .route("/v1/messages/count_tokens", post(crate::anthropic::count_tokens))
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(app)
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

fn error(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    ApiError(status, kind.to_string(), msg.into()).response()
}

/// A request's failure: its status, error type and message. The answer, or once a stream has started
/// (a stream that waited at the gate), its error event.
pub(crate) struct ApiError(pub(crate) StatusCode, pub(crate) String, pub(crate) String);

impl ApiError {
    fn json(&self) -> Value {
        json!({"error": {"message": self.2, "type": self.1, "param": null, "code": self.0.as_u16()}})
    }
    fn response(&self) -> Response {
        (self.0, [("content-type", "application/json")], self.json().to_string()).into_response()
    }
}

/// Refused at the gate: 503, with Retry-After as an answer.
pub(crate) fn refused(r: Refused) -> ApiError {
    let msg = match r {
        Refused::Held(t) => format!("the engine has been down for the {} s this request waited (the ring is lent to a development job, or the engine restarts); try again later", t.as_secs()),
        Refused::Closing => "the server is shutting down".into(),
    };
    ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), msg)
}

/// The model an active-engine alias names now: whichever family is loaded, or being switched to
/// (switch.rs `alias_family`), so that it never causes a switch.
pub(crate) fn resolve_alias(app: &App, model: &str) -> Option<String> {
    let uncensored = app.cfg.alias_of(model)?;
    let f = app.switcher().map_or(Family::DeepSeek, |s| s.alias_family());
    let to = app.cfg.alias_target(f, uncensored).to_string();
    eprintln!("alias: {model} -> {to} (the {} engine)", f.name());
    Some(to)
}

/// The request body with an active-engine alias resolved (None: it names none). The body is read
/// only when it holds an alias's name.
pub(crate) fn alias_body(app: &App, body: &Bytes) -> Option<Bytes> {
    let names = [app.cfg.active_alias.as_deref(), app.cfg.active_alias_uncensored.as_deref()];
    // with GLM, any name may be an alias (ServerConfig::alias_of): the body is read every time
    if app.cfg.glm.is_none() && !names.into_iter().flatten().any(|n| !n.is_empty() && body.windows(n.len()).any(|w| w == n.as_bytes())) {
        return None;
    }
    let mut req: Value = serde_json::from_slice(body).ok()?;
    let to = resolve_alias(app, req.get("model")?.as_str()?)?;
    req["model"] = json!(to);
    Some(Bytes::from(serde_json::to_vec(&req).ok()?))
}

/// With engine switching, a DeepSeek request's admission to its engine if it needs no wait: Ok(None)
/// without switching, Ok(Some) admitted, Err(()) it has to wait (`ds_admit`).
pub(crate) fn ds_try(app: &App) -> Result<Option<Ticket>, ()> {
    match app.switcher() {
        None => Ok(None),
        Some(sw) => sw.try_admit(Family::DeepSeek).map(Some).ok_or(()),
    }
}

/// Wait for the DeepSeek engine (a switch from GLM): refused as the gate refuses, 503.
pub(crate) async fn ds_admit(app: &App) -> Result<Option<Ticket>, ApiError> {
    match app.switcher() {
        None => Ok(None),
        Some(sw) => sw.admit(Family::DeepSeek).await.map(Some).map_err(|e| ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), e)),
    }
}

/// A stream that waits for the engine carries an SSE comment this often, so that a client's idle
/// timeout sees bytes (and so does a long prefill's wait for its first token).
pub(crate) const KEEP_ALIVE: Duration = Duration::from_secs(5);

fn bad_request(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

/// The key as a bearer token, or in `x-api-key` (Anthropic clients).
pub(crate) fn key_ok(app: &App, headers: &HeaderMap) -> bool {
    let Some(key) = &app.cfg.api_key else { return true };
    let got = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .map(str::trim)
        .unwrap_or("");
    // constant-time comparison
    got.len() == key.len() && got.bytes().zip(key.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

fn check_auth(app: &App, headers: &HeaderMap) -> Result<(), Response> {
    if key_ok(app, headers) {
        Ok(())
    } else {
        Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "invalid API key"))
    }
}

async fn health(State(app): State<Arc<App>>) -> Response {
    if let Some(sw) = app.switcher() {
        let line = sw.status();
        match (sw.switching(), sw.active()) {
            (Some(_), _) | (None, None) => return (StatusCode::SERVICE_UNAVAILABLE, format!("{line}\n")).into_response(),
            (None, Some(Family::Glm)) => {
                let url = &app.cfg.glm.as_ref().expect("glm configured").url;
                return if crate::glm::health(url, Duration::from_secs(2)).await {
                    (StatusCode::OK, format!("ok (glm) {line}\n")).into_response()
                } else {
                    (StatusCode::SERVICE_UNAVAILABLE, format!("the GLM engine does not answer at {url}; {line}\n")).into_response()
                };
            }
            (None, Some(Family::DeepSeek)) => {}
        }
    }
    if app.gate.held() {
        let msg = format!("the ring is lent to a development job: requests wait up to {} s\n", app.gate.hold_limit().as_secs());
        (StatusCode::SERVICE_UNAVAILABLE, msg).into_response()
    } else if app.engine.info().is_some() {
        (StatusCode::OK, "ok\n").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "engine not connected\n").into_response()
    }
}

async fn models(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&app, &headers) {
        return r;
    }
    let max_len = app.engine.info().map(|i| i.max_context as u64);
    // OpenAI's list, with the fields an Anthropic client reads beside them
    let ds = std::iter::once(app.cfg.model_name.as_str()).chain(app.cfg.variant_model.as_deref()).map(|n| (n, max_len));
    let glm = match &app.cfg.glm {
        None => Vec::new(),
        Some(g) => {
            let len = crate::glm::max_model_len(g).await;
            vec![(g.model.as_str(), len), (g.steered_model.as_str(), len)]
        }
    };
    let mut names: Vec<(&str, Option<u64>)> = ds.chain(glm).collect();
    // the active-engine aliases, with the window of what they name now
    let f = app.switcher().map_or(Family::DeepSeek, |s| s.alias_family());
    for (a, unc) in [(&app.cfg.active_alias, false), (&app.cfg.active_alias_uncensored, true)] {
        if let Some(a) = a {
            let to = app.cfg.alias_target(f, unc);
            let len = names.iter().find(|x| x.0 == to).and_then(|x| x.1);
            names.push((a.as_str(), len));
        }
    }
    let data: Vec<Value> = names
        .iter()
        .map(|(n, len)| {
            json!({"id": n, "object": "model", "created": 0, "owned_by": "dsgb10", "root": n, "max_model_len": len,
                   "type": "model", "display_name": n, "created_at": "1970-01-01T00:00:00Z"})
        })
        .collect();
    let names: Vec<&str> = names.iter().map(|x| x.0).collect();
    let body = json!({"object": "list", "data": data, "has_more": false, "first_id": names[0], "last_id": names[names.len() - 1]});
    (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response()
}

// ------------------------------------------------------------------ request → prompt

/// What the request asks for, with the prompt rendered.
pub struct Prepared {
    pub prompt: String,
    pub mode: ThinkingMode,
    pub tools: bool,
    pub stream: bool,
    pub include_usage: bool,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: Option<u64>,
    pub stops: Vec<String>,
    /// besides EOS, unless ignore_eos
    pub stop_ids: Vec<u32>,
    pub ignore_eos: bool,
    /// vision: the image records in prompt order (encoding.py's), one per placeholder; empty for text
    pub images: Vec<Value>,
    /// the weight variant its model name runs on (ServerConfig::variant_of)
    pub variant: u32,
}

fn num_f64(v: &Value) -> Option<f64> {
    v.as_number().and_then(|n| n.to_string().parse::<f64>().ok())
}

fn num_i64(v: &Value) -> Option<i64> {
    let s = v.as_number()?.to_string();
    if s.contains(['.', 'e', 'E']) {
        let f: f64 = s.parse().ok()?;
        return (f.fract() == 0.0 && f.abs() < 9e15).then_some(f as i64);
    }
    s.parse().ok()
}

/// The reasoning-effort budget: the kit's table (low 50, high and xhigh 75, max 100), integers 1-100
/// as they are, OpenAI fractions in [0, 1] times 100; anything else is the default.
pub fn effort_budget(v: Option<&Value>, default: i64) -> i64 {
    match v {
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "low" => 50,
            "high" | "xhigh" => 75,
            "max" => 100,
            t => t.parse::<f64>().ok().map_or(default, |f| effort_budget(Some(&json!(f)), default)),
        },
        Some(n @ Value::Number(_)) => {
            let s = n.as_number().unwrap().to_string();
            let f = num_f64(n).unwrap_or(f64::NAN);
            if !s.contains(['.', 'e', 'E']) {
                let i = f as i64;
                if (1..=100).contains(&i) {
                    i
                } else {
                    default
                }
            } else if (0.0..=1.0).contains(&f) {
                ((f * 100.0).round() as i64).clamp(1, 100)
            } else if (1.0..=100.0).contains(&f) {
                f.round() as i64
            } else {
                default
            }
        }
        _ => default,
    }
}

/// The tool payload the kit's SGLang renders: the function's name, description, parameters and
/// strict, in that order, and only those the client sent.
fn tool_payload(t: &Value) -> Result<Value, String> {
    let f = t.get("function").and_then(Value::as_object).ok_or("each tool needs a function object")?;
    if !f.get("name").map_or(false, Value::is_string) {
        return Err("each tool function needs a name".into());
    }
    let mut out = Map::new();
    for k in ["name", "description", "parameters", "strict"] {
        if let Some(v) = f.get(k) {
            if !v.is_null() {
                out.insert(k.into(), v.clone());
            }
        }
    }
    let mut tool = json!({"type": "function", "function": Value::Object(out)});
    if let Some(ns) = t.get("namespace").filter(|v| !v.is_null()) {
        tool["namespace"] = ns.clone(); // 1017's encoding qualifies the name with it; V4.1's ignores it
    }
    Ok(tool)
}

/// A tool call for the API, as 1017's tool_calls_to_openai_format makes it: a `namespace::name` the
/// model wrote is the bare name, with the namespace beside the function (api_tool_name).
fn tool_call_json(enc: Encoding, index: Option<usize>, id_hex: String, name: &str, args: String) -> Value {
    let (ns, name) = api_tool_name(enc, name);
    let mut call = json!({"id": format!("call_{id_hex}"), "type": "function", "function": {"name": name, "arguments": args}});
    if let Some(i) = index {
        call["index"] = json!(i);
    }
    if let Some(ns) = ns {
        call["namespace"] = json!(ns);
    }
    call
}

/// Messages for the encoder: OpenAI fields only, developer → system, null content → "".
fn convert_messages(msgs: &[Value], vision: bool) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let o = m.as_object().ok_or(format!("messages[{i}] is not an object"))?;
        let role = o.get("role").and_then(Value::as_str).ok_or(format!("messages[{i}] has no role"))?;
        let role = match role {
            "developer" => "system",
            r @ ("system" | "user" | "assistant" | "tool") => r,
            r => return Err(format!("messages[{i}]: unsupported role {r:?}")),
        };
        let mut n = Map::new();
        n.insert("role".into(), json!(role));
        let content = match o.get("content") {
            None | Some(Value::Null) => json!(""),
            Some(Value::Array(parts)) => {
                for p in parts {
                    match p.get("type").and_then(Value::as_str) {
                        Some("image_url" | "image") if vision => {}
                        Some("input_image" | "video_url" | "input_audio" | "audio_url") if vision => {
                            return Err("audio and video input are not supported; images go in image_url parts".into())
                        }
                        Some("image_url" | "image" | "input_image" | "video_url" | "input_audio" | "audio_url") => {
                            return Err("image, audio and video input are not supported".into())
                        }
                        _ => {}
                    }
                }
                Value::Array(parts.clone())
            }
            Some(c) => c.clone(),
        };
        n.insert("content".into(), content);
        for k in ["name", "tool_calls", "tool_call_id", "reasoning_content"] {
            if let Some(v) = o.get(k) {
                if !v.is_null() {
                    n.insert(k.into(), v.clone());
                }
            }
        }
        if !n.contains_key("reasoning_content") {
            if let Some(Value::String(r)) = o.get("reasoning") {
                n.insert("reasoning_content".into(), json!(r));
            }
        }
        out.push(Value::Object(n));
    }
    Ok(out)
}

pub fn prepare(cfg: &ServerConfig, req: &Value) -> Result<Prepared, String> {
    let r = req.as_object().ok_or("the request body must be a JSON object")?;
    let msgs = r.get("messages").and_then(Value::as_array).ok_or("messages must be a list")?;
    if msgs.is_empty() {
        return Err("messages must not be empty".into());
    }
    if let Some(n) = r.get("n").filter(|v| !v.is_null()) {
        if num_i64(n) != Some(1) {
            return Err("only n = 1 is supported".into());
        }
    }
    let mut messages = convert_messages(msgs, cfg.vision_addr.is_some())?;
    messages.iter_mut().for_each(crate::encoding::neutralize_image_placeholder);

    // tools, hosted on a leading system message
    let tool_choice_none = r.get("tool_choice").and_then(Value::as_str) == Some("none");
    let tools: Vec<Value> = match r.get("tools") {
        Some(Value::Array(ts)) if !ts.is_empty() && !tool_choice_none => ts.iter().map(tool_payload).collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };
    if !tools.is_empty() {
        if messages[0]["role"] != "system" {
            messages.insert(0, json!({"role": "system", "content": ""}));
        }
        messages[0].as_object_mut().unwrap().insert("tools".into(), Value::Array(tools.clone()));
    }

    // thinking and effort
    let ctk = r.get("chat_template_kwargs").and_then(Value::as_object);
    let reasoning = r.get("reasoning").and_then(Value::as_object);
    let top_effort = r.get("reasoning_effort").filter(|v| !v.is_null());
    let thinking = if let Some(v) = ctk.and_then(|c| c.get("thinking")).filter(|v| !v.is_null()) {
        truthy(Some(v))
    } else if let Some(v) = ctk.and_then(|c| c.get("enable_thinking")).filter(|v| !v.is_null()) {
        truthy(Some(v))
    } else if let Some(e) = top_effort.or_else(|| reasoning.and_then(|x| x.get("effort")).filter(|v| !v.is_null())) {
        e.as_str() != Some("none")
    } else if let Some(v) = reasoning.and_then(|x| x.get("enabled")).filter(|v| !v.is_null()) {
        truthy(Some(v))
    } else {
        cfg.default_thinking
    };
    let effort_src = ctk
        .and_then(|c| c.get("reasoning_effort"))
        .filter(|v| !v.is_null())
        .or(top_effort)
        .or_else(|| reasoning.and_then(|x| x.get("effort").or_else(|| x.get("reasoning_effort"))).filter(|v| !v.is_null()));
    let budget = json!(effort_budget(effort_src, cfg.default_effort));
    let mode = if thinking { ThinkingMode::Thinking } else { ThinkingMode::Chat };
    let mut o = EncodeOptions::new(mode);
    o.reasoning_effort = Some(&budget);
    o.encoding = cfg.encoding;
    let (prompt, images) = encode_messages(&messages, &o).map_err(|e| format!("cannot encode messages: {e}"))?;
    if images.len() > crate::vision::MAX_IMAGES {
        return Err(format!("at most {} images per request", crate::vision::MAX_IMAGES));
    }

    // sampling and limits
    let max_tokens = match r.get("max_completion_tokens").filter(|v| !v.is_null()).or(r.get("max_tokens").filter(|v| !v.is_null())) {
        None => cfg.max_tokens_cap,
        Some(v) => match num_i64(v) {
            Some(n) if n >= 1 => (n as u64).min(cfg.max_tokens_cap as u64) as u32,
            _ => return Err("max_tokens must be a positive integer".into()),
        },
    };
    let fnum = |k: &str| r.get(k).filter(|v| !v.is_null()).map(|v| num_f64(v).ok_or(format!("{k} must be a number"))).transpose();
    let mut temperature = fnum("temperature")?.map_or(cfg.default_temperature, |t| t as f32);
    let mut top_p = fnum("top_p")?.map_or(cfg.default_top_p, |t| t as f32);
    if !(0.0..=100.0).contains(&temperature) || !(0.0..=1.0).contains(&top_p) {
        return Err("temperature must be >= 0 and top_p in [0, 1]".into());
    }
    if top_p == 0.0 {
        temperature = 0.0;
        top_p = 1.0;
    }
    let top_k = fnum("top_k")?.map_or(0, |k| if k > 0.0 { k as u32 } else { 0 });
    let seed = r.get("seed").filter(|v| !v.is_null()).map(|v| num_i64(v).map(|s| s as u64).ok_or("seed must be an integer")).transpose()?;
    let stops = match r.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().map(|s| s.as_str().map(str::to_string).ok_or("stop must be strings")).collect::<Result<_, _>>()?,
        _ => return Err("stop must be a string or a list of strings".into()),
    };
    let stop_ids: Vec<u32> = match r.get("stop_token_ids") {
        Some(Value::Array(a)) => a.iter().filter_map(|v| num_i64(v).map(|x| x as u32)).collect(),
        _ => Vec::new(),
    };
    let ignore_eos = truthy(r.get("ignore_eos"));
    let stream = truthy(r.get("stream"));
    let include_usage = truthy(r.get("stream_options").and_then(|s| s.get("include_usage")));
    Ok(Prepared {
        prompt,
        mode,
        tools: !tools.is_empty(),
        stream,
        include_usage,
        max_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        stops,
        stop_ids,
        ignore_eos,
        images,
        variant: cfg.variant_of(r.get("model").and_then(Value::as_str)),
    })
}

// ------------------------------------------------------------------ generation

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Finish {
    Stop,
    Length,
    ToolCalls,
}

impl Finish {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
        }
    }
}

pub(crate) struct Usage {
    pub(crate) prompt: usize,
    pub(crate) cached: u32,
    pub(crate) completion: usize,
    pub(crate) reasoning: usize,
}

impl Usage {
    fn json(&self) -> Value {
        json!({"prompt_tokens": self.prompt, "completion_tokens": self.completion,
               "total_tokens": self.prompt + self.completion,
               "prompt_tokens_details": {"cached_tokens": self.cached},
               "completion_tokens_details": {"reasoning_tokens": self.reasoning}})
    }
}

/// One request in flight on the engine: decodes, stops, guards and parses its token stream.
pub(crate) struct Gen {
    app: Arc<App>,
    /// counts the request in flight at the gate until it ends (or its client goes away)
    _pass: Pass,
    req_id: u64,
    session_id: u64,
    prompt_text: String,
    pub(crate) prompt_ids: Vec<u32>,
    /// the prefix the session plan expects to reuse; the engine's count comes at the end
    pub(crate) reused: usize,
    rx: mpsc::UnboundedReceiver<EngineEvent>,
    stop_ids: Vec<u32>,
    stopper: StopMatcher,
    parser: StreamParser,
    dec: StreamDecoder,
    /// every id the engine sent (the session holds these)
    ids: Vec<u32>,
    /// ids that count as output (up to a stop string, a loop or a stop id)
    kept: usize,
    gen_bytes: Vec<u8>,
    gen_ends: Vec<usize>,
    stopped: Option<Finish>,
    done: bool,
    t0: Instant,
    first_token: Option<Instant>,
}

impl Gen {
    /// Feed engine events until something can be sent. Returns the deltas and, at the end, the
    /// finish reason and usage.
    pub(crate) async fn next(&mut self, out: &mut Vec<Delta>) -> Option<Result<(Finish, Usage), String>> {
        loop {
            let ev = match self.rx.recv().await {
                Some(ev) => ev,
                None => return Some(Err("engine stream ended".into())),
            };
            match ev {
                EngineEvent::Tokens(t) => {
                    self.first_token.get_or_insert_with(Instant::now);
                    for &id in &t.ids {
                        self.ids.push(id);
                        if self.stopped.is_some() {
                            continue;
                        }
                        self.kept = self.ids.len();
                        let tb = self.app.tok.token_bytes(id);
                        self.gen_bytes.extend_from_slice(tb);
                        self.gen_ends.push(self.gen_bytes.len());
                        if self.stop_ids.contains(&id) {
                            // a stop id ends the output; its text is not part of the answer
                            self.flush(out);
                            self.stopped = Some(Finish::Stop);
                            continue;
                        }
                        let mut text = String::new();
                        self.dec.push(&self.app.tok, id, &mut text);
                        let (pass, hit) = self.stopper.push(&text);
                        self.parser.push(&pass, out);
                        if hit {
                            self.stopped = Some(Finish::Stop);
                            self.app.engine.cancel(self.req_id);
                        }
                    }
                    if self.stopped.is_none() {
                        if let Some(g) = &self.app.cfg.loop_guard {
                            let out_ids = &self.ids[..self.kept];
                            let from = out_ids.len().saturating_sub(LINE_WINDOW_TOKENS);
                            let start = if from == 0 { 0 } else { self.gen_ends[from - 1] };
                            let tail = String::from_utf8_lossy(&self.gen_bytes[start..]);
                            if let Some(keep) = g.check(out_ids, &tail) {
                                eprintln!("req {}: repetition loop after {} tokens; keeping {keep}", self.req_id, out_ids.len());
                                self.stopped = Some(Finish::Stop);
                                self.app.engine.cancel(self.req_id);
                            }
                        }
                    }
                    if !out.is_empty() {
                        return None;
                    }
                }
                EngineEvent::Done(d) => {
                    self.done = true;
                    if d.reason == DoneReason::Error {
                        self.app.sessions.lock().unwrap().abandon(self.session_id);
                        self.app.engine.release(self.session_id);
                        return Some(Err(if d.error.is_empty() { "engine error".into() } else { d.error }));
                    }
                    if self.stopped.is_none() {
                        self.flush(out);
                    }
                    self.parser.finish(out);
                    let finish = if self.parser.tool_calls_seen() > 0 {
                        Finish::ToolCalls
                    } else if let Some(f) = self.stopped {
                        f
                    } else {
                        match d.reason {
                            DoneReason::Stop | DoneReason::Cancelled => Finish::Stop,
                            _ => Finish::Length,
                        }
                    };
                    let kept = &self.ids[..self.kept];
                    let reasoning = match self.app.think_end_id {
                        Some(te) if self.parser_was_thinking() => kept.iter().position(|&x| x == te).unwrap_or(kept.len()),
                        _ => 0,
                    };
                    self.app.sessions.lock().unwrap().finish(self.session_id, &self.prompt_text, &self.prompt_ids, &self.ids, &self.app.tok);
                    let usage = Usage { prompt: self.prompt_ids.len(), cached: d.cached_tokens, completion: self.kept, reasoning };
                    let total = self.t0.elapsed().as_secs_f64();
                    let ttft = self.first_token.map_or(total, |t| (t - self.t0).as_secs_f64());
                    if let Some(dir) = &self.app.cfg.dump_dir {
                        // the output's ids: the first `kept` count as output, the rest came after a stop
                        let rec = json!({"req_id": self.req_id, "ids": self.ids, "kept": self.kept, "finish": finish.as_str(),
                            "usage": usage.json(), "ttft_s": ttft, "total_s": total});
                        if let Err(e) = std::fs::write(dir.join(format!("{:06}.out.json", self.app.dump_base + self.req_id)), rec.to_string()) {
                            eprintln!("dump: {e}");
                        }
                    }
                    eprintln!(
                        "req {} session {}: prompt {} (cached {}), completion {}, {}; ttft {:.3} s, {:.1} tok/s",
                        self.req_id,
                        self.session_id,
                        usage.prompt,
                        usage.cached,
                        usage.completion,
                        finish.as_str(),
                        ttft,
                        if total > ttft { usage.completion as f64 / (total - ttft) } else { 0.0 }
                    );
                    return Some(Ok((finish, usage)));
                }
            }
        }
    }

    /// Pass on the text still held by the decoder and the stop matcher.
    fn flush(&mut self, out: &mut Vec<Delta>) {
        let mut rest = String::new();
        self.dec.finish(&mut rest);
        let (pass, hit) = self.stopper.push(&rest);
        self.parser.push(&pass, out);
        if !hit {
            let tail = self.stopper.finish();
            self.parser.push(&tail, out);
        }
    }

    fn parser_was_thinking(&self) -> bool {
        self.prompt_text.ends_with(crate::encoding::THINK_START)
    }
}

impl Drop for Gen {
    fn drop(&mut self) {
        if !self.done {
            // The client went away mid-request. A cancelled session keeps what it processed, of
            // which what we received is a prefix: still good for the conversation's next turn.
            self.app.engine.cancel(self.req_id);
            self.app.sessions.lock().unwrap().finish(self.session_id, &self.prompt_text, &self.prompt_ids, &self.ids, &self.app.tok);
        }
    }
}

pub(crate) async fn start(app: &Arc<App>, p: &Prepared, pass: Pass) -> Result<Gen, ApiError> {
    let unavailable = |msg: String| ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), msg);
    let Some(info) = app.engine.info() else {
        return Err(unavailable("engine not connected".into()));
    };
    if !p.images.is_empty() {
        return start_images(app, p, pass, info).await;
    }
    let (plan, released) = {
        let app2 = app.clone();
        let (text, variant) = (p.prompt.clone(), p.variant);
        tokio::task::spawn_blocking(move || {
            let mut t = app2.sessions.lock().unwrap();
            t.plan(&text, variant, &app2.tok, app2.cfg.tokenize_threads)
        })
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "server_error".into(), e.to_string()))?
    };
    for s in released {
        app.engine.release(s);
    }
    let ctx = info.max_context as usize;
    if plan.ids.len() >= ctx {
        app.sessions.lock().unwrap().abandon(plan.session_id);
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "invalid_request_error".into(),
            format!("the prompt is {} tokens; the context window is {ctx}", plan.ids.len()),
        ));
    }
    let max_new = (p.max_tokens as usize).min(ctx - plan.ids.len()) as u32;
    let mut stop_ids = p.stop_ids.clone();
    if !p.ignore_eos {
        stop_ids.push(info.eos_id);
    }
    let submit = Submit {
        req_id: 0,
        session_id: plan.session_id,
        base_session_id: plan.base_session_id,
        seed: p.seed.unwrap_or(0),
        temperature: p.temperature,
        top_p: p.top_p,
        top_k: p.top_k,
        max_new_tokens: max_new,
        flags: FLAG_KEEP | if p.seed.is_some() { FLAG_SEED } else { 0 },
        n_top_logprobs: 0,
        prefix_hint: plan.reused as u32,
        variant: p.variant,
        stop_ids: stop_ids.clone(),
        prompt: plan.ids.clone(),
        images: None,
    };
    let (req_id, rx) = match app.engine.submit(submit) {
        Ok(x) => x,
        Err(e) => {
            app.sessions.lock().unwrap().abandon(plan.session_id);
            return Err(unavailable(e));
        }
    };
    // where it starts (counts only); rank 0's "serve: req N restart:" line says what the engine found
    eprintln!("req {req_id} plan: {}", plan.describe(p.prompt.len()));
    Ok(Gen {
        app: app.clone(),
        _pass: pass,
        req_id,
        session_id: plan.session_id,
        prompt_text: p.prompt.clone(),
        reused: plan.reused,
        prompt_ids: plan.ids,
        rx,
        stop_ids,
        stopper: StopMatcher::new(p.stops.clone()),
        parser: StreamParser::with_tools(p.mode, p.tools),
        dec: StreamDecoder::default(),
        ids: Vec::new(),
        kept: 0,
        gen_bytes: Vec::new(),
        gen_ends: Vec::new(),
        stopped: None,
        done: false,
        t0: Instant::now(),
        first_token: None,
    })
}

/// A request with images (docs/vision.md): its images through the encoder, the placeholders expanded into
/// their spans, and a session of its own that the engine frees at the end (no KEEP) and the text table
/// never sees, so text requests' prefix reuse is untouched.
async fn start_images(app: &Arc<App>, p: &Prepared, pass: Pass, info: crate::protocol::HelloEngine) -> Result<Gen, ApiError> {
    let bad = |msg: String| ApiError(StatusCode::BAD_REQUEST, "invalid_request_error".into(), msg);
    let Some(addr) = app.cfg.vision_addr.clone() else {
        return Err(bad("image, audio and video input are not supported".into()));
    };
    let enc = crate::vision::encode(&addr, &p.images, Duration::from_secs(90)).await.map_err(|e| {
        if e.starts_with("cannot read") { bad(e) } else { ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), e) }
    })?;
    let ids = {
        let (app2, text) = (app.clone(), p.prompt.clone());
        tokio::task::spawn_blocking(move || app2.tok.encode_parallel(&text, app2.cfg.tokenize_threads))
            .await
            .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "server_error".into(), e.to_string()))?
    };
    let (ids, starts) = crate::vision::expand(&ids, &enc.rows).map_err(bad)?;
    let ctx = info.max_context as usize;
    if ids.len() >= ctx {
        return Err(bad(format!("the prompt is {} tokens with its images; the context window is {ctx}", ids.len())));
    }
    let max_new = (p.max_tokens as usize).min(ctx - ids.len()) as u32;
    let mut stop_ids = p.stop_ids.clone();
    if !p.ignore_eos {
        stop_ids.push(info.eos_id);
    }
    let session_id = (1u64 << 63) | u64::from_str_radix(&app.random_hex(15), 16).unwrap_or(1);
    let submit = Submit {
        req_id: 0,
        session_id,
        base_session_id: 0,
        seed: p.seed.unwrap_or(0),
        temperature: p.temperature,
        top_p: p.top_p,
        top_k: p.top_k,
        max_new_tokens: max_new,
        flags: crate::protocol::FLAG_IMAGES | if p.seed.is_some() { FLAG_SEED } else { 0 },
        n_top_logprobs: 0,
        prefix_hint: 0,
        variant: p.variant,
        stop_ids: stop_ids.clone(),
        prompt: ids.clone(),
        images: Some(crate::vision::payload(&starts, &enc)),
    };
    let (req_id, rx) = app.engine.submit(submit).map_err(|e| ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), e))?;
    eprintln!(
        "req {req_id} images: {} ({} rows), encoder {:.0} ms; prompt {} ids; session {session_id} (not kept)",
        enc.rows.len(),
        enc.rows.iter().sum::<u32>(),
        enc.ms,
        ids.len()
    );
    Ok(Gen {
        app: app.clone(),
        _pass: pass,
        req_id,
        session_id,
        prompt_text: p.prompt.clone(),
        reused: 0,
        prompt_ids: ids,
        rx,
        stop_ids,
        stopper: StopMatcher::new(p.stops.clone()),
        parser: StreamParser::with_tools(p.mode, p.tools),
        dec: StreamDecoder::default(),
        ids: Vec::new(),
        kept: 0,
        gen_bytes: Vec::new(),
        gen_ends: Vec::new(),
        stopped: None,
        done: false,
        t0: Instant::now(),
        first_token: None,
    })
}

async fn chat(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = check_auth(&app, &headers) {
        return r;
    }
    let body = alias_body(&app, &body).unwrap_or(body);
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("invalid JSON: {e}")),
    };
    if let Some(steered) = app.cfg.glm.as_ref().and_then(|g| g.route(req.get("model").and_then(Value::as_str))) {
        return crate::glm::chat(app, req, steered).await;
    }
    let p = match prepare(&app.cfg, &req) {
        Ok(p) => p,
        Err(e) => return bad_request(e),
    };
    // With engine switching, a request that has to wait for the DeepSeek engine (a switch from GLM)
    // waits as for the gate below.
    let ticket = match ds_try(&app) {
        Ok(t) => t,
        Err(()) if p.stream => return stream(app, p, req, None, None),
        Err(()) => match ds_admit(&app).await {
            Ok(t) => t,
            Err(e) => {
                let mut resp = e.response();
                resp.headers_mut().insert("retry-after", RETRY_AFTER_S.into());
                return resp;
            }
        },
    };
    // While the engine is down (the ring lent to a job, or a restart) a request waits at the gate. A
    // stream that has to wait starts at once and carries comments meanwhile, so that its client's
    // timeout sees bytes; refused, it ends with an error event. Anything else waits here, refused with
    // 503 and Retry-After.
    if p.stream && !app.gate.open(app.engine.info().is_some()) {
        return stream(app, p, req, None, ticket);
    }
    let pass = match app.gate.enter(&app.engine).await {
        Ok(pass) => pass.with_ticket(ticket),
        Err(r) => {
            let mut resp = refused(r).response();
            resp.headers_mut().insert("retry-after", RETRY_AFTER_S.into());
            return resp;
        }
    };
    if p.stream {
        return stream(app, p, req, Some(pass), None);
    }
    let mut gen = match start(&app, &p, pass).await {
        Ok(g) => g,
        Err(e) => return e.response(),
    };
    dump(&app, &req, &p, &gen);
    let id = format!("chatcmpl-{}", app.random_hex(24));
    let created = now_secs();
    let model = app.cfg.name_of(p.variant).to_string();
    // non-streaming
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    let mut out = Vec::new();
    let (finish, usage) = loop {
        let end = gen.next(&mut out).await;
        for d in out.drain(..) {
            match d {
                Delta::Reasoning(s) => reasoning.push_str(&s),
                Delta::Content(s) => content.push_str(&s),
                Delta::ToolCall(_, name, args) => calls.push(tool_call_json(app.cfg.encoding, None, app.random_hex(24), &name, args)),
            }
        }
        match end {
            None => continue,
            Some(Ok(x)) => break x,
            Some(Err(e)) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
        }
    };
    let mut message = json!({"role": "assistant",
        "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { json!(content) }});
    if p.mode == ThinkingMode::Thinking {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    let body = json!({"id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": null, "finish_reason": finish.as_str()}],
        "usage": usage.json()});
    (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response()
}

/// The request's record, with --dump-dir.
pub(crate) fn dump(app: &App, req: &Value, p: &Prepared, gen: &Gen) {
    if let Some(dir) = &app.cfg.dump_dir {
        let rec = json!({"request": req, "prompt": p.prompt, "ids": gen.prompt_ids, "session": gen.session_id, "req_id": gen.req_id});
        if let Err(e) = std::fs::write(dir.join(format!("{:06}.json", app.dump_base + gen.req_id)), rec.to_string()) {
            eprintln!("dump: {e}");
        }
    }
}

/// A streamed answer. Without a pass it waits at the gate first: the stream starts at once with a
/// comment saying why, and a refusal (the hold limit, a shutdown) is its error event. Any stream gets
/// a comment after KEEP_ALIVE without an event (waiting at the gate, or for a long prefill).
fn stream(app: Arc<App>, p: Prepared, req: Value, pass: Option<Pass>, ticket: Option<Ticket>) -> Response {
    let (tx, rx) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let last = |e: ApiError| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(Event::default().data(e.json().to_string())).await;
                let _ = tx.send(Event::default().data("[DONE]")).await;
            }
        };
        let pass = match pass {
            Some(pass) => pass,
            None => {
                // with engine switching: the DeepSeek engine's turn first
                let ticket = match (ticket, app.switcher()) {
                    (Some(t), _) => Some(t),
                    (None, None) => None,
                    (None, Some(sw)) => {
                        let note = format!("waiting for the DeepSeek engine ({})", sw.status());
                        if tx.send(Event::default().comment(note)).await.is_err() {
                            return;
                        }
                        tokio::select! {
                            r = ds_admit(&app) => match r {
                                Ok(t) => t,
                                Err(e) => return last(e).await,
                            },
                            _ = tx.closed() => return,
                        }
                    }
                };
                let switched = ticket.is_some() && app.gate.open(app.engine.info().is_some());
                let why = if app.gate.held() { "the ring is lent to a development job" } else { "the engine is not connected" };
                let note = format!("waiting for the engine ({why}), up to {} s", app.gate.hold_limit().as_secs());
                if !switched && tx.send(Event::default().comment(note)).await.is_err() {
                    return;
                }
                tokio::select! {
                    r = app.gate.enter(&app.engine) => match r {
                        Ok(pass) => pass.with_ticket(ticket),
                        Err(r) => return last(refused(r)).await,
                    },
                    _ = tx.closed() => return, // the client went away: counted out at the gate
                }
            }
        };
        let mut gen = match start(&app, &p, pass).await {
            Ok(g) => g,
            Err(e) => return last(e).await,
        };
        dump(&app, &req, &p, &gen);
        let id = format!("chatcmpl-{}", app.random_hex(24));
        let created = now_secs();
        let model = app.cfg.name_of(p.variant).to_string();
        let chunk = |delta: Value, finish: Option<&str>| {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish}]})
        };
        let send = |v: Value| tx.send(Event::default().data(v.to_string()));
        if send(chunk(json!({"role": "assistant", "content": ""}), None)).await.is_err() {
            return;
        }
        let mut out = Vec::new();
        loop {
            let end = gen.next(&mut out).await;
            for d in out.drain(..) {
                let delta = match d {
                    Delta::Reasoning(s) => json!({"reasoning_content": s}),
                    Delta::Content(s) => json!({"content": s}),
                    Delta::ToolCall(i, name, args) => {
                        json!({"tool_calls": [tool_call_json(app.cfg.encoding, Some(i), app.random_hex(24), &name, args)]})
                    }
                };
                if send(chunk(delta, None)).await.is_err() {
                    return; // client gone; dropping gen cancels the request
                }
            }
            match end {
                None => continue,
                Some(Ok((finish, usage))) => {
                    let _ = send(chunk(json!({}), Some(finish.as_str()))).await;
                    if p.include_usage {
                        let _ = send(json!({"id": id, "object": "chat.completion.chunk", "created": created,
                                            "model": model, "choices": [], "usage": usage.json()}))
                        .await;
                    }
                    break;
                }
                Some(Err(e)) => {
                    let _ = send(json!({"error": {"message": e, "type": "server_error", "param": null, "code": 500}})).await;
                    break;
                }
            }
        }
        let _ = tx.send(Event::default().data("[DONE]")).await;
    });
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let stream = futures_util::StreamExt::map(stream, Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(KEEP_ALIVE)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ServerConfig {
        ServerConfig {
            model_name: "m".into(),
            api_key: None,
            max_tokens_cap: 32768,
            default_effort: 75,
            encoding: Encoding::V41,
            default_thinking: false,
            default_temperature: 1.0,
            default_top_p: 0.95,
            loop_guard: None,
            tokenize_threads: 1,
            dump_dir: None,
            hold: Duration::from_secs(600),
            vision_addr: None,
            variant_model: None,
            glm: None,
            active_alias: None,
            active_alias_uncensored: None,
        }
    }

    #[test]
    fn efforts() {
        for (v, want) in [
            (json!("low"), 50),
            (json!("high"), 75),
            (json!("xhigh"), 75),
            (json!("max"), 100),
            (json!("medium"), 75),
            (json!(30), 30),
            (json!(0), 75),
            (json!(0.5), 50),
            (json!(80.0), 80),
            (json!("40"), 40),
            (json!(true), 75),
        ] {
            assert_eq!(effort_budget(Some(&v), 75), want, "{v}");
        }
    }

    #[test]
    fn dumps_go_on_after_the_last_run() {
        let dir = std::env::temp_dir().join(format!("dsfront-dumps-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        assert_eq!(dump_base(&dir), 0);
        for f in ["000007.json", "000041.out.json", "000040.json", "notes.json", "12a.json", "000099.json.tmp"] {
            std::fs::write(dir.join(f), "{}").unwrap();
        }
        assert_eq!(dump_base(&dir), 41);
        assert_eq!(dump_base(&dir.join("absent")), 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_model_name_picks_its_weights() {
        let c = ServerConfig { variant_model: Some("v".into()), ..cfg() };
        let req = |model: &str| json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(prepare(&c, &req("v")).unwrap().variant, 1);
        assert_eq!(prepare(&c, &req("m")).unwrap().variant, 0);
        assert_eq!(prepare(&c, &json!({"messages": [{"role": "user", "content": "hi"}]})).unwrap().variant, 0);
        assert_eq!(prepare(&cfg(), &req("v")).unwrap().variant, 0, "no variant: every name is the one model");
        assert_eq!((c.name_of(0), c.name_of(1), cfg().name_of(1)), ("m", "v", "m"));
    }

    #[test]
    fn active_engine_aliases_name_the_loaded_family() {
        let glm = crate::glm::GlmConfig {
            url: "http://127.0.0.1:8100".into(),
            upstream_model: "glm-5.3".into(),
            model: "glm-5.3".into(),
            steered_model: "glm-5.3-jspark".into(),
            profile: "ablit-behavior-v1".into(),
            scale: 1.0,
            default_effort: "high".into(),
            small_max_tokens: 4096,
            small_budget: 1024,
            small_budget_share: 0.5,
            head_wait: Duration::from_secs(3),
            min_room: 1024,
            dump_dir: None,
        };
        let c = ServerConfig {
            model_name: "deepseek-v4.1-flash".into(),
            variant_model: Some("deepseek-v4.1-flash-heretic".into()),
            glm: Some(glm),
            active_alias: Some("local-active".into()),
            active_alias_uncensored: Some("local-active-uncensored".into()),
            ..cfg()
        };
        assert_eq!((c.alias_of("local-active"), c.alias_of("local-active-uncensored"), c.alias_of("glm-5.3")), (Some(false), Some(true), None));
        // the served names stay themselves; any other name runs on the loaded engine, as local-active
        for m in ["deepseek-v4.1-flash", "deepseek-v4.1-flash-heretic", "glm-5.3-jspark"] {
            assert_eq!(c.alias_of(m), None, "{m}");
        }
        assert_eq!(c.alias_of("claude-haiku-4-5"), Some(false));
        let no_glm = ServerConfig { glm: None, ..c.clone() };
        assert_eq!(no_glm.alias_of("claude-haiku-4-5"), None, "without GLM an unknown name stays DeepSeek's");
        assert_eq!(c.alias_target(Family::DeepSeek, false), "deepseek-v4.1-flash");
        assert_eq!(c.alias_target(Family::DeepSeek, true), "deepseek-v4.1-flash-heretic");
        assert_eq!(c.alias_target(Family::Glm, false), "glm-5.3");
        assert_eq!(c.alias_target(Family::Glm, true), "glm-5.3-jspark");
        // without a variant, the uncensored alias is DeepSeek's one model
        let c = ServerConfig { variant_model: None, ..c };
        assert_eq!(c.alias_target(Family::DeepSeek, true), "deepseek-v4.1-flash");
        assert_eq!(cfg().alias_of("local-active"), None, "no alias unless configured");
    }

    #[test]
    fn prepare_like_the_kit() {
        let req: Value = serde_json::from_str(
            r#"{"messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "tools": [{"type": "function", "function": {"parameters": {"type": "object"}, "name": "f", "extra": 1}}],
                "chat_template_kwargs": {"thinking": true, "reasoning_effort": "max"}, "max_tokens": 100000, "stop": "x"}"#,
        )
        .unwrap();
        let p = prepare(&cfg(), &req).unwrap();
        assert!(p.prompt.starts_with("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 (range 1-100"));
        assert!(p.prompt.contains("be brief\n\n## Tools"));
        assert!(p.prompt.contains("{\"name\": \"f\", \"parameters\": {\"type\": \"object\"}}"));
        assert!(p.prompt.ends_with("<｜User｜>hi<｜Assistant｜><think>"));
        assert_eq!((p.max_tokens, p.tools, p.mode), (32768, true, ThinkingMode::Thinking));
        assert_eq!(p.stops, vec!["x".to_string()]);
    }
}
