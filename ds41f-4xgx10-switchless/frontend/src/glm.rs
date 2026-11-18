//! The GLM-5.3 backend (docs/glm.md): Anthropic Messages requests whose model is `--glm-model` or
//! `--glm-steered-model` go to gr8's OpenAI-style chat API (the steered name with `steering_profile`),
//! and gr8's stream comes back as Anthropic events. The request side reuses anthropic.rs's translation,
//! then sets what GLM-5.3 and Claude Code need:
//! - effort: Claude Code's levels mapped explicitly (`effort`); the effort is always sent, so gr8's
//!   fallback (max) never applies by accident;
//! - `clear_thinking: false`: every replayed assistant turn keeps its reasoning (the template's default,
//!   pinned), so a tool loop's reasoning stays in the prompt and the prompt grows only at its end;
//! - small requests (thinking disabled, a small max_tokens, or stop sequences: Claude Code's titles and
//!   other background calls) get low effort and a thinking budget that leaves room for the answer within
//!   their max_tokens: GLM-5.3 always thinks, and only the budget bounds it;
//! - stop sequences are matched here, on the visible text only: gr8 would match them in the reasoning too;
//! - a prompt over the window comes back as "prompt is too long", Claude Code's cue to compact; a prompt
//!   that fits but leaves too little room for max_tokens is retried once with max_tokens cut to the room.
//! Streams carry ping events while they wait (an engine switch, a long prefill: gr8 answers its headers
//! only once the first token is out).

use crate::anthropic::{self, error, error_json, error_kind, message_id, sse, usage_json, Blocks, PING_EVERY};
use crate::encoding::Delta;
use crate::server::{App, Usage, KEEP_ALIVE};
use crate::switch::{Family, Ticket};
use axum::body::{Body, Bytes};
use axum::http::{header, Request, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct GlmConfig {
    /// gr8's base URL: http://HOST:PORT
    pub url: String,
    /// the model name gr8 serves (its --served-model-name)
    pub upstream_model: String,
    /// the client-facing names: stock, and with the steering profile
    pub model: String,
    pub steered_model: String,
    pub profile: String,
    pub scale: f64,
    /// effort when a thinking request names none
    pub default_effort: String,
    /// a request with max_tokens up to this (or thinking disabled, or stop sequences) is small
    pub small_max_tokens: u64,
    /// a small request's thinking budget: at most this many tokens...
    pub small_budget: u64,
    /// ...and at most this share of its max_tokens, so that the answer has the rest
    pub small_budget_share: f64,
    /// how long a stream waits for gr8's first answer before it starts with pings
    pub head_wait: Duration,
    /// the least room for the answer: a prompt that leaves less is "too long"
    pub min_room: u64,
}

impl GlmConfig {
    /// Some(steered) when the model name is one of ours.
    pub fn check_url(url: &str) -> Result<(), String> {
        parse_base(url).map(|_| ())
    }

    pub fn route(&self, model: Option<&str>) -> Option<bool> {
        match model? {
            m if m == self.model => Some(false),
            m if m == self.steered_model => Some(true),
            _ => None,
        }
    }
}

// ------------------------------------------------------------------ HTTP client to gr8

struct Base {
    host: String,
    port: u16,
    authority: String,
    prefix: String,
}

fn parse_base(url: &str) -> Result<Base, String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("{url}: only http:// URLs"))?;
    let (authority, prefix) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
        None => (rest, ""),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.ends_with(']') || h.starts_with('[') => (h, p.parse::<u16>().map_err(|_| format!("{url}: bad port"))?),
        _ => (authority, 80),
    };
    let host = host.trim_start_matches('[').trim_end_matches(']').to_string();
    Ok(Base { host, port, authority: authority.to_string(), prefix: prefix.to_string() })
}

/// A response from gr8: its status and headers, its body read frame by frame. Dropping it closes the
/// connection, which ends the request on gr8 (it cancels a generation whose client went away).
pub(crate) struct Upstream {
    pub status: u16,
    pub headers: axum::http::HeaderMap,
    body: hyper::body::Incoming,
    conn: tokio::task::JoinHandle<()>,
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.conn.abort();
    }
}

impl Upstream {
    /// The next piece of the body; None at its end.
    pub async fn chunk(&mut self) -> Option<Result<Bytes, String>> {
        use http_body::Body as _;
        loop {
            let f = std::future::poll_fn(|cx| std::pin::Pin::new(&mut self.body).poll_frame(cx)).await?;
            match f {
                Err(e) => return Some(Err(e.to_string())),
                Ok(f) => {
                    if let Ok(d) = f.into_data() {
                        return Some(Ok(d));
                    }
                }
            }
        }
    }

    /// The whole body, up to `limit` bytes.
    pub async fn read_all(&mut self, limit: usize) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        while let Some(c) = self.chunk().await {
            out.extend_from_slice(&c?);
            if out.len() > limit {
                break;
            }
        }
        Ok(out)
    }
}

pub(crate) async fn send(url: &str, method: &str, path: &str, body: String) -> Result<Upstream, String> {
    let b = parse_base(url)?;
    let stream = tokio::net::TcpStream::connect((b.host.as_str(), b.port)).await.map_err(|e| format!("{}: {e}", b.authority))?;
    let _ = stream.set_nodelay(true);
    let (mut tx, conn) = hyper::client::conn::http1::handshake::<_, String>(hyper_util::rt::TokioIo::new(stream)).await.map_err(|e| e.to_string())?;
    let conn = tokio::spawn(async move {
        let _ = conn.await;
    });
    let req = Request::builder()
        .method(method)
        .uri(format!("{}{path}", b.prefix))
        .header(header::HOST, &b.authority)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .body(body)
        .map_err(|e| e.to_string())?;
    match tx.send_request(req).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            Ok(Upstream { status: parts.status.as_u16(), headers: parts.headers, body, conn })
        }
        Err(e) => {
            conn.abort();
            Err(e.to_string())
        }
    }
}

/// gr8's /health: Some(its JSON) when it answers 200 within `timeout` (it serves).
pub async fn health_json(url: &str, timeout: Duration) -> Option<Value> {
    let r = tokio::time::timeout(timeout, async {
        let mut u = send(url, "GET", "/health", String::new()).await.ok()?;
        if u.status != 200 {
            return None;
        }
        let b = u.read_all(1 << 16).await.ok()?;
        Some(serde_json::from_slice(&b).unwrap_or(Value::Null))
    })
    .await;
    r.ok().flatten()
}

/// gr8's context window from its /v1/models (None while it does not answer; remembered after).
pub async fn max_model_len(cfg: &GlmConfig) -> Option<u64> {
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let got = tokio::time::timeout(Duration::from_secs(1), async {
        let mut u = send(&cfg.url, "GET", "/v1/models", String::new()).await.ok()?;
        let v: Value = serde_json::from_slice(&u.read_all(1 << 20).await.ok()?).ok()?;
        v["data"].as_array()?.iter().find(|m| m["id"] == cfg.upstream_model.as_str())?["max_model_len"].as_u64()
    })
    .await
    .ok()
    .flatten();
    match got {
        Some(n) => {
            SEEN.store(n, std::sync::atomic::Ordering::Relaxed);
            Some(n)
        }
        None => Some(SEEN.load(std::sync::atomic::Ordering::Relaxed)).filter(|&n| n > 0),
    }
}

pub async fn health(url: &str, timeout: Duration) -> bool {
    health_json(url, timeout).await.is_some()
}

/// Server-sent events: the `data` of each event, as the bytes arrive.
#[derive(Default)]
pub(crate) struct SseData {
    buf: Vec<u8>,
}

impl SseData {
    pub fn push(&mut self, b: &[u8], out: &mut Vec<String>) {
        self.buf.extend_from_slice(b);
        loop {
            let end = self.buf.windows(2).position(|w| w == b"\n\n");
            let end_crlf = self.buf.windows(4).position(|w| w == b"\r\n\r\n");
            let (at, len) = match (end, end_crlf) {
                (Some(a), Some(c)) if c < a => (c, 4),
                (Some(a), _) => (a, 2),
                (None, Some(c)) => (c, 4),
                (None, None) => return,
            };
            let ev: Vec<u8> = self.buf.drain(..at + len).collect();
            let text = String::from_utf8_lossy(&ev[..at]);
            let mut data: Vec<&str> = Vec::new();
            for line in text.lines() {
                let line = line.trim_end_matches('\r');
                if let Some(d) = line.strip_prefix("data:") {
                    data.push(d.strip_prefix(' ').unwrap_or(d));
                }
            }
            if !data.is_empty() {
                out.push(data.join("\n"));
            }
        }
    }
}

// ------------------------------------------------------------------ request → gr8 chat request

/// Claude Code's effort level as GLM-5.3's. The template names Low, High and Max; gr8 takes Max when
/// a request names none. Every level is mapped here explicitly: low → low, medium and high → high,
/// xhigh and max → max; none, missing or a name neither side knows → the default (high), never Max
/// by accident.
pub fn effort(v: Option<&Value>, default: &str) -> String {
    match v.and_then(Value::as_str) {
        None | Some("none") => default.to_string(),
        Some("low") => "low".into(),
        Some("medium") | Some("high") => "high".into(),
        Some("xhigh") | Some("max") => "max".into(),
        Some(other) => {
            eprintln!("glm: effort {other:?} is not one of low, medium, high, xhigh, max: {default}");
            default.to_string()
        }
    }
}

/// A request translated for gr8, and what the answer needs.
pub struct Plan {
    pub body: Value,
    /// the name the client asked for, echoed back
    pub model: String,
    /// the client asked for thinking: reasoning goes out as thinking blocks; otherwise it is withheld
    pub show_thinking: bool,
    pub omit_thinking: bool,
    pub stops: Vec<String>,
    pub stream: bool,
    pub steered: bool,
    /// the client's max_tokens (the request to gr8 may carry more: the small requests' budget)
    pub max_tokens: u64,
}

pub fn to_gr8(req: &Value, cfg: &GlmConfig, steered: bool) -> Result<Plan, String> {
    let conv = anthropic::convert_for(req, false, false, true)?;
    let r = req.as_object().ok_or("the request body must be a JSON object")?;
    let Value::Object(mut chat) = conv.chat else { unreachable!() };
    for k in ["chat_template_kwargs", "reasoning_effort", "stop", "stream", "model", "max_tokens"] {
        chat.remove(k);
    }
    let thinking = r.get("thinking").and_then(Value::as_object);
    let kind = thinking.and_then(|t| t.get("type")).and_then(Value::as_str);
    let thinking_on = matches!(kind, Some("enabled" | "adaptive"));
    let max_tokens = match r.get("max_tokens").filter(|v| !v.is_null()) {
        None => 4096,
        Some(v) => v.as_u64().filter(|&n| n >= 1).ok_or("max_tokens must be a positive integer")?,
    };
    let stops: Vec<String> = match r.get("stop_sequences") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string).collect(),
        Some(_) => return Err("stop_sequences must be a list of strings".into()),
    };
    // tool_choice: none is set by the translation; any and tool force a call (gr8's insert rule)
    let mut forced = false;
    match r.get("tool_choice").and_then(|c| c.get("type")).and_then(Value::as_str) {
        Some("any") if chat.contains_key("tools") => {
            chat.insert("tool_choice".into(), json!("required"));
            forced = true;
        }
        Some("tool") if chat.contains_key("tools") => {
            let name = r["tool_choice"].get("name").cloned().unwrap_or(json!(""));
            chat.insert("tool_choice".into(), json!({"type": "function", "function": {"name": name}}));
            forced = true;
        }
        _ => {}
    }
    // structured output: gr8's response_format (enforced after generation, or by grammar without tools)
    if let Some(schema) = r.get("output_config").and_then(|o| o.get("format")).filter(|f| f.get("type").and_then(Value::as_str) == Some("json_schema")).and_then(|f| f.get("schema")) {
        chat.insert("response_format".into(), json!({"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}));
    }
    // GLM-5.3 always thinks: its template has no thinking off (gr8's enable_thinking false only
    // withholds the reasoning, which still takes its tokens), so only thinking_budget bounds it: gr8
    // closes the think block in place of the budget's Nth token. Thinking disabled (thinking.type
    // disabled, or no thinking field: Claude Code's background requests) is low effort with a budget,
    // and the reasoning stays out of the reply (`show_thinking`). A small request (that, a small
    // max_tokens, or stop sequences) gets a budget of at most --glm-small-budget tokens and
    // --glm-small-budget-share of its max_tokens, so that its answer fits in the rest.
    let small = !thinking_on || max_tokens <= cfg.small_max_tokens || !stops.is_empty();
    let budget = if forced {
        None // a forced call carries no reasoning (and gr8 refuses a budget beside it)
    } else if small {
        Some(cfg.small_budget.min((max_tokens as f64 * cfg.small_budget_share) as u64).max(1))
    } else {
        // thinking.type enabled with budget_tokens: kept when it leaves room for an answer
        thinking.and_then(|t| t.get("budget_tokens")).and_then(Value::as_u64).filter(|&b| b >= 1 && b + cfg.min_room <= max_tokens)
    };
    chat.insert("model".into(), json!(cfg.upstream_model));
    chat.insert("max_tokens".into(), json!(max_tokens));
    let level = if thinking_on { effort(r.get("output_config").and_then(|o| o.get("effort")), &cfg.default_effort) } else { "low".into() };
    chat.insert("reasoning_effort".into(), json!(level));
    chat.insert("chat_template_kwargs".into(), json!({"clear_thinking": false}));
    if let Some(b) = budget {
        chat.insert("thinking_budget".into(), json!(b));
    }
    chat.insert("stream".into(), json!(true));
    chat.insert("stream_options".into(), json!({"include_usage": true, "continuous_usage_stats": true}));
    if steered {
        chat.insert("steering_profile".into(), json!(cfg.profile));
        chat.insert("steering_scale".into(), json!(cfg.scale));
    }
    Ok(Plan {
        body: Value::Object(chat),
        model: conv.model,
        show_thinking: thinking_on,
        omit_thinking: conv.omit_thinking,
        stops,
        stream: crate::pyjson::truthy(r.get("stream")),
        steered,
        max_tokens,
    })
}

/// gr8's /tokenize body for a request: the chat request's messages, tools and template arguments (its
/// effort line opens the prompt), as the prompt gr8 would render, without what only generation uses.
pub fn tokenize_body(plan: &Plan) -> Value {
    let mut body = plan.body.clone();
    if let Some(o) = body.as_object_mut() {
        for k in ["stream", "stream_options", "max_tokens", "thinking_budget", "tool_choice", "response_format", "steering_profile", "steering_scale"] {
            o.remove(k);
        }
        o.insert("add_generation_prompt".into(), json!(true));
    }
    body
}

/// A GLM request's prompt tokens by gr8's own tokenizer (its /tokenize). None unless GLM is the
/// engine loaded and gr8 answers within a few seconds: counting never switches the engines, and the
/// caller falls back to DeepSeek's tokenizer, an estimate.
pub async fn count_tokens(app: &App, req: &Value, steered: bool) -> Option<u64> {
    let cfg = app.cfg.glm.as_ref()?;
    if app.switcher().is_some_and(|s| s.active() != Some(Family::Glm) || s.switching().is_some()) {
        return None;
    }
    let body = tokenize_body(&to_gr8(req, cfg, steered).ok()?);
    let n = tokio::time::timeout(Duration::from_secs(10), async {
        let mut u = send(&cfg.url, "POST", "/tokenize", body.to_string()).await.ok()?;
        if u.status != 200 {
            return None;
        }
        let v: Value = serde_json::from_slice(&u.read_all(1 << 26).await.ok()?).ok()?;
        v.get("count").and_then(Value::as_u64).or_else(|| v.get("tokens")?.as_array().map(|a| a.len() as u64))
    })
    .await
    .ok()
    .flatten();
    if n.is_none() {
        eprintln!("glm: gr8's /tokenize did not answer; count_tokens estimates with DeepSeek's tokenizer");
    }
    n
}

// ------------------------------------------------------------------ gr8's errors

/// "This model's maximum context length is X tokens. However, you requested Y tokens (P in the
/// messages, M in the completion)...": (X, P, M).
pub fn parse_overflow(msg: &str) -> Option<(u64, u64, u64)> {
    let num = |s: &str| s.trim().parse::<u64>().ok();
    let x = num(msg.split("maximum context length is ").nth(1)?.split(" tokens").next()?)?;
    let inner = msg.split('(').nth(1)?;
    let p = num(inner.split(" in the messages").next()?)?;
    let m = num(inner.split(", ").nth(1)?.split(" in the completion").next()?)?;
    Some((x, p, m))
}

/// A failure, in Anthropic's terms: HTTP status, error type, message, Retry-After.
pub struct Fail {
    pub status: StatusCode,
    pub message: String,
    pub retry_after: Option<String>,
}

impl Fail {
    fn new(status: u16, message: impl Into<String>) -> Fail {
        Fail { status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), message: message.into(), retry_after: None }
    }
    fn response(&self) -> Response {
        let mut r = error(self.status, self.message.clone());
        if let Some(ra) = self.retry_after.as_deref().and_then(|v| v.parse().ok()) {
            r.headers_mut().insert("retry-after", ra);
        }
        r
    }
    fn event(&self) -> Event {
        // 503 is Anthropic's 529 (overloaded) in a stream too
        let status = if self.status == StatusCode::SERVICE_UNAVAILABLE { StatusCode::from_u16(529).unwrap() } else { self.status };
        sse("error", &error_json(error_kind(status), &self.message))
    }
}

/// gr8's error body {"error": {"message", "code"}}: its message.
fn error_message(body: &[u8]) -> String {
    let v: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    v.pointer("/error/message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| String::from_utf8_lossy(body).chars().take(500).collect())
}

/// What to do with gr8's refusal: retry with max_tokens cut to the room left, or fail.
enum Refusal {
    Retry(u64),
    Fail(Fail),
}

fn refusal(status: u16, msg: String, retry_after: Option<String>, cfg: &GlmConfig, client_max: u64, retried: bool) -> Refusal {
    if status == 400 {
        if let Some((x, p, m)) = parse_overflow(&msg) {
            let need = cfg.min_room.min(m).min(client_max).max(1);
            let room = x.saturating_sub(p);
            if room >= need && room < m && !retried {
                return Refusal::Retry(room);
            }
            // Claude Code compacts on this message ("prompt is too long: N tokens > M maximum")
            let max = if p >= x { x } else { x.saturating_sub(need) };
            return Refusal::Fail(Fail::new(400, format!("prompt is too long: {p} tokens > {max} maximum")));
        }
    }
    let status = match status {
        // gr8 down, or its engine not connected: overloaded, which clients retry
        502..=504 => 503,
        s => s,
    };
    let mut f = Fail::new(status, format!("gr8: {msg}"));
    f.retry_after = retry_after;
    Refusal::Fail(f)
}

// ------------------------------------------------------------------ gr8's stream → Anthropic events

/// Stop sequences over the visible text: the text that can go out, and the stop string that ended it.
struct StopScan {
    stops: Vec<String>,
    buf: String,
}

impl StopScan {
    fn push(&mut self, s: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (s.to_string(), None);
        }
        self.buf.push_str(s);
        if let Some((p, st)) = self.stops.iter().filter_map(|st| self.buf.find(st.as_str()).map(|p| (p, st))).min_by_key(|x| x.0) {
            let out = self.buf[..p].to_string();
            let st = st.clone();
            self.buf.clear();
            return (out, Some(st));
        }
        // hold back a tail that may start a stop string
        let mut keep = 0;
        for st in &self.stops {
            for k in (1..st.len().min(self.buf.len() + 1)).rev() {
                let at = self.buf.len() - k;
                if self.buf.is_char_boundary(at) && st.starts_with(&self.buf[at..]) {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let n = self.buf.len() - keep;
        let out = self.buf[..n].to_string();
        self.buf.drain(..n);
        (out, None)
    }
}

type Ev = (&'static str, Value);

#[derive(Default, Clone, Copy, Debug)]
pub struct GrUsage {
    pub prompt: u64,
    pub completion: u64,
    pub cached: u64,
    pub reasoning: u64,
}

impl GrUsage {
    fn anthropic(&self) -> Value {
        usage_json(&Usage { prompt: self.prompt as usize, cached: self.cached as u32, completion: self.completion as usize, reasoning: self.reasoning as usize })
    }
}

/// gr8's chunks in, Anthropic content blocks and events out.
pub struct Translator {
    pub blocks: Blocks,
    show_thinking: bool,
    stop: StopScan,
    /// a tool call being streamed: (index, name, arguments so far)
    call: Option<(u64, String, String)>,
    calls: usize,
    pub usage: GrUsage,
    finish: Option<String>,
    pub stop_sequence: Option<String>,
    ids: Box<dyn FnMut() -> String + Send>,
}

pub enum Step {
    More,
    /// a stop sequence ended the answer: the rest of gr8's stream is not needed
    Stopped,
}

impl Translator {
    pub fn new(plan: &Plan, ids: Box<dyn FnMut() -> String + Send>) -> Translator {
        Translator {
            blocks: Blocks::new(plan.omit_thinking),
            show_thinking: plan.show_thinking,
            stop: StopScan { stops: plan.stops.clone(), buf: String::new() },
            call: None,
            calls: 0,
            usage: GrUsage::default(),
            finish: None,
            stop_sequence: None,
            ids,
        }
    }

    fn flush_call(&mut self, ev: &mut Vec<Ev>) {
        if let Some((i, name, args)) = self.call.take() {
            let args = if args.trim().is_empty() { "{}".to_string() } else { args };
            self.calls += 1;
            let ids = &mut self.ids;
            self.blocks.push(Delta::ToolCall(i as usize, name, args), || ids(), ev);
        }
    }

    /// One chunk of gr8's stream (the parsed `data` of an event).
    pub fn feed(&mut self, v: &Value, ev: &mut Vec<Ev>) -> Result<Step, Fail> {
        if let Some(e) = v.get("error") {
            let code = e.get("code").and_then(Value::as_u64).unwrap_or(500) as u16;
            let msg = e.get("message").and_then(Value::as_str).unwrap_or("gr8 error").to_string();
            return Err(Fail::new(if code < 400 { 500 } else { code }, format!("gr8: {msg}")));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            let n = |p: &str| u.pointer(p).and_then(Value::as_u64);
            self.usage.prompt = n("/prompt_tokens").unwrap_or(self.usage.prompt);
            self.usage.completion = n("/completion_tokens").unwrap_or(self.usage.completion);
            self.usage.cached = n("/prompt_tokens_details/cached_tokens").unwrap_or(self.usage.cached);
            self.usage.reasoning = n("/completion_tokens_details/reasoning_tokens").unwrap_or(self.usage.reasoning);
        }
        let Some(c) = v.get("choices").and_then(|c| c.get(0)) else { return Ok(Step::More) };
        if self.stop_sequence.is_some() {
            return Ok(Step::Stopped);
        }
        if let Some(d) = c.get("delta").and_then(Value::as_object) {
            // vLLM's name for it, else the older one (gr8's --reasoning-fields)
            let reasoning = d.get("reasoning").and_then(Value::as_str).or_else(|| d.get("reasoning_content").and_then(Value::as_str));
            if let Some(r) = reasoning.filter(|r| !r.is_empty()) {
                if self.show_thinking {
                    self.flush_call(ev);
                    self.blocks.push(Delta::Reasoning(r.to_string()), String::new, ev);
                }
            }
            if let Some(t) = d.get("content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                self.flush_call(ev);
                let (pass, hit) = self.stop.push(t);
                self.blocks.push(Delta::Content(pass), String::new, ev);
                if let Some(s) = hit {
                    self.stop_sequence = Some(s);
                    return Ok(Step::Stopped);
                }
            }
            if let Some(Value::Array(tcs)) = d.get("tool_calls") {
                for tc in tcs {
                    let i = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                    if self.call.as_ref().is_some_and(|c| c.0 != i) {
                        self.flush_call(ev);
                    }
                    let call = self.call.get_or_insert_with(|| (i, String::new(), String::new()));
                    if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str) {
                        call.1.push_str(n);
                    }
                    if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                        call.2.push_str(a);
                    }
                }
            }
        }
        if let Some(f) = c.get("finish_reason").and_then(Value::as_str) {
            self.finish = Some(f.to_string());
        }
        Ok(Step::More)
    }

    /// The end of the stream: what was held back goes out, the open block closes. The stop reason.
    pub fn finish(&mut self, ev: &mut Vec<Ev>) -> &'static str {
        if self.stop_sequence.is_none() {
            let tail = std::mem::take(&mut self.stop.buf);
            self.blocks.push(Delta::Content(tail), String::new, ev);
        }
        self.flush_call(ev);
        self.blocks.close(ev);
        if self.stop_sequence.is_some() {
            "stop_sequence"
        } else if self.calls > 0 {
            "tool_use"
        } else if self.finish.as_deref() == Some("length") {
            "max_tokens"
        } else {
            "end_turn"
        }
    }
}

// ------------------------------------------------------------------ the request

/// What the generation sends to the HTTP side.
enum Msg {
    /// the first chunk came: message_start's usage
    Start(GrUsage),
    Ev(&'static str, Value),
    /// the end: stop reason, stop sequence, usage, and the content blocks for an answer that is not streamed
    Done(&'static str, Option<String>, GrUsage, Vec<Value>),
    Fail(Fail),
}

/// Wait for the GLM engine (a switch), send the request to gr8 (retrying once with a smaller max_tokens
/// when only the answer's room was short), and translate its stream into `tx`.
async fn drive(app: Arc<App>, plan: Plan, tx: mpsc::Sender<Msg>) {
    let cfg = app.cfg.glm.clone().expect("glm configured");
    let _ticket: Option<Ticket> = match app.switcher() {
        None => None,
        Some(sw) => {
            let t = tokio::select! {
                t = sw.admit(Family::Glm) => t,
                _ = tx.closed() => return,
            };
            match t {
                Ok(t) => Some(t),
                Err(e) => {
                    let _ = tx.send(Msg::Fail(Fail::new(503, e))).await;
                    return;
                }
            }
        }
    };
    let mut body = plan.body.clone();
    let mut retried = false;
    let t0 = std::time::Instant::now();
    let mut up = loop {
        let r = tokio::select! {
            r = send(&cfg.url, "POST", "/v1/chat/completions", body.to_string()) => r,
            _ = tx.closed() => return,
        };
        let mut up = match r {
            Ok(u) => u,
            Err(e) => {
                if let Some(sw) = app.switcher() {
                    sw.lost(Family::Glm, &e);
                }
                let _ = tx.send(Msg::Fail(Fail::new(503, format!("the GLM engine (gr8 at {}) does not answer: {e}", cfg.url)))).await;
                return;
            }
        };
        if up.status == 200 {
            break up;
        }
        let ra = up.headers.get("retry-after").and_then(|v| v.to_str().ok()).map(str::to_string);
        let msg = error_message(&up.read_all(1 << 16).await.unwrap_or_default());
        match refusal(up.status, msg.clone(), ra, &cfg, plan.max_tokens, retried) {
            Refusal::Retry(room) => {
                eprintln!("glm: {msg}; again with max_tokens {room}");
                body["max_tokens"] = json!(room);
                if body.get("thinking_budget").and_then(Value::as_u64).is_some_and(|b| b >= room) {
                    body["thinking_budget"] = json!((room / 2).max(1));
                }
                retried = true;
            }
            Refusal::Fail(f) => {
                eprintln!("glm: gr8 answered {}: {msg}", up.status);
                let _ = tx.send(Msg::Fail(f)).await;
                return;
            }
        }
    };
    let app2 = app.clone();
    let mut tr = Translator::new(&plan, Box::new(move || app2.random_hex(24)));
    let mut sse = SseData::default();
    let mut started = false;
    let mut datas = Vec::new();
    let mut ev = Vec::new();
    let mut done = false;
    'read: loop {
        let c = tokio::select! {
            c = up.chunk() => c,
            _ = tx.closed() => return, // dropping `up` ends the generation on gr8
        };
        let bytes = match c {
            None => break,
            Some(Ok(b)) => b,
            Some(Err(e)) => {
                let _ = tx.send(Msg::Fail(Fail::new(500, format!("gr8 stream: {e}")))).await;
                return;
            }
        };
        sse.push(&bytes, &mut datas);
        for d in datas.drain(..) {
            if d.trim() == "[DONE]" {
                done = true;
                break 'read;
            }
            let v: Value = match serde_json::from_str(&d) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("glm: a chunk that is not JSON ({e}): {}", d.chars().take(200).collect::<String>());
                    continue;
                }
            };
            let step = tr.feed(&v, &mut ev);
            if !started {
                started = true;
                if tx.send(Msg::Start(tr.usage)).await.is_err() {
                    return;
                }
            }
            for (k, e) in ev.drain(..) {
                if tx.send(Msg::Ev(k, e)).await.is_err() {
                    return;
                }
            }
            match step {
                Ok(Step::More) => {}
                Ok(Step::Stopped) => break 'read, // dropping `up` ends the generation on gr8
                Err(f) => {
                    let _ = tx.send(Msg::Fail(f)).await;
                    return;
                }
            }
        }
    }
    drop(up);
    if !done && tr.stop_sequence.is_none() {
        let _ = tx.send(Msg::Fail(Fail::new(500, "gr8's stream ended early"))).await;
        return;
    }
    let reason = tr.finish(&mut ev);
    for (k, e) in ev.drain(..) {
        if tx.send(Msg::Ev(k, e)).await.is_err() {
            return;
        }
    }
    let u = tr.usage;
    eprintln!(
        "glm{}: prompt {} (cached {}), completion {} (reasoning {}), {reason}; {:.3} s",
        if plan.steered { " (steered)" } else { "" },
        u.prompt,
        u.cached,
        u.completion,
        u.reasoning,
        t0.elapsed().as_secs_f64()
    );
    let content = std::mem::take(&mut tr.blocks.content);
    let _ = tx.send(Msg::Done(reason, tr.stop_sequence.clone(), u, content)).await;
}

/// POST /v1/messages for a GLM model name.
pub async fn messages(app: Arc<App>, req: Value, steered: bool) -> Response {
    let cfg = app.cfg.glm.clone().expect("glm configured");
    let plan = match to_gr8(&req, &cfg, steered) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let (model, stream) = (plan.model.clone(), plan.stream);
    let (tx, mut rx) = mpsc::channel::<Msg>(256);
    tokio::spawn(drive(app.clone(), plan, tx));
    if !stream {
        let (mut start, mut done) = (None, None);
        while let Some(m) = rx.recv().await {
            match m {
                Msg::Start(u) => start = Some(u),
                Msg::Ev(..) => {}
                Msg::Done(r, s, u, c) => {
                    done = Some((r, s, u, c));
                    break;
                }
                Msg::Fail(f) => return f.response(),
            }
        }
        let Some((reason, stop_seq, usage, content)) = done else {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "the GLM request ended without an answer");
        };
        let _ = start;
        let body = json!({"id": message_id(&app), "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": reason, "stop_sequence": stop_seq, "usage": usage.anthropic()});
        return (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response();
    }
    // A refusal that comes at once (a prompt over the window, a bad request) is the HTTP status; past
    // head_wait (an engine switch, a long prefill) the stream starts and waits with pings.
    let first = match tokio::time::timeout(cfg.head_wait, rx.recv()).await {
        Ok(Some(Msg::Fail(f))) => return f.response(),
        Ok(Some(m)) => Some(m),
        Ok(None) => return error(StatusCode::INTERNAL_SERVER_ERROR, "the GLM request ended without an answer"),
        Err(_) => None,
    };
    let (etx, erx) = mpsc::channel::<Event>(256);
    let id = message_id(&app);
    tokio::spawn(async move {
        let mut started = false;
        let start_ev = |u: &GrUsage| {
            sse("message_start", &json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": null, "stop_sequence": null, "usage": u.anthropic()}}))
        };
        let ping = || sse("ping", &json!({"type": "ping"}));
        let mut pending = first;
        let mut quiet = tokio::time::interval_at(tokio::time::Instant::now() + PING_EVERY, PING_EVERY);
        if pending.is_none() {
            // waited head_wait already: start now, so that the client sees the stream alive
            started = true;
            if etx.send(start_ev(&GrUsage::default())).await.is_err() || etx.send(ping()).await.is_err() {
                return;
            }
        }
        loop {
            let m = match pending.take() {
                Some(m) => m,
                None => tokio::select! {
                    m = rx.recv() => match m { Some(m) => m, None => return },
                    _ = quiet.tick() => {
                        if !started {
                            started = true;
                            if etx.send(start_ev(&GrUsage::default())).await.is_err() { return; }
                        }
                        if etx.send(ping()).await.is_err() { return; }
                        continue;
                    }
                    _ = etx.closed() => return, // the client went away: dropping rx ends the generation
                },
            };
            quiet.reset();
            match m {
                Msg::Start(u) => {
                    if !started {
                        started = true;
                        if etx.send(start_ev(&u)).await.is_err() {
                            return;
                        }
                        let _ = etx.send(ping()).await;
                    }
                }
                Msg::Ev(k, v) => {
                    if etx.send(sse(k, &v)).await.is_err() {
                        return;
                    }
                }
                Msg::Done(reason, stop_seq, u, _) => {
                    let delta = json!({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": stop_seq}, "usage": u.anthropic()});
                    let _ = etx.send(sse("message_delta", &delta)).await;
                    let _ = etx.send(sse("message_stop", &json!({"type": "message_stop"}))).await;
                    return;
                }
                Msg::Fail(f) => {
                    let _ = etx.send(f.event()).await;
                    return;
                }
            }
        }
    });
    let stream = tokio_stream::wrappers::ReceiverStream::new(erx);
    let stream = futures_util::StreamExt::map(stream, Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(KEEP_ALIVE)).into_response()
}

/// POST /v1/chat/completions for a GLM model name: gr8 speaks OpenAI's chat API itself, so the request
/// goes through as it is, with gr8's model name and, for the steered name, the steering profile.
pub async fn chat(app: Arc<App>, mut req: Value, steered: bool) -> Response {
    let cfg = app.cfg.glm.clone().expect("glm configured");
    let oai = |status: StatusCode, msg: String| {
        let kind = if status.is_server_error() { "server_error" } else { "invalid_request_error" };
        (status, [("content-type", "application/json")], json!({"error": {"message": msg, "type": kind, "param": null, "code": status.as_u16()}}).to_string())
            .into_response()
    };
    let Some(o) = req.as_object_mut() else { return oai(StatusCode::BAD_REQUEST, "the request body must be a JSON object".into()) };
    o.insert("model".into(), json!(cfg.upstream_model));
    if steered {
        o.insert("steering_profile".into(), json!(cfg.profile));
        o.insert("steering_scale".into(), json!(cfg.scale));
    }
    let ticket = match app.switcher() {
        None => None,
        Some(sw) => match sw.admit(Family::Glm).await {
            Ok(t) => Some(t),
            Err(e) => return oai(StatusCode::SERVICE_UNAVAILABLE, e),
        },
    };
    let mut up = match send(&cfg.url, "POST", "/v1/chat/completions", req.to_string()).await {
        Ok(u) => u,
        Err(e) => {
            if let Some(sw) = app.switcher() {
                sw.lost(Family::Glm, &e);
            }
            return oai(StatusCode::SERVICE_UNAVAILABLE, format!("the GLM engine (gr8 at {}) does not answer: {e}", cfg.url));
        }
    };
    let status = StatusCode::from_u16(up.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let ctype = up.headers.get(header::CONTENT_TYPE).cloned();
    let (btx, brx) = mpsc::channel::<Result<Bytes, Infallible>>(64);
    tokio::spawn(async move {
        let _ticket = ticket;
        loop {
            let c = tokio::select! {
                c = up.chunk() => c,
                _ = btx.closed() => return,
            };
            match c {
                Some(Ok(b)) => {
                    if btx.send(Ok(b)).await.is_err() {
                        return;
                    }
                }
                _ => return,
            }
        }
    });
    let mut resp = Response::new(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(brx)));
    *resp.status_mut() = status;
    if let Some(c) = ctype {
        resp.headers_mut().insert(header::CONTENT_TYPE, c);
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_body_keeps_the_prompt() {
        let cfg = cfg();
        let req = json!({"model": "glm-5.3-jspark", "max_tokens": 100, "stream": true, "stop_sequences": ["x"],
            "thinking": {"type": "enabled", "budget_tokens": 50}, "output_config": {"effort": "max"},
            "tools": [{"name": "f", "input_schema": {"type": "object"}}], "messages": [{"role": "user", "content": "hi"}]});
        let b = tokenize_body(&to_gr8(&req, &cfg, true).unwrap());
        let o = b.as_object().unwrap();
        for k in ["stream", "stream_options", "max_tokens", "thinking_budget", "steering_profile", "steering_scale"] {
            assert!(!o.contains_key(k), "{k}");
        }
        assert_eq!((&b["reasoning_effort"], &b["add_generation_prompt"]), (&json!("max"), &json!(true)));
        assert_eq!(b["chat_template_kwargs"], json!({"clear_thinking": false}));
        assert!(b["messages"].is_array() && b["tools"].is_array());
    }

    pub(crate) fn cfg() -> GlmConfig {
        GlmConfig {
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
        }
    }

    #[test]
    fn efforts_are_mapped_and_unknown_ones_are_not_max() {
        let got: Vec<String> = ["low", "medium", "high", "xhigh", "max", "none", "ultra"].iter().map(|e| effort(Some(&json!(e)), "high")).collect();
        assert_eq!(got, ["low", "high", "high", "max", "max", "high", "high"]);
        assert_eq!(effort(None, "high"), "high");
        assert_eq!(effort(Some(&json!(7)), "low"), "low");
    }

    fn cc_turn() -> Value {
        // Claude Code's main loop: adaptive thinking, display omitted, effort high, a tool loop with a
        // system reminder beside the tool result, an error result, and a block the model cannot read
        json!({"model": "glm-5.3", "max_tokens": 32000, "stream": true,
            "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"},
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2; cch=abcd;"},
                       {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>"}, {"type": "text", "text": "list files"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "", "signature": anthropic::signature("I should run ls.")},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true, "content": "ls: denied"},
                    {"type": "text", "text": "<system-reminder>todo list empty</system-reminder>"},
                    {"type": "mystery_block", "x": 1}]}]})
    }

    #[test]
    fn a_claude_code_turn_for_gr8() {
        let p = to_gr8(&cc_turn(), &cfg(), false).unwrap();
        let b = &p.body;
        assert_eq!(b["model"], "glm-5.3");
        assert_eq!(b["reasoning_effort"], "high");
        assert_eq!(b["chat_template_kwargs"], json!({"clear_thinking": false}));
        assert_eq!(b["max_tokens"], 32000);
        assert!(b.get("thinking_budget").is_none(), "a main-loop turn has no budget");
        assert!(b.get("steering_profile").is_none());
        assert_eq!(b["stream"], true);
        assert_eq!(b["stream_options"]["include_usage"], true);
        let m = b["messages"].as_array().unwrap();
        // billing header gone; the system prompt is the same every turn
        assert_eq!(m[0], json!({"role": "system", "content": "You are Claude Code."}));
        // the reasoning of the tool loop is replayed from the signature
        assert_eq!(m[2]["reasoning_content"], "I should run ls.");
        assert_eq!(m[2]["tool_calls"][0]["function"]["name"], "Bash");
        // the tool error keeps its mark, and the reminder stays where it was, after the result
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "toolu_1", "content": "<tool_use_error>ls: denied</tool_use_error>"}));
        assert_eq!(m[4]["role"], "user");
        assert!(m[4]["content"].as_str().unwrap().starts_with("<system-reminder>todo list empty</system-reminder>"));
        assert!(m[4]["content"].as_str().unwrap().ends_with("[Unsupported mystery_block]"));
        assert!(p.show_thinking && p.omit_thinking && p.stream && p.stops.is_empty());
    }

    #[test]
    fn the_steered_name_names_the_profile() {
        let mut req = cc_turn();
        req["model"] = json!("glm-5.3-jspark");
        let c = cfg();
        assert_eq!(c.route(Some("glm-5.3-jspark")), Some(true));
        assert_eq!(c.route(Some("glm-5.3")), Some(false));
        assert_eq!(c.route(Some("deepseek-v4.1-flash")), None);
        let p = to_gr8(&req, &c, true).unwrap();
        assert_eq!(p.body["steering_profile"], "ablit-behavior-v1");
        assert_eq!(p.body["steering_scale"], 1.0);
        assert_eq!(p.body["model"], "glm-5.3", "gr8's own name");
        assert_eq!(p.model, "glm-5.3-jspark", "echoed back");
    }

    #[test]
    fn a_small_request_gets_a_budget_and_its_stops_stay_here() {
        // a title: no thinking field, a small max_tokens, stop sequences
        let req = json!({"model": "glm-5.3", "max_tokens": 64, "stop_sequences": ["\n", ""],
            "messages": [{"role": "user", "content": "name this chat"}]});
        let p = to_gr8(&req, &cfg(), false).unwrap();
        assert_eq!(p.body["chat_template_kwargs"], json!({"clear_thinking": false}), "no enable_thinking: it bounds nothing");
        assert_eq!(p.body["reasoning_effort"], "low");
        assert_eq!(p.body["thinking_budget"], 32, "half the max_tokens");
        assert_eq!(p.body["max_tokens"], 64, "not inflated");
        assert!(p.body.get("stop").is_none(), "gr8 would match them in the reasoning");
        assert_eq!(p.stops, ["\n"]);
        assert!(!p.show_thinking);
        // thinking disabled, a large max_tokens: low effort, the small budget, the reasoning not shown
        let req = json!({"model": "glm-5.3", "max_tokens": 21333, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "x"}]});
        let p = to_gr8(&req, &cfg(), false).unwrap();
        assert_eq!((p.body["reasoning_effort"].as_str(), p.body["thinking_budget"].as_u64(), p.show_thinking), (Some("low"), Some(1024), false));
        // the share is configurable
        let c = GlmConfig { small_budget_share: 0.25, ..cfg() };
        let req = json!({"model": "glm-5.3", "max_tokens": 2112, "messages": [{"role": "user", "content": "x"}]});
        assert_eq!(to_gr8(&req, &c, false).unwrap().body["thinking_budget"], 528);
        // thinking on but small: the budget too, the reasoning shown
        let req = json!({"model": "glm-5.3", "max_tokens": 200, "thinking": {"type": "adaptive"}, "output_config": {"effort": "max"},
            "messages": [{"role": "user", "content": "hi"}]});
        let p = to_gr8(&req, &cfg(), false).unwrap();
        assert_eq!((p.body["reasoning_effort"].as_str(), p.body["thinking_budget"].as_u64(), p.show_thinking), (Some("max"), Some(100), true));
        assert!(p.body["chat_template_kwargs"].get("enable_thinking").is_none());
        // enabled with a budget that leaves room: kept
        let req = json!({"model": "glm-5.3", "max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 8000},
            "messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(to_gr8(&req, &cfg(), false).unwrap().body["thinking_budget"], 8000);
        let req = json!({"model": "glm-5.3", "max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 31999},
            "messages": [{"role": "user", "content": "hi"}]});
        assert!(to_gr8(&req, &cfg(), false).unwrap().body.get("thinking_budget").is_none());
    }

    #[test]
    fn forced_tools_map_to_gr8s_choice() {
        let mut req = cc_turn();
        req["tool_choice"] = json!({"type": "tool", "name": "Bash"});
        let p = to_gr8(&req, &cfg(), false).unwrap();
        assert_eq!(p.body["tool_choice"], json!({"type": "function", "function": {"name": "Bash"}}));
        assert!(p.body.get("thinking_budget").is_none());
        req["tool_choice"] = json!({"type": "any"});
        assert_eq!(to_gr8(&req, &cfg(), false).unwrap().body["tool_choice"], "required");
        req["tool_choice"] = json!({"type": "none"});
        assert_eq!(to_gr8(&req, &cfg(), false).unwrap().body["tool_choice"], "none");
    }

    #[test]
    fn overflow_is_parsed_and_mapped() {
        let msg = "This model's maximum context length is 442364 tokens. However, you requested 450000 tokens (418000 in the messages, 32000 in the completion). Please reduce the length of the messages or completion.";
        assert_eq!(parse_overflow(msg), Some((442364, 418000, 32000)));
        let c = cfg();
        // the prompt fits: again with max_tokens cut to the room
        assert!(matches!(refusal(400, msg.into(), None, &c, 32000, false), Refusal::Retry(24364)));
        // once only
        let Refusal::Fail(f) = refusal(400, msg.into(), None, &c, 32000, true) else { panic!() };
        assert_eq!(f.message, "prompt is too long: 418000 tokens > 441340 maximum");
        // the prompt alone is over
        let msg = "This model's maximum context length is 442364 tokens. However, you requested 482000 tokens (450000 in the messages, 32000 in the completion). Please reduce the length of the messages or completion.";
        let Refusal::Fail(f) = refusal(400, msg.into(), None, &c, 32000, false) else { panic!() };
        assert_eq!((f.status.as_u16(), f.message.as_str()), (400, "prompt is too long: 450000 tokens > 442364 maximum"));
        // under min_room left: too long as well
        let msg = "This model's maximum context length is 442364 tokens. However, you requested 473864 tokens (441864 in the messages, 32000 in the completion).";
        let Refusal::Fail(f) = refusal(400, msg.into(), None, &c, 32000, false) else { panic!() };
        assert!(f.message.starts_with("prompt is too long: 441864 tokens > "));
        // others: their status, 5xx as overloaded
        let Refusal::Fail(f) = refusal(429, "busy".into(), Some("7".into()), &c, 10, false) else { panic!() };
        assert_eq!((f.status.as_u16(), f.retry_after.as_deref()), (429, Some("7")));
        let Refusal::Fail(f) = refusal(503, "engine not connected".into(), None, &c, 10, false) else { panic!() };
        assert_eq!(f.status.as_u16(), 503);
    }

    #[test]
    fn sse_data_split_across_reads() {
        let mut s = SseData::default();
        let mut out = Vec::new();
        s.push(b"data: {\"a\":1}\n\nda", &mut out);
        s.push(b"ta: [DONE]\r\n\r\n: comment\n\n", &mut out);
        assert_eq!(out, ["{\"a\":1}", "[DONE]"]);
    }

    fn chunk(delta: Value, finish: Option<&str>) -> Value {
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "glm-5.3",
               "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish, "token_ids": null}],
               "usage": {"prompt_tokens": 100, "total_tokens": 110, "completion_tokens": 10}})
    }

    fn plan(show: bool, stops: &[&str]) -> Plan {
        Plan { body: json!({}), model: "glm-5.3".into(), show_thinking: show, omit_thinking: true, stops: stops.iter().map(|s| s.to_string()).collect(), stream: true, steered: false, max_tokens: 100 }
    }

    #[test]
    fn a_tool_loop_turn_translates() {
        let mut t = Translator::new(&plan(true, &[]), Box::new(|| "x".into()));
        let mut ev = Vec::new();
        for c in [
            chunk(json!({"role": "assistant", "content": ""}), None),
            chunk(json!({"reasoning": "Let me "}), None),
            chunk(json!({"reasoning": "look."}), None),
            chunk(json!({"content": "Checking."}), None),
            chunk(json!({"tool_calls": [{"index": 0, "id": "chatcmpl-tool-1", "type": "function", "function": {"name": "Bash"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"command\": "}}]}), None),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"ls\"}"}}]}), None),
            chunk(json!({}), Some("tool_calls")),
            json!({"id": "chatcmpl-1", "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                   "prompt_tokens_details": {"cached_tokens": 64}, "completion_tokens_details": {"reasoning_tokens": 5}}}),
        ] {
            assert!(matches!(t.feed(&c, &mut ev), Ok(Step::More)));
        }
        assert_eq!(t.finish(&mut ev), "tool_use");
        let c = &t.blocks.content;
        assert_eq!(c[0]["type"], "thinking");
        assert_eq!(anthropic::reasoning_from_signature(c[0]["signature"].as_str().unwrap()).as_deref(), Some("Let me look."));
        assert_eq!(c[1], json!({"type": "text", "text": "Checking."}));
        assert_eq!(c[2], json!({"type": "tool_use", "id": "toolu_x", "name": "Bash", "input": {"command": "ls"}}));
        assert_eq!((t.usage.prompt, t.usage.cached, t.usage.completion), (100, 64, 20));
        assert_eq!(t.usage.anthropic()["input_tokens"], 36);
    }

    #[test]
    fn stop_sequences_fire_in_the_answer_only_and_hidden_reasoning_stays_hidden() {
        let mut t = Translator::new(&plan(false, &["\n"]), Box::new(|| "x".into()));
        let mut ev = Vec::new();
        assert!(matches!(t.feed(&chunk(json!({"reasoning": "line one\nline two\n"}), None), &mut ev), Ok(Step::More)));
        assert!(ev.is_empty(), "thinking not asked for: no thinking block");
        assert!(matches!(t.feed(&chunk(json!({"content": "Fix the "}), None), &mut ev), Ok(Step::More)));
        assert!(matches!(t.feed(&chunk(json!({"content": "parser\nmore"}), None), &mut ev), Ok(Step::Stopped)));
        assert_eq!(t.finish(&mut ev), "stop_sequence");
        assert_eq!(t.stop_sequence.as_deref(), Some("\n"));
        assert_eq!(t.blocks.content, [json!({"type": "text", "text": "Fix the parser"})]);
    }

    #[test]
    fn an_error_chunk_fails_the_stream_and_length_is_max_tokens() {
        let mut t = Translator::new(&plan(true, &[]), Box::new(|| "x".into()));
        let mut ev = Vec::new();
        let e = json!({"error": {"message": "engine connection lost", "type": "InternalServerError", "param": null, "code": 500}});
        let Err(f) = t.feed(&e, &mut ev) else { panic!() };
        assert_eq!(f.status.as_u16(), 500);
        let mut t = Translator::new(&plan(true, &[]), Box::new(|| "x".into()));
        let _ = t.feed(&chunk(json!({"content": "a"}), Some("length")), &mut ev);
        assert_eq!(t.finish(&mut ev), "max_tokens");
    }

    #[test]
    fn base_urls() {
        let b = parse_base("http://10.0.0.2:8100/").unwrap();
        assert_eq!((b.host.as_str(), b.port, b.authority.as_str(), b.prefix.as_str()), ("10.0.0.2", 8100, "10.0.0.2:8100", ""));
        let b = parse_base("http://gr8.lan/api").unwrap();
        assert_eq!((b.host.as_str(), b.port, b.prefix.as_str()), ("gr8.lan", 80, "/api"));
        assert!(parse_base("https://x").is_err());
    }
}
