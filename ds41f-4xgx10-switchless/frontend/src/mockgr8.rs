//! A gr8 stand-in for tests (`dsfront mock-gr8`): gr8's OpenAI chat API with its stream format
//! (gr8 frontend/g6front-rs/src/api.rs at the commit the GLM recipe pins), its refusals (unknown model,
//! effort, budget, steering, the context window, busy), /health and /v1/models. The prompt is
//! rendered the way GLM-5.3's template lays it out (the effort line first, then the tools, then the
//! turns, reasoning in <think>), counted at 4 bytes a token, and matched against earlier prompts per
//! steering profile (gr8 keeps stock and steered prefixes apart) in 64-token pages, so that the
//! cached_tokens it reports show whether a client keeps its prompt prefix stable.
//!
//! What it answers comes from the conversation's last turn:
//! - after tool results: "Tool result received: ..." (the loop ends);
//! - `MOCK_TOOL NAME {json}` in the user's text: a call of NAME with those arguments;
//! - `MOCK_LONG N`: N words; `MOCK_ERROR`: an error chunk mid-stream; `MOCK_BUSY`: 429;
//! - response_format json_schema: an object with each required string property "mock";
//! - otherwise: "Mock GLM answer to: ...", on two lines.
//! Every answer starts with reasoning (two lines), cut at thinking_budget as gr8's budget does. A
//! steered request's answer starts with "[steered:PROFILE] ".

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub struct MockGr8Config {
    pub ready_after: Duration,
    pub max_context: u64,
    pub delay: Duration,
    pub prefill_tps: f64,
    pub profiles: Vec<String>,
    pub dump_dir: Option<std::path::PathBuf>,
    pub served: Vec<String>,
}

struct St {
    cfg: MockGr8Config,
    t0: Instant,
    /// earlier prompts per steering namespace ("" for stock)
    cache: Mutex<HashMap<String, Vec<String>>>,
    n: AtomicU64,
}

const PAGE: usize = 64;

fn err(status: u16, kind: &str, msg: &str, param: Option<&str>) -> Response {
    let body = json!({"error": {"message": msg, "type": kind, "param": param, "code": status}});
    (StatusCode::from_u16(status).unwrap(), [("content-type", "application/json")], body.to_string()).into_response()
}

pub fn router(cfg: MockGr8Config) -> Router {
    let st = Arc::new(St { cfg, t0: Instant::now(), cache: Mutex::new(HashMap::new()), n: AtomicU64::new(0) });
    Router::new().route("/health", get(health)).route("/v1/models", get(models)).route("/v1/chat/completions", post(chat)).with_state(st)
}

async fn health(State(st): State<Arc<St>>) -> Response {
    let ready = st.t0.elapsed() >= st.cfg.ready_after;
    let steering = if st.cfg.profiles.is_empty() { Value::Null } else { json!({"profiles": st.cfg.profiles}) };
    let body = json!({"ready": ready, "steering": steering});
    let status = if ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (status, [("content-type", "application/json")], body.to_string()).into_response()
}

async fn models(State(st): State<Arc<St>>) -> Response {
    let data: Vec<Value> = st
        .cfg
        .served
        .iter()
        .map(|n| json!({"id": n, "object": "model", "created": 0, "owned_by": "gr8", "root": st.cfg.served[0], "parent": null, "max_model_len": st.cfg.max_context, "permission": []}))
        .collect();
    (StatusCode::OK, [("content-type", "application/json")], json!({"object": "list", "data": data}).to_string()).into_response()
}

fn text_of(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

/// GLM-5.3's layout: effort first, tools, the turns (chat_template.jinja), as one string.
fn render(b: &Value, effort: &str) -> String {
    let mut s = String::from("[gMASK]<sop>");
    let eff = match effort {
        "low" | "minimal" | "none" => "Low",
        "medium" | "high" => "High",
        _ => "Max",
    };
    s.push_str(&format!("<|system|>Reasoning Effort: {eff}"));
    if let Some(Value::Array(ts)) = b.get("tools") {
        s.push_str("<|system|>\n# Tools\n<tools>\n");
        for t in ts {
            s.push_str(&t.get("function").unwrap_or(t).to_string());
            s.push('\n');
        }
        s.push_str("</tools>");
    }
    let clear = b.pointer("/chat_template_kwargs/clear_thinking").and_then(Value::as_bool).unwrap_or(false);
    let msgs = b.get("messages").and_then(Value::as_array).cloned().unwrap_or_default();
    let last_user = msgs.iter().rposition(|m| m["role"] == "user");
    let mut prev_tool = false;
    for (i, m) in msgs.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or("");
        match role {
            "user" => s.push_str(&format!("<|user|>{}", text_of(&m["content"]))),
            "system" => s.push_str(&format!("<|system|>{}", text_of(&m["content"]))),
            "assistant" => {
                s.push_str("<|assistant|>");
                let r = m.get("reasoning_content").and_then(Value::as_str);
                match r {
                    Some(r) if !clear || last_user.is_none_or(|u| i > u) => s.push_str(&format!("<think>{r}</think>")),
                    _ => s.push_str("<think></think>"),
                }
                s.push_str(text_of(&m["content"]).trim());
                if let Some(Value::Array(tcs)) = m.get("tool_calls") {
                    for tc in tcs {
                        s.push_str(&format!("\n<tool_call>{}{}</tool_call>", tc["function"]["name"].as_str().unwrap_or(""), tc["function"]["arguments"]));
                    }
                }
            }
            "tool" => {
                if !prev_tool {
                    s.push_str("<|observation|>");
                }
                s.push_str(&format!("<tool_response>{}</tool_response>", text_of(&m["content"])));
            }
            _ => {}
        }
        prev_tool = role == "tool";
    }
    s.push_str("<|assistant|><think>");
    s
}

fn tokens(s: &str) -> u64 {
    (s.len() as u64).div_ceil(4)
}

/// The tokens an earlier prompt in the namespace shares with this one, in whole pages.
fn cached(st: &St, ns: &str, prompt: &str) -> u64 {
    let mut c = st.cache.lock().unwrap();
    let list = c.entry(ns.to_string()).or_default();
    let best = list.iter().map(|p| p.bytes().zip(prompt.bytes()).take_while(|(a, b)| a == b).count()).max().unwrap_or(0);
    list.push(prompt.to_string());
    if list.len() > 32 {
        list.remove(0);
    }
    let toks = (best / 4) / PAGE * PAGE;
    (toks as u64).min(tokens(prompt).saturating_sub(1))
}

enum Piece {
    Reasoning(String),
    Content(String),
    Call(String, String),
    Error,
}

fn words(s: &str) -> Vec<String> {
    // a token per word, its trailing space or newline kept
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        cur.push(ch);
        if ch == ' ' || ch == '\n' {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The answer, from the conversation's last turn.
fn script(b: &Value, profile: Option<&str>) -> Vec<Piece> {
    let msgs = b.get("messages").and_then(Value::as_array).cloned().unwrap_or_default();
    let last_asst = msgs.iter().rposition(|m| m["role"] == "assistant").map_or(0, |i| i + 1);
    let tail = &msgs[last_asst.min(msgs.len())..];
    let tool = tail.iter().find(|m| m["role"] == "tool").map(|m| text_of(&m["content"]));
    let user: String = tail.iter().filter(|m| m["role"] == "user").map(|m| text_of(&m["content"])).collect::<Vec<_>>().join("\n");
    let ask = user.lines().filter(|l| !l.contains("system-reminder>")).collect::<Vec<_>>().join(" ");
    let short: String = ask.chars().take(60).collect();
    let mut out = vec![Piece::Reasoning(format!("The user asks: {short}\nI will answer briefly.\n"))];
    let pre = profile.map_or(String::new(), |p| format!("[steered:{p}] "));
    if let Some(t) = tool {
        let t: String = t.chars().take(80).collect::<String>().replace('\n', " ");
        out.push(Piece::Content(format!("{pre}Tool result received: {t}")));
        return out;
    }
    if let Some(i) = user.find("MOCK_TOOL ") {
        let rest = &user[i + 10..];
        let (name, args) = rest.split_once(' ').unwrap_or((rest, "{}"));
        let args = args.lines().next().unwrap_or("{}").trim();
        out.push(Piece::Content(format!("{pre}Calling {name}.")));
        out.push(Piece::Call(name.trim().to_string(), args.to_string()));
        return out;
    }
    if let Some(i) = user.find("MOCK_LONG ") {
        let n: usize = user[i + 10..].split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(100);
        out.push(Piece::Content(pre + &(0..n).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ")));
        return out;
    }
    if user.contains("MOCK_ERROR") {
        out.push(Piece::Content(format!("{pre}Starting")));
        out.push(Piece::Error);
        return out;
    }
    if let Some(schema) = b.pointer("/response_format/json_schema/schema") {
        let mut o = serde_json::Map::new();
        for k in schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
            o.insert(k.into(), json!("mock"));
        }
        out.push(Piece::Content(Value::Object(o).to_string()));
        return out;
    }
    out.push(Piece::Content(format!("{pre}Mock GLM answer to: {short}\nSecond line.")));
    out
}

async fn chat(State(st): State<Arc<St>>, body: Bytes) -> Response {
    let b: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return err(400, "BadRequestError", &format!("invalid JSON: {e}"), None),
    };
    let n = st.n.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(d) = &st.cfg.dump_dir {
        let _ = std::fs::write(d.join(format!("gr8-{n:04}.json")), serde_json::to_string_pretty(&b).unwrap());
    }
    if st.t0.elapsed() < st.cfg.ready_after {
        return err(503, "ServiceUnavailableError", "engine not connected", None);
    }
    let model = b.get("model").and_then(Value::as_str).unwrap_or("");
    if !st.cfg.served.iter().any(|m| m == model) {
        return err(404, "NotFoundError", &format!("The model `{model}` does not exist."), Some("model"));
    }
    let effort = match b.get("reasoning_effort") {
        None | Some(Value::Null) => "max".to_string(),
        Some(Value::String(e)) if ["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&e.as_str()) => e.clone(),
        _ => return err(400, "BadRequestError", "reasoning_effort must be none, minimal, low, medium, high, xhigh or max", Some("reasoning_effort")),
    };
    let budget = match b.get("thinking_budget") {
        None | Some(Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(n) => Some(n),
            None => return err(400, "BadRequestError", "thinking_budget must be an integer or null", Some("thinking_budget")),
        },
    };
    let forced = matches!(b.get("tool_choice"), Some(Value::String(s)) if s == "required") || b.get("tool_choice").is_some_and(Value::is_object);
    if forced && budget.is_some() {
        return err(400, "BadRequestError", "thinking_budget cannot be combined with a forced tool_choice", Some("thinking_budget"));
    }
    let profile = match b.get("steering_profile") {
        None | Some(Value::Null) => None,
        Some(Value::String(p)) => {
            if st.cfg.profiles.is_empty() {
                return err(400, "BadRequestError", "steering is not enabled on this server", Some("steering_profile"));
            }
            if !st.cfg.profiles.contains(p) {
                return err(400, "BadRequestError", "unknown steering_profile", Some("steering_profile"));
            }
            Some(p.clone())
        }
        Some(_) => return err(400, "BadRequestError", "steering_profile must be a string", Some("steering_profile")),
    };
    if let Some(s) = b.get("steering_scale").and_then(Value::as_f64) {
        if !(0.0..=4.0).contains(&s) {
            return err(400, "BadRequestError", "steering_scale must be in [0, 4]", Some("steering_scale"));
        }
    }
    if text_of(&b["messages"]).contains("MOCK_BUSY") || b["messages"].to_string().contains("MOCK_BUSY") {
        let mut r = err(429, "RateLimitError", "the server is busy", None);
        r.headers_mut().insert("retry-after", "3".parse().unwrap());
        return r;
    }
    // as gr8 (thinking.rs): enable_thinking false (else thinking false) or effort none renders Low and withholds the
    // reasoning, which still takes its tokens; GLM-5.3 has no thinking off
    let switch = b.pointer("/chat_template_kwargs/enable_thinking").or_else(|| b.pointer("/chat_template_kwargs/thinking")).and_then(Value::as_bool);
    let off = switch == Some(false) || (switch.is_none() && effort == "none");
    let effort = if off { "low".to_string() } else if effort == "none" { "max".into() } else { effort };
    let budget = budget.filter(|&n| n > 0); // 0: the server default (none here)
    let prompt = render(&b, &effort);
    let p = tokens(&prompt);
    let max = b.get("max_tokens").and_then(Value::as_u64).unwrap_or(st.cfg.max_context.saturating_sub(p));
    if p + max > st.cfg.max_context {
        let msg = format!(
            "This model's maximum context length is {} tokens. However, you requested {} tokens ({p} in the messages, {max} in the completion). Please reduce the length of the messages or completion.",
            st.cfg.max_context,
            p + max
        );
        return err(400, "BadRequestError", &msg, Some("max_tokens"));
    }
    let ns = profile.clone().unwrap_or_default();
    let cached = cached(&st, &ns, &prompt);
    // gr8 answers its headers after the first engine frame: prefill first
    tokio::time::sleep(Duration::from_secs_f64((p - cached) as f64 / st.cfg.prefill_tps)).await;
    eprintln!(
        "mock-gr8 #{n}: prompt {p} cached {cached} effort {effort}{} budget {budget:?} max_tokens {max} steering {}",
        if off { " (reasoning withheld)" } else { "" },
        profile.as_deref().unwrap_or("-")
    );

    // the reply's tokens, with gr8's budget: the Nth reply token ends the reasoning
    let mut pieces = Vec::new();
    let mut n_reason = 0u64;
    for pc in script(&b, profile.as_deref()) {
        match pc {
            Piece::Reasoning(r) => {
                for w in words(&r) {
                    if budget.is_some_and(|bd| n_reason + 1 >= bd) {
                        break;
                    }
                    n_reason += 1;
                    pieces.push(Piece::Reasoning(if off { String::new() } else { w }));
                }
            }
            Piece::Content(c) => pieces.extend(words(&c).into_iter().map(Piece::Content)),
            other => pieces.push(other),
        }
    }
    let stream = b.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let so = b.get("stream_options");
    let include_usage = so.and_then(|s| s.get("include_usage")).and_then(Value::as_bool).unwrap_or(false);
    let continuous = include_usage && so.and_then(|s| s.get("continuous_usage_stats")).and_then(Value::as_bool).unwrap_or(false);
    let id = format!("chatcmpl-mock{n:08x}");
    let served = st.cfg.served[0].clone();
    let delay = st.cfg.delay;
    if !stream {
        let (mut r, mut c, mut calls, mut done) = (String::new(), String::new(), Vec::new(), 0u64);
        for pc in pieces {
            if done >= max {
                break;
            }
            done += 1;
            match pc {
                Piece::Reasoning(w) => r.push_str(&w),
                Piece::Content(w) => c.push_str(&w),
                Piece::Call(name, args) => calls.push(json!({"id": format!("chatcmpl-tool-{n:032x}"), "type": "function", "function": {"name": name, "arguments": args}})),
                Piece::Error => return err(500, "InternalServerError", "engine connection lost", None),
            }
        }
        let finish = if !calls.is_empty() { "tool_calls" } else if done >= max { "length" } else { "stop" };
        let mut msg = json!({"role": "assistant", "content": c, "reasoning": r});
        if !calls.is_empty() {
            msg["tool_calls"] = json!(calls);
        }
        let body = json!({"id": id, "object": "chat.completion", "created": 0, "model": served,
            "choices": [{"index": 0, "message": msg, "logprobs": null, "finish_reason": finish, "stop_reason": null}],
            "usage": {"prompt_tokens": p, "total_tokens": p + done, "completion_tokens": done, "prompt_tokens_details": {"cached_tokens": cached}}});
        return (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response();
    }
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(64);
    tokio::spawn(async move {
        let send = |v: Value| {
            let tx = tx.clone();
            async move { tx.send(Ok(Bytes::from(format!("data: {v}\n\n")))).await.is_ok() }
        };
        let head = |choices: Value| json!({"id": id, "object": "chat.completion.chunk", "created": 0, "model": served, "choices": choices});
        let usage = |done: u64| json!({"prompt_tokens": p, "total_tokens": p + done, "completion_tokens": done});
        let mut first = head(json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}]));
        if continuous {
            first["usage"] = usage(0);
        }
        if !send(first).await {
            return;
        }
        let (mut done, mut calls) = (0u64, 0u64);
        let mut finish = "stop";
        for pc in pieces {
            if done >= max {
                finish = "length";
                break;
            }
            tokio::time::sleep(delay).await;
            let deltas: Vec<Value> = match pc {
                // withheld reasoning takes its tokens and sends nothing (gr8 skips an empty step)
                Piece::Reasoning(w) if w.is_empty() => {
                    done += 1;
                    continue;
                }
                Piece::Reasoning(w) => vec![json!({"reasoning": w})],
                Piece::Content(w) => vec![json!({"content": w})],
                Piece::Call(name, args) => {
                    let id = format!("chatcmpl-tool-{:032x}", n * 16 + calls);
                    let mid = args.len() / 2;
                    let mid = (0..=mid).rev().find(|&i| args.is_char_boundary(i)).unwrap_or(0);
                    let d = vec![
                        json!({"tool_calls": [{"index": calls, "id": id, "type": "function", "function": {"name": name}}]}),
                        json!({"tool_calls": [{"index": calls, "function": {"arguments": &args[..mid]}}]}),
                        json!({"tool_calls": [{"index": calls, "function": {"arguments": &args[mid..]}}]}),
                    ];
                    calls += 1;
                    d
                }
                Piece::Error => {
                    let _ = send(json!({"error": {"message": "engine connection lost", "type": "InternalServerError", "param": null, "code": 500}})).await;
                    let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
                    return;
                }
            };
            for d in deltas {
                done += 1;
                let mut c = head(json!([{"index": 0, "delta": d, "logprobs": null, "finish_reason": null, "token_ids": null}]));
                if continuous {
                    c["usage"] = usage(done);
                }
                if !send(c).await {
                    eprintln!("mock-gr8 #{n}: client went away after {done} tokens");
                    return;
                }
            }
        }
        if calls > 0 {
            finish = "tool_calls";
        }
        let mut c = head(json!([{"index": 0, "delta": {}, "logprobs": null, "finish_reason": finish, "token_ids": null, "stop_reason": null}]));
        if continuous {
            c["usage"] = usage(done);
        }
        if !send(c).await {
            return;
        }
        if include_usage {
            let mut u = usage(done);
            u["prompt_tokens_details"] = json!({"cached_tokens": cached});
            u["completion_tokens_details"] = json!({"reasoning_tokens": n_reason.min(done)});
            if !send(head(json!([])).as_object().cloned().map(|mut o| {
                o.insert("usage".into(), u);
                Value::Object(o)
            }).unwrap())
            .await
            {
                return;
            }
        }
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
    });
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)))
        .unwrap()
}
