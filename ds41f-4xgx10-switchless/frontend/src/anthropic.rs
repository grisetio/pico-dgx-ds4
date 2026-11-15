//! Anthropic Messages API on the chat path: POST /v1/messages (streaming and not) and
//! POST /v1/messages/count_tokens, for Claude Code and other Anthropic clients.
//!
//! A request is translated into the chat request `server::prepare` takes, so its prompt comes from
//! the same encoder and reuses sessions the same way. The translation:
//! - system blocks are joined; Claude Code's `x-anthropic-billing-header` block is dropped (client
//!   metadata that would otherwise lead every prompt);
//! - `system` messages inside `messages` stay where they are: the encoding's mid-conversation system;
//! - tool results become tool messages, which the encoder merges back into their user turn, in the
//!   order the blocks came;
//! - reasoning is replayed from the thinking block's text or, with `display: omitted` (Claude Code's
//!   default), from its signature, which carries the reasoning itself (`SIG_PREFIX` + base64);
//! - deferred tools (`defer_loading`) stay out of the tool list. A `tool_reference` that loads one is
//!   rendered in its tool result as the tool's schema, so the prompt's leading tool list stays the
//!   same for the whole conversation;
//! - effort: `output_config.effort` low and medium → budget 50, high and xhigh → 75, max → 100 (`effort`);
//! - images become image_url parts with --vision-addr, otherwise a placeholder; other blocks the
//!   model cannot read (documents, server tool results) become placeholders, never an error.

use crate::encoding::Delta;
use crate::gate::Pass;
use crate::pyjson::{dumps, truthy};
use crate::server::{dump, key_ok, prepare, refused, start, ApiError, App, Finish, Gen, Prepared, Usage, KEEP_ALIVE};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;

/// A thinking block's signature: this prefix, then the reasoning in base64.
pub const SIG_PREFIX: &str = "dsfront-r1:";

const BILLING_HEADER: &str = "x-anthropic-billing-header:";

// ------------------------------------------------------------------ errors

fn error_kind(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 413 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    }
}

fn error_json(kind: &str, msg: &str) -> Value {
    json!({"type": "error", "error": {"type": kind, "message": msg}})
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    // 503 (the engine down) is Anthropic's 529, which its clients retry with backoff
    let status = if status == StatusCode::SERVICE_UNAVAILABLE { StatusCode::from_u16(529).unwrap() } else { status };
    let body = error_json(error_kind(status), &msg.into()).to_string();
    let mut resp = (status, [("content-type", "application/json")], body).into_response();
    if status.as_u16() == 529 {
        resp.headers_mut().insert("retry-after", crate::gate::RETRY_AFTER_S.into());
    }
    resp
}

/// The chat path's failure in Anthropic's terms. A prompt over the context window says "prompt is
/// too long", which Claude Code reads as the cue to compact.
fn from_api(e: &ApiError) -> (StatusCode, String) {
    let msg = match e.2.strip_prefix("the prompt is ").and_then(|r| r.split_once(" tokens")) {
        Some((n, rest)) if e.0 == StatusCode::BAD_REQUEST => {
            let max = rest.rsplit(' ').next().unwrap_or("?");
            format!("prompt is too long: {n} tokens > {max} maximum")
        }
        _ => e.2.clone(),
    };
    (e.0, msg)
}

// ------------------------------------------------------------------ base64

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= c.len() {
                out.push(B64[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for ch in s.bytes() {
        if ch == b'=' {
            break;
        }
        let v = B64.iter().position(|&b| b == ch)? as u32;
        acc = acc << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits & 0xff) as u8);
        }
    }
    Some(out)
}

pub fn signature(reasoning: &str) -> String {
    format!("{SIG_PREFIX}{}", b64_encode(reasoning.as_bytes()))
}

/// The reasoning a signature of ours carries; None for anyone else's.
pub fn reasoning_from_signature(sig: &str) -> Option<String> {
    sig.strip_prefix(SIG_PREFIX).and_then(b64_decode).and_then(|b| String::from_utf8(b).ok())
}

// ------------------------------------------------------------------ request → chat request

/// What the translation keeps beside the chat request.
pub struct Converted {
    pub chat: Value,
    /// `thinking.display: omitted`: thinking blocks go out with empty text (the signature carries it)
    pub omit_thinking: bool,
    /// the model name the client asked for, echoed back
    pub model: String,
}

fn bad(msg: impl Into<String>) -> String {
    msg.into()
}

fn block_type(b: &Value) -> &str {
    b.get("type").and_then(Value::as_str).unwrap_or("")
}

/// A system prompt or system message's text: a string, or its text blocks joined by blank lines,
/// without the billing header.
fn system_text(v: Option<&Value>) -> Result<String, String> {
    match v {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for b in blocks {
                if block_type(b) != "text" {
                    continue;
                }
                let t = b.get("text").and_then(Value::as_str).unwrap_or("");
                if t.starts_with(BILLING_HEADER) {
                    continue;
                }
                parts.push(t);
            }
            Ok(parts.join("\n\n"))
        }
        Some(_) => Err(bad("system must be a string or a list of text blocks")),
    }
}

/// An image block as an image_url part: a data URL for base64, the URL as it is otherwise.
fn image_part(b: &Value) -> Option<Value> {
    let src = b.get("source")?;
    let url = match src.get("type").and_then(Value::as_str)? {
        "base64" => format!(
            "data:{};base64,{}",
            src.get("media_type").and_then(Value::as_str).unwrap_or("image/png"),
            src.get("data").and_then(Value::as_str)?
        ),
        "url" => src.get("url").and_then(Value::as_str)?.to_string(),
        _ => return None,
    };
    Some(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// A tool result's content: text blocks joined by blank lines (as the encoder joins a result's
/// blocks), loaded tools as their schemas, anything else as a placeholder. With vision on, a result
/// holding images stays a list of parts, whose images the encoder takes as it does a user's.
fn tool_result_content(content: Option<&Value>, deferred: &HashMap<String, Value>, vision: bool) -> Value {
    let blocks = match content {
        None | Some(Value::Null) => return json!(""),
        Some(Value::String(s)) => return json!(s),
        Some(Value::Array(b)) => b,
        Some(v) => return json!(v.to_string()),
    };
    let mut parts: Vec<Value> = Vec::new();
    let mut loaded: Vec<String> = Vec::new();
    let text = |t: String| json!({"type": "text", "text": t});
    for b in blocks {
        match block_type(b) {
            "text" => parts.push(text(b.get("text").and_then(Value::as_str).unwrap_or("").to_string())),
            "tool_reference" => {
                let name = b.get("tool_name").and_then(Value::as_str).unwrap_or("");
                loaded.push(match deferred.get(name) {
                    Some(schema) => format!("<function>{}</function>", dumps(schema)),
                    None => format!("<function>{}</function>", dumps(&json!({"name": name}))),
                });
            }
            "image" => match image_part(b) {
                Some(p) if vision => parts.push(p),
                _ => parts.push(text("[Unsupported image]".into())),
            },
            t => parts.push(text(format!("[Unsupported {t}]"))),
        }
    }
    if !loaded.is_empty() {
        parts.push(text(format!("<functions>\n{}\n</functions>", loaded.join("\n"))));
    }
    if parts.iter().all(|p| block_type(p) == "text") {
        let texts: Vec<&str> = parts.iter().map(|p| p["text"].as_str().unwrap_or("")).collect();
        return json!(texts.join("\n\n"));
    }
    Value::Array(parts)
}

/// A user message's blocks as chat messages: text and images as user messages, each tool result as
/// a tool message, in block order (the encoder merges them back into one turn).
fn push_user(content: &Value, vision: bool, deferred: &HashMap<String, Value>, out: &mut Vec<Value>) -> Result<(), String> {
    let blocks = match content {
        Value::String(s) => {
            out.push(json!({"role": "user", "content": s}));
            return Ok(());
        }
        Value::Array(b) => b,
        _ => return Err(bad("message content must be a string or a list of blocks")),
    };
    let mut parts: Vec<Value> = Vec::new();
    let flush = |parts: &mut Vec<Value>, out: &mut Vec<Value>| {
        if parts.is_empty() {
            return;
        }
        let content = if parts.iter().all(|p| block_type(p) == "text") {
            // one string, as the encoder would join the blocks
            let texts: Vec<&str> = parts.iter().map(|p| p["text"].as_str().unwrap_or("")).collect();
            json!(texts.join("\n\n"))
        } else {
            Value::Array(std::mem::take(parts))
        };
        parts.clear();
        out.push(json!({"role": "user", "content": content}));
    };
    for b in blocks {
        match block_type(b) {
            "text" => parts.push(json!({"type": "text", "text": b.get("text").and_then(Value::as_str).unwrap_or("")})),
            "image" => match image_part(b) {
                Some(p) if vision => parts.push(p),
                _ => parts.push(json!({"type": "text", "text": "[Unsupported image]"})),
            },
            "tool_result" => {
                flush(&mut parts, out);
                let mut content = tool_result_content(b.get("content"), deferred, vision);
                if let Value::String(text) = &content {
                    if truthy(b.get("is_error")) && !text.contains("<tool_use_error>") {
                        content = json!(format!("<tool_use_error>{text}</tool_use_error>"));
                    }
                }
                out.push(json!({"role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(json!("")), "content": content}));
            }
            "document" => {
                let src = b.get("source");
                match src.and_then(|s| s.get("type")).and_then(Value::as_str) {
                    Some("text") => parts.push(json!({"type": "text", "text": src.unwrap().get("data").and_then(Value::as_str).unwrap_or("")})),
                    _ => parts.push(json!({"type": "text", "text": "[Unsupported document]"})),
                }
            }
            t => parts.push(json!({"type": "text", "text": format!("[Unsupported {t}]")})),
        }
    }
    flush(&mut parts, out);
    if blocks.is_empty() {
        out.push(json!({"role": "user", "content": ""}));
    }
    Ok(())
}

/// An assistant message's blocks as one chat message: reasoning, content and tool calls. Merged into
/// the previous message when that one is an assistant's too.
fn push_assistant(content: &Value, out: &mut Vec<Value>) -> Result<(), String> {
    let (mut reasoning, mut text, mut calls) = (String::new(), String::new(), Vec::new());
    let mut has_reasoning = false;
    match content {
        Value::String(s) => text.push_str(s),
        Value::Array(blocks) => {
            for b in blocks {
                match block_type(b) {
                    "thinking" => {
                        has_reasoning = true;
                        let t = b.get("thinking").and_then(Value::as_str).unwrap_or("");
                        if !t.is_empty() {
                            reasoning.push_str(t);
                        } else if let Some(r) = b.get("signature").and_then(Value::as_str).and_then(reasoning_from_signature) {
                            reasoning.push_str(&r);
                        }
                    }
                    "redacted_thinking" => {
                        if let Some(r) = b.get("data").and_then(Value::as_str).and_then(reasoning_from_signature) {
                            has_reasoning = true;
                            reasoning.push_str(&r);
                        }
                    }
                    "text" => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                    "tool_use" => {
                        let input = b.get("input").cloned().unwrap_or(json!({}));
                        calls.push(json!({"id": b.get("id").cloned().unwrap_or(json!("")), "type": "function",
                            "function": {"name": b.get("name").cloned().unwrap_or(json!("")), "arguments": dumps(&input)}}));
                    }
                    _ => {} // server tool blocks: nothing the model wrote
                }
            }
        }
        _ => return Err(bad("message content must be a string or a list of blocks")),
    }
    if let Some(last) = out.last_mut().filter(|m| m["role"] == "assistant") {
        let m = last.as_object_mut().unwrap();
        if has_reasoning {
            let r = m.get("reasoning_content").and_then(Value::as_str).unwrap_or("").to_string() + &reasoning;
            m.insert("reasoning_content".into(), json!(r));
        }
        let c = m.get("content").and_then(Value::as_str).unwrap_or("").to_string() + &text;
        m.insert("content".into(), json!(c));
        if !calls.is_empty() {
            let mut all = m.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
            all.extend(calls);
            m.insert("tool_calls".into(), Value::Array(all));
        }
        return Ok(());
    }
    let mut m = Map::new();
    m.insert("role".into(), json!("assistant"));
    m.insert("content".into(), json!(text));
    if has_reasoning {
        m.insert("reasoning_content".into(), json!(reasoning));
    }
    if !calls.is_empty() {
        m.insert("tool_calls".into(), Value::Array(calls));
    }
    out.push(Value::Object(m));
    Ok(())
}

/// The effort budget for `output_config.effort`; None leaves the server's default. The kit's table
/// has only 50, 75 and 100, so medium and xhigh snap down to the level below (as the chat path does
/// with xhigh): the budget opens the prompt, and an untrained value is a risk for no gain.
pub fn effort(v: Option<&Value>) -> Option<i64> {
    match v?.as_str()? {
        "low" | "medium" => Some(50),
        "high" | "xhigh" => Some(75),
        "max" => Some(100),
        _ => None,
    }
}

/// An Anthropic Messages request as the chat request `prepare` takes.
pub fn to_chat(req: &Value, vision: bool) -> Result<Converted, String> {
    convert(req, vision, false)
}

/// As `to_chat`; for counting, a conversation may end with the assistant's turn, as the API allows.
fn convert(req: &Value, vision: bool, counting: bool) -> Result<Converted, String> {
    let r = req.as_object().ok_or("the request body must be a JSON object")?;
    let msgs = r.get("messages").and_then(Value::as_array).ok_or("messages must be a list")?;

    // tools: client tools in the list; deferred ones kept for the tool results that load them
    let mut tools = Vec::new();
    let mut deferred: HashMap<String, Value> = HashMap::new();
    if let Some(Value::Array(ts)) = r.get("tools") {
        for t in ts {
            if !matches!(t.get("type").and_then(Value::as_str), None | Some("custom")) {
                continue; // server tools run at Anthropic, not here
            }
            let name = t.get("name").and_then(Value::as_str).ok_or("each tool needs a name")?;
            let mut f = Map::new();
            f.insert("name".into(), json!(name));
            if let Some(d) = t.get("description").filter(|v| !v.is_null()) {
                f.insert("description".into(), d.clone());
            }
            f.insert("parameters".into(), t.get("input_schema").cloned().unwrap_or(json!({"type": "object"})));
            if truthy(t.get("defer_loading")) {
                deferred.insert(name.to_string(), Value::Object(f));
            } else {
                tools.push(json!({"type": "function", "function": Value::Object(f)}));
            }
        }
    }

    let mut out: Vec<Value> = Vec::new();
    let system = system_text(r.get("system"))?;
    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system}));
    }
    for (i, m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Value::as_str).ok_or(format!("messages[{i}] has no role"))?;
        let content = m.get("content").cloned().unwrap_or(json!(""));
        match role {
            "user" => push_user(&content, vision, &deferred, &mut out)?,
            "assistant" => push_assistant(&content, &mut out)?,
            "system" => out.push(json!({"role": "system", "content": system_text(Some(&content))?})),
            r => return Err(format!("messages[{i}]: unsupported role {r:?}")),
        }
    }
    if out.is_empty() || (!counting && out.last().map_or(true, |m| m["role"] == "assistant")) {
        return Err(bad("the last message must be the user's (assistant prefill is not supported)"));
    }

    let thinking = r.get("thinking").and_then(Value::as_object);
    let thinking_on = matches!(thinking.and_then(|t| t.get("type")).and_then(Value::as_str), Some("enabled" | "adaptive"));
    let omit_thinking = thinking.and_then(|t| t.get("display")).and_then(Value::as_str) == Some("omitted");

    let mut chat = Map::new();
    chat.insert("messages".into(), Value::Array(out));
    if !tools.is_empty() {
        chat.insert("tools".into(), Value::Array(tools));
    }
    match r.get("tool_choice").and_then(|c| c.get("type")).and_then(Value::as_str) {
        Some("none") => {
            chat.insert("tool_choice".into(), json!("none"));
        }
        Some(t @ ("any" | "tool")) if !counting => eprintln!("anthropic: tool_choice {t:?} is not enforced"),
        _ => {}
    }
    if !counting && r.get("output_config").and_then(|o| o.get("format")).is_some_and(|f| !f.is_null()) {
        eprintln!("anthropic: output_config.format is not enforced");
    }
    chat.insert("chat_template_kwargs".into(), json!({"thinking": thinking_on}));
    if thinking_on {
        if let Some(e) = effort(r.get("output_config").and_then(|o| o.get("effort"))) {
            chat.insert("reasoning_effort".into(), json!(e));
        }
    }
    for (from, to) in [("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop")] {
        if let Some(v) = r.get(from).filter(|v| !v.is_null()) {
            chat.insert(to.into(), v.clone());
        }
    }
    chat.insert("stream".into(), json!(truthy(r.get("stream"))));
    let model = r.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    Ok(Converted { chat: Value::Object(chat), omit_thinking, model })
}

// ------------------------------------------------------------------ output → content blocks

enum Open {
    Thinking(String),
    Text(String),
}

/// Turns the chat path's deltas into Anthropic content blocks: the stream's events as they come,
/// and the finished blocks for a non-streaming answer.
pub struct Blocks {
    omit_thinking: bool,
    open: Option<Open>,
    index: usize,
    pub content: Vec<Value>,
}

type Ev = (&'static str, Value);

impl Blocks {
    pub fn new(omit_thinking: bool) -> Self {
        Blocks { omit_thinking, open: None, index: 0, content: Vec::new() }
    }

    fn start(&mut self, block: Value, ev: &mut Vec<Ev>) {
        ev.push(("content_block_start", json!({"type": "content_block_start", "index": self.index, "content_block": block})));
    }

    pub fn close(&mut self, ev: &mut Vec<Ev>) {
        let Some(open) = self.open.take() else { return };
        match open {
            Open::Thinking(text) => {
                let sig = signature(&text);
                ev.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index,
                    "delta": {"type": "signature_delta", "signature": sig}})));
                let shown = if self.omit_thinking { String::new() } else { text };
                self.content.push(json!({"type": "thinking", "thinking": shown, "signature": sig}));
            }
            Open::Text(text) => self.content.push(json!({"type": "text", "text": text})),
        }
        ev.push(("content_block_stop", json!({"type": "content_block_stop", "index": self.index})));
        self.index += 1;
    }

    pub fn push(&mut self, d: Delta, tool_id: impl FnOnce() -> String, ev: &mut Vec<Ev>) {
        match d {
            Delta::Reasoning(s) => {
                if !matches!(self.open, Some(Open::Thinking(_))) {
                    self.close(ev);
                    self.start(json!({"type": "thinking", "thinking": "", "signature": ""}), ev);
                    self.open = Some(Open::Thinking(String::new()));
                }
                if let Some(Open::Thinking(t)) = &mut self.open {
                    t.push_str(&s);
                }
                if !self.omit_thinking && !s.is_empty() {
                    ev.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index,
                        "delta": {"type": "thinking_delta", "thinking": s}})));
                }
            }
            Delta::Content(s) => {
                if s.is_empty() {
                    return;
                }
                if !matches!(self.open, Some(Open::Text(_))) {
                    self.close(ev);
                    self.start(json!({"type": "text", "text": ""}), ev);
                    self.open = Some(Open::Text(String::new()));
                }
                if let Some(Open::Text(t)) = &mut self.open {
                    t.push_str(&s);
                }
                ev.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index,
                    "delta": {"type": "text_delta", "text": s}})));
            }
            Delta::ToolCall(_, name, args) => {
                self.close(ev);
                let id = format!("toolu_{}", tool_id());
                self.start(json!({"type": "tool_use", "id": id, "name": name, "input": {}}), ev);
                ev.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index,
                    "delta": {"type": "input_json_delta", "partial_json": args}})));
                ev.push(("content_block_stop", json!({"type": "content_block_stop", "index": self.index})));
                let input: Value = serde_json::from_str(&args).unwrap_or(json!({}));
                self.content.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
                self.index += 1;
            }
        }
    }
}

fn stop_reason(f: Finish) -> &'static str {
    match f {
        Finish::ToolCalls => "tool_use",
        Finish::Length => "max_tokens",
        Finish::Stop => "end_turn",
    }
}

fn usage_json(u: &Usage) -> Value {
    json!({"input_tokens": u.prompt.saturating_sub(u.cached as usize), "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": u.cached, "output_tokens": u.completion})
}

// ------------------------------------------------------------------ handlers

fn parse(app: &App, headers: &HeaderMap, body: &Bytes, counting: bool) -> Result<(Value, Converted, Prepared), Response> {
    if !key_ok(app, headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "invalid API key"));
    }
    let req: Value = serde_json::from_slice(body).map_err(|e| error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;
    let conv = convert(&req, app.cfg.vision_addr.is_some(), counting).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let p = prepare(&app.cfg, &conv.chat).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    Ok((req, conv, p))
}

pub async fn count_tokens(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    let (_, _, p) = match parse(&app, &headers, &body, true) {
        Ok(x) => x,
        Err(r) => return r,
    };
    let app2 = app.clone();
    let n = match tokio::task::spawn_blocking(move || app2.tok.encode_parallel(&p.prompt, app2.cfg.tokenize_threads).len()).await {
        Ok(n) => n,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    (StatusCode::OK, [("content-type", "application/json")], json!({"input_tokens": n}).to_string()).into_response()
}

pub async fn messages(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    let (req, conv, p) = match parse(&app, &headers, &body, false) {
        Ok(x) => x,
        Err(r) => return r,
    };
    let record = json!({"anthropic": req, "chat": conv.chat});
    // While the engine is down a stream starts at once and waits with keep-alives (see server.rs);
    // anything else waits here and is refused with 529.
    if p.stream && !app.gate.open(app.engine.info().is_some()) {
        return stream(app, p, conv, record, None);
    }
    let pass = match app.gate.enter(&app.engine).await {
        Ok(pass) => pass,
        Err(r) => {
            let e = refused(r);
            return error(e.0, e.2);
        }
    };
    // started before the answer, so that a prompt over the window is a 400, not a stream's error
    let gen = match start(&app, &p, pass).await {
        Ok(g) => g,
        Err(e) => {
            let (s, m) = from_api(&e);
            return error(s, m);
        }
    };
    dump(&app, &record, &p, &gen);
    if p.stream {
        return stream(app, p, conv, record, Some(gen));
    }
    answer(app, conv, gen).await
}

/// How long a stream may go without an event before it gets a ping.
const PING_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

fn message_id(app: &App) -> String {
    format!("msg_{}", app.random_hex(24))
}

async fn answer(app: Arc<App>, conv: Converted, mut gen: Gen) -> Response {
    let mut blocks = Blocks::new(conv.omit_thinking);
    let (mut out, mut ev) = (Vec::new(), Vec::new());
    let (finish, usage) = loop {
        let end = gen.next(&mut out).await;
        for d in out.drain(..) {
            blocks.push(d, || app.random_hex(24), &mut ev);
        }
        ev.clear();
        match end {
            None => continue,
            Some(Ok(x)) => break x,
            Some(Err(e)) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    };
    blocks.close(&mut ev);
    let body = json!({"id": message_id(&app), "type": "message", "role": "assistant", "model": conv.model,
        "content": blocks.content, "stop_reason": stop_reason(finish), "stop_sequence": null, "usage": usage_json(&usage)});
    (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response()
}

fn sse(kind: &str, data: &Value) -> Event {
    Event::default().event(kind).data(data.to_string())
}

/// A streamed answer. Without a started request it waits at the gate first, as the chat path does.
fn stream(app: Arc<App>, p: Prepared, conv: Converted, record: Value, gen: Option<Gen>) -> Response {
    let (tx, rx) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let fail = |status: StatusCode, msg: String| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(sse("error", &error_json(error_kind(status), &msg))).await;
            }
        };
        let mut gen = match gen {
            Some(g) => g,
            None => {
                let note = format!("waiting for the engine, up to {} s", app.gate.hold_limit().as_secs());
                if tx.send(Event::default().comment(note)).await.is_err() {
                    return;
                }
                let pass: Pass = tokio::select! {
                    r = app.gate.enter(&app.engine) => match r {
                        Ok(pass) => pass,
                        Err(r) => { let e = refused(r); return fail(e.0, e.2).await; }
                    },
                    _ = tx.closed() => return,
                };
                match start(&app, &p, pass).await {
                    Ok(g) => {
                        dump(&app, &record, &p, &g);
                        g
                    }
                    Err(e) => {
                        let (s, m) = from_api(&e);
                        return fail(s, m).await;
                    }
                }
            }
        };
        let start_msg = json!({"type": "message_start", "message": {"id": message_id(&app), "type": "message",
            "role": "assistant", "model": conv.model, "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": usage_json(&Usage { prompt: gen.prompt_ids.len(), cached: gen.reused as u32, completion: 0, reasoning: 0 })}});
        if tx.send(sse("message_start", &start_msg)).await.is_err() {
            return;
        }
        let ping = || sse("ping", &json!({"type": "ping"}));
        let _ = tx.send(ping()).await;
        let mut blocks = Blocks::new(conv.omit_thinking);
        let (mut out, mut ev) = (Vec::new(), Vec::new());
        // Hidden thinking and long prefills send nothing for minutes: a ping event (not a comment) keeps
        // the client's idle watchdog quiet, and watching the channel notices a client that left.
        let mut quiet = tokio::time::interval_at(tokio::time::Instant::now() + PING_EVERY, PING_EVERY);
        loop {
            let end = tokio::select! {
                end = gen.next(&mut out) => end, // cancel-safe: it only awaits the engine channel
                _ = quiet.tick() => {
                    if tx.send(ping()).await.is_err() {
                        return;
                    }
                    continue;
                }
                _ = tx.closed() => return, // client gone; dropping gen cancels the request
            };
            for d in out.drain(..) {
                blocks.push(d, || app.random_hex(24), &mut ev);
            }
            if let Some(Ok(_)) = &end {
                blocks.close(&mut ev);
            }
            if !ev.is_empty() {
                quiet.reset();
            }
            for (kind, data) in ev.drain(..) {
                if tx.send(sse(kind, &data)).await.is_err() {
                    return; // client gone; dropping gen cancels the request
                }
            }
            match end {
                None => continue,
                Some(Ok((finish, usage))) => {
                    let delta = json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(finish), "stop_sequence": null},
                        "usage": usage_json(&usage)});
                    let _ = tx.send(sse("message_delta", &delta)).await;
                    let _ = tx.send(sse("message_stop", &json!({"type": "message_stop"}))).await;
                    break;
                }
                Some(Err(e)) => {
                    fail(StatusCode::INTERNAL_SERVER_ERROR, e).await;
                    break;
                }
            }
        }
    });
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let stream = futures_util::StreamExt::map(stream, Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(KEEP_ALIVE)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip() {
        for s in ["", "a", "ab", "abc", "abcd", "思考 · reasoning\n\"quoted\""] {
            assert_eq!(reasoning_from_signature(&signature(s)).as_deref(), Some(s));
        }
        assert_eq!(b64_encode(b"Man"), "TWFu");
        assert_eq!(b64_encode(b"Ma"), "TWE=");
        assert_eq!(reasoning_from_signature("EqQBCkYIBxgCKkA"), None);
    }

    #[test]
    fn billing_header_dropped_and_blocks_joined() {
        let req = json!({"model": "m", "max_tokens": 10,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cc_entrypoint=cli;"},
                       {"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "B"}],
            "messages": [{"role": "user", "content": "hi"}]});
        let c = to_chat(&req, false).unwrap();
        assert_eq!(c.chat["messages"][0], json!({"role": "system", "content": "A\n\nB"}));
    }

    #[test]
    fn tool_turn_and_reasoning_from_signature() {
        let req = json!({"model": "m", "max_tokens": 10, "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": "low"},
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}},
                      {"name": "Later", "description": "deferred", "input_schema": {"type": "object"}, "defer_loading": true}],
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "system", "content": [{"type": "text", "text": "env"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "", "signature": signature("plan it")},
                    {"type": "text", "text": "Running."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls", "timeout": 5}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a"}, {"type": "tool_reference", "tool_name": "Later"}]},
                    {"type": "text", "text": "<system-reminder>r</system-reminder>"}]}]});
        let c = to_chat(&req, false).unwrap();
        assert!(c.omit_thinking);
        let m = c.chat["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "user", "content": "go"}));
        assert_eq!(m[1], json!({"role": "system", "content": "env"}));
        assert_eq!(m[2]["reasoning_content"], "plan it");
        assert_eq!(m[2]["content"], "Running.");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"command\": \"ls\", \"timeout\": 5}");
        assert_eq!(m[3]["role"], "tool");
        assert_eq!(m[3]["tool_call_id"], "toolu_1");
        assert!(m[3]["content"].as_str().unwrap().starts_with("a\n\n<functions>\n<function>{\"name\": \"Later\""));
        assert_eq!(m[4], json!({"role": "user", "content": "<system-reminder>r</system-reminder>"}));
        assert_eq!(c.chat["tools"].as_array().unwrap().len(), 1);
        assert_eq!(c.chat["reasoning_effort"], 50);
        assert_eq!(c.chat["chat_template_kwargs"]["thinking"], true);
    }

    #[test]
    fn no_thinking_field_is_chat_mode() {
        let req = json!({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "title?"}], "stop_sequences": ["\n"]});
        let c = to_chat(&req, false).unwrap();
        assert_eq!(c.chat["chat_template_kwargs"]["thinking"], false);
        assert_eq!(c.chat["stop"], json!(["\n"]));
        assert!(c.chat.get("reasoning_effort").is_none());
    }

    #[test]
    fn prefill_refused_and_error_mapped() {
        let req = json!({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]});
        assert!(to_chat(&req, false).is_err());
        assert!(convert(&req, false, true).is_ok(), "counting takes a trailing assistant turn");
        let e = ApiError(StatusCode::BAD_REQUEST, "invalid_request_error".into(), "the prompt is 1200 tokens; the context window is 1000".into());
        assert_eq!(from_api(&e).1, "prompt is too long: 1200 tokens > 1000 maximum");
    }

    #[test]
    fn blocks_from_deltas() {
        let mut b = Blocks::new(true);
        let mut ev = Vec::new();
        b.push(Delta::Reasoning("think".into()), || "x".into(), &mut ev);
        b.push(Delta::Content("Hi".into()), || "x".into(), &mut ev);
        b.push(Delta::ToolCall(0, "Bash".into(), "{\"command\": \"ls\"}".into()), || "abc".into(), &mut ev);
        b.close(&mut ev);
        assert_eq!(b.content[0]["thinking"], "");
        assert_eq!(reasoning_from_signature(b.content[0]["signature"].as_str().unwrap()).as_deref(), Some("think"));
        assert_eq!(b.content[1], json!({"type": "text", "text": "Hi"}));
        assert_eq!(b.content[2], json!({"type": "tool_use", "id": "toolu_abc", "name": "Bash", "input": {"command": "ls"}}));
        // omitted: no thinking_delta goes out, only the signature
        assert!(!ev.iter().any(|(_, d)| d["delta"]["type"] == "thinking_delta"));
        let kinds: Vec<&str> = ev.iter().map(|(k, _)| *k).collect();
        assert_eq!(kinds.iter().filter(|k| **k == "content_block_start").count(), 3);
        assert_eq!(kinds.iter().filter(|k| **k == "content_block_stop").count(), 3);
    }

    #[test]
    fn effort_snaps_to_the_kit_table() {
        let got: Vec<Option<i64>> = ["low", "medium", "high", "xhigh", "max", "?"].iter().map(|e| effort(Some(&json!(e)))).collect();
        assert_eq!(got, [Some(50), Some(50), Some(75), Some(75), Some(100), None]);
    }

    #[test]
    fn tool_result_images_pass_through_with_vision() {
        let img = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}});
        let content = json!([{"type": "text", "text": "shot"}, img]);
        assert_eq!(tool_result_content(Some(&content), &HashMap::new(), false), json!("shot\n\n[Unsupported image]"));
        let parts = tool_result_content(Some(&content), &HashMap::new(), true);
        assert_eq!(parts[0], json!({"type": "text", "text": "shot"}));
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(tool_result_content(Some(&json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])), &HashMap::new(), true), json!("a\n\nb"));
    }
}
