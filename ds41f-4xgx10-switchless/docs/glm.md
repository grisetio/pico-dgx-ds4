# GLM-5.3 beside DeepSeek: the GLM backend and engine switching

dsfront serves a second model family, GLM-5.3 run by gr8 (the recipe kindlingai/glm-5.3-exlr8-k4-4x-spark at
fb9a0b4, gr8 at 2f3b287), on the same Anthropic endpoint as DeepSeek. Claude Code picks the engine by the
model name alone. The boxes hold one engine at a time, so dsfront switches them when a request needs the
other one.

| model name | runs on |
|---|---|
| `--model-name` (deepseek-v4.1-flash), `--variant-model` (…-heretic) | pico's engine, as before (variants.md) |
| `--glm-model` (glm-5.3) | gr8, stock |
| `--glm-steered-model` (glm-5.3-jspark) | gr8 with `steering_profile` `--glm-steering-profile` (ablit-behavior-v1), scale `--glm-steering-scale` (1.0) |
| `--active-alias` (local-active) | whichever family is loaded: the DeepSeek model, or glm-5.3 |
| `--active-alias-uncensored` (local-active-uncensored) | the same: DeepSeek's variant, or glm-5.3-jspark |

`/v1/models` lists them all; the GLM names carry gr8's `max_model_len`. Without `--glm-url` nothing of
this is on, and dsfront is what it was.

## The GLM backend (src/glm.rs)

A request naming a GLM name goes to gr8's `/v1/chat/completions` (`--glm-url`, model `--glm-upstream-model`).
The request is translated by anthropic.rs's converter, the same as DeepSeek's (the billing header dropped,
system reminders kept where they are, reasoning replayed from the signature, `is_error` as
`<tool_use_error>`, unknown blocks as placeholders, deferred tools in their tool results). Then, for GLM-5.3:

- **The prefix.** `chat_template_kwargs.clear_thinking: false` (the template's default, pinned): every replayed
  assistant turn keeps its reasoning, so a tool loop's reasoning stays in the prompt and the prompt only grows
  at its end. The effort line opens the prompt, so the effort is always sent.
- **Effort.** The template names Low, High and Max; gr8 renders Max when a request names none. Claude Code's
  `output_config.effort`: low → low; medium, high → high; xhigh, max → max; none, missing or unknown →
  `--glm-effort` (high), logged when unknown. Never Max by accident.
- **Small requests.** GLM-5.3 always thinks: its template has no thinking off (gr8's `enable_thinking: false`
  only withholds the reasoning, which still takes its tokens), so only `thinking_budget` bounds it: gr8 closes
  the think block in place of the budget's Nth token. A request with thinking disabled (`thinking.type`
  disabled, or no `thinking` field: Claude Code's titles and other background calls) is sent with
  `reasoning_effort: low`, and its reasoning stays out of the reply. A small request (thinking disabled,
  `max_tokens` ≤ `--glm-small-max-tokens` 4096, or stop sequences) gets `thinking_budget` = the smaller of
  `--glm-small-budget` (1024) and `--glm-small-budget-share` (0.5) of its max_tokens, so its answer fits in
  the rest; max_tokens is passed as asked. `thinking.type: enabled` with `budget_tokens` passes the budget
  when it leaves `--glm-min-room` for the answer. A forced tool choice (`any`, `tool`) gets no budget (gr8
  refuses one).
- **Stop sequences** are not sent to gr8, which would match them in the reasoning too. dsfront matches them in
  the visible text, ends the answer there (`stop_reason: stop_sequence`) and closes gr8's stream, which ends the
  generation.
- **Overflow.** gr8's "maximum context length is X … (P in the messages, M in the completion)": when the prompt
  fits with at least `--glm-min-room` (1024) left, the request goes again once with max_tokens = X − P;
  otherwise it is a 400 `prompt is too long: P tokens > N maximum`, Claude Code's cue to compact.
- **Other refusals.** 429 → `rate_limit_error` with gr8's Retry-After; 503, or gr8 not answering → 529
  `overloaded_error` (clients retry); 404 → `not_found_error`; an error chunk mid-stream → an `error` event.
- **Keep-alives.** gr8 sends its response headers only after the prefill. A stream waits `--glm-head-wait-ms`
  (3000) for gr8's first answer, so that a quick refusal is still an HTTP status; then it starts with
  `message_start` and a `ping` every 10 s until gr8's tokens come (a switch, a long prefill), as well as
  SSE comments every 5 s.
- **Usage.** `input_tokens` = prompt − cached, `cache_read_input_tokens` = gr8's `cached_tokens`.
- `/v1/chat/completions` with a GLM name goes through to gr8 as it is, with gr8's model name and, for the
  steered name, the steering fields.
- count_tokens for a GLM name counts with DeepSeek's tokenizer: an estimate.

## Engine switching (src/switch.rs), `--switch-cmd`

One family's engine is active. A request for it passes; one for the other family waits until nothing is in
flight, and the requests that come after it wait their turn behind it (both families), so neither starves the
other (as variants.md does for weight variants). Then dsfront runs `--switch-cmd PATH glm|deepseek`
(`DSFRONT_SWITCH_FROM` set), waits until the new engine is ready (pico's socket answers; gr8's `/health` is
200), and serves the waiting requests in order. While a request waits, a stream carries `message_start` and
pings (Anthropic) or comments (OpenAI).

- **Damping.** For `--switch-min-s` (60) after a switch, its family's requests pass even while the other
  family waits, and no switch starts: mixed traffic runs in batches, not a switch per turn.
- **The log.** `switch: deepseek -> glm: 2 requests waiting for it; switching`, then
  `switch: deepseek -> glm done in 24.1 s (command 5.2 s, then ready 18.9 s later); serving 2 waiting requests`.
- **Failures.** The command and the readiness wait have `--switch-timeout-s` (1200) together. A failed switch
  leaves no engine taken for active; the requests that waited for it get 529, and the next try waits
  `--switch-retry-s` (30). A request waits at most the switch timeout plus `--hold-s`. If gr8 stops answering
  while active, the GLM engine counts as lost, and the next GLM request runs the command again (which starts
  gr8 again). dsfront itself survives either engine being down.
- **Start.** The active engine at dsfront's start: GLM if gr8's `/health` answers, else DeepSeek.
- **`--switch-home-s N`.** After GLM was idle N s, switch back to DeepSeek (0, the default: stay).
- **/health.** 503 while switching, or with no engine after a failure; `ok (glm) …` when GLM is active and
  gr8 answers; DeepSeek's answer as before otherwise.
- **The ring.** service.sh lends the ring to a job through dsfront's admin `hold`. With switching, the hold is
  refused while a GLM request waits or a switch to DeepSeek runs; it is allowed while DeepSeek is idle (a
  development job) and while a switch to GLM runs or GLM is active (gr8 is then the ring's job:
  engine/server/switch.sh). No switch away from DeepSeek starts while the ring is lent to a development job.

**The active-engine aliases.** Claude Code sends its background (haiku-tier) requests to one fixed model per
provider, and those must never switch engines. `local-active` and `local-active-uncensored` resolve, per
request, to the family that is loaded: the engine being switched to during a switch; the active one if a
request of its family would pass now; else the family that a waiting request is about to switch to. So an
alias request never starts a switch of its own. dsfront logs `alias: local-active -> glm-5.3 (the glm engine)`;
the answer names the model it ran on.

## On A2: engine/server/switch.sh

`switch.sh glm` starts a holder that waits for the ring as a job (ringlock.sh) and asks the service to lend
it at once: `service.sh lend-now PID` skips `DSGB10_YIELD_IDLE_S` for that one want, only while it is the
only want and dsfront's `hold 0` succeeds, so other ring jobs still wait for the idle time. The service stops
its engine (dsfront holds DeepSeek requests); the holder then runs gr8 with docker compose on the four boxes
(`GLM_ENV`, default `.env.jspark`, installed as `.env`) and holds the ring. Nothing preempts it: only
`switch.sh deepseek` ends it, which runs `glm53 down` on the head, `docker compose down` on every box and
waits until no `g6m4` is left (`GLM_STOP_S`, 60), before the lock goes. The service then takes the ring back,
starts its engine and releases dsfront's gate. `switch.sh status` shows the holder, gr8's health and the
ring's owner. Settings: `GLM_BOXES` (head first; `local` is this box), `GLM_DIR`, `GLM_ENV`, `GLM_STOP_S`,
`RING_WAIT_S`.

```bash
DSGB10_FRONT_ARGS="--glm-url http://127.0.0.1:8100 --switch-cmd $SRC/engine/server/switch.sh" service.sh start
```

## Tests

- `cargo test --release` in frontend/: the converter, the stream translation, stop sequences, overflow, the
  effort map, the switcher's queue, dwell, failures, blocking and aliases.
- `tools/glm_e2e.sh [DIR]` (TOKENIZER=… the DeepSeek tokenizer.json; CLAUDE_E2E=1 for `claude -p` too): dsfront
  with `--switch-cmd tools/mock_switch.sh`, which swaps `dsfront mock-engine` (pico) and `dsfront mock-gr8`, a
  stand-in that replays gr8's stream, usage, error and overflow formats, its steering checks, and a prefix
  cache per steering profile.
