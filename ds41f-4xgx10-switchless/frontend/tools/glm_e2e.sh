#!/usr/bin/env bash
# The GLM backend and engine switching end to end, on the host: dsfront with --switch-cmd
# tools/mock_switch.sh, which swaps dsfront's mock pico engine and its mock gr8 (docs/glm.md).
#   tools/glm_e2e.sh [WORKDIR]       (after cargo build --release; TOKENIZER=... the DeepSeek tokenizer.json)
# With CLAUDE_E2E=1 it also runs `claude -p` against the local dsfront (a throwaway config dir and a
# dummy key: nothing goes to Anthropic).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
W=${1:-$(mktemp -d)}
mkdir -p "$W"
export MOCK_DIR=$W DSFRONT=$here/target/release/dsfront ENGINE_SOCKET=$W/engine.sock GR8_LISTEN=127.0.0.1:18100
export TOKENIZER=${TOKENIZER:?the DeepSeek tokenizer.json}
PORT=${PORT:-18990}
B=http://127.0.0.1:$PORT
fails=0
ok() { echo "ok   $*"; }
no() { echo "FAIL $*"; fails=$((fails + 1)); }
check() { # name, command...: the command's success
	local n=$1; shift
	if "$@"; then ok "$n"; else no "$n"; fi
}
cleanup() {
	for p in dsfront pico gr8; do kill "$(cat "$W/$p.pid" 2>/dev/null)" 2>/dev/null || true; done
}
trap cleanup EXIT

"$here/tools/mock_switch.sh" deepseek
"$DSFRONT" serve --listen 127.0.0.1:$PORT --engine-socket "$ENGINE_SOCKET" --tokenizer "$TOKENIZER" --encoding 1017 \
	--model-name deepseek-v4.1-flash --variant-model deepseek-v4.1-flash-heretic \
	--glm-url "http://$GR8_LISTEN" --switch-cmd "$here/tools/mock_switch.sh" --switch-min-s 3 --switch-timeout-s 60 \
	--switch-retry-s 5 --admin-socket "$W/admin.sock" >"$W/dsfront.log" 2>&1 &
echo $! >"$W/dsfront.pid"
for _ in $(seq 50); do curl -sf "$B/health" >/dev/null && break; sleep 0.2; done

# the body on stdin: Linux caps one argument at 128 KiB (MAX_ARG_STRLEN), and the overflow checks' are larger
msg() { curl -s -N "$B/v1/messages" -H 'content-type: application/json' --data-binary @- <<<"$1"; }
code() { curl -s -o "$W/body" -w '%{http_code}' "$B/v1/messages" -H 'content-type: application/json' --data-binary @- <<<"$1"; }
has() { grep -q -- "$1" <<<"$2"; }
last_dump() { ls -t "$W"/gr8-dump/*.json | head -1; }

models=$(curl -s "$B/v1/models")
check "/v1/models lists all four names" has '"glm-5.3-jspark"' "$models"
check "/v1/models: DeepSeek names" has '"deepseek-v4.1-flash-heretic"' "$models"
check "/v1/models: the active-engine aliases" has '"local-active-uncensored"' "$models"
switches() { grep -c ' done in ' "$W/dsfront.log" || true; }

r=$(msg '{"model":"deepseek-v4.1-flash","max_tokens":20,"stream":true,"messages":[{"role":"user","content":"hi"}]}')
check "DeepSeek streams while it is active" has 'event: message_stop' "$r"
r=$(msg '{"model":"local-active-uncensored","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}')
check "alias on DeepSeek: the heretic variant" has '"model":"deepseek-v4.1-flash-heretic"' "$r"

r=$(msg '{"model":"glm-5.3","max_tokens":32000,"stream":true,"thinking":{"type":"adaptive","display":"omitted"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hello"}]}')
check "GLM after a switch: a whole message" has 'event: message_stop' "$r"
check "GLM: thinking signed, text omitted" has '"signature_delta","signature":"dsfront-r1:' "$r"
check "the switch is logged with its duration" grep -q 'switch: deepseek -> glm done in' "$W/dsfront.log"
d=$(cat "$(last_dump)")
check "gr8 gets effort high, clear_thinking false" has '"reasoning_effort": "high"' "$d"
check "gr8 gets clear_thinking false" has '"clear_thinking": false' "$d"
check "/health while GLM is active" has 'ok (glm)' "$(curl -s "$B/health")"
check "the ring may be lent to gr8 while GLM is active" test "$("$DSFRONT" admin --socket "$W/admin.sock" hold 0)" = ok
"$DSFRONT" admin --socket "$W/admin.sock" release >/dev/null

r=$(msg '{"model":"glm-5.3-jspark","max_tokens":64,"stop_sequences":["\n"],"messages":[{"role":"user","content":"name this chat"}]}')
check "steered name: the profile reaches gr8" has '\[steered:ablit-behavior-v1\]' "$r"
check "a stop sequence ends the answer, not the reasoning" has '"stop_reason":"stop_sequence","stop_sequence":"\\n"' "$r"
d=$(cat "$(last_dump)")
check "a small request: low effort" has '"reasoning_effort": "low"' "$d"
check "... no enable_thinking" eval '! grep -q enable_thinking "$(last_dump)"'
check "... its reasoning capped at half its max_tokens" has '"thinking_budget": 32' "$d"
check "... max_tokens not inflated" has '"max_tokens": 64' "$d"
n=$(switches)
r=$(msg '{"model":"local-active","max_tokens":64,"messages":[{"role":"user","content":"title?"}]}')
check "alias on GLM: glm-5.3" has '"model":"glm-5.3"' "$r"
r=$(msg '{"model":"local-active-uncensored","max_tokens":64,"messages":[{"role":"user","content":"title?"}]}')
check "uncensored alias on GLM: steered" has 'steered:ablit-behavior-v1' "$r"
check "the aliases switched nothing" test "$(switches)" = "$n"
check "the resolution is logged" grep -q 'alias: local-active -> glm-5.3 (the glm engine)' "$W/dsfront.log"
check "stop sequences stay in dsfront" eval '! grep -q "\"stop\"" "$(last_dump)"'

r=$(msg '{"model":"glm-5.3","max_tokens":32000,"thinking":{"type":"adaptive"},"output_config":{"effort":"ultra"},"tools":[{"name":"Bash","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"MOCK_TOOL Bash {\"command\": \"ls\"}"}]}')
check "a tool call comes back as tool_use" has '"stop_reason":"tool_use"' "$r"
check "an unknown effort is the default, not max" has '"reasoning_effort": "high"' "$(cat "$(last_dump)")"

r=$(msg '{"model":"glm-5.3","max_tokens":32000,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","is_error":true,"content":"denied"},{"type":"weird_block"}]}]}')
check "is_error kept, an unknown block does not fail" has '<tool_use_error>denied</tool_use_error>' "$(cat "$(last_dump)")"
check "... and the turn is answered" has '"stop_reason":"end_turn"' "$r"

big=$(python3 -c "print('x '*420000)")
c=$(code "{\"model\":\"glm-5.3\",\"max_tokens\":32000,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$big\"}]}")
check "overflow: 400 prompt is too long (Claude Code compacts)" has '400 .*prompt is too long: ' "$c $(cat "$W/body")"
mid=$(python3 -c "print('y '*340000)")
r=$(msg "{\"model\":\"glm-5.3\",\"max_tokens\":64000,\"stream\":true,\"thinking\":{\"type\":\"adaptive\"},\"messages\":[{\"role\":\"user\",\"content\":\"$mid\"}]}")
check "room short: retried with less max_tokens" has 'event: message_stop' "$r"

r=$(msg '{"model":"glm-5.3","max_tokens":1000,"stream":true,"messages":[{"role":"user","content":"MOCK_ERROR"}]}')
check "an error mid-stream is an error event" has 'event: error' "$r"
c=$(code '{"model":"glm-5.3","max_tokens":100,"messages":[{"role":"user","content":"MOCK_BUSY"}]}')
check "gr8 busy: 429 rate_limit_error" has '429 .*rate_limit_error' "$c $(cat "$W/body")"

r=$(curl -s -N "$B/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"glm-5.3-jspark","max_tokens":200,"messages":[{"role":"user","content":"hey"}]}')
check "OpenAI chat to GLM passes through, steered" has 'steered:ablit-behavior-v1' "$r"

# mixed traffic: d1 waits for the switch, g1 behind it, d2 behind g1 (passes in d1's dwell)
: >"$W/order"
pids=()
for x in d1:deepseek-v4.1-flash g1:glm-5.3 d2:deepseek-v4.1-flash-heretic; do
	(msg "{\"model\":\"${x#*:}\",\"max_tokens\":30,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" >/dev/null; echo "${x%%:*}" >>"$W/order") &
	pids+=($!)
	sleep 0.3
done
wait "${pids[@]}"
check "mixed traffic: every request answered, GLM last" test "$(tail -1 "$W/order")" = g1

# gr8 crashes while active: 529, then the next GLM request runs the switch command again
kill "$(cat "$W/gr8.pid")"; sleep 0.5
c=$(code '{"model":"glm-5.3","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}')
check "gr8 gone: 529 overloaded" has '529 .*overloaded_error' "$c $(cat "$W/body")"
c=$(code '{"model":"glm-5.3","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}')
check "... and the next request restarts it" test "$c" = 200

if [ "${CLAUDE_E2E:-}" = 1 ]; then
	mkdir -p "$W/cc-cfg" "$W/work"
	cc() {
		(cd "$W/work" && CLAUDE_CONFIG_DIR=$W/cc-cfg ANTHROPIC_BASE_URL=$B ANTHROPIC_API_KEY=dummy-local ANTHROPIC_MODEL=$1 \
			ANTHROPIC_DEFAULT_HAIKU_MODEL=$1 ANTHROPIC_SMALL_FAST_MODEL=$1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
			DISABLE_TELEMETRY=1 timeout 300 claude -p "$2" --allowedTools Bash --output-format json 2>/dev/null)
	}
	r=$(cc glm-5.3 'MOCK_TOOL Bash {"command": "echo hello-from-tool", "description": "echo"}')
	check "claude -p on GLM: a tool loop" has 'Tool result received: hello-from-tool' "$r"
	r=$(cc deepseek-v4.1-flash 'say hi')
	check "claude -p on DeepSeek after it" has '"subtype":"success"' "$r"
	r=$(cc glm-5.3-jspark 'say hi')
	check "claude -p on the steered name" has 'steered:ablit-behavior-v1' "$r"
fi

grep -E '^switch:' "$W/dsfront.log" || true
echo "$fails failed; logs in $W"
[ "$fails" = 0 ]
