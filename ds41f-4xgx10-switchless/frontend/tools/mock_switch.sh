#!/usr/bin/env bash
# dsfront --switch-cmd for tests: the two engines are dsfront's mocks (mock-engine for pico's, mock-gr8
# for gr8), and a switch stops the one and starts the other, as switch.sh does on A2.
#   mock_switch.sh deepseek|glm
# Environment: MOCK_DIR (pid files and logs), DSFRONT (the binary), TOKENIZER, ENGINE_SOCKET,
# GR8_LISTEN (127.0.0.1:18100), GR8_READY_S (mock-gr8's start-up time: 2), STOP_S (a stop's time: 1).
set -euo pipefail
to=${1:?usage: mock_switch.sh deepseek|glm}
: "${MOCK_DIR:?}" "${DSFRONT:?}" "${TOKENIZER:?}" "${ENGINE_SOCKET:?}"
GR8_LISTEN=${GR8_LISTEN:-127.0.0.1:18100}
echo "mock_switch: ${DSFRONT_SWITCH_FROM:-?} -> $to" >&2

stop() { # name
	local pid
	pid=$(cat "$MOCK_DIR/$1.pid" 2>/dev/null) || return 0
	kill "$pid" 2>/dev/null || true
	for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
	rm -f "$MOCK_DIR/$1.pid"
	sleep "${STOP_S:-1}" # the memory coming back
}

running() { kill -0 "$(cat "$MOCK_DIR/$1.pid" 2>/dev/null)" 2>/dev/null; }

start_pico() {
	"$DSFRONT" mock-engine --socket "$ENGINE_SOCKET" --tokenizer "$TOKENIZER" --delay-ms 2 \
		>>"$MOCK_DIR/pico.log" 2>&1 &
	echo $! >"$MOCK_DIR/pico.pid"
}

start_gr8() {
	"$DSFRONT" mock-gr8 --listen "$GR8_LISTEN" --ready-after-s "${GR8_READY_S:-2}" \
		--steering-profile ablit-behavior-v1 --dump-dir "$MOCK_DIR/gr8-dump" \
		>>"$MOCK_DIR/gr8.log" 2>&1 &
	echo $! >"$MOCK_DIR/gr8.pid"
}

mkdir -p "$MOCK_DIR/gr8-dump"
case $to in
deepseek) stop gr8; running pico || start_pico ;;
glm) stop pico; running gr8 || start_gr8 ;;
*) echo "mock_switch: unknown family $to" >&2; exit 2 ;;
esac
# readiness is dsfront's to wait for (pico's socket, gr8's /health)
