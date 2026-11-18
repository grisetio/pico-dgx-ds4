#!/bin/bash
# DRAFT, not run on the boxes yet: dsfront's --switch-cmd on A2 (docs/glm.md, "Engine switching"). It
# moves the four boxes between pico's DeepSeek engine (service.sh's ranks) and gr8's GLM-5.3 (the recipe
# kindlingai/glm-5.3-exlr8-k4-4x-spark at fb9a0b4: docker compose on every box, the head on A2).
#   engine/server/switch.sh deepseek|glm      DSFRONT_SWITCH_FROM: what dsfront takes for active (or none)
#   engine/server/switch.sh status            which engine runs, and who holds the ring
# dsfront runs it once the requests in flight are done, then waits by itself until the new engine is
# ready (pico's socket answers, or gr8's /health is 200), up to --switch-timeout-s. So the script returns
# once the stop is over and the start is under way: exit 0, or non-zero with the reason on stderr (the
# switch fails, and its waiting requests get 529). It must be idempotent: dsfront runs it again after a
# failure, and after gr8 stopped answering while active (to start it again).
#
# The ring: gr8 runs as a ring job (ringlock.sh, docs/engine.md "Borrowing the ring"). The switch to GLM
# starts a holder, `switch.sh _hold`, that waits in the ring's queue like any job; the service lends the
# ring to it (its engine stops, dsfront holds DeepSeek requests at its gate); the holder runs gr8 while it
# has the ring. The switch to DeepSeek stops gr8 and ends the holder: the service takes the ring back and
# starts its engine, and releases dsfront's gate. So development jobs and gr8 never share the boxes, and
# the service's supervisor stays the one that runs pico's engine.
#
# PLACEHOLDERS, to settle at the deploy (marked PLACEHOLDER below):
# - lend_now: service.sh lends the ring only once no request came for DSGB10_YIELD_IDLE_S (600) and its
#   engine has run that long. A switch needs it at once; service.sh has no command for that yet.
# - gr8_start / gr8_stop: the recipe's compose commands on each box (paths, env file, the order).
# - the box names and the recipe's directory.
set -u
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
SRC=$(cd "$(dirname "$0")/../.." && pwd)
. "$(dirname "$0")/ringlock.sh"  # FLEET, RING_LOCK, ring_take, ring_run

# PLACEHOLDER: the boxes gr8 runs on, head first (the recipe's rank order), as ssh targets
GLM_BOXES=${GLM_BOXES:-a2 b2 c2 d2}
# PLACEHOLDER: the recipe's compose directory, the same path on every box
GLM_DIR=${GLM_DIR:-/home/kgriset/glm-5.3-exlr8-k4-4x-spark}
GLM_HEALTH=${GLM_HEALTH:-http://127.0.0.1:8100/health}
# gr8 stores its sessions' KV at a stop (GR8_STOP_GRACE, 150 s in its SWITCHING.md): a stop takes that long
GLM_STOP_S=${GLM_STOP_S:-200}
# dsfront's admin socket (service.sh's run/admin.sock) and binary
SERVICE_RUN=${SERVICE_RUN:-$HOME/dsgb10/service/run}
DSFRONT=${DSFRONT:-$HOME/dsgb10/service/bin/dsfront}
HOLDER_PID=$SERVICE_RUN/glm-holder.pid
HOLDER_LOG=${HOLDER_LOG:-$HOME/dsgb10/service/logs/glm-holder.log}
# how long the switch to GLM waits for the ring (the service's lend; a development job on it)
RING_WAIT_S=${RING_WAIT_S:-900}

log() { echo "$(date '+%F %T') switch.sh: $*" >&2; }
die() { log "$*"; exit 1; }
admin() { "$DSFRONT" admin --socket "$SERVICE_RUN/admin.sock" "$@" 2> /dev/null; }
on_boxes() {  # shell text: run it on every gr8 box at once; fails if any fails
  local b pids=() rc=0
  for b in $GLM_BOXES; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$b" "$1" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
gr8_healthy() { [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$GLM_HEALTH")" = 200 ]; }
holder() { local p; p=$(cat "$HOLDER_PID" 2> /dev/null) && kill -0 "$p" 2> /dev/null && echo "$p"; }
holding() { [ -n "$(holder)" ] && [ "$(sed -n 1p "$SERVICE_RUN/glm-holder.state" 2> /dev/null)" = holding ]; }

gr8_start() {
  # PLACEHOLDER: the recipe's start on every box (workers first if its compose needs them up before the head)
  on_boxes "cd '$GLM_DIR' && docker compose up -d"
}
gr8_stop() {
  # PLACEHOLDER: the recipe's orderly stop on every box (it stores the KV store's sessions first)
  on_boxes "cd '$GLM_DIR' && timeout $GLM_STOP_S docker compose down"
}
gr8_running() {
  # PLACEHOLDER: any of gr8's containers up on any box
  on_boxes "cd '$GLM_DIR' && [ -z \"\$(docker compose ps -q)\" ]" > /dev/null 2>&1 && return 1
  return 0
}
lend_now() {
  # PLACEHOLDER: make service.sh lend the ring at once to the waiting holder. dsfront's hold already
  # succeeds now (it allows it while a switch to GLM runs); the supervisor's own conditions
  # (YIELD_IDLE_S without a request, and since its engine's start) do not. Needs, say, a service.sh
  # command `lend-now` that skips them for a want whose line says "glm-5.3 (switch.sh)".
  admin hold 0 > /dev/null
  "$SRC/engine/server/service.sh" lend-now 2> /dev/null || true
}

# The holder: waits for the ring, then holds it while gr8 runs; stopped by `switch.sh deepseek` (TERM)
# or preempted by the service (TERM to its process group, from ring_run): gr8 stops in both cases.
cmd_hold() {
  echo waiting > "$SERVICE_RUN/glm-holder.state"
  OWNER="glm-5.3 (switch.sh)" LOCK_WAIT=$RING_WAIT_S ring_take "glm-5.3 (switch.sh): gr8 on $GLM_BOXES" || exit 1
  echo "$(date '+%F %T') pid $$ glm-5.3 (switch.sh): gr8 on $GLM_BOXES" > "$RING_OWNER"
  ring_run bash -c '
    trap "exit 0" TERM
    echo holding > "'"$SERVICE_RUN"'/glm-holder.state"
    while :; do sleep 3600 & wait $!; done'
  rc=$?
  echo stopping > "$SERVICE_RUN/glm-holder.state"
  # a preemption (76) or a stop: gr8 must be down before the lock goes (the service starts its engine next)
  gr8_stop || log "gr8's stop failed on a box: look at it before the service's engine starts"
  rm -f "$SERVICE_RUN/glm-holder.state" "$HOLDER_PID"
  exit $rc
}

to_glm() {
  if holding && gr8_healthy; then
    log "gr8 already runs on the ring"
    return 0
  fi
  if [ -z "$(holder)" ]; then
    setsid nohup "$SELF" _hold >> "$HOLDER_LOG" 2>&1 < /dev/null &
    echo $! > "$HOLDER_PID"
  fi
  lend_now
  local t0=$SECONDS
  until holding; do
    [ -n "$(holder)" ] || die "the ring holder ended (see $HOLDER_LOG)"
    ((SECONDS - t0 < RING_WAIT_S)) || die "no ring after $RING_WAIT_S s (owner: $(cat "$RING_OWNER" 2> /dev/null))"
    sleep 1
  done
  log "the ring is ours after $((SECONDS - t0)) s; starting gr8"
  gr8_start || die "gr8's start failed"
  # readiness (gr8's /health 200) is dsfront's to wait for
}

to_deepseek() {
  local p t0=$SECONDS
  p=$(holder)
  if [ -n "$p" ]; then
    kill -TERM "$p"  # its gr8_stop runs, then the lock goes
    while kill -0 "$p" 2> /dev/null; do
      ((SECONDS - t0 < GLM_STOP_S + 30)) || die "the ring holder did not end in $((GLM_STOP_S + 30)) s"
      sleep 1
    done
  elif gr8_running; then
    log "gr8 runs without a holder: stopping it"
    gr8_stop || die "gr8's stop failed"
  fi
  log "gr8 is down after $((SECONDS - t0)) s; the service takes the ring back and starts its engine"
  # the service's supervisor sees the lock free (within 2 s), retakes it, starts the ranks and releases
  # dsfront's gate; readiness (the engine's socket) is dsfront's to wait for
}

case ${1:-} in
  glm) to_glm ;;
  deepseek) to_deepseek ;;
  _hold) cmd_hold ;;
  status)
    echo "holder: $(holder || echo none) ($(cat "$SERVICE_RUN/glm-holder.state" 2> /dev/null || echo -)); gr8 health: $(gr8_healthy && echo 200 || echo no)"
    echo "ring: $(cat "$RING_OWNER" 2> /dev/null)"
    ;;
  *) echo "usage: switch.sh deepseek|glm|status" >&2; exit 2 ;;
esac
