#!/bin/bash
# dsfront's --switch-cmd on A2 (docs/glm.md, "Engine switching"). It moves the four boxes between pico's
# DeepSeek engine (service.sh's ranks) and gr8's GLM-5.3 (the recipe kindlingai/glm-5.3-exlr8-k4-4x-spark
# at fb9a0b4: docker compose on every box, the head on A2).
#   engine/server/switch.sh deepseek|glm      DSFRONT_SWITCH_FROM: what dsfront takes for active (or none)
#   engine/server/switch.sh status            which engine runs, and who holds the ring
# dsfront runs it once the requests in flight are done, then waits by itself until the new engine is
# ready (pico's socket answers, or gr8's /health is 200), up to --switch-timeout-s. So the script returns
# once the stop is over and the start is under way: exit 0, or non-zero with the reason on stderr (the
# switch fails, and its waiting requests get 529). It is idempotent: dsfront runs it again after a
# failure, and after gr8 stopped answering while active (to start it again).
#
# The ring: gr8 runs as a ring job (ringlock.sh, docs/engine.md "Borrowing the ring"). The switch to GLM
# starts a holder, `switch.sh _hold`, that waits in the ring's queue like any job and asks the service to
# lend it the ring at once (service.sh lend-now: without DSGB10_YIELD_IDLE_S, only for this want, only
# while it is the only one); the service stops its engine (dsfront holds DeepSeek requests at its gate)
# and the holder has the ring while gr8 runs. Nothing preempts it (PREEMPT=0's semantics: the service's
# reclaim does not stop it): only `switch.sh deepseek` ends it, which stops gr8 first. Then the service
# takes the ring back, starts its engine and releases dsfront's gate. So development jobs and gr8 never
# share the boxes, and the service's supervisor stays the one that runs pico's engine.
#
# Settings (environment): GLM_BOXES, GLM_DIR, GLM_ENV, GLM_HEALTH, GLM_STOP_S, RING_WAIT_S, SERVICE_RUN.
set -u
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
SRC=$(cd "$(dirname "$0")/../.." && pwd)
export FLEET=${FLEET:-$HOME/dsgb10/fleet-sparks.conf}
. "$(dirname "$0")/ringlock.sh"  # FLEET, RING_LOCK, RING_OWNER, RING_WANTS, ring_take

# the boxes gr8 runs on in the recipe's rank order, head first: `local` is this box (A2, rank 0), the
# others ssh targets as in the fleet file (A4, A3, A1)
GLM_BOXES=${GLM_BOXES:-local 192.168.1.59 192.168.1.106 192.168.1.81}
# the recipe's compose directory, the same path on every box
GLM_DIR=${GLM_DIR:-$HOME/glm-5.3-exlr8-k4-4x-spark}
# the env file installed as .env before each start (.env.stock: stock; .env.jspark: the same plus the
# JSPARK release with its steering profile loaded, which serves stock requests too). One with an hf:
# STEERING_PROFILE starts with HF_TOKEN from ~/.config/gr8/hf-token (mode 600) through compose.hf-token.yaml
GLM_ENV=${GLM_ENV:-.env.jspark}
GLM_HEALTH=${GLM_HEALTH:-http://127.0.0.1:8100/health}
# gr8's stop: `glm53 down` on the head, then compose down everywhere, measured at 6.4 s (2026-11-18);
# each of the three waits is bounded by this
GLM_STOP_S=${GLM_STOP_S:-60}
SERVICE_RUN=${SERVICE_RUN:-$HOME/dsgb10/service/run}
HOLDER_PID=$SERVICE_RUN/glm-holder.pid HOLDER_STATE=$SERVICE_RUN/glm-holder.state
HOLDER_LOG=${HOLDER_LOG:-$HOME/dsgb10/service/logs/glm-holder.log}
# how long the switch to GLM waits for the ring (the service's lend; a development job on it)
RING_WAIT_S=${RING_WAIT_S:-900}

log() { echo "$(date '+%F %T') switch.sh: $*" >&2; }
die() { log "$*"; exit 1; }
on_boxes() {  # shell text: run it on every gr8 box at once; fails if any fails
  local b pids=() rc=0
  for b in $GLM_BOXES; do
    if [ "$b" = local ]; then
      bash -c "$1" < /dev/null &
    else
      ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$b" "$1" &
    fi
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
gr8_healthy() { [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$GLM_HEALTH")" = 200 ]; }
holder() { local p; p=$(cat "$HOLDER_PID" 2> /dev/null) && kill -0 "$p" 2> /dev/null && echo "$p"; }
holding() { [ -n "$(holder)" ] && [ "$(cat "$HOLDER_STATE" 2> /dev/null)" = holding ]; }

gr8_start() {
  local up="docker compose up -d"
  grep -q '^STEERING_PROFILE=hf:' "$GLM_DIR/$GLM_ENV" &&
    up="HF_TOKEN=\"\$(cat ~/.config/gr8/hf-token)\" docker compose -f compose.yaml -f compose.hf-token.yaml up -d"
  on_boxes "cd '$GLM_DIR' && cp -p '$GLM_ENV' .env && $up > /dev/null 2>&1"
}
gr8_left() {  # true while a gr8 container or engine is left on any box
  ! on_boxes "cd '$GLM_DIR' && [ -z \"\$(docker compose ps -q 2> /dev/null)\" ] && ! pgrep -f 'bin/g6m[4]' > /dev/null"
}
gr8_stop() {  # the head's orderly stop of every rank, then the containers (restart: on-failure would bring a failed rank back)
  local t0=$SECONDS
  (cd "$GLM_DIR" && [ -n "$(docker compose ps -q --status running 2> /dev/null)" ] &&
    timeout "$GLM_STOP_S" docker compose exec -T gr8 /opt/gr8/gr8 glm53 down < /dev/null > /dev/null 2>&1)
  on_boxes "cd '$GLM_DIR' && timeout $GLM_STOP_S docker compose down > /dev/null 2>&1"
  while gr8_left; do
    ((SECONDS - t0 < 2 * GLM_STOP_S)) || return 1
    sleep 1
  done
}

# The holder: waits for the ring, then holds it until `switch.sh deepseek` (TERM), which it answers by
# stopping gr8 before the lock goes. Its pid is its want's name (ring.wants/<pid>), which lend-now names.
cmd_hold() {
  echo $$ > "$HOLDER_PID"
  echo waiting > "$HOLDER_STATE"
  OWNER="glm-5.3 (switch.sh)" LOCK_WAIT=$RING_WAIT_S ring_take "glm-5.3 (switch.sh): gr8 on $GLM_BOXES" || {
    rm -f "$HOLDER_STATE" "$HOLDER_PID"
    exit 1
  }
  echo "$(date '+%F %T') pid $$ glm-5.3 (switch.sh): gr8 on $GLM_BOXES" > "$RING_OWNER"
  STOPPING=
  trap 'STOPPING=1' TERM INT
  echo holding > "$HOLDER_STATE"
  log "holding the ring"
  while [ -z "$STOPPING" ]; do
    sleep 1 &
    wait $!
  done
  echo stopping > "$HOLDER_STATE"
  # gr8 must be down before the lock goes (the service starts its engine next)
  gr8_stop || log "gr8's stop did not finish on every box: look at it before the service's engine starts"
  log "gr8 is down; releasing the ring"
  rm -f "$HOLDER_STATE" "$HOLDER_PID"
  exit 0  # the lock (fd 9) goes with this process
}

to_glm() {
  local p t0=$SECONDS
  if holding && gr8_healthy; then
    log "gr8 already runs on the ring"
    return 0
  fi
  if [ -z "$(holder)" ]; then
    rm -f "$HOLDER_PID" "$HOLDER_STATE"
    setsid nohup "$SELF" _hold >> "$HOLDER_LOG" 2>&1 < /dev/null &
  fi
  until holding; do
    p=$(holder)
    if [ -z "$p" ] && ((SECONDS - t0 > 5)); then die "the ring holder ended (see $HOLDER_LOG)"; fi
    # the service lends at once to this want (it needs nothing in flight at dsfront: dsfront's switch waits for that)
    [ -n "$p" ] && [ -e "$RING_WANTS/$p" ] && "$SRC/engine/server/service.sh" lend-now "$p" > /dev/null 2>&1
    ((SECONDS - t0 < RING_WAIT_S)) || die "no ring after $RING_WAIT_S s (owner: $(cat "$RING_OWNER" 2> /dev/null))"
    sleep 1
  done
  log "the ring is ours after $((SECONDS - t0)) s; starting gr8 ($GLM_ENV)"
  gr8_start || die "gr8's start failed on a box"
  # readiness (gr8's /health 200) is dsfront's to wait for
}

to_deepseek() {
  local p t0=$SECONDS
  p=$(holder)
  if [ -n "$p" ]; then
    kill -TERM "$p"  # it stops gr8, then the lock goes
    while kill -0 "$p" 2> /dev/null; do
      ((SECONDS - t0 < 3 * GLM_STOP_S + 30)) || die "the ring holder did not end in $((3 * GLM_STOP_S + 30)) s"
      sleep 0.5
    done
  fi
  if gr8_left; then
    log "gr8 runs without a holder: stopping it"
    gr8_stop || die "gr8's stop did not finish on every box"
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
    echo "holder: $(holder || echo none) ($(cat "$HOLDER_STATE" 2> /dev/null || echo -)); gr8 health: $(gr8_healthy && echo 200 || echo no); gr8 left on a box: $(gr8_left && echo yes || echo no)"
    echo "ring: $(cat "$RING_OWNER" 2> /dev/null)"
    ;;
  *) echo "usage: switch.sh deepseek|glm|status" >&2; exit 2 ;;
esac
