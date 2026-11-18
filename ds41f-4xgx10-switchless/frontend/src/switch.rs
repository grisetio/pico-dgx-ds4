//! Engine switching (docs/glm.md): the boxes hold one engine at a time, pico's for DeepSeek or gr8 for
//! GLM-5.3, and one dsfront port serves both families. A request for the active family passes; one for
//! the other family waits until nothing of the active family is in flight, and the requests after it
//! wait their turn behind it, so neither family starves the other (the weight variants' rule,
//! docs/variants.md). Then the switch command runs (`--switch-cmd PATH FAMILY`), dsfront waits until
//! the new engine is ready, and serves the waiting requests in order.
//!
//! Damping: for `min_dwell` after a switch, requests of the family just switched to pass even while
//! requests of the other family wait, and no switch starts. Mixed traffic so runs in batches of at
//! least that long instead of switching at every turn of an agent loop.
//!
//! No switch starts while the driver says it is blocked (the ring lent to a development job: gate.rs),
//! and with `home_after` an idle engine of the other family gives the boxes back to the home family
//! (DeepSeek), so that the ring can be lent again.

use crate::engine::EngineClient;
use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// pico's engine behind the rank-0 socket
    DeepSeek,
    /// gr8's front end over HTTP
    Glm,
}

impl Family {
    /// The switch command's argument and the logs' name.
    pub fn name(self) -> &'static str {
        match self {
            Family::DeepSeek => "deepseek",
            Family::Glm => "glm",
        }
    }
}

fn name(f: Option<Family>) -> &'static str {
    f.map_or("none", Family::name)
}

pub type BoxFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// What a switch does to the world: run the command, and tell whether an engine is ready.
pub trait Driver: Send + Sync + 'static {
    fn run(&self, to: Family, from: Option<Family>) -> BoxFut<Result<(), String>>;
    fn ready(&self, f: Family) -> BoxFut<bool>;
    /// Why no switch away from `from` may start now (None: it may).
    fn blocked(&self, _from: Option<Family>) -> Option<String> {
        None
    }
}

/// The production driver: `--switch-cmd PATH`, pico's socket answering, gr8's /health 200.
pub struct CmdDriver {
    pub cmd: PathBuf,
    pub engine: EngineClient,
    pub glm_url: String,
    pub gate: Arc<crate::gate::Gate>,
}

impl Driver for CmdDriver {
    fn run(&self, to: Family, from: Option<Family>) -> BoxFut<Result<(), String>> {
        let cmd = self.cmd.clone();
        Box::pin(async move {
            // its output goes to dsfront's log; killed if the switch times out
            let status = tokio::process::Command::new(&cmd)
                .arg(to.name())
                .env("DSFRONT_SWITCH_FROM", name(from))
                .stdin(std::process::Stdio::null())
                .kill_on_drop(true)
                .status()
                .await
                .map_err(|e| format!("{}: {e}", cmd.display()))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("{} {} exited with {status}", cmd.display(), to.name()))
            }
        })
    }

    fn ready(&self, f: Family) -> BoxFut<bool> {
        match f {
            Family::DeepSeek => {
                let up = self.engine.info().is_some();
                Box::pin(async move { up })
            }
            Family::Glm => {
                let url = self.glm_url.clone();
                Box::pin(async move { crate::glm::health(&url, Duration::from_secs(3)).await })
            }
        }
    }

    /// While DeepSeek is active, a held gate is the ring lent to a development job: no switch until it
    /// is back. From GLM, the held gate is GLM's own lend (switch.sh runs gr8 as a ring job).
    fn blocked(&self, from: Option<Family>) -> Option<String> {
        (from == Some(Family::DeepSeek) && self.gate.held()).then(|| "the ring is lent to a development job".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct SwitchConfig {
    /// after a switch, its family's requests pass for this long even while the other family waits
    pub min_dwell: Duration,
    /// the command and the readiness wait together; past it the switch failed
    pub timeout: Duration,
    /// after a failed switch to a family, the next try waits this long
    pub retry_after: Duration,
    /// a request that waited this long for its engine is refused (529)
    pub max_wait: Duration,
    /// switch back to DeepSeek after the other engine was idle this long (None: stay)
    pub home_after: Option<Duration>,
}

struct St {
    active: Option<Family>,
    switching: Option<Family>,
    in_flight: u32,
    /// waiting requests in arrival order: (id, family, when it came)
    queue: VecDeque<(u64, Family, Instant)>,
    next_id: u64,
    /// when the active engine became active (the dwell's start)
    since: Instant,
    /// the last failed switch: its target, when, why
    failed: Option<(Family, Instant, String)>,
    switches: u64,
    /// the last admission or request end
    last: Instant,
    /// the reason no switch may start, logged once
    blocked: Option<String>,
}

pub struct Switcher {
    st: Mutex<St>,
    wake: Notify,
    cfg: SwitchConfig,
    driver: Arc<dyn Driver>,
}

/// A request admitted to its family's engine; dropping it counts the request done.
pub struct Ticket {
    sw: Arc<Switcher>,
    pub family: Family,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let mut s = self.sw.st.lock().unwrap();
        s.in_flight -= 1;
        s.last = Instant::now();
        drop(s);
        self.sw.wake.notify_waiters();
    }
}

/// A queued request; leaves the queue when admitted, refused, or its client goes away.
struct Queued<'a> {
    sw: &'a Switcher,
    id: u64,
}

impl Drop for Queued<'_> {
    fn drop(&mut self) {
        let mut s = self.sw.st.lock().unwrap();
        let before = s.queue.len();
        s.queue.retain(|q| q.0 != self.id);
        if s.queue.len() != before {
            drop(s);
            self.sw.wake.notify_waiters();
        }
    }
}

/// How often a waiting request looks again without being woken (the dwell's end, a retry's backoff).
const TICK: Duration = Duration::from_millis(500);

impl Switcher {
    pub fn new(cfg: SwitchConfig, driver: Arc<dyn Driver>, active: Option<Family>) -> Arc<Switcher> {
        let since = Instant::now().checked_sub(cfg.min_dwell).unwrap_or_else(Instant::now);
        Arc::new(Switcher {
            st: Mutex::new(St {
                active,
                switching: None,
                in_flight: 0,
                queue: VecDeque::new(),
                next_id: 1,
                since,
                failed: None,
                switches: 0,
                last: Instant::now(),
                blocked: None,
            }),
            wake: Notify::new(),
            cfg,
            driver,
        })
    }

    pub fn active(&self) -> Option<Family> {
        self.st.lock().unwrap().active
    }

    pub fn switching(&self) -> Option<Family> {
        self.st.lock().unwrap().switching
    }

    /// One line for /health and the logs.
    pub fn status(&self) -> String {
        let s = self.st.lock().unwrap();
        let waiting = |f: Family| s.queue.iter().filter(|q| q.1 == f).count();
        format!(
            "active {} switching {} in_flight {} queued deepseek {} glm {} switches {}{}",
            name(s.active),
            name(s.switching),
            s.in_flight,
            waiting(Family::DeepSeek),
            waiting(Family::Glm),
            s.switches,
            s.failed.as_ref().map_or(String::new(), |(f, t, e)| format!(" last_failure {} {} s ago: {e}", f.name(), t.elapsed().as_secs()))
        ) + &self.driver.blocked(s.active).map_or(String::new(), |b| format!(" blocked: {b}"))
    }

    /// The family a request naming an active-engine alias runs on (server.rs `resolve_alias`): one that
    /// never causes a switch. The engine being switched to; else the active one if a request of its
    /// family would pass now; else the family a waiting request is about to switch to; else the active
    /// one (DeepSeek when none is).
    pub fn alias_family(&self) -> Family {
        let s = self.st.lock().unwrap();
        if let Some(t) = s.switching {
            return t;
        }
        if let Some(a) = s.active.filter(|&a| self.admissible(&s, None, a)) {
            return a;
        }
        if let Some(q) = s.queue.iter().find(|q| Some(q.1) != s.active) {
            return q.1;
        }
        s.active.unwrap_or(Family::DeepSeek)
    }

    /// The engine of family `f` stopped answering while active (gr8 crashed): it is no longer taken
    /// for running, so the next request for it runs the switch command again (which starts it) once
    /// the requests in flight are done.
    pub fn lost(&self, f: Family, why: &str) {
        let mut s = self.st.lock().unwrap();
        if s.active == Some(f) && s.switching.is_none() {
            eprintln!("switch: the {} engine is lost ({why}); the next request for it runs the switch command", f.name());
            s.active = None;
            drop(s);
            self.wake.notify_waiters();
        }
    }

    /// Why the ring may not be lent now (gate.rs `hold`). It may while DeepSeek is active and nothing
    /// waits for GLM (to a development job), and while a switch to GLM runs or GLM is active (to gr8:
    /// switch.sh runs it as a ring job). Not during a switch to DeepSeek, nor while a GLM request
    /// waits for its turn (requests first).
    pub fn lend_refusal(&self) -> Option<String> {
        let s = self.st.lock().unwrap();
        let ok = match (s.switching, s.active) {
            (Some(Family::Glm), _) | (None, Some(Family::Glm)) => true,
            (None, Some(Family::DeepSeek)) => !s.queue.iter().any(|q| q.1 != Family::DeepSeek),
            _ => false,
        };
        (!ok).then(|| format!("engine switching: active {} switching {}", name(s.active), name(s.switching)))
    }

    fn status_of(&self, s: &St) -> String {
        format!("active {} switching {}{}", name(s.active), name(s.switching), self.driver.blocked(s.active).map_or(String::new(), |b| format!(", blocked: {b}")))
    }

    fn in_dwell(&self, s: &St) -> bool {
        s.active.is_some() && s.since.elapsed() < self.cfg.min_dwell
    }

    /// Whether the request `id` of family `f` may go now: its engine is active and nothing of the other
    /// family waits ahead of it (or the dwell lets it pass them). `id` None: a request not yet queued.
    fn admissible(&self, s: &St, id: Option<u64>, f: Family) -> bool {
        if s.switching.is_some() || s.active != Some(f) {
            return false;
        }
        self.in_dwell(s) || !s.queue.iter().take_while(|q| Some(q.0) != id).any(|q| q.1 != f)
    }

    /// Admitted at once, without waiting, or None (the caller then waits in `admit`, with keep-alives).
    pub fn try_admit(self: &Arc<Self>, f: Family) -> Option<Ticket> {
        let mut s = self.st.lock().unwrap();
        if self.admissible(&s, None, f) {
            s.in_flight += 1;
            s.last = Instant::now();
            return Some(Ticket { sw: self.clone(), family: f });
        }
        None
    }

    /// Wait until a request of family `f` may run on its engine, switching engines when it is its
    /// turn. Err: a switch to `f` failed while it waited (the caller answers 529, which clients retry).
    pub async fn admit(self: &Arc<Self>, f: Family) -> Result<Ticket, String> {
        let (id, t0) = {
            let mut s = self.st.lock().unwrap();
            let id = s.next_id;
            s.next_id += 1;
            let t0 = Instant::now();
            s.queue.push_back((id, f, t0));
            (id, t0)
        };
        let _q = Queued { sw: self, id };
        loop {
            let woken = self.wake.notified();
            tokio::pin!(woken);
            woken.as_mut().enable();
            {
                let mut s = self.st.lock().unwrap();
                if let Some((ff, t, e)) = &s.failed {
                    if *ff == f && *t >= t0 {
                        return Err(format!("switching to the {} engine failed: {e}", f.name()));
                    }
                }
                if self.admissible(&s, Some(id), f) {
                    s.queue.retain(|q| q.0 != id);
                    s.in_flight += 1;
                    s.last = Instant::now();
                    return Ok(Ticket { sw: self.clone(), family: f });
                }
                if t0.elapsed() >= self.cfg.max_wait {
                    return Err(format!("waited {} s for the {} engine ({})", t0.elapsed().as_secs(), f.name(), self.status_of(&s)));
                }
                self.maybe_switch(&mut s);
            }
            tokio::select! {
                _ = &mut woken => {}
                _ = tokio::time::sleep(TICK) => {}
            }
        }
    }

    /// Start a switch when it is the turn of the first request of the other family: nothing in
    /// flight, the dwell over, no request of the active family ahead of it, no recent failure.
    fn maybe_switch(self: &Arc<Self>, s: &mut St) {
        if s.switching.is_some() || s.in_flight > 0 || self.in_dwell(s) {
            return;
        }
        let Some(pos) = s.queue.iter().position(|q| Some(q.1) != s.active) else { return };
        if s.queue.iter().take(pos).any(|q| Some(q.1) == s.active) {
            return; // they pass first (they are admissible and about to wake)
        }
        let to = s.queue[pos].1;
        if let Some((ff, t, _)) = &s.failed {
            if *ff == to && t.elapsed() < self.cfg.retry_after {
                return;
            }
        }
        let waiting = s.queue.iter().filter(|q| q.1 == to).count();
        self.start_switch(s, to, &format!("{waiting} requests waiting for it"));
    }

    fn start_switch(self: &Arc<Self>, s: &mut St, to: Family, why: &str) {
        if let Some(b) = self.driver.blocked(s.active) {
            if s.blocked.as_deref() != Some(b.as_str()) {
                eprintln!("switch: {} -> {} waits: {b}", name(s.active), to.name());
                s.blocked = Some(b);
            }
            return;
        }
        s.blocked = None;
        let from = s.active;
        s.switching = Some(to);
        eprintln!("switch: {} -> {}: {why}; switching", name(from), to.name());
        let sw = self.clone();
        tokio::spawn(async move { sw.run_switch(from, to).await });
    }

    /// With `home_after`: every TICK, an idle engine of the other family gives way to DeepSeek.
    pub fn spawn_home(self: &Arc<Self>) {
        let Some(after) = self.cfg.home_after else { return };
        let sw = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(TICK).await;
                let Some(sw) = sw.upgrade() else { return };
                let mut s = sw.st.lock().unwrap();
                let idle = s.last.elapsed();
                if s.active != Some(Family::DeepSeek)
                    && s.switching.is_none()
                    && s.in_flight == 0
                    && s.queue.is_empty()
                    && idle >= after
                    && !s.failed.as_ref().is_some_and(|f| f.1.elapsed() < sw.cfg.retry_after)
                {
                    sw.start_switch(&mut s, Family::DeepSeek, &format!("idle {} s", idle.as_secs()));
                }
            }
        });
    }

    async fn run_switch(self: Arc<Self>, from: Option<Family>, to: Family) {
        let t0 = Instant::now();
        let driver = self.driver.clone();
        let r = tokio::time::timeout(self.cfg.timeout, async {
            driver.run(to, from).await?;
            let t_cmd = t0.elapsed();
            while !driver.ready(to).await {
                tokio::time::sleep(TICK).await;
            }
            Ok::<_, String>(t_cmd)
        })
        .await
        .unwrap_or_else(|_| Err(format!("not ready within {} s", self.cfg.timeout.as_secs())));
        let total = t0.elapsed().as_secs_f64();
        {
            let mut s = self.st.lock().unwrap();
            s.switching = None;
            match r {
                Ok(t_cmd) => {
                    s.active = Some(to);
                    s.since = Instant::now();
                    s.last = Instant::now();
                    s.switches += 1;
                    s.failed = None;
                    let waiting = s.queue.iter().filter(|q| q.1 == to).count();
                    eprintln!(
                        "switch: {} -> {} done in {total:.1} s (command {:.1} s, then ready {:.1} s later); serving {waiting} waiting requests",
                        name(from),
                        to.name(),
                        t_cmd.as_secs_f64(),
                        total - t_cmd.as_secs_f64()
                    );
                }
                Err(e) => {
                    // which engine runs now is unknown: the next switch, either way, runs the command again
                    s.active = None;
                    eprintln!("switch: {} -> {} FAILED after {total:.1} s: {e}", name(from), to.name());
                    s.failed = Some((to, Instant::now(), e));
                }
            }
        }
        self.wake.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Switches take `delay`; records them; `fail` makes the next one fail.
    struct Fake {
        log: Mutex<Vec<String>>,
        delay: Duration,
        fail: AtomicBool,
        block: Mutex<Option<String>>,
    }

    impl Driver for Fake {
        fn run(&self, to: Family, from: Option<Family>) -> BoxFut<Result<(), String>> {
            self.log.lock().unwrap().push(format!("{}->{}", name(from), to.name()));
            let (d, fail) = (self.delay, self.fail.swap(false, Ordering::SeqCst));
            Box::pin(async move {
                tokio::time::sleep(d).await;
                if fail {
                    Err("boom".into())
                } else {
                    Ok(())
                }
            })
        }
        fn ready(&self, _: Family) -> BoxFut<bool> {
            Box::pin(async { true })
        }
        fn blocked(&self, from: Option<Family>) -> Option<String> {
            self.block.lock().unwrap().clone().filter(|_| from == Some(Family::DeepSeek))
        }
    }

    fn setup(dwell_ms: u64) -> (Arc<Switcher>, Arc<Fake>) {
        setup_with(dwell_ms, None)
    }

    fn setup_with(dwell_ms: u64, home_after: Option<Duration>) -> (Arc<Switcher>, Arc<Fake>) {
        let fake = Arc::new(Fake { log: Mutex::new(Vec::new()), delay: Duration::from_millis(50), fail: AtomicBool::new(false), block: Mutex::new(None) });
        let cfg = SwitchConfig {
            min_dwell: Duration::from_millis(dwell_ms),
            timeout: Duration::from_secs(5),
            retry_after: Duration::from_millis(200),
            max_wait: Duration::from_secs(10),
            home_after,
        };
        (Switcher::new(cfg, fake.clone(), Some(Family::DeepSeek)), fake)
    }

    /// Admit f in the background; the order of admissions goes into `order`.
    fn spawn_admit(sw: &Arc<Switcher>, f: Family, tag: &'static str, order: &Arc<Mutex<Vec<&'static str>>>, hold: Duration) -> tokio::task::JoinHandle<()> {
        let (sw, order) = (sw.clone(), order.clone());
        tokio::spawn(async move {
            let t = sw.admit(f).await.unwrap();
            order.lock().unwrap().push(tag);
            tokio::time::sleep(hold).await;
            drop(t);
        })
    }

    #[tokio::test]
    async fn the_other_family_waits_for_the_drain_and_requests_behind_it_wait_their_turn() {
        let (sw, fake) = setup(0);
        let order = Arc::new(Mutex::new(Vec::new()));
        let d1 = sw.try_admit(Family::DeepSeek).expect("active family passes at once");
        assert!(sw.try_admit(Family::Glm).is_none());
        let g1 = spawn_admit(&sw, Family::Glm, "g1", &order, Duration::from_millis(100));
        tokio::time::sleep(Duration::from_millis(20)).await;
        // a DeepSeek request after the waiting GLM one waits behind it, though DeepSeek is active
        assert!(sw.try_admit(Family::DeepSeek).is_none());
        let d2 = spawn_admit(&sw, Family::DeepSeek, "d2", &order, Duration::ZERO);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(order.lock().unwrap().is_empty(), "nothing passes while d1 is in flight");
        assert!(fake.log.lock().unwrap().is_empty(), "no switch while d1 is in flight");
        drop(d1);
        g1.await.unwrap();
        d2.await.unwrap();
        assert_eq!(*order.lock().unwrap(), ["g1", "d2"]);
        assert_eq!(*fake.log.lock().unwrap(), ["deepseek->glm", "glm->deepseek"]);
        assert_eq!(sw.active(), Some(Family::DeepSeek));
    }

    #[tokio::test]
    async fn within_the_dwell_the_family_switched_to_passes_the_waiting_other_family() {
        let (sw, fake) = setup(400);
        let order = Arc::new(Mutex::new(Vec::new()));
        let g1 = spawn_admit(&sw, Family::Glm, "g1", &order, Duration::from_millis(50));
        g1.await.unwrap(); // switched to GLM; the dwell runs from here
        let d1 = spawn_admit(&sw, Family::DeepSeek, "d1", &order, Duration::ZERO);
        tokio::time::sleep(Duration::from_millis(20)).await;
        // GLM requests still pass ahead of the waiting DeepSeek one
        let g2 = sw.try_admit(Family::Glm).expect("passes within the dwell");
        drop(g2);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(fake.log.lock().unwrap().len(), 1, "no switch back within the dwell");
        d1.await.unwrap();
        assert_eq!(*order.lock().unwrap(), ["g1", "d1"]);
        assert_eq!(*fake.log.lock().unwrap(), ["deepseek->glm", "glm->deepseek"]);
        // after the dwell, a GLM request queues behind the waiting DeepSeek one again
        let _d = sw.try_admit(Family::DeepSeek).unwrap();
        tokio::time::sleep(Duration::from_millis(450)).await;
        let _q = spawn_admit(&sw, Family::Glm, "g3", &order, Duration::ZERO);
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(sw.try_admit(Family::DeepSeek).is_none(), "the waiting GLM request is ahead");
    }

    #[tokio::test]
    async fn a_failed_switch_refuses_its_waiters_and_the_next_request_tries_again() {
        let (sw, fake) = setup(0);
        fake.fail.store(true, Ordering::SeqCst);
        let r = sw.admit(Family::Glm).await;
        assert!(r.err().unwrap().contains("boom"));
        assert_eq!(sw.active(), None);
        assert!(sw.status().contains("last_failure glm"));
        // within retry_after no new try; then the next one runs the command again
        let t = sw.admit(Family::Glm).await.unwrap();
        assert_eq!(*fake.log.lock().unwrap(), ["deepseek->glm", "none->glm"]);
        drop(t);
        assert_eq!(sw.active(), Some(Family::Glm));
    }

    #[tokio::test]
    async fn a_waiting_client_that_leaves_is_no_longer_counted() {
        let (sw, fake) = setup(0);
        let d = sw.try_admit(Family::DeepSeek).unwrap();
        let sw2 = sw.clone();
        let w = tokio::spawn(async move { sw2.admit(Family::Glm).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(sw.status().contains("glm 1"));
        w.abort();
        let _ = w.await;
        assert!(sw.status().contains("glm 0"));
        drop(d);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(fake.log.lock().unwrap().is_empty(), "nobody waits: no switch");
        assert!(sw.try_admit(Family::DeepSeek).is_some());
    }

    #[tokio::test]
    async fn no_switch_while_blocked_and_an_idle_glm_engine_goes_home() {
        let (sw, fake) = setup_with(0, Some(Duration::from_millis(300)));
        sw.spawn_home();
        *fake.block.lock().unwrap() = Some("the ring is lent".into());
        let sw2 = sw.clone();
        let w = tokio::spawn(async move { sw2.admit(Family::Glm).await.map(|t| t.family) });
        tokio::time::sleep(Duration::from_millis(700)).await;
        assert!(fake.log.lock().unwrap().is_empty(), "blocked: no switch");
        assert!(sw.status().contains("blocked: the ring is lent"));
        *fake.block.lock().unwrap() = None;
        assert_eq!(w.await.unwrap(), Ok(Family::Glm));
        assert_eq!(sw.active(), Some(Family::Glm));
        // idle for home_after: back to DeepSeek without a request asking
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert_eq!(*fake.log.lock().unwrap(), ["deepseek->glm", "glm->deepseek"]);
        assert_eq!(sw.active(), Some(Family::DeepSeek));
    }

    #[tokio::test]
    async fn a_request_waits_at_most_max_wait() {
        let (sw, fake) = setup(0);
        *fake.block.lock().unwrap() = Some("held".into());
        let mut s = SwitchConfig { ..sw.cfg.clone() };
        s.max_wait = Duration::from_millis(300);
        let sw = Switcher::new(s, fake.clone(), Some(Family::DeepSeek));
        let e = sw.admit(Family::Glm).await.err().unwrap();
        assert!(e.contains("waited 0 s for the glm engine") && e.contains("blocked: held"), "{e}");
        assert!(sw.status().contains("glm 0"));
    }

    #[tokio::test]
    async fn the_alias_follows_the_engine_and_never_switches() {
        let (sw, fake) = setup(0);
        assert_eq!(sw.alias_family(), Family::DeepSeek);
        let d = sw.try_admit(Family::DeepSeek).unwrap();
        // a GLM request waits for the drain: the switch is coming, the alias goes with it
        let sw2 = sw.clone();
        let g = tokio::spawn(async move { sw2.admit(Family::Glm).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(sw.alias_family(), Family::Glm);
        drop(d);
        // during the switch: the engine being switched to
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(sw.switching(), Some(Family::Glm));
        assert_eq!(sw.alias_family(), Family::Glm);
        g.await.unwrap().unwrap();
        assert_eq!(sw.alias_family(), Family::Glm);
        // an alias request resolved so passes at once: no switch back
        let t = sw.try_admit(sw.alias_family()).expect("the alias's family passes");
        drop(t);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(*fake.log.lock().unwrap(), ["deepseek->glm"]);
        // a failed switch leaves no engine: DeepSeek
        fake.fail.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = sw.admit(Family::DeepSeek).await;
        assert_eq!((sw.active(), sw.alias_family()), (None, Family::DeepSeek));
    }

    #[tokio::test]
    async fn the_ring_is_lent_to_gr8_during_a_switch_to_glm_and_a_dev_job_only_from_an_idle_deepseek() {
        let (sw, fake) = setup(0);
        assert_eq!(sw.lend_refusal(), None, "DeepSeek idle: a development job may have it");
        let d = sw.try_admit(Family::DeepSeek).unwrap();
        let sw2 = sw.clone();
        let g = tokio::spawn(async move { sw2.admit(Family::Glm).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(sw.lend_refusal().is_some(), "a GLM request waits: not to a job");
        drop(d);
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(sw.switching(), Some(Family::Glm));
        assert_eq!(sw.lend_refusal(), None, "the switch lends it to gr8");
        g.await.unwrap().unwrap();
        // GLM active with the gate held (its own lend): the switch back is not blocked
        *fake.block.lock().unwrap() = Some("held".into());
        assert!(sw.status().find("blocked").is_none());
        let t = sw.admit(Family::DeepSeek).await.unwrap();
        drop(t);
        // DeepSeek active and the gate held: that is a development job's lend, which blocks
        assert!(sw.status().contains("blocked: held"));
    }
}
