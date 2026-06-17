//! Bounded shutdown orchestration: register shutdown-ordered tasks, propagate
//! cancellation, drain in bounded time (no restart/health supervision).
//!
//! Cancelling registered tokens is the ONLY way `spawn_blocking` workers (engine,
//! training epoch loops) observe shutdown -- `abort()` cannot stop a running
//! blocking closure. The mic arbitrator (not a `JoinHandle`, so outside this
//! registry) must be silenced before [`DrainRegistry::shutdown_and_drain`] so
//! consumers drain into a quiet pipeline.

use std::time::Duration;
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;

/// Task tier -- controls per-task drain budget.
#[derive(Debug, Clone, Copy)]
enum Tier {
    /// Long-running consumers; 5 s budget covers a blocking worker checking the
    /// token between ~250 ms iterations plus in-flight settling.
    Major,
    /// Heartbeats/reaper; 1 s budget = one interval tick to hit the cancel arm.
    Background,
}

struct Slot {
    name: &'static str,
    tier: Tier,
    inner: Box<dyn DrainableHandle + Send + 'static>,
}

/// Type-erases each `JoinHandle` so the registry's vec is heterogeneous.
trait DrainableHandle {
    fn abort_handle(&self) -> AbortHandle;

    fn drain(self: Box<Self>) -> futures_util::future::BoxFuture<'static, TaskOutcome>;
}

enum TaskOutcome {
    Clean,
    Error(String),
    /// Aborted externally or by a racing abort -- never our per-task-timeout abort,
    /// which fires only after `timeout` dropped this await so its `JoinError` is
    /// never seen here.
    Cancelled,
    /// Panic payload can't cross `JoinError`; stringify its `Display` (carries
    /// the panic location).
    Panicked(String),
}

struct ResultHandle<T, E>
where
    T: Send + 'static,
    E: Send + std::fmt::Display + 'static,
{
    handle: JoinHandle<Result<T, E>>,
}

impl<T, E> DrainableHandle for ResultHandle<T, E>
where
    T: Send + 'static,
    E: Send + std::fmt::Display + 'static,
{
    fn abort_handle(&self) -> AbortHandle {
        self.handle.abort_handle()
    }

    fn drain(self: Box<Self>) -> futures_util::future::BoxFuture<'static, TaskOutcome> {
        Box::pin(async move {
            match self.handle.await {
                Ok(Ok(_)) => TaskOutcome::Clean,
                Ok(Err(e)) => TaskOutcome::Error(e.to_string()),
                Err(je) if je.is_cancelled() => TaskOutcome::Cancelled,
                Err(je) => TaskOutcome::Panicked(je.to_string()),
            }
        })
    }
}

struct UnitHandle {
    handle: JoinHandle<()>,
}

impl DrainableHandle for UnitHandle {
    fn abort_handle(&self) -> AbortHandle {
        self.handle.abort_handle()
    }

    fn drain(self: Box<Self>) -> futures_util::future::BoxFuture<'static, TaskOutcome> {
        Box::pin(async move {
            match self.handle.await {
                Ok(()) => TaskOutcome::Clean,
                Err(je) if je.is_cancelled() => TaskOutcome::Cancelled,
                Err(je) => TaskOutcome::Panicked(je.to_string()),
            }
        })
    }
}

/// Owns registered handles and (optionally) per-task tokens; [`Self::shutdown_and_drain`]
/// cancels every token then joins every handle under per-task budgets capped by an
/// outer total budget.
pub struct DrainRegistry {
    slots: Vec<Slot>,
    /// Cancelled in bulk before any per-handle drain.
    cancel_tokens: Vec<CancellationToken>,
    pre_drain_hooks: Vec<Box<dyn FnOnce() -> usize + Send + 'static>>,
}

impl std::fmt::Debug for DrainRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrainRegistry")
            .field("registered", &self.slots.len())
            .field("cancel_tokens", &self.cancel_tokens.len())
            .field("pre_drain_hooks", &self.pre_drain_hooks.len())
            .finish_non_exhaustive()
    }
}

impl Default for DrainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DrainRegistry {
    /// Best-effort SYNC cleanup for early-return (`?`) paths that never reach the
    /// async `shutdown_and_drain`: without cancelling, the engine `spawn_blocking`
    /// worker blocks runtime drop until the OS kills the process. Idempotent with
    /// `shutdown_and_drain`.
    fn drop(&mut self) {
        self.cancel_all();
        // catch_unwind each hook: a hook panic while Drop runs during unwinding
        // (poisoned JobRegistry mutex) would double-panic and abort. Hooks own
        // their state, so AssertUnwindSafe holds.
        let mut hook_total = 0usize;
        for hook in self.pre_drain_hooks.drain(..) {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(hook));
            match result {
                Ok(count) => hook_total = hook_total.saturating_add(count),
                Err(payload) => {
                    let detail = crate::common::error::panic_payload_to_string(&*payload);
                    tracing::error!(
                        target: "acoustics",
                        panic = %detail,
                        "pre_drain hook panicked during Drop; continuing teardown",
                    );
                }
            }
        }
        if hook_total > 0 {
            tracing::info!(
                target: "acoustics",
                cancelled = hook_total,
                "drain (Drop): pre_drain hooks cancelled subsystems",
            );
        }
    }
}

impl DrainRegistry {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            cancel_tokens: Vec::new(),
            pre_drain_hooks: Vec::new(),
        }
    }

    /// Register a token with no handle so an early-return still has [`Drop`] cancel
    /// it. Re-pushing the same shared token is harmless (`cancel()` is a no-op).
    pub fn register_cancel_token(&mut self, token: CancellationToken) {
        self.cancel_tokens.push(token);
    }

    /// Register a major-tier task (5 s budget). `spawn_blocking` wrappers must
    /// prefer [`Self::register_major_with_token`]: `abort()` cannot stop a running
    /// blocking closure.
    pub fn register_major<T, E>(&mut self, name: &'static str, handle: JoinHandle<Result<T, E>>)
    where
        T: Send + 'static,
        E: Send + std::fmt::Display + 'static,
    {
        self.slots.push(Slot {
            name,
            tier: Tier::Major,
            inner: Box::new(ResultHandle { handle }),
        });
    }

    /// Register a major-tier task AND its token; the token is cancelled before any
    /// handle is awaited so the task and its blocking closures exit within budget.
    pub fn register_major_with_token<T, E>(
        &mut self,
        name: &'static str,
        handle: JoinHandle<Result<T, E>>,
        token: CancellationToken,
    ) where
        T: Send + 'static,
        E: Send + std::fmt::Display + 'static,
    {
        self.cancel_tokens.push(token);
        self.register_major(name, handle);
    }

    /// Register a background-tier task (1 s budget) whose only shutdown signal is
    /// the cancellation token.
    pub fn register_bg(&mut self, name: &'static str, handle: JoinHandle<()>) {
        self.slots.push(Slot {
            name,
            tier: Tier::Background,
            inner: Box::new(UnitHandle { handle }),
        });
    }

    /// Register a hook run BEFORE per-task drain and AFTER `cancel_all`; it sets
    /// the cancel flag on active training jobs and returns its count.
    pub fn register_pre_drain_hook<F>(&mut self, hook: F)
    where
        F: FnOnce() -> usize + Send + 'static,
    {
        self.pre_drain_hooks.push(Box::new(hook));
    }

    /// Cancel every registered token (idempotent) and return the count; does not
    /// await handles.
    pub fn cancel_all(&self) -> usize {
        for token in &self.cancel_tokens {
            token.cancel();
        }
        self.cancel_tokens.len()
    }

    /// Bounded drain: cancel tokens, run pre-drain hooks, then await every handle
    /// under per-task budgets capped by `outer_budget` (per-task timeouts warn +
    /// abort). Returns `true` on clean drain.
    ///
    /// On `false` the caller MUST flush non-supervised tail work (log appender
    /// guard, mic-arbitrator stop) then hard-exit: `spawn_blocking` workers can't
    /// be aborted from async, so dropping the runtime would block until they finish,
    /// defeating the bound. This fn does NOT exit itself (that would skip the
    /// caller's cleanup). The caller must also silence non-supervised producers (mic
    /// arbitrator, not a `JoinHandle`, so the registry can't enforce it) first.
    #[must_use = "outer-budget overrun requires the caller to flush \
                  non-supervised tail work and then std::process::exit(1)"]
    pub async fn shutdown_and_drain(mut self, outer_budget: Duration) -> bool {
        const MAJOR_BUDGET: Duration = Duration::from_secs(5);
        const BG_BUDGET: Duration = Duration::from_secs(1);

        // Take slots out (Drop blocks partial-moves) so end-of-scope Drop sees an
        // empty Vec and is a no-op.
        let slots = std::mem::take(&mut self.slots);
        let cancelled = self.cancel_all();
        if cancelled > 0 {
            tracing::debug!(
                target: "acoustics",
                tokens = cancelled,
                "drain: cancelled supervised cancellation tokens",
            );
        }
        // Hooks run before the outer-budget timeout (which wraps only
        // `drain_all_inner`), so cap each via spawn_blocking + timeout, else a
        // synchronously-wedged hook (poisoned mutex) overshoots unbounded. A hung
        // hook leaks its thread until exit.
        const HOOK_BUDGET: Duration = Duration::from_secs(1);
        let mut hook_total = 0usize;
        for hook in self.pre_drain_hooks.drain(..) {
            let spawn = tokio::task::spawn_blocking(hook);
            match tokio::time::timeout(HOOK_BUDGET, spawn).await {
                Ok(Ok(n)) => hook_total = hook_total.saturating_add(n),
                Ok(Err(e)) => {
                    tracing::warn!(
                        target: "acoustics",
                        err = %e,
                        "drain: pre-drain hook panicked; continuing",
                    );
                }
                Err(_) => {
                    tracing::warn!(
                        target: "acoustics",
                        budget_secs = HOOK_BUDGET.as_secs(),
                        "drain: pre-drain hook did not return within budget; abandoning",
                    );
                }
            }
        }
        if hook_total > 0 {
            tracing::info!(
                target: "acoustics",
                cancelled = hook_total,
                "drain: pre-drain hooks cancelled subsystems",
            );
        }

        // Collect abort handles before slots move into the drain futures: an
        // outer-budget timeout drops every pending `drain_one` before its inner
        // per-task `abort()` runs, so without this a budget overrun leaves async
        // tasks running across the hard-exit.
        let aborts: Vec<AbortHandle> = slots.iter().map(|s| s.inner.abort_handle()).collect();

        // `true` iff drained within the per-task budget; `false` on a per-task
        // timeout (force-aborted = not clean, and abort can't stop a
        // `spawn_blocking` worker).
        let drain_one = |slot: Slot| async move {
            let Slot { name, tier, inner } = slot;
            let budget = match tier {
                Tier::Major => MAJOR_BUDGET,
                Tier::Background => BG_BUDGET,
            };
            let abort = inner.abort_handle();
            match tokio::time::timeout(budget, inner.drain()).await {
                Ok(outcome) => {
                    log_outcome(name, outcome);
                    true
                }
                Err(_) => {
                    tracing::warn!(
                        target: "acoustics",
                        task = name,
                        budget_secs = budget.as_secs(),
                        "task did not exit within shutdown budget; aborting",
                    );
                    abort.abort();
                    false
                }
            }
        };

        let drain_all_inner = futures_util::future::join_all(slots.into_iter().map(drain_one));
        match tokio::time::timeout(outer_budget, drain_all_inner).await {
            Ok(results) => {
                // Within the outer budget, but a per-task force-abort isn't clean
                // (its `spawn_blocking` worker keeps running), so any per-task
                // timeout reports unclean.
                let timed_out = results.iter().filter(|completed| !**completed).count();
                if timed_out > 0 {
                    tracing::warn!(
                        target: "acoustics",
                        timed_out,
                        "drain: task(s) exceeded their per-task budget and were force-aborted; reporting unclean drain",
                    );
                    return false;
                }
                true
            }
            Err(_) => {
                // Outer budget expired with tasks pending: abort every collected
                // handle so async tasks stop holding state across the caller's
                // hard-exit (no-op for already-drained handles).
                for abort in &aborts {
                    abort.abort();
                }
                tracing::warn!(
                    target: "acoustics",
                    outer_budget_secs = outer_budget.as_secs(),
                    aborted = aborts.len(),
                    "drain did not complete within outer budget; aborted pending async tasks",
                );
                false
            }
        }
    }
}

fn log_outcome(name: &'static str, outcome: TaskOutcome) {
    match outcome {
        TaskOutcome::Clean => {
            tracing::debug!(target: "acoustics", task = name, "task ended cleanly");
        }
        TaskOutcome::Error(e) => {
            tracing::warn!(target: "acoustics", task = name, err = %e, "task returned an error");
        }
        TaskOutcome::Cancelled => {
            tracing::debug!(target: "acoustics", task = name, "task was cancelled");
        }
        TaskOutcome::Panicked(je) => {
            tracing::error!(target: "acoustics", task = name, err = %je, "task panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    #[tokio::test]
    async fn cancel_all_cancels_registered_tokens() {
        let mut reg = DrainRegistry::new();
        let token_a = CancellationToken::new();
        let token_b = CancellationToken::new();
        let handle_a: JoinHandle<Result<(), &'static str>> = {
            let token = token_a.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                Ok(())
            })
        };
        let handle_b: JoinHandle<Result<(), &'static str>> = {
            let token = token_b.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                Ok(())
            })
        };
        reg.register_major_with_token("a", handle_a, token_a.clone());
        reg.register_major_with_token("b", handle_b, token_b.clone());

        let cancelled = reg.cancel_all();
        assert_eq!(cancelled, 2);
        assert!(token_a.is_cancelled());
        assert!(token_b.is_cancelled());
    }

    #[tokio::test]
    async fn pre_drain_hooks_run_during_shutdown() {
        let mut reg = DrainRegistry::new();
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let c = counter.clone();
            reg.register_pre_drain_hook(move || {
                c.fetch_add(7, Ordering::SeqCst);
                7
            });
        }
        {
            let c = counter.clone();
            reg.register_pre_drain_hook(move || {
                c.fetch_add(3, Ordering::SeqCst);
                3
            });
        }
        // Empty drain (no slots) still fires hooks.
        let drained_clean = reg.shutdown_and_drain(Duration::from_secs(5)).await;
        assert!(drained_clean, "empty drain must complete within budget");
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[tokio::test]
    async fn token_cancelled_task_drains_cleanly_within_budget() {
        let mut reg = DrainRegistry::new();
        let token = CancellationToken::new();
        let started = Arc::new(Notify::new());
        let started_for_task = started.clone();
        let handle: JoinHandle<Result<(), &'static str>> = {
            let token = token.clone();
            tokio::spawn(async move {
                started_for_task.notify_one();
                token.cancelled().await;
                Ok(())
            })
        };
        reg.register_major_with_token("cooperative", handle, token);
        // Park on the token first so the drain race is deterministic.
        started.notified().await;

        let start = std::time::Instant::now();
        let drained_clean = reg.shutdown_and_drain(Duration::from_secs(2)).await;
        let elapsed = start.elapsed();
        assert!(
            drained_clean,
            "cooperative drain must complete within budget"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "cooperative drain took {elapsed:?}; expected <<2 s outer budget",
        );
    }

    /// Guards the `timed_out > 0` branch: a task blowing its per-task budget
    /// reports unclean even when it finishes within the outer budget.
    #[tokio::test]
    async fn drain_reports_unclean_when_task_exceeds_per_task_budget() {
        let mut reg = DrainRegistry::new();
        // Bg tier (1 s budget), no token: only the per-task timeout can stop it.
        let handle: JoinHandle<()> = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        reg.register_bg("runaway", handle);

        let start = std::time::Instant::now();
        // 4 s outer >> 1 s per-task budget exercises the per-task path.
        let drained_clean = reg.shutdown_and_drain(Duration::from_secs(4)).await;
        let elapsed = start.elapsed();
        assert!(
            !drained_clean,
            "a task that blew its per-task budget must report an unclean drain",
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "expected the ~1 s per-task timeout, not the outer budget; got {elapsed:?}",
        );
    }

    /// Guards Drop against leaking the engine `spawn_blocking` worker behind an
    /// un-cancelled token: dropping without `shutdown_and_drain` (early `?`) still
    /// cancels every token and runs every hook.
    #[tokio::test]
    async fn drop_cancels_tokens_and_runs_pre_drain_hooks() {
        let token_a = CancellationToken::new();
        let token_b = CancellationToken::new();
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let mut reg = DrainRegistry::new();
            let handle_a: JoinHandle<Result<(), &'static str>> = {
                let token = token_a.clone();
                tokio::spawn(async move {
                    token.cancelled().await;
                    Ok(())
                })
            };
            reg.register_major_with_token("a", handle_a, token_a.clone());
            // Bare token (master-shutdown pattern), no handle.
            reg.register_cancel_token(token_b.clone());
            let c = counter.clone();
            reg.register_pre_drain_hook(move || {
                c.fetch_add(11, Ordering::SeqCst);
                11
            });
        }
        assert!(
            token_a.is_cancelled(),
            "registered cancel token must fire on Drop"
        );
        assert!(
            token_b.is_cancelled(),
            "bare cancel token must fire on Drop"
        );
        assert_eq!(
            counter.load(Ordering::SeqCst),
            11,
            "pre-drain hook must run on Drop",
        );
    }
}
