//! Daemon wiring: single-binary boot (config, logging, audio capture,
//! status monitor + boot recovery, stream router, optional inference
//! engine, opus encoder, API/stream listeners), then wait-for-signal +
//! bounded drain. `--check` (debug only) boots, runs `check_seconds()`,
//! prints a [`crate::status::StatusSnapshot`] JSON, exits 0 iff healthy.

// `#[global_allocator]` (mimalloc) lives on the binary, NOT here: a library
// allocator conflicts with test binaries linking `acoustics_lab`. mimalloc =
// aggressive OS-return (vs ptmalloc fragmentation under the converter/training
// spike) + no background thread contesting the audio capture thread.
use crate::daemon::drain_registry;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use clap::Parser;
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::audio_buffer::AudioBuffer;
use crate::audio_io::mic_arbitrator::{
    CandidateSource, ChannelSelection, MicArbitrator, MicArbitratorConfig, MicCandidate,
    MicCatalogue, MicPolicy, MicSelection, MicSettingsStore,
};
use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use crate::config::{
    Config, ConfigCell, LaunchConfig, MicSettingsCell, MicSettingsHandle,
    validate_policy_against_catalogue,
};
use crate::file_mgr::{AdmissionCfg, FsService, FsServiceImpl};
use crate::inference::{HotHead, InferenceEngine};
use crate::opus_stream as opus;
use crate::status::{Heartbeat, StatusMonitor};
use crate::stream_io::StreamRouter;
use crate::training::JobRegistry;

// Per-thread CPU topology + SCHED_FIFO priorities. Core 0 unpinned for
// kernel/IRQ. audio(50) > inference(30) > tokio (SCHED_OTHER): audio drops are
// unrecoverable, inference frames recoverable, tokio jitter-tolerant; cap audio
// below kernel RT (99). Without CAP_SYS_NICE the RT calls fall back to
// SCHED_OTHER and pid=0 self-pins still succeed.

const MIC_ARBITRATOR_PIN_CORE: usize = 1;
const MIC_ARBITRATOR_RT_PRIORITY: i32 = 50;

const INFERENCE_PIN_CORE: usize = 2;
const INFERENCE_RT_PRIORITY: i32 = 30;

const TOKIO_PIN_CORE: usize = 3;

/// 1 s: fresh enough for `GET /api/v1/status` within the HTTP budget, slow
/// enough that per-tick `compose` stays off the idle CPU profile.
const HEARTBEAT_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// Inference frame channel capacity, distinct from the audio-only
/// `ApiCfg::broadcast_capacity` (inference emits at 1-4 Hz vs audio's 50 Hz);
/// 64 slots ~= 16-64 s of history. If made operator-configurable MUST stay
/// <= `MAX_BROADCAST_CAPACITY` (1024) to avoid boot-time OOM.
const INFERENCE_BROADCAST_CAPACITY: usize = 64;

#[derive(Parser, Debug)]
#[command(
    name = "acoustics_lab",
    version,
    about = "On-device audio classification daemon."
)]
struct Cli {
    /// Workspace root the daemon owns wholly (config, backbone, workspaces,
    /// active head, logs, default UDS); `config.toml` auto-materialized within
    /// it on first boot.
    #[arg(long)]
    workspace: PathBuf,

    /// Launch-time TOML (mic + backbone catalogues, stream binds,
    /// `[head.default]`), outside the workspace tree. Read once at boot, edits
    /// ignored until restart.
    #[arg(long)]
    config: PathBuf,

    /// Tokio worker count. 2 leaves headroom for the std-thread mic
    /// arbitrator + the inference `spawn_blocking` task.
    #[arg(long, default_value_t = 2)]
    worker_threads: usize,

    /// Override `[api].tcp_bind`. Test-harness escape hatch: each test passes
    /// `127.0.0.1:0` for an ephemeral port (else parallel test binaries race
    /// the fixed port).
    #[arg(long)]
    tcp_bind: Option<String>,

    /// Synthesize a 1 kHz tone: override the catalogue with one in-memory
    /// `mock:0` candidate AND pin policy to `Fixed { id: "mock:0" }`.
    #[cfg(debug_assertions)]
    #[arg(long)]
    mock_audio: bool,

    /// Skip InferenceEngine startup (hosts without librknnrt). Debug only.
    #[cfg(debug_assertions)]
    #[arg(long)]
    no_inference: bool,

    /// Boot, run `--check-seconds` (default 5), print one `StatusSnapshot`
    /// JSON, exit 0 iff all subsystems healthy. Debug only.
    #[cfg(debug_assertions)]
    #[arg(long)]
    check: bool,

    #[cfg(debug_assertions)]
    #[arg(long, default_value_t = 5)]
    check_seconds: u64,
}

/// Release builds collapse the debug-only flags to const accessors so
/// `async_main` reads them via `args.<flag>()` without `#[cfg]` at each site.
#[cfg(not(debug_assertions))]
impl Cli {
    const fn mock_audio(&self) -> bool {
        false
    }
    const fn no_inference(&self) -> bool {
        false
    }
    const fn check(&self) -> bool {
        false
    }
    const fn check_seconds(&self) -> u64 {
        0
    }
}

#[cfg(debug_assertions)]
impl Cli {
    fn mock_audio(&self) -> bool {
        self.mock_audio
    }
    fn no_inference(&self) -> bool {
        self.no_inference
    }
    fn check(&self) -> bool {
        self.check
    }
    fn check_seconds(&self) -> u64 {
        self.check_seconds
    }
}

/// Top-level entry point the thin `acousticsd` binary calls.
pub fn run() -> Result<()> {
    let args = Cli::parse();
    // `max_blocking_threads` / `thread_stack_size` stay at tokio defaults
    // (512 / 2 MiB): a smaller stack risks SIGSEGV-without-traceback under deep
    // recursion (Burn recorder, prost, serde, axum/tower); a smaller blocking
    // cap deadlocks `tokio::fs::*` (config write + status sample + uploads +
    // inference already saturate a small cap).
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.worker_threads.max(1))
        .thread_name("al-rt")
        .enable_all()
        .on_thread_start(|| {
            // `eprintln!` not `tracing`: fires inside `runtime.build()`, before
            // the subscriber is installed. Pin failure non-fatal.
            if let Err(e) = crate::sched::pin_to_core(TOKIO_PIN_CORE) {
                eprintln!(
                    "acousticsd: tokio worker pin_to_core({}) failed: {}; \
                     continuing on default placement",
                    TOKIO_PIN_CORE, e,
                );
            }
        })
        .build()
        .context("build tokio runtime")?;
    runtime.block_on(async_main(args))
}

async fn async_main(args: Cli) -> Result<()> {
    // Two config layers cross-validated at every boundary (boot, `watch_with`
    // reload, `POST /mic/policy`): launch (immutable `--config`) and user-pref
    // (hot-reloadable + API-mutable `<workspace>/config.toml`: mic policy +
    // inference cadence only). Workspace root comes from `--workspace`, NOT
    // persisted (a stored copy would drift from the CLI on next boot).
    std::fs::create_dir_all(&args.workspace)
        .with_context(|| format!("create workspace dir {}", args.workspace.display()))?;
    let workspace_root = args.workspace.clone();
    let user_config_path = workspace_root.join("config.toml");
    if paths_may_alias(&user_config_path, &args.config) {
        anyhow::bail!(
            "--config (launch TOML) must not point at <workspace>/config.toml \
             (the user-pref TOML lives there); pass distinct paths",
        );
    }
    let launch = load_or_init_launch_config(&args.config)?;
    // Outer `Arc` so the API (`Arc<dyn ConfigHandle>`) and in-crate consumers
    // (`MicSettingsCell`, watcher) share one pointer (dyn coercion at the API
    // boundary).
    let config = Arc::new(load_or_init_config(&user_config_path)?);
    let snap = config.snapshot();
    let api = launch.api.clone();
    let output_inference = launch.output.inference.clone();
    let default_head = launch.head.default.clone();

    // Fatal: a `Fixed { id }` for a missing catalogue entry spins the
    // arbitrator inert with rate-limited warns.
    if !args.mock_audio()
        && let Err(e) = validate_policy_against_catalogue(&snap.mic, &launch.mic, &user_config_path)
    {
        anyhow::bail!(
            "{e}; either fix the policy in {} or add the candidate in {}",
            user_config_path.display(),
            args.config.display(),
        );
    }
    // The config watcher installs below, AFTER the live ArcSwaps exist so it
    // can capture them; otherwise file edits would update only `config.inner`
    // and never reach the arbitrator / inference engine.
    let log_dir = workspace_root.join("logs");
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("create log dir {}", log_dir.display()))?;
    // Plaintext rolling log (no systemd/journald; operators tail it). Daily
    // rotation, max 7 files, auto-pruned by the appender.
    let appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("acousticsd")
        .filename_suffix("log")
        .max_log_files(7)
        .build(&log_dir)
        .with_context(|| format!("build rolling appender at {}", log_dir.display()))?;
    // Bounded (2048 lines ~= 8 MiB) + `lossy(true)`: the default is unbounded
    // so a panic dump grows without limit, and blocking the producer deadlocks
    // if the producer IS the panic-dump path.
    let (writer, log_guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
        .buffered_lines_limit(2048)
        .lossy(true)
        .finish(appender);
    // Guard must outlive `async_main`; dropping it kills the appender.
    let _log_guard_holder = LogGuardHolder { _guard: log_guard };

    // Leading bare `info` is the GLOBAL default unlisted targets inherit.
    // Override via `ACOUSTICS_LOG` (NOT `RUST_LOG`, so operator filters don't
    // hit bundled crates). Canonical target `"acoustics"` is operator-facing
    // and must stay stable across renames.
    let env_filter = EnvFilter::try_from_env("ACOUSTICS_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info,acoustics=info,inference=info,opus_stream=info"));

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(writer)
                .with_ansi(false),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr));
    registry.try_init().context("install tracing subscriber")?;

    // Global panic hook, installed BEFORE any task spawn. Step order survives
    // panic-formatter failure (OOM, panic in a `Debug` impl): `eprintln!` one
    // line FIRST (locked stderr is the most resilient sink), THEN structured
    // `tracing::error!` with backtrace. Default hook NOT chained (would double
    // stderr volume during panic storms).
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let payload = info.payload();
        let msg = crate::common::error::panic_payload_to_string(payload);
        eprintln!("acousticsd: PANIC at {location}: {msg}");
        // Backtrace gated on RUST_BACKTRACE. Flat snake_case field names:
        // `tracing`'s macro matcher rejects dotted/quoted forms and snake_case
        // is portable across fmt + json.
        tracing::error!(
            target: "acoustics",
            panic_location = %location,
            panic_payload  = %msg,
            backtrace = ?std::backtrace::Backtrace::capture(),
            "thread panicked",
        );
    }));

    tracing::info!(
        target: "acoustics",
        version = env!("CARGO_PKG_VERSION"),
        workspace = %args.workspace.display(),
        config = %args.config.display(),
        log_dir = %log_dir.display(),
        mock_audio = args.mock_audio(),
        no_inference = args.no_inference(),
        "daemon starting",
    );

    // Trust posture: the daemon terminates no auth, so a non-loopback bind
    // warns loudly at boot. Either transport may be absent (UDS-only `[api]`).
    {
        match args.tcp_bind.as_deref().or(api.tcp_bind.as_deref()) {
            Some(bind) => {
                let resolved_loopback = bind_is_loopback(bind);
                tracing::info!(
                    target: "acoustics",
                    tcp_bind = %bind,
                    tcp_loopback = resolved_loopback,
                    "trust posture: no in-daemon auth",
                );
                if !resolved_loopback {
                    tracing::warn!(
                        target: "acoustics",
                        tcp_bind = %bind,
                        "non-loopback bind: HTTP + WebSocket API exposed to the network with no in-daemon auth",
                    );
                }
            }
            None => {
                tracing::info!(
                    target: "acoustics",
                    "trust posture: UDS-only (no TCP bind)",
                );
            }
        }
    }

    // Defense in depth on `bind_uds`'s parent-confinement: mkdir the UDS parent
    // 0o700 if absent, BEFORE `try_bind_uds` (which re-checks + warns on a
    // world-writable dir). An early-return `?` orphaning the dir is benign
    // (deterministic daemon-owned paths, idempotent mkdir). Covers both `[api]`
    // and `[output.inference]`, either optional.
    for (label, uds_path) in [
        ("api", api.uds_path.as_deref()),
        (
            "output.inference",
            output_inference.as_ref().map(|o| o.uds_path.as_path()),
        ),
    ] {
        if let Some(path) = uds_path {
            ensure_uds_parent_dir(path)
                .with_context(|| format!("ensure {label} UDS parent for {}", path.display()))?;
        }
    }

    // 262_144 = 2^18 ~= 5.94 s @ 44.1 kHz mono. AudioBuffer needs power-of-two
    // capacity (wrap via `head & (cap - 1)`); rounded up from the 5 s target
    // with headroom for the longest peek window.
    let audio_buf = AudioBuffer::new(262_144);
    let writer = audio_buf.take_writer();

    // Single shared capture-timing anchor: the arbitrator publishes a fresh one
    // after each `Writer::push`; the opus encoder + inference engine read it to
    // stamp frames with their covered audio's first-sample capture time (not
    // emit time).
    let timing_anchor = crate::common::time::shared_timing_anchor();

    // `MicSettings` bundles the read-only launch catalogue (Arc, so a policy
    // update rebuilds without cloning `Vec<MicCandidate>`) and the hot-swappable
    // policy. `--mock-audio` hijacks BOTH at boot only; on-disk TOMLs untouched,
    // later hot-reloads flow into `policy` only.
    let (catalogue_arc, policy) = if args.mock_audio() {
        let catalogue = MicCatalogue {
            candidates: vec![MicCandidate {
                id: MicId::from_static("mock:0"),
                source: CandidateSource::Mock {
                    waveforms: vec![Waveform::Sine {
                        freq_hz: 1_000.0,
                        amplitude: 0.25,
                    }],
                    period_size: 512,
                    sample_rate: crate::common::dims::SampleRate::VALUE,
                },
                channels: vec![0],
            }],
        };
        // Catch a daemon synthesis bug before the arbitrator silently fails at
        // first read.
        if let Err((id, err)) = catalogue.validate() {
            anyhow::bail!("daemon-built mock-audio catalogue invalid (candidate {id}: {err})");
        }
        let policy = MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("mock:0"),
            },
            channel: ChannelSelection::Auto,
        };
        (Arc::new(catalogue), policy)
    } else {
        if launch.mic.candidates.is_empty() {
            tracing::warn!(
                target: "acoustics",
                "launch catalogue is empty; the arbitrator will run without an active source. \
                 Add at least one [[mic.candidates]] entry to {} (debug builds may also pass \
                 --mock-audio).",
                args.config.display(),
            );
        }
        (Arc::new(launch.mic.clone()), snap.mic.clone())
    };
    // `MicSettingsCell` projects to `Arc<dyn MicSettingsStore>` (wait-free
    // reads -> arbitrator) and `Arc<dyn MicSettingsHandle>` (read+write+persist
    // -> API + watcher); both alias the same cell, so a handle mutation shows
    // on the next store snapshot.
    let mic_settings_cell = Arc::new(MicSettingsCell::new(
        catalogue_arc.clone(),
        policy,
        config.clone(),
    ));
    let mic_settings_store: Arc<dyn MicSettingsStore> = mic_settings_cell.clone();
    let mic_settings_handle: Arc<dyn MicSettingsHandle> = mic_settings_cell.clone();
    let inference_cfg_arcswap = Arc::new(ArcSwap::from_pointee(snap.inference));

    // Hot-reload of the user TOML must update the live ArcSwaps the arbitrator
    // + inference engine read (without this hook only API mutations would).
    // Runs with `mutate_lock` held: keep cheap and do NOT re-enter
    // `config.mutate*` (deadlock). All-or-nothing -- on Err the watcher discards
    // the WHOLE reload, else a partial apply strands `inner` behind the engine
    // or commits a bogus policy `GET /config` misreports. `--mock-audio` skips
    // only the policy update (on-disk may disagree with the mock).
    let mic_handle_for_reload = mic_settings_handle.clone();
    let inference_for_reload = inference_cfg_arcswap.clone();
    let mock_audio = args.mock_audio();
    // Format once so the move-closure captures owned strings.
    let user_config_path_for_log = user_config_path.display().to_string();
    let launch_config_path_for_log = args.config.display().to_string();
    let _config_watcher = config
        .watch_with(
            move |cfg| -> Result<(), crate::config::ConfigValidationError> {
                // Pre-allocate (the only OOM-prone step) BEFORE any side effect
                // so an OOM panic leaves policy + inner at OLD (catch_unwind ->
                // Err -> discard); after the set it would leave live ArcSwaps
                // NEW while inner stayed OLD.
                let next_inference = Arc::new(cfg.inference);

                // `try_set_policy_no_persist`, not a persisting mutate: the
                // observed TOML IS the truth, and re-entering `mutate` (holding
                // `mutate_lock`) trips `ReentrantMutate`; the cell's validator
                // re-runs the catalogue cross-check, on Err we discard. Gate on
                // actual change since the call unconditionally bumps the
                // ResourceVersion + swaps a fresh Arc, so a cadence-only reload
                // would spuriously satisfy a `GET /mic?min_version=N` wait.
                if !mock_audio
                    && cfg.mic != mic_handle_for_reload.snapshot().policy
                    && let Err(e) = mic_handle_for_reload.try_set_policy_no_persist(cfg.mic.clone())
                {
                    return Err(crate::config::ConfigValidationError::Callback(format!(
                        "{e}; edit the policy in {} to match the catalogue, OR add the missing \
                         candidate to {} and restart the daemon",
                        user_config_path_for_log, launch_config_path_for_log,
                    )));
                }
                // Independent ArcSwap, no cross-store atomicity needed.
                inference_for_reload.store(next_inference);
                Ok(())
            },
        )
        .context("install config watcher")?;

    let monitor = StatusMonitor::new();
    // Background sampler keeps `/api/v1/status` reads wait-free; 500 ms balances
    // freshness vs sysinfo refresh cost. Aborted by `Drop for Inner` when the
    // last clone drops at exit.
    monitor.start_sampler(
        Some(workspace_root.clone()),
        std::time::Duration::from_millis(500),
    );

    // Process-wide `WorkspaceMetrics` global (`OnceLock::set`, as are the 7
    // `install_*_hook` calls). A second in-process daemon Errs `install_global`
    // and credits the FIRST daemon's handle (its own local Arc gets zero
    // increments); counter-isolation tests run in separate processes.
    let workspace_metrics = std::sync::Arc::new(crate::status::WorkspaceMetrics::new());
    let _ =
        crate::status::workspace_metrics::install_global(std::sync::Arc::clone(&workspace_metrics));
    install_workspace_metrics_hooks(&workspace_metrics);
    // Config-reload counters on `/status` let operators detect a rejected
    // reload: a parse/validate/callback fail leaves the snapshot at OLD, so a
    // later API mutation writes OLD+delta, silently clobbering the operator's
    // NEW disk edit.
    let (config_reload_succeeded, config_reload_rejected) = config.reload_counter_arcs();
    let _ = crate::status::config_metrics::install_global(crate::status::ConfigReloadHandles::new(
        config_reload_succeeded,
        config_reload_rejected,
    ));
    let shutdown = CancellationToken::new();

    // The mic arbitrator (not a `JoinHandle`) is silenced separately BEFORE the
    // registry drain.
    let mut drain_registry = drain_registry::DrainRegistry::new();
    // Enrol the master token UP FRONT so any early-return before the first
    // `register_major_with_token` still cancels on `Drop`, honoring the
    // registry's "cancel tokens before dropping handles" contract.
    drain_registry.register_cancel_token(shutdown.clone());

    // The arbitrator thread IS the audio producer (no MPSC channel): it writes
    // the AudioBuffer directly.
    let arb_cfg = MicArbitratorConfig {
        hysteresis_db: 3.0,
        dwell: Duration::from_millis(250),
        rms_window: Duration::from_millis(100),
        mic_failover_after: Duration::from_secs(2),
        failover_retry_interval: Duration::from_secs(1),
        // Best-effort pin + RT bump; failure logs WARN and the thread
        // continues on default placement.
        sched_pin: Some(MIC_ARBITRATOR_PIN_CORE),
        sched_priority: Some(MIC_ARBITRATOR_RT_PRIORITY),
        timing_anchor: Some(timing_anchor.clone()),
    };
    // `start` self-validates and panics on rejection, so a new spawn site can't
    // bypass the gate.
    let arb_handle = MicArbitrator::start(writer, mic_settings_store.clone(), arb_cfg);

    let capture_hb = monitor
        .register("audio_capture")
        .context("register audio_capture subsystem")?;
    {
        // Three regimes: mock (healthy); no candidates (empty catalogue ->
        // `degraded(_, "no_device")`, a misconfig that shouldn't flip healthy
        // off); candidate-driven (normal healthy/unhealthy switching via the
        // head-advance pump).
        let no_mic_configured = !args.mock_audio()
            && mic_settings_store
                .snapshot()
                .catalogue
                .candidates
                .is_empty();
        let initial_detail: &'static str = if args.mock_audio() {
            "mock:0 / 44.1 k"
        } else if no_mic_configured {
            "no candidates configured"
        } else {
            "candidate-driven"
        };
        let initial = if no_mic_configured {
            Heartbeat::degraded(initial_detail, "no_device")
        } else {
            Heartbeat::ok(initial_detail)
        };
        capture_hb.send(initial).ok();
        let buf_for_watch = audio_buf.clone();
        let mut last_head = buf_for_watch.head();
        drain_registry.register_bg(
            "audio_capture_hb",
            spawn_heartbeat_loop(
                shutdown.clone(),
                capture_hb.clone(),
                HEARTBEAT_REFRESH_INTERVAL,
                move || {
                    let cur_head = buf_for_watch.head();
                    let advanced = cur_head > last_head;
                    let delta = cur_head.saturating_sub(last_head);
                    last_head = cur_head;
                    if advanced {
                        // Per-tick delta, no "/s": under Skip missed-tick a tick
                        // may span >1 s so "/s" over-counts.
                        Heartbeat::ok(format!("{initial_detail}; head={cur_head} (+{delta})"))
                    } else if no_mic_configured {
                        // Misconfig stays degraded ("unhealthy" is transient).
                        Heartbeat::degraded(
                            format!("{initial_detail}; head stuck at {cur_head}"),
                            "no_device",
                        )
                    } else {
                        Heartbeat::unhealthy(format!(
                            "no audio for >=1 s; head stuck at {cur_head}"
                        ))
                    }
                },
            ),
        );
    }

    // StreamRouter built before inference so the engine publishes into its
    // `infer_tx`. Per-listener policies from `ApiCfg::{tcp,uds}_policy`.
    let stream_router = StreamRouter::with_capacities_and_policy(
        api.broadcast_capacity,
        INFERENCE_BROADCAST_CAPACITY,
        api.tcp_policy.clone(),
    );
    let opus_audio_tx: broadcast::Sender<bytes::Bytes> = stream_router.audio_tx();
    let audio_subs_rx: watch::Receiver<usize> = stream_router.audio_subscribers();
    let infer_tx_for_engine: broadcast::Sender<bytes::Bytes> = stream_router.infer_tx();

    // HotHead + InferenceEngine + listener binds parallelized: `boot_inference`
    // (~80-200 ms Burn `.mpk` parse) runs concurrently with the independent
    // bind syscalls; the router mounts after all three futures resolve. Boot
    // recovery keeps the daemon up on failure (boot-without-inference);
    // `boot_recovery_unhealthy` -> heartbeat.
    let (mut boot_recovery_report, boot_recovery_unhealthy) =
        run_boot_recovery(&workspace_root, default_head.as_ref(), &workspace_metrics);

    let mut head = synthetic_head_for_dev()?;
    let inference_hb = monitor
        .register("inference")
        .context("register inference subsystem")?;

    // Parse `tcp_bind` early so the bind future starts alongside boot_inference
    // and a parse failure surfaces here, not buried in join!'s tuple.
    // `--tcp-bind` overrides `[api].tcp_bind`; `None` = no TCP listener
    // (`ApiCfg::validate` guarantees >= 1 transport).
    let resolved_tcp_bind = args.tcp_bind.as_deref().or(api.tcp_bind.as_deref());
    let tcp_addr: Option<std::net::SocketAddr> = match resolved_tcp_bind {
        Some(bind) => Some(
            bind.parse()
                .with_context(|| format!("parse tcp_bind {bind}"))?,
        ),
        None => None,
    };

    let want_inference = !args.no_inference()
        && boot_recovery_unhealthy.is_none()
        && !launch.backbone.is_empty()
        && head_files_present(&workspace_root);
    let inference_fut = async {
        if want_inference {
            boot_inference(
                &workspace_root,
                launch.backbone.clone(),
                &audio_buf,
                inference_hb.clone(),
                inference_cfg_arcswap.clone(),
                infer_tx_for_engine.clone(),
                shutdown.clone(),
                timing_anchor.clone(),
            )
            .await
            .map(Some)
        } else {
            Ok(None)
        }
    };
    // Each bind is optional and concurrent with `boot_inference`; an absent
    // listener resolves to `None` to keep the join tuple shape.
    let tcp_bind_fut = async {
        match tcp_addr {
            Some(addr) => Some(tokio::net::TcpListener::bind(addr).await),
            None => None,
        }
    };
    let api_uds_bind_fut = async {
        match api.uds_path.as_deref() {
            Some(path) => Some(try_bind_uds(path, api.uds_mode).await),
            None => None,
        }
    };
    let output_uds_bind_fut = async {
        match output_inference.as_ref() {
            Some(out) => Some(try_bind_uds(&out.uds_path, out.uds_mode).await),
            None => None,
        }
    };

    let (inference_outcome, tcp_bind_res, api_uds_bind_res, output_uds_bind_res) = tokio::join!(
        inference_fut,
        tcp_bind_fut,
        api_uds_bind_fut,
        output_uds_bind_fut
    );

    // Process inference FIRST so `head` is set before any consumer (opus_stream,
    // AppState) reads it. Some=succeeded, None=skipped, Err=boot failed ->
    // continue without inference.
    match inference_outcome {
        Ok(Some((engine_handle, hb_pump_handle, real_head))) => {
            head = real_head;
            // The spawn_blocking closure sees shutdown only via the token it
            // polls between iterations, so register the token alongside the
            // handle for a clean exit within the 5 s budget.
            drain_registry.register_major_with_token("inference", engine_handle, shutdown.clone());
            // MAJOR-tier so its 5 s drain budget covers the engine's (the pump
            // exits on `engine_hb_tx` channel close); a bg-tier 1 s budget would
            // abort it before the engine's clean exit, defeating the
            // wedge-protection contract.
            drain_registry.register_major("inference_hb_pump", hb_pump_handle);
            inference_hb.send(Heartbeat::ok("engine spawned")).ok();
        }
        Err(e) => {
            tracing::error!(
                target: "acoustics",
                err = %e,
                "inference boot failed; daemon will continue without it",
            );
            let reason: Arc<str> = format!("boot failed: {e}").into();
            inference_hb
                .send(Heartbeat::unhealthy(reason.to_string()))
                .ok();
            // 1 Hz refresh so the unhealthy entry shows a current age, not a
            // stale timestamp suggesting further breakage.
            let reason_arc = reason.clone();
            drain_registry.register_bg(
                "inference_status_refresh",
                spawn_heartbeat_loop(
                    shutdown.clone(),
                    inference_hb.clone(),
                    HEARTBEAT_REFRESH_INTERVAL,
                    move || Heartbeat::unhealthy(reason_arc.to_string()),
                ),
            );
        }
        Ok(None) => {
            // Skipped; the synthetic head stays so the API + /inference/*
            // endpoints respond. Voluntary -> healthy; involuntary -> degraded
            // (no backbone/head) or unhealthy (recovery failed). Reason strings
            // are operator-API contract; tests pin exact text.
            enum SkipKind {
                Voluntary,
                NoBackbone,
                NoHead,
                RecoveryUnhealthy(Arc<str>),
            }
            // Precedence: voluntary > NoBackbone (pure config, reported
            // regardless of recovery) > RecoveryUnhealthy (bundled-default
            // fallback failed) > NoHead.
            let kind = if args.no_inference() {
                SkipKind::Voluntary
            } else if launch.backbone.is_empty() {
                SkipKind::NoBackbone
            } else if let Some(reason) = boot_recovery_unhealthy {
                SkipKind::RecoveryUnhealthy(reason.into())
            } else {
                SkipKind::NoHead
            };
            let detail: &'static str = match kind {
                SkipKind::Voluntary => "skipped via --no-inference",
                SkipKind::NoBackbone => {
                    "backbone catalogue is empty -- daemon running without inference"
                }
                SkipKind::NoHead => "head files missing -- daemon running without inference",
                SkipKind::RecoveryUnhealthy(_) => {
                    "boot recovery unhealthy -- daemon running without inference"
                }
            };
            tracing::info!(
                target: "acoustics",
                detail,
                "inference engine NOT started",
            );
            let make_hb = move || match &kind {
                SkipKind::Voluntary => Heartbeat::ok(detail),
                SkipKind::NoBackbone => Heartbeat::degraded(detail, "no_backbone"),
                SkipKind::NoHead => Heartbeat::degraded(detail, "no_head"),
                SkipKind::RecoveryUnhealthy(reason) => {
                    Heartbeat::unhealthy(format!("{detail}: {reason}"))
                }
            };
            inference_hb.send(make_hb()).ok();
            // 1 Hz refresh so the skip-state entry doesn't go stale.
            drain_registry.register_bg(
                "inference_skip_refresh",
                spawn_heartbeat_loop(
                    shutdown.clone(),
                    inference_hb.clone(),
                    HEARTBEAT_REFRESH_INTERVAL,
                    make_hb,
                ),
            );
        }
    }

    let opus_reader = audio_buf.reader();
    let opus_token = shutdown.clone();
    let opus_hb = monitor
        .register("opus_stream")
        .context("register opus_stream subsystem")?;
    opus_hb.send(Heartbeat::ok("waiting for subscriber")).ok();
    // Stalled-encoder detection: encoder bumps this once per packet; the
    // heartbeat reads the delta and reports unhealthy when subscribers are
    // present but no packet emerged for >= 2 s (paused 0-subs stays healthy).
    // Counts encoder-progress, not delivery.
    let opus_packets_encoded = Arc::new(std::sync::atomic::AtomicU64::new(0));
    {
        let mut audio_subs_rx_for_hb = stream_router.audio_subscribers();
        let opus_packets_for_hb = opus_packets_encoded.clone();
        let mut last_packets: u64 = 0;
        let mut last_advance_at = std::time::Instant::now();
        drain_registry.register_bg("opus_status_refresh", spawn_heartbeat_loop(
            shutdown.clone(),
            opus_hb.clone(),
            HEARTBEAT_REFRESH_INTERVAL,
            move || {
                let n = *audio_subs_rx_for_hb.borrow_and_update();
                let cur_packets =
                    opus_packets_for_hb.load(std::sync::atomic::Ordering::Relaxed);
                let now = std::time::Instant::now();
                if cur_packets != last_packets {
                    last_packets = cur_packets;
                    last_advance_at = now;
                }
                // Advance the stall anchor on EVERY paused tick so paused time
                // never counts toward the 2 s budget and the first active tick
                // after idle starts clean (no spurious stall while
                // `OpusEngine::new` builds its first packet). Anchoring to the
                // LAST paused tick (NOT the resume edge) is load-bearing: a
                // resume-edge reset would grant a fresh window each flap cycle
                // and mask a real stall, so even a flapping `1,1,0` trips the
                // gate. Flap-to-zero every tick is a 1 Hz-sampler blind spot.
                if n == 0 {
                    last_advance_at = now;
                }
                let stalled_for = now.saturating_duration_since(last_advance_at);
                if n == 0 {
                    Heartbeat::ok("paused (0 subscribers)")
                } else if stalled_for >= Duration::from_secs(2) {
                    Heartbeat::unhealthy(format!(
                        "no packets for {}ms with {n} subscriber{}; encoder stalled at packets={cur_packets}",
                        stalled_for.as_millis(),
                        plural_s(n),
                    ))
                } else {
                    Heartbeat::ok(format!(
                        "encoding ({n} subscriber{}, packets={cur_packets})",
                        plural_s(n),
                    ))
                }
            },
        ));
    }
    let opus_packets_for_run = opus_packets_encoded.clone();
    let opus_timing_anchor = timing_anchor.clone();
    drain_registry.register_major(
        "opus_stream",
        tokio::spawn(async move {
            opus::run(
                opus_reader,
                audio_subs_rx,
                opus_audio_tx,
                opus_token,
                opus_packets_for_run,
                // Producer's anchor -> per-packet first-sample capture time, not
                // emit time.
                Some(opus_timing_anchor),
            )
            .await
        }),
    );

    // Admission caps from the launch `[file]` block (default 256 MiB / 4
    // uploads); saturating cast usize -> u32 at the boundary.
    let admission = AdmissionCfg {
        max_upload_bytes: launch.file.max_upload_bytes,
        max_concurrent_uploads: u32::try_from(launch.file.max_concurrent_uploads)
            .unwrap_or(u32::MAX),
    };
    // One `JobRegistry` shared by workspace-side admission paths and the
    // api-side `GET /jobs` / SSE routes.
    let jobs_registry = std::sync::Arc::new(crate::file_mgr::JobRegistry::new(
        crate::file_mgr::JobRegistryCfg::default(),
    ));
    // Single-shot: registry accepts only the first boot-recovery report.
    if let Some(report) = boot_recovery_report.take() {
        jobs_registry.record_boot_recovery(report);
    }
    // One Arc shared between `FsServiceImpl` (internal on delete_head /
    // publish_*) and `AppState::active_mutex` (`POST /active` holds it
    // end-to-end) -> single serialization order.
    let active_mutex: Arc<parking_lot::Mutex<()>> = Arc::new(parking_lot::Mutex::new(()));
    let fs_impl = FsServiceImpl::with_admission_jobs_and_active_mutex(
        workspace_root.clone(),
        admission,
        jobs_registry.clone(),
        active_mutex.clone(),
    );
    let files: Arc<dyn FsService> = Arc::new(fs_impl);
    let training = JobRegistry::new();
    let training_hb = monitor
        .register("training")
        .context("register training subsystem")?;
    training_hb.send(Heartbeat::ok("idle")).ok();
    // Job-aware heartbeat ("idle" / "running N" / "cancelling N") so a
    // running-during-shutdown job is visible vs idle.
    {
        let training_for_hb = training.clone();
        let shutdown_for_training_hb = shutdown.clone();
        drain_registry.register_bg(
            "training_status_refresh",
            spawn_heartbeat_loop(
                shutdown.clone(),
                training_hb.clone(),
                HEARTBEAT_REFRESH_INTERVAL,
                move || {
                    let active = training_for_hb.active_count();
                    let cancelling = shutdown_for_training_hb.is_cancelled();
                    match (active, cancelling) {
                        (0, _) => Heartbeat::ok("idle"),
                        (n, false) => Heartbeat::ok(format!("running {n} job{}", plural_s(n),)),
                        (n, true) => Heartbeat::ok(format!(
                            "cancelling {n} job{} (shutdown in progress)",
                            plural_s(n),
                        )),
                    }
                },
            ),
        );
    }
    // Pre-drain hook sets the cancel flag on every active training job BEFORE
    // any handle is awaited -- the ONLY signal reaching the spawn_blocking
    // `finetune::run` workers (the master async-only CancellationToken doesn't
    // enter the blocking closure).
    {
        let training_for_drain = training.clone();
        drain_registry
            .register_pre_drain_hook(move || training_for_drain.cancel_all_for_shutdown());
    }

    // Reaper: every 5 min drop finished training entries older than 1 h (never
    // a running job). Cheap idempotent DashMap walk; a missed Skip tick is
    // harmless.
    {
        let registry = training.clone();
        drain_registry.register_bg(
            "training_reaper",
            spawn_interval_loop(shutdown.clone(), Duration::from_secs(300), move || {
                let n = registry.reap_finished(Duration::from_secs(3600));
                if n > 0 {
                    tracing::info!(
                        target: "acoustics",
                        reaped = n,
                        "training: pruned finished job entries older than 1 h",
                    );
                }
                async {}
            }),
        );
    }

    // Storage reaper: hourly `.tmp/` orphan sweep. Safety net for a long-running
    // daemon that leaked (boot recovery covers the crashed case); the 24 h
    // `.tmp/` age is far above any in-flight op (uploads finish in seconds) so
    // the sweep can't race a producer.
    {
        const STORAGE_REAP_INTERVAL: Duration = Duration::from_secs(3600);
        const TMP_AGE_THRESHOLD: Duration = Duration::from_secs(24 * 3600);
        let workspace_root_for_reap = workspace_root.clone();
        let metrics_for_storage = workspace_metrics.clone();
        drain_registry.register_bg(
            "storage_reaper",
            spawn_interval_loop(shutdown.clone(), STORAGE_REAP_INTERVAL, move || {
                // Per-tick clones move into the async block; outer originals
                // stay live for the next tick (cheap, 1/hr).
                let root = workspace_root_for_reap.clone();
                let metrics = metrics_for_storage.clone();
                async move {
                    let cfg = crate::file_mgr::SweepConfig {
                        tmp_age: TMP_AGE_THRESHOLD,
                    };
                    // Blocking I/O on the spawn_blocking pool keeps the async
                    // worker free; log a `JoinError` panic.
                    let outcome = tokio::task::spawn_blocking(move || {
                        crate::file_mgr::sweep_once(&root, &cfg)
                    })
                    .await;
                    match outcome {
                        Ok(Ok(report)) => {
                            metrics
                                .record_storage_sweep(report.tmp_orphans_reaped, report.failures);
                            if report.did_work() || report.failures > 0 {
                                tracing::info!(
                                    target: "acoustics",
                                    tmp_orphans_reaped = report.tmp_orphans_reaped,
                                    workspaces_scanned = report.workspaces_scanned,
                                    failures = report.failures,
                                    "storage reaper sweep completed",
                                );
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(
                                target: "acoustics",
                                err = %e,
                                "storage reaper sweep failed",
                            );
                        }
                        Err(je) => {
                            tracing::error!(
                                target: "acoustics",
                                err = %je,
                                "storage reaper blocking task panicked",
                            );
                        }
                    }
                }
            }),
        );
    }

    // `.snapshot()` per `/status` is Relaxed atomic reads.
    let broadcast_lag_reader: std::sync::Arc<dyn crate::common::traits::lag_source::LagSource> =
        std::sync::Arc::new(stream_router.lag_counters());

    // Trait objects so API code + tests see only the trait surface.
    let head_store: std::sync::Arc<dyn crate::common::traits::head_store::HeadStore> =
        std::sync::Arc::new(head);
    let monitor_reporter: std::sync::Arc<dyn crate::status::StatusReporter> =
        std::sync::Arc::new(monitor.clone());
    let training_registry: std::sync::Arc<dyn crate::training::TrainingRegistry> =
        std::sync::Arc::new(training);
    // Trainer's backbone = the first `kind = "burn"` candidate (training always
    // needs a Burn `.mpk` regardless of inference build features). `None` ->
    // the training route 404s at request time so the daemon still boots.
    let training_backbone_path = launch
        .backbone
        .candidates
        .iter()
        .find(|c| c.kind == crate::inference::BackboneKind::Burn)
        .map(|c| c.path.clone());
    let app_state = crate::api::AppState {
        config: config.clone(),
        head: head_store,
        mic_settings: mic_settings_handle,
        inference_cfg: inference_cfg_arcswap,
        files,
        monitor: monitor_reporter,
        training: training_registry,
        broadcast_lag_reader,
        active_mutex,
        default_head: default_head.clone(),
        training_backbone_path,
        jobs: jobs_registry,
    };
    let api_router = crate::api::router_v1_nested(app_state);
    // Same `api_router` on every listener (no CORS; the reverse proxy owns
    // cross-origin). Per-listener WS routers share the broadcast channels but
    // carry their own `TransportPolicy` (TCP strict, UDS relaxes the
    // subprotocol check). The shutdown token threads in so detached WS tasks
    // wind down within the drain budget.
    let tcp_local: Option<std::net::SocketAddr> = match (tcp_addr, tcp_bind_res) {
        (Some(addr), Some(bind_res)) => {
            // TCP bind failure is fatal (`?`): an operator who asked for a TCP
            // listener gets a hard error, not a silent API-less daemon.
            let tcp = bind_res.with_context(|| format!("bind {addr}"))?;
            let local = tcp
                .local_addr()
                .with_context(|| format!("local addr for {addr}"))?;
            tracing::info!(target: "acoustics", addr = %local, "TCP listener bound");
            let tcp_app: axum::Router = api_router.clone().merge(
                stream_router
                    .router_with_policy_and_shutdown(api.tcp_policy.clone(), shutdown.clone()),
            );
            let tcp_token = shutdown.clone();
            // Token alongside the handle so `DrainRegistry::Drop` cancels it on
            // an early-return `?` below; else the serve task runs un-signaled
            // until runtime-drop aborts it abruptly, killing in-flight requests.
            drain_registry.register_major_with_token(
                "stream_io_tcp",
                tokio::spawn({
                    let tcp_token = tcp_token.clone();
                    async move {
                        crate::stream_io::serve_tcp(tcp, tcp_app, tcp_token)
                            .await
                            .map_err(|e| anyhow::anyhow!("tcp serve: {e}"))
                    }
                }),
                tcp_token,
            );
            Some(local)
        }
        _ => None,
    };

    // [api] UDS listener. Non-fatal on failure: the daemon continues with
    // whatever other listeners came up.
    let api_uds_bound = match api_uds_bind_res {
        Some(Ok(uds)) => {
            let uds_app: axum::Router = api_router.clone().merge(
                stream_router
                    .router_with_policy_and_shutdown(api.uds_policy.clone(), shutdown.clone()),
            );
            let uds_token = shutdown.clone();
            // Same Drop-cancellation rationale as the TCP arm above.
            drain_registry.register_major_with_token(
                "stream_io_uds",
                tokio::spawn({
                    let uds_token = uds_token.clone();
                    async move {
                        crate::stream_io::serve_uds(uds, uds_app, uds_token)
                            .await
                            .map_err(|e| anyhow::anyhow!("uds serve: {e}"))
                    }
                }),
                uds_token,
            );
            true
        }
        Some(Err(e)) => {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                path = ?api.uds_path,
                "api uds bind failed; continuing without the [api] UDS listener",
            );
            false
        }
        None => false,
    };

    // [output.inference] raw push listener, WebSocket-free:
    // `serve_inference_uds` streams length-prefixed `Envelope` frames. Non-fatal
    // on failure.
    let output_uds_bound = match output_uds_bind_res {
        Some(Ok(uds)) => {
            let infer_tx_for_raw = stream_router.infer_tx();
            let out_token = shutdown.clone();
            drain_registry.register_major_with_token(
                "stream_io_inference_uds",
                tokio::spawn({
                    let out_token = out_token.clone();
                    async move {
                        crate::stream_io::serve_inference_uds(uds, infer_tx_for_raw, out_token)
                            .await
                            .map_err(|e| anyhow::anyhow!("inference-uds serve: {e}"))
                    }
                }),
                out_token,
            );
            true
        }
        Some(Err(e)) => {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                path = ?output_inference.as_ref().map(|o| o.uds_path.display().to_string()),
                "output.inference uds bind failed; continuing without the raw inference socket",
            );
            false
        }
        None => false,
    };

    // No `[api]` listener -> no HTTP/WS surface. Only reachable when a UDS-only
    // `[api]`'s bind failed (TCP bind failure is fatal via `?`).
    if tcp_local.is_none() && !api_uds_bound {
        tracing::warn!(
            target: "acoustics",
            "no [api] listener bound; the HTTP API + WebSocket streams are unreachable",
        );
    }

    let stream_io_hb = monitor
        .register("stream_io")
        .context("register stream_io subsystem")?;
    let initial_stream_detail = {
        let mut parts: Vec<String> = Vec::new();
        if let Some(addr) = tcp_local {
            parts.push(format!("TCP {addr}"));
        }
        if api_uds_bound && let Some(path) = api.uds_path.as_deref() {
            parts.push(format!("API-UDS {}", path.display()));
        }
        if output_uds_bound && let Some(out) = output_inference.as_ref() {
            parts.push(format!("OUT-UDS {}", out.uds_path.display()));
        }
        if parts.is_empty() {
            "no listeners bound".to_string()
        } else {
            parts.join("; ")
        }
    };
    stream_io_hb
        .send(Heartbeat::ok(initial_stream_detail.clone()))
        .ok();
    {
        let mut audio_subs = stream_router.audio_subscribers();
        let mut infer_subs = stream_router.infer_subscribers();
        let initial = initial_stream_detail.clone();
        drain_registry.register_bg(
            "stream_io_status_refresh",
            spawn_heartbeat_loop(
                shutdown.clone(),
                stream_io_hb.clone(),
                HEARTBEAT_REFRESH_INTERVAL,
                move || {
                    let a = *audio_subs.borrow_and_update();
                    let i = *infer_subs.borrow_and_update();
                    Heartbeat::ok(format!("{initial} | audio={a} infer={i}"))
                },
            ),
        );
    }

    let exit_code = if args.check() {
        let res = run_check_mode(
            &monitor,
            &shutdown,
            Duration::from_secs(args.check_seconds()),
        )
        .await;
        if res.is_err() { 1 } else { 0 }
    } else {
        wait_for_signal().await;
        0
    };

    // Second-signal escalator: "Ctrl-C, Ctrl-C" hard-kills a wedged drain
    // (tokio buffers only one signal per kind, so a fresh listener is needed);
    // runtime-drop aborts this task on a clean drain. DEBOUNCE because a
    // supervisor forwards the terminal Ctrl-C as SIGTERM AND the terminal's
    // SIGINT also reaches a same-process-group child, so ONE keystroke arrives
    // as TWO signals ms apart; without debounce the escalator misreads the echo
    // and hard-exits mid-drain, aborting the engine's cooperative shutdown
    // (librknnrt "invalid device fd!"). A real escalation is seconds apart, so
    // ignore a second signal within `SECOND_SIGNAL_DEBOUNCE` of drain start;
    // then stderr log + `process::exit(1)` (operator hard-exit, not `abort`).
    if !args.check() {
        // Above supervisor forward latency (sub-ms) yet below a human re-press
        // (>=1 s): suppresses the echo, allows a real double-tap.
        const SECOND_SIGNAL_DEBOUNCE: Duration = Duration::from_secs(1);
        let drain_started = std::time::Instant::now();
        tokio::spawn(async move {
            loop {
                wait_for_signal().await;
                if drain_started.elapsed() >= SECOND_SIGNAL_DEBOUNCE {
                    break;
                }
                // Spurious echo: drop and re-arm. A fresh `wait_for_signal()`
                // blocks on a NEW signal (the consumed echo isn't redelivered),
                // so no busy-spin.
            }
            eprintln!("acousticsd: second signal received during drain; hard-exit");
            std::process::exit(1);
        });
    }

    tracing::info!(target: "acoustics", "shutdown requested; cancelling tasks");
    // Silence the producer BEFORE draining consumers so it stops appending to
    // the audio + lag buffers; else it keeps capturing for the whole drain
    // window, filling ALSA buffers and spamming overruns. `signal_stop` is
    // non-blocking; the run loop observes it within ~one capture period (~12 ms).
    arb_handle.signal_stop();
    shutdown.cancel();

    // Concurrent drain under a 10 s outer budget (5 s major / 1 s bg per-task).
    // Outer-cap expiry returns `false`; the non-registered tail (mic stop +
    // log-guard flush) still runs before hard-exit so logs aren't truncated.
    let drained_clean = drain_registry
        .shutdown_and_drain(Duration::from_secs(10))
        .await;

    // `MicArbitrator::stop` does a synchronous `thread::join()`, so run it on
    // the blocking pool (a direct call would block the tokio worker); the run
    // loop saw `signal_stop` during the consumer drain so the join completes
    // within ~one capture period. Cap the wall-clock so a wedged producer (stuck
    // `snd_pcm_close`) can't block shutdown past the drain envelope / skip the
    // appender flush below.
    const ARB_STOP_BUDGET: std::time::Duration = std::time::Duration::from_secs(2);
    let arb_wedged = match tokio::time::timeout(
        ARB_STOP_BUDGET,
        tokio::task::spawn_blocking(move || arb_handle.stop()),
    )
    .await
    {
        Ok(Ok(())) => false,
        Ok(Err(e)) => {
            eprintln!("acousticsd: mic arbitrator stop join error: {e}");
            tracing::warn!(target: "acoustics", err = %e, "mic arbitrator stop join error");
            false
        }
        Err(_elapsed) => {
            eprintln!(
                "acousticsd: mic arbitrator stop exceeded {:?}; skipping",
                ARB_STOP_BUDGET
            );
            tracing::warn!(
                target: "acoustics",
                budget_ms = ARB_STOP_BUDGET.as_millis() as u64,
                "mic arbitrator stop wall-clock budget exceeded; skipping",
            );
            true
        }
    };
    drop(_log_guard_holder);

    if arb_wedged {
        // spawn_blocking is non-abortable; returning would wedge runtime-drop
        // joining the blocking pool. Force-exit so systemd restarts.
        std::process::exit(1);
    }
    if !drained_clean {
        // Drain expiry takes precedence over `exit_code`: a partial drain is the
        // more load-bearing supervisor diagnostic.
        std::process::exit(1);
    }
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
    Ok(())
}

struct LogGuardHolder {
    _guard: tracing_appender::non_blocking::WorkerGuard,
}

/// Periodic task firing `tick` every `period`, exiting on
/// `shutdown.cancelled()`: `Skip` missed-tick (a runtime stall must not burst
/// on resume), biased select against shutdown, initial-tick discard so the
/// first real tick lands at `period` not t=0. Cancel-safe; `tick` is NOT raced
/// -- an in-flight call runs to completion, shutdown is seen at the next
/// iteration, an over-long body is aborted at the drain budget.
fn spawn_interval_loop<F, Fut>(
    shutdown: CancellationToken,
    period: Duration,
    mut tick: F,
) -> JoinHandle<()>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                _ = interval.tick() => {}
            }
            tick().await;
        }
    })
}

/// Heartbeat specialisation of [`spawn_interval_loop`]: each tick sends
/// `compose()` to `sender`.
fn spawn_heartbeat_loop<F>(
    shutdown: CancellationToken,
    sender: watch::Sender<Heartbeat>,
    period: Duration,
    mut compose: F,
) -> JoinHandle<()>
where
    F: FnMut() -> Heartbeat + Send + 'static,
{
    spawn_interval_loop(shutdown, period, move || {
        let _ = sender.send(compose());
        async {}
    })
}

/// Install all 7 `file_mgr::metrics_hooks::install_*` forwarders into
/// `WorkspaceMetrics`. Lives in `daemon`, the only layer allowed to name BOTH
/// `status` AND `file_mgr` (`dependency_edge_guard` forbids both edges
/// elsewhere).
fn install_workspace_metrics_hooks(
    workspace_metrics: &std::sync::Arc<crate::status::WorkspaceMetrics>,
) {
    let m = std::sync::Arc::clone(workspace_metrics);
    crate::file_mgr::metrics_hooks::install_workspace_core_write_hook(move |d| {
        m.record_workspace_core_write(d);
    });
    let m = std::sync::Arc::clone(workspace_metrics);
    crate::file_mgr::metrics_hooks::install_head_index_write_hook(move |d| {
        m.record_head_index_write(d);
    });
    let m = std::sync::Arc::clone(workspace_metrics);
    crate::file_mgr::metrics_hooks::install_upload_hook(move |bytes| {
        m.record_upload(bytes);
    });
    let m = std::sync::Arc::clone(workspace_metrics);
    crate::file_mgr::metrics_hooks::install_dataset_mutation_rejected_hook(move || {
        m.record_dataset_mutation_rejected();
    });
    // Separate counters so operators distinguish per-tree contention.
    let m = std::sync::Arc::clone(workspace_metrics);
    crate::file_mgr::metrics_hooks::install_converter_mutation_rejected_hook(move || {
        m.record_converter_mutation_rejected();
    });
    let m = std::sync::Arc::clone(workspace_metrics);
    crate::file_mgr::metrics_hooks::install_job_events_dropped_hook(move |n| {
        m.record_job_events_dropped(n);
    });
    // `JsonlEventLog::open` runs `enforce_keep_last_n` on a new
    // `<job_id>.jsonl`; forward pruned + failure counts.
    let m = std::sync::Arc::clone(workspace_metrics);
    crate::file_mgr::metrics_hooks::install_logs_pruned_hook(move |pruned, failures| {
        m.record_logs_pruned(pruned, failures);
    });
}

/// Ensure the parent dir of a first-boot config TOML exists. No-op for a bare
/// `path` (cwd-relative; handled by load+persist) or an existing parent.
/// `label` is spliced into the `with_context` chain on failure.
fn ensure_config_parent_dir(path: &std::path::Path, label: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} parent dir {}", parent.display()))?;
    }
    Ok(())
}

/// Load `<workspace>/config.toml` (user-pref TOML), or write defaults on first
/// boot. A stale `workspace_root` key fails `deny_unknown_fields` (the root
/// lives in CLI state, not here).
fn load_or_init_config(path: &std::path::Path) -> Result<ConfigCell> {
    if path.exists() {
        ConfigCell::load(path).with_context(|| format!("load config {}", path.display()))
    } else {
        ensure_config_parent_dir(path, "config")?;
        let cfg = Config::default_for();
        let h = ConfigCell::from_value(cfg, path.to_path_buf())
            .context("first-boot default config failed validation")?;
        h.persist().context("persist initial config")?;
        Ok(h)
    }
}

/// Load the launch-time config, or materialize defaults if absent. Plain
/// `LaunchConfig` (no watcher/mutate machinery; the launch layer is immutable).
fn load_or_init_launch_config(path: &std::path::Path) -> Result<LaunchConfig> {
    if path.exists() {
        // Repair launch-owned UDS parent dirs BEFORE `LaunchConfig::load`
        // validates, so a TOML whose socket parent was swept still boots. Probes
        // `[api]` + `[output.inference]` uds_path; best-effort (parse failure
        // falls through to the loader).
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(value) = toml::from_str::<toml::Value>(&text)
        {
            let probe = |table: Option<&toml::Value>| {
                if let Some(uds_str) = table
                    .and_then(|t| t.get("uds_path"))
                    .and_then(|p| p.as_str())
                {
                    let _ = ensure_uds_parent_dir(&PathBuf::from(uds_str));
                }
            };
            probe(value.get("api"));
            probe(value.get("output").and_then(|o| o.get("inference")));
        }
        LaunchConfig::load(path).with_context(|| format!("load launch config {}", path.display()))
    } else {
        ensure_config_parent_dir(path, "launch config")?;
        let cfg = LaunchConfig::default_for();
        // Prepare UDS parent dirs for whichever sockets the first-boot defaults
        // declare (today none).
        for uds_path in [
            cfg.api.uds_path.as_deref(),
            cfg.output.inference.as_ref().map(|o| o.uds_path.as_path()),
        ]
        .into_iter()
        .flatten()
        {
            ensure_uds_parent_dir(uds_path)?;
        }
        cfg.persist(path).context("persist initial launch config")?;
        tracing::info!(
            target: "acoustics",
            path = %path.display(),
            "launch config absent; wrote first-boot defaults",
        );
        Ok(cfg)
    }
}

/// `""` for exactly one, `"s"` otherwise (heartbeat detail strings).
fn plural_s(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// True when two CLI paths clearly point at the same file. The user-pref config
/// and launch catalogue are separate schemas; passing one TOML to both flags
/// would only produce a confusing parse error.
fn paths_may_alias(a: &std::path::Path, b: &std::path::Path) -> bool {
    if a == b {
        return true;
    }
    // Strongest check (handles symlinks, `.`/`..`, case-folding).
    if let (Ok(ca), Ok(cb)) = (std::fs::canonicalize(a), std::fs::canonicalize(b))
        && ca == cb
    {
        return true;
    }
    // Canonicalize fails pre-init (paths not yet created), missing e.g.
    // `--config ./../foo/cfg.toml` aliasing `--workspace /tmp/foo`. Fall back to
    // canonicalizing each PARENT (usually exists) + re-join the file_name; no
    // parent/name falls back to the byte-equal above.
    fn lexical_canonical(p: &std::path::Path) -> Option<std::path::PathBuf> {
        let parent = p.parent()?;
        let name = p.file_name()?;
        std::fs::canonicalize(parent).ok().map(|c| c.join(name))
    }
    match (lexical_canonical(a), lexical_canonical(b)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// True iff `<workspace_root>/active/current.json` exists, parses, and
/// the pointed generation's manifest validates.
fn head_files_present(workspace_root: &std::path::Path) -> bool {
    matches!(resolve_active_head_paths(workspace_root), Ok(Some(_)))
}

/// Run the boot-time sweep (`ensure_root_layout` + `recover_all`), returning
/// the `RecoveryReport` (-> `/status`) and the `boot_recovery_unhealthy` reason
/// (gates the inference spawn + surfaces in the heartbeat).
fn run_boot_recovery(
    workspace_root: &std::path::Path,
    default_head: Option<&crate::config::DefaultHeadRef>,
    workspace_metrics: &crate::status::WorkspaceMetrics,
) -> (Option<crate::file_mgr::RecoveryReport>, Option<String>) {
    // Sync `WorkspaceMgr` for a fixed-cost layout pass before any FsService
    // consumer (built later in `async_main`).
    let layout_mgr = crate::file_mgr::WorkspaceMgr::new(workspace_root.to_path_buf());
    if let Err(e) = layout_mgr.ensure_root_layout() {
        tracing::error!(
            target: "acoustics",
            err = %e,
            "ensure_root_layout failed; daemon will boot without inference",
        );
        return (None, Some(format!("ensure_root_layout failed: {e}")));
    }

    // Transient map: recovery's eviction hook fires against this; the real
    // FsService (built later, empty) inherits on-disk state lazily.
    let caches: dashmap::DashMap<
        crate::common::ids::WorkspaceId,
        std::sync::Arc<crate::file_mgr::WorkspaceCacheCell>,
    > = dashmap::DashMap::new();
    // Absent `head.default` still runs recovery; only active-head
    // materialization -> `Unhealthy`.
    let default_source =
        default_head.map(|h| crate::file_mgr::active_head_writer::DefaultHeadSource {
            path: &h.path,
            labels_path: &h.labels_path,
        });
    if default_source.is_none() {
        tracing::warn!(
            target: "acoustics",
            "head.default not configured in launch config; bundled-default fallback disabled \
             (workspace + staging recovery still runs)",
        );
    }
    // Captures nothing; `&loader` coerces to `&HeadInnerLoader` without a Box.
    let loader = |head_mpk: &std::path::Path,
                  labels: &std::path::Path,
                  head_id: crate::common::ids::HeadId|
     -> Result<Box<dyn std::any::Any + Send>, String> {
        let head = HotHead::load(head_mpk, labels, head_id).map_err(|e| format!("{e}"))?;
        let inner = (*head.snapshot()).clone();
        Ok(Box::new(inner) as Box<dyn std::any::Any + Send>)
    };
    let report =
        match crate::file_mgr::recover_all(workspace_root, default_source, &caches, &loader) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    target: "acoustics",
                    err = %e,
                    "boot recovery failed; daemon will boot without inference",
                );
                return (None, Some(format!("boot recovery failed: {e}")));
            }
        };

    tracing::info!(
        target: "acoustics",
        workspaces_scanned = report.workspaces.workspaces_scanned,
        workspace_recovery_failures = report.workspaces.workspace_recovery_failures,
        workspace_enumeration_failures = report.workspaces.workspace_enumeration_failures,
        head_orphans_swept = report.workspaces.head_orphans_swept,
        head_count_repaired = report.workspaces.head_count_repaired,
        dataset_tombstones_completed = report.workspaces.dataset_tombstones_completed,
        dataset_stage_orphans_swept = report.workspaces.dataset_stage_orphans_swept,
        converter_tombstones_completed = report.workspaces.converter_tombstones_completed,
        converter_stage_orphans_swept = report.workspaces.converter_stage_orphans_swept,
        incomplete_creates_removed = report.workspaces.incomplete_creates_removed,
        workspace_tombstones_completed = report.root_staging.workspace_tombstones_completed,
        workspace_stage_orphans_swept = report.root_staging.workspace_stage_orphans_swept,
        "boot recovery completed",
    );
    let orphans = (report.workspaces.head_orphans_swept
        + report.workspaces.dataset_stage_orphans_swept
        + report.workspaces.converter_stage_orphans_swept
        + report.root_staging.workspace_stage_orphans_swept) as u64;
    workspace_metrics.record_boot_orphans_swept(orphans);
    // Per-workspace recovery failures (heads.json parse, sweep IO) as a
    // dashboard aggregate; the `recover_workspaces` warn stays authoritative.
    workspace_metrics.record_boot_workspace_recovery_failures(
        report.workspaces.workspace_recovery_failures as u64,
    );
    // Dirent-level enumeration failures on a separate counter so triage splits
    // "couldn't read the entry" from "read it, sweep errored".
    workspace_metrics.record_boot_workspace_enumeration_failures(
        report.workspaces.workspace_enumeration_failures as u64,
    );

    let unhealthy_reason = match &report.active {
        crate::file_mgr::RecoveryActiveResult::Current { activation_id, .. } => {
            tracing::info!(
                target: "acoustics",
                activation_id = %activation_id,
                "active head verified at boot",
            );
            None
        }
        crate::file_mgr::RecoveryActiveResult::PromotedPrevious { activation_id, .. } => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                "current generation failed verify; previous promoted",
            );
            None
        }
        crate::file_mgr::RecoveryActiveResult::DefaultedFromBundle { activation_id, .. } => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                "no valid generation; bundled default activated",
            );
            None
        }
        crate::file_mgr::RecoveryActiveResult::Unhealthy { reason } => {
            tracing::error!(
                target: "acoustics",
                reason = %reason,
                "boot recovery unhealthy; daemon will boot without inference",
            );
            Some(reason.clone())
        }
    };

    (Some(report), unhealthy_reason)
}

/// Resolve the live active-head dir under `<workspace_root>/active/`, returning
/// `(gen_dir, head_mpk, labels, runtime_head_id)`. `None` when nothing is
/// activated (no `current.json`); `Err` only on a corrupt/unreadable
/// `current.json` pointer (a missing/invalid manifest degrades to `None` after
/// the previous-generation fallback fails). On a current-gen hash mismatch,
/// falls back to the newest valid previous generation (warns) so a torn
/// current.json doesn't take inference offline; the full corrupt-everything ->
/// bundled-default sweep lives in boot recovery.
fn resolve_active_head_paths(
    workspace_root: &std::path::Path,
) -> Result<Option<(PathBuf, PathBuf, PathBuf, crate::common::ids::HeadId)>> {
    use crate::file_mgr::schema as fm_schema;
    let root = workspace_root;
    let pointer_path = fm_schema::active_current_path(root);
    if !pointer_path.exists() {
        return Ok(None);
    }
    let pointer = fm_schema::read_active_current(root)
        .with_context(|| format!("read active pointer {}", pointer_path.display()))?;
    if let Some(triple) = try_resolve_generation(root, &pointer.activation_id)? {
        return Ok(Some(triple));
    }

    // Current failed verify; pick the newest-by-mtime other generation whose
    // manifest validates + bytes hash-match (deterministic).
    tracing::warn!(
        target: "acoustics",
        activation_id = %pointer.activation_id,
        "active generation hash verify failed; trying previous generation",
    );
    let generations_root = fm_schema::active_generations_dir(root);
    if !generations_root.is_dir() {
        return Ok(None);
    }
    let mut candidates: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in std::fs::read_dir(&generations_root)
        .with_context(|| format!("read {}", generations_root.display()))?
    {
        let entry = entry?;
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if name == pointer.activation_id {
            continue;
        }
        let metadata = entry.metadata()?;
        if !metadata.is_dir() {
            continue;
        }
        let mtime = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        candidates.push((mtime, name));
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.0));
    for (_, candidate) in candidates {
        if let Some(triple) = try_resolve_generation(root, &candidate)? {
            tracing::warn!(
                target: "acoustics",
                fallback_activation_id = %candidate,
                "fell back to previous active generation; boot recovery should rewrite current.json",
            );
            return Ok(Some(triple));
        }
    }
    Ok(None)
}

/// Resolve one generation: read + validate manifest, hash `head.mpk` against
/// the manifest sha256, verify `labels.txt` exists (not hashed; rebuildable
/// from `manifest.labels`). `Ok(None)` on any verify failure so the caller can
/// try another generation.
fn try_resolve_generation(
    root: &std::path::Path,
    activation_id: &str,
) -> Result<Option<(PathBuf, PathBuf, PathBuf, crate::common::ids::HeadId)>> {
    use crate::file_mgr::schema as fm_schema;
    use sha2::Digest;
    let manifest = match fm_schema::read_active_manifest(root, activation_id) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                err = %e,
                "active manifest read/parse failed",
            );
            return Ok(None);
        }
    };
    if let Err(e) = manifest.validate() {
        tracing::warn!(
            target: "acoustics",
            activation_id = %activation_id,
            err = %e,
            "active manifest validation failed",
        );
        return Ok(None);
    }
    let gen_dir = fm_schema::active_generation_dir(root, activation_id);
    let head_mpk = gen_dir.join(fm_schema::ACTIVE_HEAD_FILENAME);
    let labels = gen_dir.join(fm_schema::ACTIVE_LABELS_FILENAME);
    let head_bytes = match std::fs::read(&head_mpk) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                path = %head_mpk.display(),
                err = %e,
                "active head.mpk read failed",
            );
            return Ok(None);
        }
    };
    let digest = sha2::Sha256::digest(&head_bytes);
    if crate::common::hex::hex_lowercase(digest.as_slice()) != manifest.sha256 {
        tracing::warn!(
            target: "acoustics",
            activation_id = %activation_id,
            "active head.mpk hash mismatch",
        );
        return Ok(None);
    }
    // labels.txt may lag `labels_sha256` (boot recovery regenerates it from
    // `manifest.labels[]`); accept it as long as it exists.
    if !labels.is_file() {
        tracing::warn!(
            target: "acoustics",
            activation_id = %activation_id,
            "active labels.txt missing",
        );
        return Ok(None);
    }
    Ok(Some((gen_dir, head_mpk, labels, manifest.runtime_head_id)))
}

/// Synthetic 2-class head for dev hosts lacking a real head.mpk, so
/// `GET /api/v1/active` and downstream see a populated slot before a real
/// activation. Validated via `try_from_inner` (production checks); a failure is
/// a daemon bug surfaced via `anyhow!` (not `expect`) so it chains into the
/// boot Result without aborting.
fn synthetic_head_for_dev() -> Result<HotHead> {
    let inner = crate::inference::HeadInner {
        weight: vec![0.0; crate::common::dims::BackboneFeatureDim::USIZE * 2],
        bias: vec![0.0; 2],
        labels: vec!["bg".into(), "voice".into()],
        head_id: crate::common::ids::HeadId::new(),
        n_classes: 2,
    };
    HotHead::try_from_inner(inner)
        .map_err(|e| anyhow::anyhow!("synthetic head failed validation: {e}"))
}

/// Build + spawn the inference engine and its heartbeat-pump, returning the
/// engine's `spawn_blocking` handle, the pump handle, and the loaded `HotHead`
/// (threaded into API state for `POST /active`). The pump re-publishes the
/// engine's heartbeat AND owns the wedge-watchdog that `process::abort()`s on
/// `STALE_ABORT_AFTER` of silence; its lifecycle ties to `engine_hb_tx`
/// channel-close, NOT the master shutdown token, so exiting on
/// `shutdown.cancelled()` won't DISARM the watchdog across the 5 s engine drain.
#[allow(clippy::too_many_arguments)]
async fn boot_inference(
    workspace_root: &std::path::Path,
    backbone_catalogue: crate::inference::BackboneCatalogue,
    audio_buf: &AudioBuffer,
    status_tx: tokio::sync::watch::Sender<crate::status::Heartbeat>,
    inference_cfg: Arc<ArcSwap<crate::inference::InferenceCfg>>,
    infer_tx: broadcast::Sender<bytes::Bytes>,
    shutdown: CancellationToken,
    timing_anchor: crate::common::time::SharedTimingAnchor,
) -> Result<(
    JoinHandle<Result<()>>,
    JoinHandle<Result<(), std::convert::Infallible>>,
    HotHead,
)> {
    let backbone = build_backbone_pipeline(backbone_catalogue).await?;
    tracing::info!(
        target: "acoustics",
        backbone = backbone.description(),
        "inference backbone selected",
    );

    // Resolve again (the `head_files_present` gate confirmed it) so head +
    // labels paths come from the on-disk source of truth.
    let (_gen_dir, head_mpk, labels_path, head_id) = resolve_active_head_paths(workspace_root)
        .with_context(|| "resolve active head paths for boot")?
        .ok_or_else(|| anyhow::anyhow!("active generation absent at boot"))?;
    let head = tokio::task::spawn_blocking(move || HotHead::load(&head_mpk, &labels_path, head_id))
        .await??;

    // Engine heartbeat watch (per-iteration liveness from the hot loop),
    // re-published as the daemon's `inference` entry, plus a 1 Hz floor so a
    // quiet `Waiting` engine's entry doesn't go stale (>5 s).
    let (engine_hb_tx, engine_hb_rx) =
        tokio::sync::watch::channel(crate::inference::Heartbeat::default());
    let hb_pump_handle: JoinHandle<Result<(), std::convert::Infallible>> = {
        let mut hb_rx = engine_hb_rx.clone();
        // No master-shutdown clone: completion observed via channel close keeps
        // the watchdog armed through the engine's drain window.
        tokio::spawn(async move {
            let mut floor = tokio::time::interval(std::time::Duration::from_secs(1));
            // Floor refreshes the status entry only when the engine is quiet;
            // Skip avoids a backlog burst after a stall.
            floor.set_missed_tick_behavior(MissedTickBehavior::Skip);
            floor.tick().await; // skip immediate first tick
            // Two signals. STALE_AFTER (2 s "quiet") uses `frames_emitted`. The
            // wedge-watchdog (abort on STALE_ABORT_AFTER silence, covers a wedged
            // `rknn_run`) uses heartbeat-RECEIPT time: a `Waiting` engine still
            // heartbeats (~2 Hz, throttled via
            // inference::engine::WAITING_HEARTBEAT_INTERVAL = 500 ms) without
            // advancing frames, so receipt-time distinguishes wedged from idle.
            // Lives in the async pump since the
            // engine is sync `spawn_blocking`. Abort gates: eligible only after
            // the first hb OR `BOOT_GRACE` (catches a wedge BEFORE any hb, e.g.
            // RKNN init hang); silence >= STALE_ABORT_AFTER; skip when state
            // terminal -- the engine sets Stopped/Failed BEFORE dropping
            // `engine_hb_tx`.
            let mut last_emitted_observed: u64 = 0;
            let mut last_advance_at = std::time::Instant::now();
            let mut last_hb_received_at: Option<std::time::Instant> = None;
            // Silence-clock reference until the first hb arrives.
            let pump_started_at = std::time::Instant::now();
            const STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(2);
            const STALE_ABORT_AFTER: std::time::Duration = std::time::Duration::from_secs(5);
            const BOOT_GRACE: std::time::Duration = std::time::Duration::from_secs(10);
            loop {
                tokio::select! {
                    biased;
                    changed = hb_rx.changed() => {
                        if changed.is_err() {
                            // `engine_hb_tx` dropped: engine closure returned
                            // (clean exit, panic, final-tick). Exit cleanly;
                            // abort is for silence wedges only.
                            break;
                        }
                        last_hb_received_at = Some(std::time::Instant::now());
                    }
                    _ = floor.tick() => {
                        // No hb this period: do NOT touch `last_hb_received_at`
                        // -- the gap IS the wedge.
                    }
                }
                let snap = *hb_rx.borrow_and_update();
                let now = std::time::Instant::now();
                if snap.frames_emitted != last_emitted_observed {
                    last_emitted_observed = snap.frames_emitted;
                    last_advance_at = now;
                }
                let stalled_for = now.saturating_duration_since(last_advance_at);
                let stalled = stalled_for >= STALE_AFTER;

                let engine_terminal = matches!(
                    snap.state,
                    crate::inference::EngineState::Failed | crate::inference::EngineState::Stopped
                );
                let hb_silence_for = match last_hb_received_at {
                    Some(t) => now.saturating_duration_since(t),
                    // No hb yet: silence runs from pump start, minus boot grace.
                    None => now
                        .saturating_duration_since(pump_started_at)
                        .saturating_sub(BOOT_GRACE),
                };
                let abort_eligible = last_hb_received_at.is_some()
                    || now.saturating_duration_since(pump_started_at) > BOOT_GRACE;
                let should_abort =
                    abort_eligible && hb_silence_for >= STALE_ABORT_AFTER && !engine_terminal;
                if should_abort {
                    tracing::error!(
                        target: "acoustics",
                        hb_silence_ms = hb_silence_for.as_millis() as u64,
                        last_emitted = snap.frames_emitted,
                        last_state = ?snap.state,
                        "inference engine wedged > {:?} (no heartbeat); aborting for \
                         external supervisor restart",
                        STALE_ABORT_AFTER,
                    );
                    let _ = status_tx.send(Heartbeat::unhealthy(format!(
                        "inference wedged {} ms; aborting",
                        hb_silence_for.as_millis()
                    )));
                    // Direct stderr write: the `tracing::error!` above routes
                    // through the non-blocking appender, dropped unflushed
                    // before `process::abort()` (abort skips
                    // `WorkerGuard::drop`), so this is the only sink guaranteed
                    // to reach journald.
                    eprintln!(
                        "acousticsd: ABORT -- inference engine wedged \
                         {hb_silence_ms} ms (last_state={state:?}, \
                         frames_emitted={emitted}); external supervisor \
                         must restart",
                        hb_silence_ms = hb_silence_for.as_millis(),
                        state = snap.state,
                        emitted = snap.frames_emitted,
                    );
                    // `std::thread::sleep` (not `tokio::time::sleep`): the
                    // appender drains on its own thread, and a cooperative yield
                    // may not fire promptly under the bg-tier scheduling pressure
                    // we're aborting from.
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    std::process::abort();
                }
                let state_healthy = matches!(
                    snap.state,
                    crate::inference::EngineState::Running | crate::inference::EngineState::Waiting
                );
                // `Starting` is healthy only inside the boot grace window (else
                // a `--check` probe during boot false-negatives); after that
                // it's a real problem (abort fires at +15 s).
                let in_boot_grace = now.saturating_duration_since(pump_started_at) <= BOOT_GRACE;
                let starting_in_grace =
                    in_boot_grace && matches!(snap.state, crate::inference::EngineState::Starting);
                let healthy = (state_healthy || starting_in_grace) && !stalled;
                // `snap.state` via Display so the detail starts lowercase like
                // every other heartbeat detail.
                let detail = if stalled {
                    format!(
                        "{} seq={} emitted={} drop_nan={} drop_lag={} (stalled {}ms; no new windows)",
                        snap.state,
                        snap.last_seq,
                        snap.frames_emitted,
                        snap.frames_dropped_nan,
                        snap.frames_dropped_lag,
                        stalled_for.as_millis(),
                    )
                } else {
                    format!(
                        "{} seq={} emitted={} drop_nan={} drop_lag={}",
                        snap.state,
                        snap.last_seq,
                        snap.frames_emitted,
                        snap.frames_dropped_nan,
                        snap.frames_dropped_lag,
                    )
                };
                let hb = if healthy {
                    Heartbeat::ok(detail)
                } else {
                    Heartbeat::unhealthy(detail)
                };
                let _ = status_tx.send(hb);
            }
            // Reached only via channel-close break. `Infallible` because the
            // abort path leaves via `process::abort()`, not `Err`.
            Ok(())
        })
    };

    let head_clone = head.clone();
    let engine = InferenceEngine::new(
        backbone.into_boxed(),
        head_clone,
        inference_cfg,
        engine_hb_tx,
        // Producer's anchor -> per-window FIRST sample capture time
        // (window-start), not emit time (lags ~1 window, ~1 s).
        Some(timing_anchor),
    );

    let reader = audio_buf.reader();
    let join: JoinHandle<Result<()>> = tokio::task::spawn_blocking(move || {
        // Best-effort pin + RT bump; failure logs WARN and the engine
        // continues on default placement.
        if let Err(e) = crate::sched::pin_to_core(INFERENCE_PIN_CORE) {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                core = INFERENCE_PIN_CORE,
                "inference pin_to_core failed; continuing on default placement",
            );
        }
        if let Err(e) = crate::sched::set_realtime(INFERENCE_RT_PRIORITY) {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                priority = INFERENCE_RT_PRIORITY,
                "inference set_realtime failed (likely missing CAP_SYS_NICE); \
                 continuing at SCHED_OTHER",
            );
        }
        engine
            .run_blocking(reader, infer_tx, shutdown)
            .map_err(|e| anyhow::anyhow!("inference run: {e}"))
    });
    Ok((join, hb_pump_handle, head))
}

/// Pick the inference backbone by walking `[[backbone.candidates]]` in
/// declaration order, returning the first that loads on this build. Heavy work
/// (I/O, sha256, RKNN FFI init, Burn `.mpk` parse) runs in `spawn_blocking` so
/// the async worker isn't stalled. RKNN library discovery is owned by
/// `RknnBackbone::load` (`RKNN_LIB` / `LD_LIBRARY_PATH` search), not the daemon.
async fn build_backbone_pipeline(
    catalogue: crate::inference::BackboneCatalogue,
) -> Result<crate::inference::BackbonePipeline> {
    tokio::task::spawn_blocking(move || catalogue.load_first_supported())
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join (backbone): {e}"))?
        .map_err(|e| anyhow::anyhow!("backbone selection: {e}"))
}

async fn try_bind_uds(path: &std::path::Path, mode: u32) -> Result<tokio::net::UnixListener> {
    let listener = crate::stream_io::bind_uds(path)
        .await
        .map_err(|e| anyhow::anyhow!("uds bind {}: {e}", path.display()))?;
    if let Err(e) = crate::stream_io::set_uds_permissions(path, mode) {
        // Unlink the just-bound socket so the next boot's `bind_uds` doesn't see
        // a stale file at looser-than-`mode` perms (the exact trust-posture leak
        // the chmod closes). Removal failure is logged but doesn't displace the
        // chmod Err.
        if let Err(rm_err) = std::fs::remove_file(path) {
            tracing::warn!(
                target: "acoustics",
                err = %rm_err,
                path = %path.display(),
                "uds chmod failed AND post-failure unlink failed; \
                 stale socket may persist with default perms until next boot's bind_uds",
            );
        }
        return Err(anyhow::anyhow!("uds chmod {}: {e}", path.display()));
    }
    Ok(listener)
}

/// True iff `bind` resolves to a loopback host (`127.0.0.0/8`, `::1`,
/// `localhost`). Fails closed: unparseable -> "not loopback" so the
/// trust-posture WARN is louder, not silently missed (plain `SocketAddr::parse`
/// misclassifies `localhost:8787`).
fn bind_is_loopback(bind: &str) -> bool {
    let Some((host_raw, _port)) = bind.rsplit_once(':') else {
        return false;
    };
    let host = host_raw
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host_raw);
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Ensure the UDS parent dir exists with safe perms before any bind. Defense in
/// depth on `bind_uds`'s parent-confinement: mkdir 0o700 if missing (operators
/// needing group access pre-create the dir), and hard-reject a symlinked
/// parent. Permission diagnostics (incl. the world-writable-no-sticky warn) are
/// left to `bind_uds`. Idempotent.
fn ensure_uds_parent_dir(uds_path: &std::path::Path) -> Result<()> {
    let parent = match uds_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        // No/empty parent is a misconfig `bind_uds` rejects.
        _ => return Ok(()),
    };
    // `symlink_metadata` not `exists()` so a dangling symlink is rejected, not
    // silently followed by `create_dir_all` (which could create the dir under a
    // writable symlink target).
    let parent_present = match std::fs::symlink_metadata(parent) {
        Ok(md) => {
            if md.file_type().is_symlink() {
                anyhow::bail!(
                    "UDS parent {} is a symlink; refuse to chmod or bind through it",
                    parent.display(),
                );
            }
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(e).with_context(|| format!("stat UDS parent {}", parent.display())),
    };
    if !parent_present {
        // create_dir_all has no mode arg, so dirs come up under the umask;
        // tighten only the leaf to 0o700 below, leaving grandparents at their
        // operator-controlled umask perms.
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create UDS parent dir {}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // 0o700: owner-only, the safe default for a daemon-private dir.
            let perms = std::fs::Permissions::from_mode(0o700);
            std::fs::set_permissions(parent, perms).with_context(|| {
                format!(
                    "chmod 0o700 on freshly-created UDS parent {}",
                    parent.display()
                )
            })?;
            tracing::info!(
                target: "acoustics",
                path = %parent.display(),
                mode = "0o700",
                "uds parent dir created with private permissions",
            );
        }
    }
    // Existing parent: `bind_uds` owns the safety checks (symlink/non-dir
    // hard-reject, world-writable-no-sticky warn).
    Ok(())
}

async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        // SIGTERM handler install can fail (sandboxing, resource pressure, slot
        // taken); an `expect` would panic-then-abort a healthy daemon, so fall
        // back to ctrl_c-only with a warn.
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!(target: "acoustics", "ctrl-c received");
                    }
                    _ = term.recv() => {
                        tracing::info!(target: "acoustics", "SIGTERM received");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "acoustics",
                    err = %e,
                    "SIGTERM handler install failed; falling back to ctrl_c-only",
                );
                // ctrl_c install can also fail; the daemon then won't respond to
                // signals at all (exits only via runtime-drop / OS kill) -- warn
                // so the operator sees it at boot.
                if let Err(e2) = tokio::signal::ctrl_c().await {
                    tracing::warn!(
                        target: "acoustics",
                        err = %e2,
                        "ctrl_c handler install also failed; daemon will not \
                         respond to ctrl-c/SIGTERM (graceful shutdown disabled)",
                    );
                    // No signal source: this future keeps the daemon alive (main
                    // path + escalator both await it); returning would
                    // self-shutdown and trip the escalator's exit(1), so park
                    // forever.
                    std::future::pending::<()>().await;
                } else {
                    tracing::info!(target: "acoustics", "ctrl-c received");
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        // Same contract as the unix double-failure arm: park forever
        // on install failure rather than self-shutdown.
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                "ctrl_c handler install failed; graceful shutdown disabled",
            );
            std::future::pending::<()>().await;
        }
    }
}

async fn run_check_mode(
    monitor: &StatusMonitor,
    shutdown: &CancellationToken,
    duration: Duration,
) -> Result<()> {
    // Race the probe duration against signals so Ctrl-C/SIGTERM during `--check`
    // engages the orderly drain, not Tokio's hard kill. The cancel branch
    // short-circuits report generation (no useful snapshot after an operator
    // abort).
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => return Ok(()),
        _ = wait_for_signal() => return Ok(()),
        _ = tokio::time::sleep(duration) => {}
    }
    // No WS broadcast paths in --check, so pass a default lag snapshot.
    let snap = monitor.snapshot(crate::status::BroadcastLagSnapshot::default());
    let json = serde_json::to_string_pretty(&snap).unwrap_or_else(|_| "{}".into());
    println!("{json}");

    let unhealthy: Vec<_> = snap
        .subsystems
        .iter()
        .filter(|(_, v)| !v.healthy)
        .map(|(k, v)| format!("{k}: {} (age {} ms)", v.detail, v.age_ms))
        .collect();
    if unhealthy.is_empty() {
        eprintln!(
            "daemon: --check OK ({} subsystems healthy)",
            snap.subsystems.len()
        );
        Ok(())
    } else {
        eprintln!("daemon: --check FAIL -- unhealthy: {unhealthy:?}");
        Err(anyhow::anyhow!("subsystems unhealthy: {unhealthy:?}"))
    }
}

#[cfg(test)]
mod bind_is_loopback_tests {
    use super::{bind_is_loopback, paths_may_alias};

    #[test]
    fn ipv4_loopback_accepts() {
        assert!(bind_is_loopback("127.0.0.1:8787"));
        assert!(bind_is_loopback("127.0.0.1:0"));
        assert!(
            bind_is_loopback("127.5.5.5:8787"),
            "all of 127.0.0.0/8 is loopback"
        );
    }

    #[test]
    fn ipv6_loopback_accepts_with_brackets() {
        assert!(bind_is_loopback("[::1]:8787"));
    }

    #[test]
    fn localhost_case_insensitive() {
        assert!(bind_is_loopback("localhost:8787"));
        assert!(bind_is_loopback("Localhost:9000"));
        assert!(bind_is_loopback("LOCALHOST:80"));
    }

    #[test]
    fn non_loopback_rejects() {
        assert!(!bind_is_loopback("0.0.0.0:8787"));
        assert!(!bind_is_loopback("[::]:8787"));
        assert!(!bind_is_loopback("192.168.1.10:8787"));
        assert!(!bind_is_loopback("8.8.8.8:8787"));
    }

    #[test]
    fn fails_closed_on_unparseable() {
        assert!(!bind_is_loopback("myhost.local:8787"));
        assert!(!bind_is_loopback("example.com:443"));
        assert!(!bind_is_loopback("garbage"));
    }

    #[test]
    fn paths_may_alias_detects_same_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        #[allow(clippy::disallowed_methods)]
        std::fs::write(&path, b"fixture").expect("write fixture");

        assert!(paths_may_alias(&path, &path));
        assert!(paths_may_alias(
            &path,
            &dir.path().join(".").join("config.toml")
        ));
        assert!(!paths_may_alias(&path, &dir.path().join("launch.toml")));
    }
}
