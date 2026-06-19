//! Acoustics Lab daemon (`acousticslabd`): thin binary wrapper around
//! [`acousticslab::daemon::run`].
//!
//! The `#[global_allocator]` (mimalloc v3) lives here, not in the lib: a
//! lib-level decl would conflict with every test binary that links it.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Bare `eprintln!` panic hook for boot steps that run before `daemon::run`
/// installs its `tracing` hook; uses `eprintln!` rather than `tracing` because
/// the subscriber is not up yet this early in boot.
fn install_pre_init_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let payload = info.payload();
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic>");
        eprintln!("acousticslabd: PANIC during boot at {location}: {msg}");
    }));
}

fn main() -> anyhow::Result<()> {
    install_pre_init_panic_hook();
    acousticslab::daemon::run()
}
