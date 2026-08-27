//! `acousticslab`: management CLI for a local AcousticsLab install.
//!
//! Companion to the `acousticslabd` daemon and `acousticslab-webd` web front.
//! It does the things an operator would otherwise do by hand-editing config or
//! shelling out: enumerate capture devices, point the daemon at one, download
//! an NPU backbone, and read daemon status over its Unix socket.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};

const DEFAULT_CONFIG: &str = "/etc/acousticslab/launch.toml";
const DEFAULT_WORKSPACE: &str = "/var/lib/acousticslab";
const DEFAULT_API_SOCKET: &str = "/run/acousticslab/api.sock";
const DEFAULT_RKNN: &str = "/usr/share/acousticslab/backbones/backbone.rknn";
/// A wedged daemon must not hang `status` forever; the response is a few KiB,
/// so any stall this long is a fault worth reporting.
const RESPONSE_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Parser)]
#[command(
    name = "acousticslab",
    version,
    about = "Manage a local AcousticsLab install (devices, backbones, status)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Microphone (ALSA capture device) management.
    Mic {
        #[command(subcommand)]
        cmd: MicCmd,
    },
    /// Backbone (feature-extractor) weight management.
    Backbone {
        #[command(subcommand)]
        cmd: BackboneCmd,
    },
    /// Query the running daemon's status over its Unix socket.
    Status {
        /// Daemon API socket.
        #[arg(long, default_value = DEFAULT_API_SOCKET)]
        socket: PathBuf,
        /// Print the raw status JSON instead of a summary.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum MicCmd {
    /// List available ALSA capture devices (like `arecord -L`).
    List,
    /// Set the ALSA capture device(s) in the launch config.
    ///
    /// The daemon uses one mic at a time. A single spec sets the first ALSA
    /// candidate's device; `all` rebuilds the candidate list with every
    /// detected capture device, so the daemon auto-selects whichever is live
    /// and fails over between them (it does NOT capture them all at once).
    Use {
        /// PCM spec (e.g. `hw:1,0`, `hw:CARD=Device,DEV=0`), or `all` to use
        /// every detected capture device as a failover candidate.
        spec: String,
        /// Launch config to edit.
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        /// Daemon workspace; `all` checks its config.toml mic policy.
        #[arg(long, default_value = DEFAULT_WORKSPACE)]
        workspace: PathBuf,
        /// Restart acousticslabd afterwards (needs systemd + privileges).
        #[arg(long)]
        restart: bool,
    },
}

#[derive(Subcommand)]
enum BackboneCmd {
    /// Download an NPU (RKNN) backbone from a URL and install it.
    Fetch {
        /// Source URL (https/http/file; fetched with curl).
        url: String,
        /// Expected SHA-256 (hex) of the download; verified before install.
        #[arg(long)]
        sha256: Option<String>,
        /// Destination path.
        #[arg(long, default_value = DEFAULT_RKNN)]
        output: PathBuf,
        /// Launch config to check for an rknn candidate referencing the install.
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Mic { cmd } => match cmd {
            MicCmd::List => mic_list(),
            MicCmd::Use {
                spec,
                config,
                workspace,
                restart,
            } => {
                if spec.eq_ignore_ascii_case("all") {
                    mic_use_all(&config, &workspace, restart)
                } else {
                    mic_use(&spec, &config, restart)
                }
            }
        },
        Cmd::Backbone { cmd } => match cmd {
            BackboneCmd::Fetch {
                url,
                sha256,
                output,
                config,
            } => backbone_fetch(&url, sha256.as_deref(), &output, &config),
        },
        Cmd::Status { socket, json } => status(&socket, json),
    }
}

// MARK: mic list

#[cfg(target_os = "linux")]
fn mic_list() -> Result<()> {
    use alsa::Direction;
    use alsa::device_name::HintIter;

    let hints = HintIter::new_str(None, "pcm").context("enumerate ALSA PCM devices")?;
    let mut count = 0usize;
    for hint in hints {
        // Keep capture-capable PCMs (explicit Capture, or bidirectional/None).
        if matches!(hint.direction, Some(Direction::Playback)) {
            continue;
        }
        let Some(name) = hint.name else { continue };
        count += 1;
        println!("{name}");
        if let Some(desc) = hint.desc {
            for line in desc.lines() {
                println!("    {line}");
            }
        }
    }
    if count == 0 {
        println!("No ALSA capture devices found.");
        println!("(Is a capture card present? Check `arecord -l`.)");
    } else {
        println!();
        println!("Set one with: acousticslab mic use <spec>");
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn mic_list() -> Result<()> {
    bail!("`mic list` enumerates ALSA devices and is only supported on Linux")
}

// MARK: mic use

fn mic_use(hw_spec: &str, config: &Path, restart: bool) -> Result<()> {
    ensure!(!hw_spec.trim().is_empty(), "hw_spec must not be empty");

    let text = std::fs::read_to_string(config)
        .with_context(|| format!("read {} (need privileges?)", config.display()))?;
    let mut doc = text
        .parse::<toml_edit::DocumentMut>()
        .with_context(|| format!("parse {}", config.display()))?;

    let candidates = doc
        .get_mut("mic")
        .and_then(|m| m.get_mut("candidates"))
        .and_then(|c| c.as_array_of_tables_mut())
        .with_context(|| format!("no [[mic.candidates]] table in {}", config.display()))?;

    // `as_table_like_mut` handles both block `[..source]` and inline
    // `source = {..}` (the latter is what `mic use all` writes).
    let mut applied = false;
    for cand in candidates.iter_mut() {
        let Some(source) = cand.get_mut("source").and_then(|s| s.as_table_like_mut()) else {
            continue;
        };
        if source.get("kind").and_then(|k| k.as_str()) == Some("alsa") {
            source.insert("hw_spec", toml_edit::value(hw_spec));
            applied = true;
            break;
        }
    }
    ensure!(
        applied,
        "no ALSA candidate (source kind = \"alsa\") in {}; add one or edit directly",
        config.display()
    );

    atomic_write(config, &doc.to_string())
        .with_context(|| format!("write {}", config.display()))?;
    println!("Set ALSA hw_spec = {hw_spec:?} in {}", config.display());

    if restart {
        restart_daemon()?;
    } else {
        println!("Restart to apply: systemctl restart acousticslabd");
    }
    Ok(())
}

// MARK: mic use all

/// A detected capture device, keyed by ALSA card name (stable across hot-plug,
/// unlike the numeric index).
struct HwDevice {
    card: String,
    dev: String,
    /// `plughw:` form: the `plug` plugin format/rate-converts, tolerating quirky
    /// USB devices the raw `hw:` open would reject.
    hw_spec: String,
    desc: String,
}

/// Enumerate physical capture devices, one per card:dev.
#[cfg(target_os = "linux")]
fn discover_capture_hw() -> Result<Vec<HwDevice>> {
    use alsa::Direction;
    use alsa::device_name::HintIter;
    use std::collections::HashSet;

    let hints = HintIter::new_str(None, "pcm").context("enumerate ALSA PCM devices")?;
    let mut devices = Vec::new();
    let mut seen = HashSet::new();
    for hint in hints {
        if matches!(hint.direction, Some(Direction::Playback)) {
            continue;
        }
        let Some(name) = hint.name else { continue };
        // Admit only raw `hw:CARD=` names -- `plughw:`/`sysdefault:`/`default`
        // alias the same hardware; `seen` collapses repeat (card, dev) hints.
        let Some(rest) = name.strip_prefix("hw:CARD=") else {
            continue;
        };
        let card = rest.split(',').next().unwrap_or(rest).to_string();
        let dev = kv_after(&name, "DEV=").unwrap_or_else(|| "0".to_string());
        if !seen.insert((card.clone(), dev.clone())) {
            continue;
        }
        let hw_spec = format!("plughw:CARD={card},DEV={dev}");
        let desc = hint
            .desc
            .unwrap_or_default()
            .lines()
            .next()
            .unwrap_or_default()
            .to_string();
        devices.push(HwDevice {
            card,
            dev,
            hw_spec,
            desc,
        });
    }
    Ok(devices)
}

#[cfg(not(target_os = "linux"))]
fn discover_capture_hw() -> Result<Vec<HwDevice>> {
    bail!("device discovery enumerates ALSA and is only supported on Linux")
}

/// Extract the value following the first occurrence of `key`, up to the next
/// `,` (or end). E.g. `kv_after("hw:CARD=Mic,DEV=0", "DEV=") -> Some("0")`.
/// Only `discover_capture_hw` (Linux-gated) calls it; `test` keeps its unit test building.
#[cfg(any(target_os = "linux", test))]
fn kv_after(haystack: &str, key: &str) -> Option<String> {
    let i = haystack.find(key)?;
    let rest = &haystack[i + key.len()..];
    let end = rest.find(',').unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// The mic id a workspace policy pins (`[mic.mic] kind = "fixed"`), if any.
fn pinned_mic_id(config_toml: &str) -> Option<String> {
    let doc = config_toml.parse::<toml_edit::DocumentMut>().ok()?;
    let mic = doc.get("mic")?.get("mic")?;
    (mic.get("kind")?.as_str()? == "fixed").then(|| {
        mic.get("id")
            .and_then(|i| i.as_str())
            .unwrap_or_default()
            .to_string()
    })
}

/// A pinned policy (console-set) never fails over, so the candidate list `all`
/// rebuilds would be dead config whose fresh ids leave the pin dangling --
/// fatal at daemon boot. Unreadable/absent config: daemon never ran, nothing
/// to strand.
fn ensure_failover_allowed(workspace: &Path) -> Result<()> {
    let path = workspace.join("config.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(());
    };
    let Some(id) = pinned_mic_id(&text) else {
        return Ok(());
    };
    bail!(
        "{} pins the microphone to {id:?} (kind = \"fixed\"), which never fails over.\n\
         Switch it to Auto in the console, or set kind = \"first_available\" there, then re-run.",
        path.display()
    )
}

fn mic_use_all(config: &Path, workspace: &Path, restart: bool) -> Result<()> {
    // Policy first: a pin contradicts `all` regardless of what's plugged in.
    ensure_failover_allowed(workspace)?;
    let devices = discover_capture_hw()?;
    ensure!(
        !devices.is_empty(),
        "no ALSA capture hardware detected (check `arecord -l`)"
    );

    let text = std::fs::read_to_string(config)
        .with_context(|| format!("read {} (need privileges?)", config.display()))?;
    let mut doc = text
        .parse::<toml_edit::DocumentMut>()
        .with_context(|| format!("parse {}", config.display()))?;

    let (ids, had_mock) = rebuild_candidates(&mut doc, &devices)?;
    atomic_write(config, &doc.to_string())
        .with_context(|| format!("write {}", config.display()))?;

    println!(
        "Configured {} capture candidate(s) in {} (first-available failover):",
        ids.len(),
        config.display()
    );
    for (id, d) in ids.iter().zip(&devices) {
        let desc = if d.desc.is_empty() {
            String::new()
        } else {
            format!("  ({})", d.desc)
        };
        println!("  {id}: {}{desc}", d.hw_spec);
    }
    if had_mock {
        println!("  + mock fallback (retained)");
    }
    println!("The daemon uses one mic at a time: it opens the first that succeeds");
    println!("and fails over between them -- it does not capture all at once.");

    if restart {
        restart_daemon()?;
    } else {
        println!("Restart to apply: systemctl restart acousticslabd");
    }
    Ok(())
}

/// Replace `[[mic.candidates]]` with one ALSA candidate per device (failover
/// order), reusing the first existing ALSA period/buffer and keeping a mock
/// fallback if present. Returns ids index-aligned with `devices`, plus whether
/// a mock was kept.
fn rebuild_candidates(
    doc: &mut toml_edit::DocumentMut,
    devices: &[HwDevice],
) -> Result<(Vec<String>, bool)> {
    let (mut period, mut buffer) = (960i64, 3840i64);
    let (mut had_mock, mut took_pb) = (false, false);
    if let Some(existing) = doc
        .get("mic")
        .and_then(|m| m.get("candidates"))
        .and_then(|c| c.as_array_of_tables())
    {
        for cand in existing.iter() {
            let Some(src) = cand.get("source").and_then(|s| s.as_table_like()) else {
                continue;
            };
            match src.get("kind").and_then(|k| k.as_str()) {
                Some("mock") => had_mock = true,
                Some("alsa") if !took_pb => {
                    period = src
                        .get("period_size")
                        .and_then(|x| x.as_integer())
                        .unwrap_or(period);
                    buffer = src
                        .get("buffer_size")
                        .and_then(|x| x.as_integer())
                        .unwrap_or(buffer);
                    took_pb = true;
                }
                _ => {}
            }
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut ids = Vec::with_capacity(devices.len());
    let mut candidates = toml_edit::ArrayOfTables::new();
    for d in devices {
        let mut id = format!("alsa-{}", sanitize_id(&d.card));
        if !seen.insert(id.clone()) {
            id = format!("alsa-{}-{}", sanitize_id(&d.card), d.dev);
            seen.insert(id.clone());
        }
        candidates.push(alsa_candidate_table(&id, &d.hw_spec, period, buffer));
        ids.push(id);
    }
    if had_mock {
        candidates.push(mock_candidate_table());
    }

    doc.get_mut("mic")
        .and_then(|m| m.as_table_mut())
        .context("no [mic] table in config")?["candidates"] =
        toml_edit::Item::ArrayOfTables(candidates);
    Ok((ids, had_mock))
}

/// `[[mic.candidates]]` entry for an ALSA device (inline `source` table).
fn alsa_candidate_table(id: &str, hw_spec: &str, period: i64, buffer: i64) -> toml_edit::Table {
    use toml_edit::{Array, InlineTable, Item, Table, Value, value};
    let mut t = Table::new();
    t["id"] = value(id);
    let mut chans = Array::new();
    chans.push(0i64);
    t["channels"] = Item::Value(Value::Array(chans));
    let mut src = InlineTable::new();
    src.insert("kind", "alsa".into());
    src.insert("hw_spec", hw_spec.into());
    src.insert("period_size", period.into());
    src.insert("buffer_size", buffer.into());
    t["source"] = Item::Value(Value::InlineTable(src));
    t
}

/// A mock (1 kHz sine) fallback candidate, so the daemon still boots if every
/// real mic fails to open.
fn mock_candidate_table() -> toml_edit::Table {
    use toml_edit::{Array, InlineTable, Item, Table, Value, value};
    let mut t = Table::new();
    t["id"] = value("default-mock");
    let mut chans = Array::new();
    chans.push(0i64);
    t["channels"] = Item::Value(Value::Array(chans));
    let mut wave = InlineTable::new();
    wave.insert("kind", "sine".into());
    wave.insert("freq_hz", 1000.0.into());
    wave.insert("amplitude", 0.25.into());
    let mut waves = Array::new();
    waves.push(Value::InlineTable(wave));
    let mut src = InlineTable::new();
    src.insert("kind", "mock".into());
    src.insert("period_size", 512i64.into());
    src.insert("sample_rate", 44100i64.into());
    src.insert("waveforms", Value::Array(waves));
    t["source"] = Item::Value(Value::InlineTable(src));
    t
}

/// Keep a card name usable as a candidate id: `[A-Za-z0-9_-]`, others -> `-`.
fn sanitize_id(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Atomic file replace: write a sibling temp then rename over the target, so a
/// crash mid-write can't leave a truncated config. (Avoids `std::fs::write`,
/// which the workspace disallows for exactly this reason.)
fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    use std::io::Write;
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config".into());
    let tmp = dir.join(format!(".{name}.tmp"));
    {
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        file.write_all(contents.as_bytes())
            .context("write temp config")?;
        file.sync_all().ok();
    }
    // Carry the target's mode: `File::create` obeys the umask, so a tight umask
    // would silently narrow a world-readable config on rewrite.
    if let Ok(meta) = std::fs::metadata(path) {
        std::fs::set_permissions(&tmp, meta.permissions())
            .with_context(|| format!("carry mode onto {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

fn restart_daemon() -> Result<()> {
    let status = std::process::Command::new("systemctl")
        .args(["restart", "acousticslabd.service"])
        .status()
        .context("run systemctl (is systemd present?)")?;
    ensure!(status.success(), "systemctl restart acousticslabd failed");
    println!("Restarted acousticslabd.service");
    Ok(())
}

// MARK: backbone fetch

fn backbone_fetch(
    url: &str,
    expected_sha256: Option<&str>,
    output: &Path,
    config: &Path,
) -> Result<()> {
    let dir = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create {} (need privileges?)", dir.display()))?;

    // Download to a sibling temp path so the install is an atomic rename.
    let file_name = output
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "backbone.rknn".into());
    let tmp = dir.join(format!(".{file_name}.download"));

    println!("Downloading {url}");
    let status = std::process::Command::new("curl")
        .arg("-fsSL")
        .arg("-o")
        .arg(&tmp)
        .arg(url)
        .status()
        .context("run curl (is it installed?)")?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!("curl failed to download {url}");
    }

    let bytes = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    if let Some(expected) = expected_sha256 {
        let got = sha256_file(&tmp)?;
        if !got.eq_ignore_ascii_case(expected.trim()) {
            let _ = std::fs::remove_file(&tmp);
            bail!("sha256 mismatch: expected {expected}, got {got}");
        }
        println!("sha256 verified: {got}");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644)).ok();
    }
    std::fs::rename(&tmp, output).with_context(|| format!("install to {}", output.display()))?;
    println!("Installed backbone -> {} ({bytes} bytes)", output.display());

    // The file alone is inert: only [[backbone.candidates]] paths load. Warn
    // when none reference it; `None` (unreadable config) stays quiet.
    if config_loads_rknn(config, output) == Some(false) {
        eprintln!(
            "warning: no rknn backbone candidate in {} points at this file;",
            config.display()
        );
        eprintln!("         the daemon keeps using the CPU backbone until you add one:");
        eprintln!("           [[backbone.candidates]]");
        eprintln!("           kind = \"rknn\"");
        eprintln!("           path = \"{}\"", output.display());
        eprintln!("         then restart: systemctl restart acousticslabd");
    } else {
        println!("Restart to load it: systemctl restart acousticslabd");
    }
    Ok(())
}

/// Whether `config` has an rknn candidate whose path resolves to `output`;
/// `None` when the config can't be read or parsed.
fn config_loads_rknn(config: &Path, output: &Path) -> Option<bool> {
    let text = std::fs::read_to_string(config).ok()?;
    let doc = text.parse::<toml_edit::DocumentMut>().ok()?;
    Some(doc_declares_rknn_at(&doc, output))
}

/// True if any `[[backbone.candidates]]` -- block or inline form -- is an rknn
/// entry pointing at `output`. Compares canonicalized paths, mirroring the
/// daemon's own `path.exists()` resolution (so symlink/`..` aliases still match).
fn doc_declares_rknn_at(doc: &toml_edit::DocumentMut, output: &Path) -> bool {
    let want = output
        .canonicalize()
        .unwrap_or_else(|_| output.to_path_buf());
    let is_match = |kind: Option<&str>, path: Option<&str>| {
        kind == Some("rknn")
            && path.is_some_and(|p| {
                Path::new(p)
                    .canonicalize()
                    .unwrap_or_else(|_| PathBuf::from(p))
                    == want
            })
    };
    let Some(cands) = doc.get("backbone").and_then(|b| b.get("candidates")) else {
        return false;
    };
    if let Some(aot) = cands.as_array_of_tables() {
        return aot.iter().any(|t| {
            is_match(
                t.get("kind").and_then(|k| k.as_str()),
                t.get("path").and_then(|p| p.as_str()),
            )
        });
    }
    if let Some(arr) = cands.as_array() {
        return arr.iter().any(|v| {
            v.as_inline_table().is_some_and(|t| {
                is_match(
                    t.get("kind").and_then(|k| k.as_str()),
                    t.get("path").and_then(|p| p.as_str()),
                )
            })
        });
    }
    false
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).context("read download")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

// MARK: status

fn status(socket: &Path, json: bool) -> Result<()> {
    let body = uds_get(socket, "/api/v1/status")?;
    if json {
        println!("{body}");
        return Ok(());
    }
    let value: serde_json::Value = serde_json::from_str(&body).context("parse status JSON")?;
    print_status_summary(&value);
    Ok(())
}

/// Minimal blocking HTTP/1.0 GET over the daemon's Unix socket. HTTP/1.0 makes
/// the server close the connection after the body (no chunked framing), so we
/// just read to EOF -- no async runtime or HTTP client dependency needed.
fn uds_get(socket: &Path, path: &str) -> Result<String> {
    use std::io::{ErrorKind, Read, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket).map_err(|e| {
        // The socket is 0660 acousticslab:acousticslab, so EACCES is the common
        // failure for a normal user -- don't report it as "daemon not running".
        let hint = match e.kind() {
            ErrorKind::PermissionDenied => {
                "run as root, or add your user to the `acousticslab` group"
            }
            _ => "is acousticslabd running?",
        };
        anyhow::anyhow!("connect {}: {e} ({hint})", socket.display())
    })?;
    stream
        .set_read_timeout(Some(RESPONSE_BUDGET))
        .context("set read timeout")?;
    stream
        .set_write_timeout(Some(RESPONSE_BUDGET))
        .context("set write timeout")?;
    let request = format!("GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .context("send request")?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| match e.kind() {
        // A tripped read timeout surfaces as WouldBlock or TimedOut per platform.
        ErrorKind::WouldBlock | ErrorKind::TimedOut => anyhow::anyhow!(
            "no response within {}s: acousticslabd is up but not answering",
            RESPONSE_BUDGET.as_secs()
        ),
        _ => anyhow::Error::new(e).context("read response"),
    })?;

    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .context("malformed HTTP response (no header terminator)")?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let body = String::from_utf8_lossy(&raw[split + 4..]).into_owned();

    let status_line = head.lines().next().unwrap_or_default();
    let code = status_line.split_whitespace().nth(1).unwrap_or("");
    ensure!(
        code == "200",
        "daemon returned {status_line:?}: {}",
        body.trim()
    );
    Ok(body)
}

fn print_status_summary(v: &serde_json::Value) {
    let g = |k: &str| v.get(k);
    if let Some(up) = g("uptime_s").and_then(|x| x.as_u64()) {
        println!("uptime:    {up}s");
    }
    if let Some(cpu) = g("cpu_pct").and_then(|x| x.as_f64()) {
        println!("cpu:       {cpu:.1}%");
    }
    if let Some(rss) = g("mem_rss_kb").and_then(|x| x.as_u64()) {
        println!("mem rss:   {} MiB", rss / 1024);
    }
    if let Some(subsystems) = g("subsystems").and_then(|x| x.as_object()) {
        println!("subsystems:");
        for (name, sub) in subsystems {
            let healthy = sub.get("healthy").and_then(|h| h.as_bool());
            let mark = match healthy {
                Some(true) => "ok",
                Some(false) => "FAIL",
                None => "?",
            };
            let detail = sub
                .get("detail")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .trim();
            if detail.is_empty() {
                println!("  [{mark:>4}] {name}");
            } else {
                println!("  [{mark:>4}] {name}: {detail}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_after_extracts_values() {
        assert_eq!(kv_after("hw:CARD=Mic,DEV=0", "DEV="), Some("0".into()));
        assert_eq!(
            kv_after("hw:CARD=USB Audio,DEV=1", "DEV="),
            Some("1".into())
        );
        assert_eq!(kv_after("hw:CARD=X", "DEV="), None);
    }

    #[test]
    fn sanitize_id_keeps_only_safe_chars() {
        assert_eq!(sanitize_id("USB_Audio-2"), "USB_Audio-2");
        assert_eq!(sanitize_id("Foo Bar!"), "Foo-Bar-");
    }

    // Shape matches what the daemon serialises for `MicPolicy` (internally
    // tagged `MicSelection`) into <workspace>/config.toml.
    #[test]
    fn pinned_mic_id_detects_only_fixed_policies() {
        assert_eq!(
            pinned_mic_id("[mic.mic]\nkind = \"fixed\"\nid = \"linux-default\"\n").as_deref(),
            Some("linux-default")
        );
        assert_eq!(
            pinned_mic_id(
                "[mic.mic]\nkind = \"first_available\"\n\n[mic.channel]\nkind = \"auto\"\n"
            ),
            None
        );
        assert_eq!(pinned_mic_id("[inference]\ntop_k = 3\n"), None);
        assert_eq!(pinned_mic_id("not : valid : toml"), None);
    }

    // Nonexistent paths make `canonicalize` fall back to the raw string, so the
    // match reduces to exact-path equality -- enough to exercise the routing.
    #[test]
    fn doc_declares_rknn_at_matches_block_and_inline_forms() {
        let out = Path::new("/opt/al/backbone.rknn");
        let parse = |s: &str| s.parse::<toml_edit::DocumentMut>().unwrap();

        // Block `[[backbone.candidates]]`, matching rknn entry beside a burn one.
        assert!(doc_declares_rknn_at(
            &parse(
                "[[backbone.candidates]]\nkind = \"rknn\"\npath = \"/opt/al/backbone.rknn\"\n\n\
                 [[backbone.candidates]]\nkind = \"burn\"\npath = \"/opt/al/backbone.mpk\"\n"
            ),
            out
        ));
        // Inline array form.
        assert!(doc_declares_rknn_at(
            &parse(
                "backbone.candidates = [{ kind = \"rknn\", path = \"/opt/al/backbone.rknn\" }]\n"
            ),
            out
        ));
        // Right path, wrong kind: a burn-loaded .rknn would be rejected at boot.
        assert!(!doc_declares_rknn_at(
            &parse("[[backbone.candidates]]\nkind = \"burn\"\npath = \"/opt/al/backbone.rknn\"\n"),
            out
        ));
        // rknn candidate, but pointing elsewhere.
        assert!(!doc_declares_rknn_at(
            &parse("[[backbone.candidates]]\nkind = \"rknn\"\npath = \"/somewhere/else.rknn\"\n"),
            out
        ));
        // No backbone section.
        assert!(!doc_declares_rknn_at(
            &parse("[api]\nuds_path = \"/x\"\n"),
            out
        ));
    }

    // The generated candidate list must be valid TOML whose fields match the
    // daemon's MicCandidate/CandidateSource schema (serde is layout-agnostic, so
    // inline `source = {..}` deserializes exactly like the block form).
    #[test]
    fn generated_candidates_round_trip() {
        let mut aot = toml_edit::ArrayOfTables::new();
        aot.push(alsa_candidate_table(
            "alsa-Mic",
            "plughw:CARD=Mic,DEV=0",
            960,
            3840,
        ));
        aot.push(mock_candidate_table());

        let mut doc = "[mic]\n".parse::<toml_edit::DocumentMut>().unwrap();
        doc["mic"]["candidates"] = toml_edit::Item::ArrayOfTables(aot);
        let rendered = doc.to_string();

        let parsed: toml_edit::DocumentMut = rendered.parse().expect("generated TOML is valid");
        let cands = parsed["mic"]["candidates"]
            .as_array_of_tables()
            .expect("candidates is an array of tables");
        assert_eq!(cands.len(), 2);

        let alsa = cands.get(0).unwrap();
        assert_eq!(alsa["id"].as_str(), Some("alsa-Mic"));
        assert_eq!(
            alsa["channels"].as_array().unwrap().len(),
            1,
            "channels whitelist is non-empty"
        );
        let src = alsa["source"].as_table_like().unwrap();
        assert_eq!(src.get("kind").and_then(|k| k.as_str()), Some("alsa"));
        assert_eq!(
            src.get("hw_spec").and_then(|k| k.as_str()),
            Some("plughw:CARD=Mic,DEV=0")
        );
        assert_eq!(
            src.get("period_size").and_then(|k| k.as_integer()),
            Some(960)
        );
        assert_eq!(
            src.get("buffer_size").and_then(|k| k.as_integer()),
            Some(3840)
        );

        let mock = cands.get(1).unwrap()["source"].as_table_like().unwrap();
        assert_eq!(mock.get("kind").and_then(|k| k.as_str()), Some("mock"));
        assert!(
            mock.get("waveforms").and_then(|w| w.as_array()).is_some(),
            "mock has a waveforms array"
        );
    }

    // Exercises the real doc-editing path: reuse the first ALSA period/buffer,
    // dedupe ids by appending the device index, and keep the mock fallback.
    #[test]
    fn rebuild_candidates_replaces_with_failover_list() {
        let cfg = "\
[[mic.candidates]]
id = \"old\"
channels = [0]
[mic.candidates.source]
kind = \"alsa\"
hw_spec = \"hw:9,0\"
period_size = 480
buffer_size = 1920

[[mic.candidates]]
id = \"m\"
channels = [0]
[mic.candidates.source]
kind = \"mock\"
period_size = 512
sample_rate = 44100
waveforms = [{ kind = \"sine\", freq_hz = 1000.0, amplitude = 0.25 }]
";
        let mut doc = cfg.parse::<toml_edit::DocumentMut>().unwrap();
        let devices = vec![
            HwDevice {
                card: "USB".into(),
                dev: "0".into(),
                hw_spec: "plughw:CARD=USB,DEV=0".into(),
                desc: String::new(),
            },
            HwDevice {
                card: "USB".into(),
                dev: "1".into(),
                hw_spec: "plughw:CARD=USB,DEV=1".into(),
                desc: String::new(),
            },
        ];

        let (ids, had_mock) = rebuild_candidates(&mut doc, &devices).unwrap();
        assert!(had_mock);
        assert_eq!(ids, vec!["alsa-USB", "alsa-USB-1"]);

        let parsed = doc.to_string().parse::<toml_edit::DocumentMut>().unwrap();
        let cands = parsed["mic"]["candidates"].as_array_of_tables().unwrap();
        assert_eq!(cands.len(), 3, "2 ALSA + retained mock");
        let s0 = cands.get(0).unwrap()["source"].as_table_like().unwrap();
        assert_eq!(
            s0.get("hw_spec").and_then(|v| v.as_str()),
            Some("plughw:CARD=USB,DEV=0")
        );
        assert_eq!(
            s0.get("period_size").and_then(|v| v.as_integer()),
            Some(480),
            "reused the existing ALSA period"
        );
        let s2 = cands.get(2).unwrap()["source"].as_table_like().unwrap();
        assert_eq!(s2.get("kind").and_then(|v| v.as_str()), Some("mock"));
    }
}
