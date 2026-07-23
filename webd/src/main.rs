//! `acousticslab-webd`: the TCP-facing front for the UDS-only `acousticslabd`.
//!
//! The daemon serves its whole HTTP/WebSocket/SSE surface over a Unix domain
//! socket (`[api].uds_path`), which a browser cannot reach. This utility
//! bridges that gap: it serves the browser SPA as static files and
//! reverse-proxies the dynamic surface to the daemon socket --
//!
//!   * `/api/**`    REST + SSE (`text/event-stream`, streamed, never buffered)
//!   * `/stream/**` WebSocket (`/stream/audio`, `/stream/infer`)
//!   * everything else -> static file, with a client-routing fallback to
//!     `index.html`.
//!
//! Config comes from the environment so a systemd unit owns the knobs; the
//! variables and defaults are listed in [`USAGE`].

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use hyper::upgrade::OnUpgrade;
use hyper_util::rt::TokioIo;
use tokio::net::{TcpListener, UnixStream};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

// Loopback-only: the control plane is unauthenticated, so it is not published
// to the network unless ACOUSTICSLAB_WEB_LISTEN opts in (packaging/etc/web.env).
const DEFAULT_LISTEN: &str = "127.0.0.1:8080";
const DEFAULT_API_SOCKET: &str = "/run/acousticslab/api.sock";
const DEFAULT_WEB_ROOT: &str = "/usr/share/acousticslab/web";

const USAGE: &str = "\
acousticslab-webd -- TCP front for the UDS-only acousticslabd: serves the SPA
and reverse-proxies /api/** and /stream/** to the daemon socket.

Takes no options; configured via the environment:
  ACOUSTICSLAB_WEB_LISTEN   TCP ip:port to bind        (default 127.0.0.1:8080)
  ACOUSTICSLAB_API_SOCKET   daemon UDS to proxy to     (default /run/acousticslab/api.sock)
  ACOUSTICSLAB_WEB_ROOT     SPA static root            (default /usr/share/acousticslab/web)
  ACOUSTICSLAB_WEB_LOG      tracing env-filter         (default info)
";

/// Shared handler state: the daemon socket every proxied request dials.
struct Proxy {
    api_socket: PathBuf,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    // No options: systemd owns the knobs via the environment. Still answer
    // --version/--help and reject strays, so a typo can't silently bind a port.
    if let Some(arg) = std::env::args().nth(1) {
        match arg.as_str() {
            "-V" | "--version" => {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => bail!("unexpected argument {other:?}\n\n{USAGE}"),
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("ACOUSTICSLAB_WEB_LOG")
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // `SocketAddr` resolves no names, hence `<ip>:<port>` in the message.
    let listen: SocketAddr = env_or("ACOUSTICSLAB_WEB_LISTEN", DEFAULT_LISTEN)
        .parse()
        .context("ACOUSTICSLAB_WEB_LISTEN must be <ip>:<port>, e.g. 127.0.0.1:8080")?;
    let api_socket = PathBuf::from(env_or("ACOUSTICSLAB_API_SOCKET", DEFAULT_API_SOCKET));
    let web_root = PathBuf::from(env_or("ACOUSTICSLAB_WEB_ROOT", DEFAULT_WEB_ROOT));
    // Non-fatal: proxying still works, and the bridge is useful without the SPA.
    if !web_root.is_dir() {
        tracing::warn!(path = %web_root.display(), "web root is not a directory; static requests will 404");
    }

    // SPA static serving: unknown paths fall back to `index.html` so the
    // client-side router owns deep links (`/workspaces/...` etc.).
    let static_service = ServeDir::new(&web_root)
        .append_index_html_on_directories(true)
        .fallback(ServeFile::new(web_root.join("index.html")));

    let proxy = Arc::new(Proxy {
        api_socket: api_socket.clone(),
    });

    let app = Router::new()
        // Wildcard captures the whole subtree; the daemon sees the path
        // verbatim (webd is root-mounted, matching the daemon's own root).
        .route("/api/{*rest}", any(proxy_handler))
        .route("/stream/{*rest}", any(proxy_handler))
        .with_state(proxy)
        .fallback_service(static_service)
        .layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    tracing::info!(
        %listen,
        api_socket = %api_socket.display(),
        web_root = %web_root.display(),
        "acousticslab-webd listening",
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;
    Ok(())
}

/// Resolve on SIGTERM (systemd stop) or Ctrl-C so in-flight streams drain.
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "cannot install SIGTERM handler; Ctrl-C only");
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = term.recv() => {}
        r = tokio::signal::ctrl_c() => { let _ = r; }
    }
    tracing::info!("shutdown signal received");
}

/// Reverse-proxy one request to the daemon UDS. Normal responses (including
/// streamed SSE) are handed straight back; a `101 Switching Protocols` splices
/// the two upgraded byte streams for WebSocket.
async fn proxy_handler(State(proxy): State<Arc<Proxy>>, mut req: Request) -> Response {
    // The inbound (client-side) upgrade future fires only after we send the 101
    // back; capture it before the request is consumed.
    let client_upgrade = req.extensions_mut().remove::<OnUpgrade>();
    let is_upgrade = req.headers().contains_key(header::UPGRADE);

    let stream = match UnixStream::connect(&proxy.api_socket).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                error = %e,
                socket = %proxy.api_socket.display(),
                "connect to acousticslabd failed",
            );
            return (StatusCode::BAD_GATEWAY, "acousticslabd unavailable\n").into_response();
        }
    };

    let (mut sender, conn) = match hyper::client::conn::http1::handshake(TokioIo::new(stream)).await
    {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!(error = %e, "http/1 handshake with acousticslabd failed");
            return (StatusCode::BAD_GATEWAY, "acousticslabd handshake failed\n").into_response();
        }
    };

    // hyper only advances the request/response (and any upgrade) while its
    // connection future is polled; drive it alongside this handler.
    // `.with_upgrades()` keeps the socket usable as a raw stream after a 101.
    let conn_task = tokio::spawn(async move {
        if let Err(e) = conn.with_upgrades().await {
            tracing::debug!(error = %e, "acousticslabd connection ended");
        }
    });

    // Let hyper re-derive framing from the body (drop stale Content-Length to
    // avoid duplicate framing / smuggling); strip hop-by-hop headers, keeping the
    // WebSocket upgrade set. HTTP/1 origin-form needs a Host, so synthesize one.
    strip_hop_headers(req.headers_mut(), is_upgrade);
    req.headers_mut().remove(header::CONTENT_LENGTH);
    if !req.headers().contains_key(header::HOST) {
        req.headers_mut()
            .insert(header::HOST, HeaderValue::from_static("localhost"));
    }

    let mut res = match sender.send_request(req).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "forwarding request to acousticslabd failed");
            conn_task.abort();
            return (StatusCode::BAD_GATEWAY, "acousticslabd request failed\n").into_response();
        }
    };

    if res.status() == StatusCode::SWITCHING_PROTOCOLS {
        // WebSocket: grab the daemon-side upgraded stream, then splice it to the
        // browser-side one once the 101 we return below completes its handshake.
        let daemon_upgrade = hyper::upgrade::on(&mut res);
        match client_upgrade {
            Some(client_upgrade) => {
                tokio::spawn(async move {
                    match tokio::join!(client_upgrade, daemon_upgrade) {
                        (Ok(client_io), Ok(daemon_io)) => {
                            let mut client_io = TokioIo::new(client_io);
                            let mut daemon_io = TokioIo::new(daemon_io);
                            if let Err(e) =
                                tokio::io::copy_bidirectional(&mut client_io, &mut daemon_io).await
                            {
                                tracing::debug!(error = %e, "websocket relay closed");
                            }
                        }
                        (client, daemon) => {
                            tracing::warn!(
                                client_ok = client.is_ok(),
                                daemon_ok = daemon.is_ok(),
                                "websocket upgrade handoff failed",
                            );
                        }
                    }
                });
            }
            None => tracing::warn!("daemon sent 101 but the client did not request an upgrade"),
        }
        // Return the daemon's 101 (status + Upgrade/Connection/Sec-WebSocket-*
        // headers) to the browser; an empty body lets hyper perform the upgrade.
        let (parts, _body) = res.into_parts();
        return Response::from_parts(parts, Body::empty());
    }

    // Normal or streaming (SSE) response: `Incoming` streams as `conn_task`
    // drives the connection. Strip hop-by-hop + stale chunked framing (the body
    // is already transfer-decoded) so the server reframes it cleanly.
    let (mut parts, body) = res.into_parts();
    strip_hop_headers(&mut parts.headers, false);
    Response::from_parts(parts, Body::new(body))
}

/// Remove hop-by-hop headers (RFC 9110 §7.6.1) so they don't leak across the
/// proxy hop. `keep_upgrade` preserves `Connection`/`Upgrade` for the WebSocket
/// handshake. `Transfer-Encoding` goes too: the peer reframes from the body.
fn strip_hop_headers(headers: &mut HeaderMap, keep_upgrade: bool) {
    use axum::http::header::{
        PROXY_AUTHENTICATE, PROXY_AUTHORIZATION, TE, TRAILER, TRANSFER_ENCODING,
    };
    for h in [
        TRANSFER_ENCODING,
        TE,
        TRAILER,
        PROXY_AUTHENTICATE,
        PROXY_AUTHORIZATION,
    ] {
        headers.remove(h);
    }
    headers.remove(HeaderName::from_static("keep-alive"));
    if !keep_upgrade {
        headers.remove(header::CONNECTION);
        headers.remove(header::UPGRADE);
    }
}
