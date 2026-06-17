//! Compile + smoke gate for the `stream_io_helpers` WS scaffolding, so the
//! helper module stays compilable even with no other consumer.

#[path = "stream_io_helpers/mod.rs"]
mod stream_io_helpers;

use bytes::Bytes;
use stream_io_helpers::{connect_tcp_ws, recv_binary, spawn_tcp_router};

/// Minimal audio-WS round trip pinned here purely to exercise (and thus gate
/// the compile of) the shared helper.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn helper_round_trip_audio() {
    let harness = spawn_tcp_router().await;
    let mut ws = connect_tcp_ws(&harness, "/stream/audio").await;

    let payload = Bytes::from_static(b" helper smoke");
    harness.audio_tx.send(payload.clone()).expect("send");

    let received = recv_binary(&mut ws).await;
    assert_eq!(received.as_ref(), payload.as_ref());
}
