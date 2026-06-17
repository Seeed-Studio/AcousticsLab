//! prost-build wire-format codegen: one file per proto `package` into
//! `$OUT_DIR/<package>.rs`, `include!`d by [`crate::proto`]. Needs `protoc`.

use std::io::Result;

fn main() -> Result<()> {
    // Custom cfg (not bare `loom`) so loom test builds don't elide tokio's
    // `net` via its `#![cfg(not(loom))]` gates and break sibling dev-deps.
    println!("cargo:rustc-check-cfg=cfg(audio_buffer_loom)");
    let protos: &[&str] = &[
        "modules/proto/audio_stream.proto",
        "modules/proto/inference_stream.proto",
        // Wrapper imports the two above; listed last for stable codegen order.
        "modules/proto/envelope.proto",
    ];
    // All `bytes` fields as refcount-on-clone `Bytes`: broadcast fan-out avoids
    // a per-frame heap copy.
    let mut config = prost_build::Config::new();
    config.bytes(["."]);
    config.compile_protos(protos, &["modules/proto/"])?;
    for p in protos {
        println!("cargo:rerun-if-changed={p}");
    }
    // Emitting `rerun-if-changed` opts into an explicit watch list, so this
    // script must list itself to retrigger codegen on its own edits.
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
