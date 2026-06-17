//! Minimal safe wrapper around `librknnrt.so` / `librknnmrt.so` (Rockchip NPU
//! runtime) for single-input/single-output inference with zero-copy borrowed
//! buffers at the FFI boundary.
//!
//! Failure model: an `extern "C"` SIGSEGV is uncatchable by
//! [`std::panic::catch_unwind`] (signal precedes unwind) and aborts, on which the
//! external supervisor restarts. A wedged-not-crashed NPU (firmware deadlock in
//! `rknn_run`, no signal) is caught at the engine level not per-call, since the
//! engine runs in [`tokio::task::spawn_blocking`] with no reachable
//! [`tokio::time::timeout`] and a per-call timeout would cost on the hot path.

#![warn(missing_debug_implementations)]

// Frozen FFI bindings, `pub(crate)` to force callers through the checked
// `Session`/`InputSlice`/`OutputSlice` wrappers; raw structs would reintroduce
// the swallowed-error pattern this module exists to prevent.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all
)]
pub(crate) mod sys {
    include!("rknn_runtime/bindings.rs");
}

mod error;
mod ffi;
mod infer;
mod session;
// Mirror the rknpu cfg so `utils` is not flagged dead on hosts that never reach it.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
pub(crate) mod utils;

pub use error::{Error, Result};
pub use infer::{InputSlice, OutputSlice};
pub use session::{DataType, IoCount, QntType, SdkVersion, Session, TensorAttr, TensorFormat};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_rknn_input_output_num() {
        assert_eq!(std::mem::size_of::<sys::rknn_input_output_num>(), 8);
        assert_eq!(std::mem::align_of::<sys::rknn_input_output_num>(), 4);
    }

    /// A size mismatch means the FFI ABI diverged from the vendored library, corrupting every call.
    #[test]
    fn layout_rknn_input_is_32_bytes_on_64bit() {
        assert_eq!(std::mem::size_of::<sys::rknn_input>(), 32);
        assert_eq!(std::mem::align_of::<sys::rknn_input>(), 8);
    }

    #[test]
    fn layout_rknn_output_is_24_bytes_on_64bit() {
        assert_eq!(std::mem::size_of::<sys::rknn_output>(), 24);
        assert_eq!(std::mem::align_of::<sys::rknn_output>(), 8);
    }

    #[test]
    fn rknn_context_is_u64() {
        assert_eq!(std::mem::size_of::<sys::rknn_context>(), 8);
    }

    /// `rknn_query` passes this size and rejects any mismatch with `RKNN_ERR_PARAM_INVALID`; catch drift here, not at runtime.
    #[test]
    fn layout_rknn_tensor_attr_is_376_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_tensor_attr>(), 376);
        assert_eq!(std::mem::align_of::<sys::rknn_tensor_attr>(), 4);
    }

    #[test]
    fn layout_rknn_sdk_version_is_512_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_sdk_version>(), 512);
        assert_eq!(std::mem::align_of::<sys::rknn_sdk_version>(), 1);
    }

    /// Always passed `null_mut` today; layout assert catches a future caller populating it at host build, not on device.
    #[test]
    fn layout_rknn_init_extend_is_136_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_init_extend>(), 136);
        assert_eq!(std::mem::align_of::<sys::rknn_init_extend>(), 8);
    }

    /// `rknn_run_extend`/`rknn_output_extend` passed `null_mut` today; layout assert guards silent ABI drift if a future caller populates them.
    #[test]
    fn layout_rknn_run_extend_is_24_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_run_extend>(), 24);
        assert_eq!(std::mem::align_of::<sys::rknn_run_extend>(), 8);
    }

    #[test]
    fn layout_rknn_output_extend_is_8_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_output_extend>(), 8);
        assert_eq!(std::mem::align_of::<sys::rknn_output_extend>(), 8);
    }

    #[test]
    fn datatype_to_raw_all_distinct() {
        let raws: Vec<_> = [
            DataType::Float32,
            DataType::Float16,
            DataType::Int8,
            DataType::Uint8,
            DataType::Int16,
            DataType::Uint16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int64,
            DataType::Bool,
            DataType::Int4,
            DataType::Bfloat16,
        ]
        .iter()
        .map(|d| d.to_raw())
        .collect();
        let mut uniq = raws.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(
            raws.len(),
            uniq.len(),
            "DataType variants must map to distinct raw values"
        );
    }

    /// Every `RKNN_TENSOR_*` (bar the `RKNN_TENSOR_TYPE_MAX` sentinel) must round-trip through `to_raw`; an unmapped dtype silently skips host-buffer size validation.
    #[test]
    fn datatype_round_trip_covers_all_known_constants() {
        let constants: &[(sys::rknn_tensor_type, &str)] = &[
            (sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32, "FLOAT32"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16, "FLOAT16"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT8, "INT8"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_UINT8, "UINT8"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT16, "INT16"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_UINT16, "UINT16"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT32, "INT32"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_UINT32, "UINT32"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT64, "INT64"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_BOOL, "BOOL"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT4, "INT4"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16, "BFLOAT16"),
        ];
        for (raw, name) in constants {
            // Explicit match: compiler enforces this list stays in sync with the enum.
            let variant = match *raw {
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32 => DataType::Float32,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16 => DataType::Float16,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT8 => DataType::Int8,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_UINT8 => DataType::Uint8,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT16 => DataType::Int16,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_UINT16 => DataType::Uint16,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT32 => DataType::Int32,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_UINT32 => DataType::Uint32,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT64 => DataType::Int64,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_BOOL => DataType::Bool,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT4 => DataType::Int4,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16 => DataType::Bfloat16,
                x => panic!("RKNN_TENSOR_{name} (raw={x}) has no matching DataType variant"),
            };
            assert_eq!(
                variant.to_raw(),
                *raw,
                "round-trip mismatch for RKNN_TENSOR_{name}",
            );
        }
        assert!(
            sys::_rknn_tensor_type::RKNN_TENSOR_TYPE_MAX as usize >= constants.len(),
            "RKNN_TENSOR_TYPE_MAX shrank below the known constant list -- bindings drift?",
        );
    }

    #[test]
    fn tensor_format_to_raw_is_correct() {
        assert_eq!(
            TensorFormat::Nchw.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_NCHW
        );
        assert_eq!(
            TensorFormat::Nhwc.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_NHWC
        );
        assert_eq!(
            TensorFormat::Nc1Hwc2.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_NC1HWC2
        );
        assert_eq!(
            TensorFormat::Undefined.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_UNDEFINED
        );
    }

    /// Every header error code must decode to a known name, not "UNKNOWN" (catches decoder-vs-enum drift).
    #[test]
    fn error_name_covers_every_known_code() {
        for (code, expected) in [
            (sys::RKNN_ERR_FAIL, "RKNN_ERR_FAIL"),
            (sys::RKNN_ERR_TIMEOUT, "RKNN_ERR_TIMEOUT"),
            (
                sys::RKNN_ERR_DEVICE_UNAVAILABLE,
                "RKNN_ERR_DEVICE_UNAVAILABLE",
            ),
            (sys::RKNN_ERR_MALLOC_FAIL, "RKNN_ERR_MALLOC_FAIL"),
            (sys::RKNN_ERR_PARAM_INVALID, "RKNN_ERR_PARAM_INVALID"),
            (sys::RKNN_ERR_MODEL_INVALID, "RKNN_ERR_MODEL_INVALID"),
            (sys::RKNN_ERR_CTX_INVALID, "RKNN_ERR_CTX_INVALID"),
            (sys::RKNN_ERR_INPUT_INVALID, "RKNN_ERR_INPUT_INVALID"),
            (sys::RKNN_ERR_OUTPUT_INVALID, "RKNN_ERR_OUTPUT_INVALID"),
            (sys::RKNN_ERR_DEVICE_UNMATCH, "RKNN_ERR_DEVICE_UNMATCH"),
            (
                sys::RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL,
                "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL",
            ),
            (
                sys::RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION,
                "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION",
            ),
            (
                sys::RKNN_ERR_TARGET_PLATFORM_UNMATCH,
                "RKNN_ERR_TARGET_PLATFORM_UNMATCH",
            ),
        ] {
            assert_eq!(error::rknn_error_name(code), expected, "code = {code}");
        }
        assert_eq!(error::rknn_error_name(sys::RKNN_SUCC as _), "RKNN_SUCC");
        assert_eq!(error::rknn_error_name(-9999), "UNKNOWN");
    }

    #[test]
    fn load_nonexistent_library_errors_cleanly() {
        let mut fake_model = vec![0u8; 128];
        // SAFETY: path cannot exist, so load fails before any FFI symbol is resolved or invoked.
        let err =
            unsafe { Session::load(std::path::Path::new("/no/such/library.so"), &mut fake_model) }
                .expect_err("should fail");
        match err {
            Error::LibraryLoad { .. } => {}
            other => panic!("expected LibraryLoad, got {other:?}"),
        }
    }
}
