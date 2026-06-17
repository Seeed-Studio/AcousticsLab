//! Zero-copy inference API: caller owns all buffers, `Session::infer` never
//! allocates. `rknn_outputs_release` runs after every `rknn_outputs_get`
//! (success or failure) since the C contract is ambiguous on per-call internal
//! state and skipping it could leak runtime-side accounting.

use crate::rknn_runtime::{
    error::{Error, Result, check},
    session::{DataType, Session, TensorFormat},
    sys,
};
use std::os::raw::c_void;

/// rknn ABI sizes are `u32`; error above the limit instead of truncating.
fn n_bytes_to_u32(what: &'static str, n_bytes: usize) -> Result<u32> {
    u32::try_from(n_bytes).map_err(|_| Error::ShapeMismatch {
        what,
        expected: u32::MAX as usize,
        got: n_bytes,
    })
}

/// Borrowed input buffer + metadata for one input tensor. Holds `&mut [T]`
/// (not `&[T]`) because `rknn_inputs_set` takes non-const `void *buf`: an
/// exclusive borrow stays sound even if the runtime mutates `buf`.
#[derive(Debug)]
pub struct InputSlice<'a> {
    pub(crate) index: u32,
    pub(crate) ptr: *mut c_void,
    pub(crate) n_bytes: usize,
    pub(crate) dtype: DataType,
    pub(crate) fmt: TensorFormat,
    pub(crate) pass_through: bool,
    _marker: std::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> InputSlice<'a> {
    /// Fp32 host buffer; librknnrt converts to the model's native dtype
    /// (`pass_through=false`).
    pub fn f32(index: u32, data: &'a mut [f32]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Float32,
            fmt: TensorFormat::Undefined,
            pass_through: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Fp16 host buffer (`&mut [u16]` = fp16 bit-pattern); `pass_through=true`
    /// (bytes already match the NPU native dtype).
    pub fn f16(index: u32, data: &'a mut [u16]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Float16,
            fmt: TensorFormat::Undefined,
            pass_through: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Int8 host buffer (quantized models).
    pub fn i8(index: u32, data: &'a mut [i8]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Int8,
            fmt: TensorFormat::Undefined,
            pass_through: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Uint8 host buffer (typically image inputs).
    pub fn u8(index: u32, data: &'a mut [u8]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Uint8,
            fmt: TensorFormat::Undefined,
            pass_through: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Source-layout hint. Required with `pass_through=false` on rv1126b: its
    /// normalize pipeline rejects `Undefined` ("only support NHWC src layout!").
    pub fn with_format(mut self, fmt: TensorFormat) -> Self {
        self.fmt = fmt;
        self
    }

    /// Override the dtype-specific `pass_through` default set by the constructor.
    pub fn with_pass_through(mut self, pass_through: bool) -> Self {
        self.pass_through = pass_through;
        self
    }
}

/// Borrowed output buffer + metadata; librknnrt writes into `ptr` in place.
#[derive(Debug)]
pub struct OutputSlice<'a> {
    pub(crate) index: u32,
    pub(crate) ptr: *mut c_void,
    pub(crate) n_bytes: usize,
    pub(crate) want_float: bool,
    _marker: std::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> OutputSlice<'a> {
    /// Pre-allocated fp32 output (`want_float=true`: librknnrt dequantizes
    /// native fp16 to fp32). `buf.len()` must equal the tensor's num_elements.
    pub fn f32_preallocated(index: u32, buf: &'a mut [f32]) -> Self {
        OutputSlice {
            index,
            n_bytes: std::mem::size_of_val(buf),
            ptr: buf.as_mut_ptr() as *mut c_void,
            want_float: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Pre-allocated fp16 output (`&mut [u16]`, `want_float=false`: raw native
    /// fp16 write, no dequant).
    pub fn f16_preallocated(index: u32, buf: &'a mut [u16]) -> Self {
        OutputSlice {
            index,
            n_bytes: std::mem::size_of_val(buf),
            ptr: buf.as_mut_ptr() as *mut c_void,
            want_float: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl Session {
    /// Run one inference cycle, zero-alloc; on return `output` holds the
    /// dequantized (or raw) result.
    pub fn infer(&mut self, input: InputSlice<'_>, output: OutputSlice<'_>) -> Result<()> {
        let in_size = n_bytes_to_u32("input buffer bytes", input.n_bytes)?;
        let out_size = n_bytes_to_u32("output buffer bytes", output.n_bytes)?;
        // Expected host bytes: pass_through=false uses caller dtype
        // `n_elems * sizeof(dtype)` NOT `attr.size` (the latter breaks fp32->fp16);
        // true uses native `attr.size`. Sub-byte (`Int4`)/`Unknown` lack a
        // byte-per-elem, so skip and defer to the caller-layer guard.
        if let Some(attr) = self.input_attr_cached(input.index) {
            let expected = if input.pass_through {
                attr.size as usize
            } else {
                match input.dtype.bytes_per_elem() {
                    Some(b) => attr.n_elems as usize * b,
                    None => input.n_bytes,
                }
            };
            if input.n_bytes != expected {
                return Err(Error::ShapeMismatch {
                    what: "input buffer bytes",
                    expected,
                    got: input.n_bytes,
                });
            }
        }
        if let Some(attr) = self.output_attr_cached(output.index) {
            let expected = if output.want_float {
                attr.n_elems as usize * 4
            } else {
                attr.size as usize
            };
            if output.n_bytes != expected {
                return Err(Error::ShapeMismatch {
                    what: "output buffer bytes",
                    expected,
                    got: output.n_bytes,
                });
            }
        }

        let mut rk_in = sys::rknn_input {
            index: input.index,
            buf: input.ptr,
            size: in_size,
            pass_through: input.pass_through as u8,
            type_: input.dtype.to_raw(),
            fmt: input.fmt.to_raw(),
        };
        // SAFETY: `input.ptr` is the exclusively-borrowed `&mut [T]` from an
        // InputSlice ctor (no aliasing, holds even if the runtime writes buf);
        // `rk_in` is a stack-local.
        let code = unsafe {
            (self.table().inputs_set)(self.context(), 1, &mut rk_in as *mut sys::rknn_input)
        };
        check("rknn_inputs_set", "set input 0", code)?;

        // SAFETY: `self.context()` is a live rknn_context held by `self`; NULL
        // second arg selects the default blocking synchronous run.
        let code = unsafe { (self.table().run)(self.context(), std::ptr::null_mut()) };
        check("rknn_run", "run inference", code)?;

        let mut rk_out = sys::rknn_output {
            want_float: output.want_float as u8,
            is_prealloc: 1, // our buffer; librknnrt writes in place
            index: output.index,
            buf: output.ptr,
            size: out_size,
        };
        // SAFETY: `output.ptr` is the exclusively-borrowed `&mut [T]` from an
        // OutputSlice ctor; `rk_out` is a stack-local. With `is_prealloc=1`
        // librknnrt writes only our buf and the stack-local header fields.
        let get_code = unsafe {
            (self.table().outputs_get)(
                self.context(),
                1,
                &mut rk_out as *mut sys::rknn_output,
                std::ptr::null_mut(),
            )
        };

        // Release ALWAYS, even on a failed get: the contract is ambiguous on
        // whether a failed get still owns internal accounting, so skipping could
        // leak per call. is_prealloc=TRUE frees only internal state, not the user
        // buf. Check get before release below so a get failure is the root cause.
        // SAFETY: same invariants as `outputs_get`; safe after a failed get.
        let release_code = unsafe {
            (self.table().outputs_release)(self.context(), 1, &mut rk_out as *mut sys::rknn_output)
        };
        check("rknn_outputs_get", "get output 0", get_code)?;
        check("rknn_outputs_release", "release output 0", release_code)?;
        Ok(())
    }
}
