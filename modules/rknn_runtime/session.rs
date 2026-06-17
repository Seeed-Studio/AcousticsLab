//! Loaded RKNN context + runtime library exposing safe query and inference.

use crate::rknn_runtime::{
    error::{Error, Result, check},
    ffi::SymbolTable,
    sys,
};
use std::{marker::PhantomData, mem::MaybeUninit, os::raw::c_char, path::Path};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct IoCount {
    pub n_input: u32,
    pub n_output: u32,
}

#[derive(Debug, Clone)]
pub struct SdkVersion {
    pub api: String,
    pub driver: String,
}

/// Decoded tensor attribute; `dims` is truncated to the valid `n_dims`.
#[derive(Debug, Clone)]
pub struct TensorAttr {
    pub index: u32,
    pub name: String,
    pub dims: Vec<u32>,
    /// Logical element count, excluding NPU-internal padding.
    pub n_elems: u32,
    /// Logical bytes: `n_elems * size_of::<type>()`.
    pub size: u32,
    pub dtype: DataType,
    pub format: TensorFormat,
    pub qnt_type: QntType,
}

/// Tensor element type; unrecognized header values map to `Unknown(raw)` so new
/// SDK additions round-trip instead of panicking.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Bool,
    Int4,
    Bfloat16,
    Unknown(u32),
}

impl DataType {
    pub(crate) fn to_raw(self) -> sys::rknn_tensor_type {
        match self {
            DataType::Float32 => sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32,
            DataType::Float16 => sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16,
            DataType::Int8 => sys::_rknn_tensor_type::RKNN_TENSOR_INT8,
            DataType::Uint8 => sys::_rknn_tensor_type::RKNN_TENSOR_UINT8,
            DataType::Int16 => sys::_rknn_tensor_type::RKNN_TENSOR_INT16,
            DataType::Uint16 => sys::_rknn_tensor_type::RKNN_TENSOR_UINT16,
            DataType::Int32 => sys::_rknn_tensor_type::RKNN_TENSOR_INT32,
            DataType::Uint32 => sys::_rknn_tensor_type::RKNN_TENSOR_UINT32,
            DataType::Int64 => sys::_rknn_tensor_type::RKNN_TENSOR_INT64,
            DataType::Bool => sys::_rknn_tensor_type::RKNN_TENSOR_BOOL,
            DataType::Int4 => sys::_rknn_tensor_type::RKNN_TENSOR_INT4,
            DataType::Bfloat16 => sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16,
            DataType::Unknown(x) => x,
        }
    }

    fn from_raw(raw: sys::rknn_tensor_type) -> Self {
        match raw {
            sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32 => DataType::Float32,
            sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16 => DataType::Float16,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT8 => DataType::Int8,
            sys::_rknn_tensor_type::RKNN_TENSOR_UINT8 => DataType::Uint8,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT16 => DataType::Int16,
            sys::_rknn_tensor_type::RKNN_TENSOR_UINT16 => DataType::Uint16,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT32 => DataType::Int32,
            sys::_rknn_tensor_type::RKNN_TENSOR_UINT32 => DataType::Uint32,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT64 => DataType::Int64,
            sys::_rknn_tensor_type::RKNN_TENSOR_BOOL => DataType::Bool,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT4 => DataType::Int4,
            sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16 => DataType::Bfloat16,
            x => DataType::Unknown(x),
        }
    }

    /// HOST-buffer bytes per element (librknnrt converts host->model when
    /// `pass_through=false`), not the model's; `None` for `Int4`/`Unknown(_)`.
    pub fn bytes_per_elem(self) -> Option<usize> {
        match self {
            DataType::Bool | DataType::Int8 | DataType::Uint8 => Some(1),
            DataType::Float16 | DataType::Bfloat16 | DataType::Int16 | DataType::Uint16 => Some(2),
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => Some(4),
            DataType::Int64 => Some(8),
            DataType::Int4 | DataType::Unknown(_) => None,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TensorFormat {
    Nchw,
    Nhwc,
    Nc1Hwc2,
    Undefined,
    Unknown(u32),
}

impl TensorFormat {
    pub(crate) fn to_raw(self) -> sys::rknn_tensor_format {
        match self {
            TensorFormat::Nchw => sys::_rknn_tensor_format::RKNN_TENSOR_NCHW,
            TensorFormat::Nhwc => sys::_rknn_tensor_format::RKNN_TENSOR_NHWC,
            TensorFormat::Nc1Hwc2 => sys::_rknn_tensor_format::RKNN_TENSOR_NC1HWC2,
            TensorFormat::Undefined => sys::_rknn_tensor_format::RKNN_TENSOR_UNDEFINED,
            TensorFormat::Unknown(x) => x,
        }
    }

    fn from_raw(raw: sys::rknn_tensor_format) -> Self {
        match raw {
            sys::_rknn_tensor_format::RKNN_TENSOR_NCHW => TensorFormat::Nchw,
            sys::_rknn_tensor_format::RKNN_TENSOR_NHWC => TensorFormat::Nhwc,
            sys::_rknn_tensor_format::RKNN_TENSOR_NC1HWC2 => TensorFormat::Nc1Hwc2,
            sys::_rknn_tensor_format::RKNN_TENSOR_UNDEFINED => TensorFormat::Undefined,
            x => TensorFormat::Unknown(x),
        }
    }
}

/// Quantization type; surfaced for diagnostics, not consumed by this crate.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum QntType {
    None,
    Dfp,
    AffineAsymmetric,
    Unknown(u32),
}

impl QntType {
    fn from_raw(raw: sys::rknn_tensor_qnt_type) -> Self {
        match raw {
            sys::_rknn_tensor_qnt_type::RKNN_TENSOR_QNT_NONE => QntType::None,
            sys::_rknn_tensor_qnt_type::RKNN_TENSOR_QNT_DFP => QntType::Dfp,
            sys::_rknn_tensor_qnt_type::RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => {
                QntType::AffineAsymmetric
            }
            x => QntType::Unknown(x),
        }
    }
}

/// Loaded RKNN model context. Concurrent calls on one context are UB, so this
/// is `Send` but not `Sync`; `PhantomData<Cell<()>>` opts out of the auto `Sync`.
pub struct Session {
    /// I/O attrs cached for byte-count validation (e.g. fp32 buffer into a
    /// fp16-input model).
    input_attrs: Vec<TensorAttr>,
    output_attrs: Vec<TensorAttr>,
    context: sys::rknn_context,
    table: SymbolTable,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("context", &format_args!("{:#x}", self.context))
            .field("table", &self.table)
            .finish()
    }
}

impl Session {
    /// Load the runtime library and init an RKNN context. `model` is `&mut`
    /// because `rknn_init` takes non-const `void*` and may mutate it.
    ///
    /// # Safety
    ///
    /// Callers must satisfy ALL of: (1) `lib_path` is a trusted Rockchip
    /// librknnrt; (2) its ABI matches vendored `bindings.rs` (host `layout_*`
    /// tests verify; a lib bump needs a re-gen); (3) the resolved path is pinned
    /// between boot and load (perms / immutable rootfs / `RKNN_LIB`). Mismatched
    /// ABI is UB despite the innocent per-call `unsafe`; once this returns the
    /// `&mut self` methods are safe.
    ///
    /// `flag = 0` internalizes the bytes so `model` may be dropped after; the
    /// zero-copy flag would instead require `model` to outlive the context.
    pub unsafe fn load(lib_path: &Path, model: &mut [u8]) -> Result<Self> {
        let table = SymbolTable::load(lib_path)?;
        // `size` is u32: guard silent truncation on a hypothetical >4 GiB buffer.
        let model_size = u32::try_from(model.len()).map_err(|_| Error::ShapeMismatch {
            what: "model bytes",
            expected: u32::MAX as usize,
            got: model.len(),
        })?;
        let mut ctx: sys::rknn_context = 0;
        let code = unsafe {
            (table.init)(
                &mut ctx,
                model.as_mut_ptr() as *mut _,
                model_size,
                0,
                std::ptr::null_mut(),
            )
        };
        check("rknn_init", "load model", code)?;
        // Partial Session (empty Vecs) so `&self` query helpers can fill the attr
        // cache; a `?` failure here drops it, releasing the context via Drop.
        let mut session = Session {
            table,
            context: ctx,
            input_attrs: Vec::new(),
            output_attrs: Vec::new(),
            _not_sync: PhantomData,
        };
        let io = session.io_count()?;
        // Bound a garbage count (corrupt model / ABI mismatch): near-u32::MAX
        // would make `reserve_exact` request hundreds of GiB and abort.
        const SANE_MAX_IO_TENSORS: u32 = 256;
        if io.n_input > SANE_MAX_IO_TENSORS || io.n_output > SANE_MAX_IO_TENSORS {
            return Err(Error::Rknn {
                name: "rknn_query",
                code: sys::RKNN_ERR_PARAM_INVALID,
                context: "io_count exceeds tensor limit",
            });
        }
        session.input_attrs.reserve_exact(io.n_input as usize);
        session.output_attrs.reserve_exact(io.n_output as usize);
        for i in 0..io.n_input {
            session.input_attrs.push(session.input_attr(i)?);
        }
        for i in 0..io.n_output {
            session.output_attrs.push(session.output_attr(i)?);
        }
        Ok(session)
    }

    /// Cached input tensor attr (no FFI); hot-path alternative to `input_attr`.
    pub fn input_attr_cached(&self, index: u32) -> Option<&TensorAttr> {
        self.input_attrs.get(index as usize)
    }

    /// Cached output tensor attr (no FFI).
    pub fn output_attr_cached(&self, index: u32) -> Option<&TensorAttr> {
        self.output_attrs.get(index as usize)
    }

    pub fn sdk_version(&self) -> Result<SdkVersion> {
        let mut ver = MaybeUninit::<sys::rknn_sdk_version>::zeroed();
        let code = unsafe {
            (self.table.query)(
                self.context,
                sys::_rknn_query_cmd::RKNN_QUERY_SDK_VERSION,
                ver.as_mut_ptr() as *mut _,
                std::mem::size_of::<sys::rknn_sdk_version>() as u32,
            )
        };
        check("rknn_query", "SDK version", code)?;
        let ver = unsafe { ver.assume_init() };
        Ok(SdkVersion {
            api: c_fixed_str_to_string(&ver.api_version)?,
            driver: c_fixed_str_to_string(&ver.drv_version)?,
        })
    }

    pub fn io_count(&self) -> Result<IoCount> {
        let mut io = MaybeUninit::<sys::rknn_input_output_num>::zeroed();
        let code = unsafe {
            (self.table.query)(
                self.context,
                sys::_rknn_query_cmd::RKNN_QUERY_IN_OUT_NUM,
                io.as_mut_ptr() as *mut _,
                std::mem::size_of::<sys::rknn_input_output_num>() as u32,
            )
        };
        check("rknn_query", "input/output count", code)?;
        let io = unsafe { io.assume_init() };
        Ok(IoCount {
            n_input: io.n_input,
            n_output: io.n_output,
        })
    }

    pub fn input_attr(&self, index: u32) -> Result<TensorAttr> {
        self.query_tensor_attr(
            index,
            sys::_rknn_query_cmd::RKNN_QUERY_INPUT_ATTR,
            "input attr",
        )
    }

    pub fn output_attr(&self, index: u32) -> Result<TensorAttr> {
        self.query_tensor_attr(
            index,
            sys::_rknn_query_cmd::RKNN_QUERY_OUTPUT_ATTR,
            "output attr",
        )
    }

    fn query_tensor_attr(
        &self,
        index: u32,
        cmd: sys::rknn_query_cmd,
        context: &'static str,
    ) -> Result<TensorAttr> {
        // C API protocol: caller writes `info.index`, the library fills the rest.
        let mut attr = MaybeUninit::<sys::rknn_tensor_attr>::zeroed();
        // SAFETY: `rknn_tensor_attr` is `#[repr(C)]` POD; writing `index` via the
        // raw ptr mutates one field without materializing a `&mut` to uninit data.
        unsafe {
            (*attr.as_mut_ptr()).index = index;
        }
        // SAFETY: live `rknn_context` owned for `&self`; ABI-verified `rknn_query`;
        // `attr` exclusively borrowed; `size_of` cast fits u32 (struct < 4 GiB).
        let code = unsafe {
            (self.table.query)(
                self.context,
                cmd,
                attr.as_mut_ptr() as *mut _,
                std::mem::size_of::<sys::rknn_tensor_attr>() as u32,
            )
        };
        check("rknn_query", context, code)?;
        // SAFETY: `check` short-circuited on non-success, so the library fully
        // populated `*attr`; all POD fields make `assume_init` sound.
        let attr = unsafe { attr.assume_init() };

        let n_dims = attr.n_dims as usize;
        // n_dims past `dims[RKNN_MAX_DIMS]` signals a C-side write-past-end; reject
        // as param-invalid rather than truncating silently.
        if n_dims > sys::RKNN_MAX_DIMS as usize {
            return Err(Error::Rknn {
                name: "rknn_query",
                code: sys::RKNN_ERR_PARAM_INVALID,
                context: "n_dims exceeds RKNN_MAX_DIMS",
            });
        }

        Ok(TensorAttr {
            index: attr.index,
            name: c_fixed_str_to_string(&attr.name)?,
            dims: attr.dims[..n_dims].to_vec(),
            n_elems: attr.n_elems,
            size: attr.size,
            dtype: DataType::from_raw(attr.type_),
            format: TensorFormat::from_raw(attr.fmt),
            qnt_type: QntType::from_raw(attr.qnt_type),
        })
    }

    pub(crate) fn table(&self) -> &SymbolTable {
        &self.table
    }

    pub(crate) fn context(&self) -> sys::rknn_context {
        self.context
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Drop must not panic, so capture the code instead of `?`; non-success
        // leaks the NPU context and persistent leaks exhaust the device.
        // SAFETY: `self.context` and destroy fn ptr come from this Session's own
        // library, valid for its lifetime.
        let code = unsafe { (self.table.destroy)(self.context) };
        if code != 0 {
            tracing::warn!(
                target: "rknn",
                code,
                "rknn_destroy failed during Session Drop; context may have leaked",
            );
        }
    }
}

/// Decode a C `char[N]` into `String`, taking bytes up to the first `\0` (a
/// missing terminator means the whole buffer is text).
fn c_fixed_str_to_string<const N: usize>(buf: &[c_char; N]) -> Result<String> {
    // SAFETY: `c_char` matches `u8` size/align, so the `N`-wide reinterpret is
    // in-bounds and shares `buf`'s lifetime.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, N) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(N);
    let s = std::str::from_utf8(&bytes[..end])?;
    Ok(s.to_string())
}
