// Frozen FFI bindings for librknnrt / librknnmrt, committed verbatim from bindgen
// of Rockchip's `rknn_api.h`; C ABI is stable across librknnrt 2.x and sizeof/alignof
// asserts in `rknn_runtime.rs`'s `tests` catch silent drift. The `extern "C"` fns are
// resolved at runtime via libloading (never linked as Rust items), so the linker need
// not bind them.

pub const RKNN_FLAG_PRIOR_HIGH: u32 = 0;
pub const RKNN_FLAG_PRIOR_MEDIUM: u32 = 1;
pub const RKNN_FLAG_PRIOR_LOW: u32 = 2;
pub const RKNN_FLAG_ASYNC_MASK: u32 = 4;
pub const RKNN_FLAG_COLLECT_PERF_MASK: u32 = 8;
pub const RKNN_FLAG_MEM_ALLOC_OUTSIDE: u32 = 16;
pub const RKNN_FLAG_SHARE_WEIGHT_MEM: u32 = 32;
pub const RKNN_FLAG_FENCE_IN_OUTSIDE: u32 = 64;
pub const RKNN_FLAG_FENCE_OUT_OUTSIDE: u32 = 128;
pub const RKNN_FLAG_COLLECT_MODEL_INFO_ONLY: u32 = 256;
pub const RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE: u32 = 512;
pub const RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU: u32 = 1024;
pub const RKNN_FLAG_ENABLE_SRAM: u32 = 2048;
pub const RKNN_FLAG_SHARE_SRAM: u32 = 4096;
pub const RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY: u32 = 8192;
pub const RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE: u32 = 16384;
pub const RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE: u32 = 32768;
pub const RKNN_FLAG_MODEL_BUFFER_ZERO_COPY: u32 = 65536;
pub const RKNN_SUCC: u32 = 0;
pub const RKNN_ERR_FAIL: i32 = -1;
pub const RKNN_ERR_TIMEOUT: i32 = -2;
pub const RKNN_ERR_DEVICE_UNAVAILABLE: i32 = -3;
pub const RKNN_ERR_MALLOC_FAIL: i32 = -4;
pub const RKNN_ERR_PARAM_INVALID: i32 = -5;
pub const RKNN_ERR_MODEL_INVALID: i32 = -6;
pub const RKNN_ERR_CTX_INVALID: i32 = -7;
pub const RKNN_ERR_INPUT_INVALID: i32 = -8;
pub const RKNN_ERR_OUTPUT_INVALID: i32 = -9;
pub const RKNN_ERR_DEVICE_UNMATCH: i32 = -10;
pub const RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL: i32 = -11;
pub const RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION: i32 = -12;
pub const RKNN_ERR_TARGET_PLATFORM_UNMATCH: i32 = -13;
pub const RKNN_MAX_DIMS: u32 = 16;
pub const RKNN_MAX_NAME_LEN: u32 = 256;
pub type rknn_context = u64;
pub mod _rknn_query_cmd {
    pub type Type = ::std::os::raw::c_uint;
    pub const RKNN_QUERY_IN_OUT_NUM: Type = 0;
    pub const RKNN_QUERY_INPUT_ATTR: Type = 1;
    pub const RKNN_QUERY_OUTPUT_ATTR: Type = 2;
    pub const RKNN_QUERY_PERF_DETAIL: Type = 3;
    pub const RKNN_QUERY_PERF_RUN: Type = 4;
    pub const RKNN_QUERY_SDK_VERSION: Type = 5;
    pub const RKNN_QUERY_MEM_SIZE: Type = 6;
    pub const RKNN_QUERY_CUSTOM_STRING: Type = 7;
    pub const RKNN_QUERY_NATIVE_INPUT_ATTR: Type = 8;
    pub const RKNN_QUERY_NATIVE_OUTPUT_ATTR: Type = 9;
    pub const RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR: Type = 8;
    pub const RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR: Type = 9;
    pub const RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR: Type = 10;
    pub const RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR: Type = 11;
    pub const RKNN_QUERY_DEVICE_MEM_INFO: Type = 12;
    pub const RKNN_QUERY_INPUT_DYNAMIC_RANGE: Type = 13;
    pub const RKNN_QUERY_CURRENT_INPUT_ATTR: Type = 14;
    pub const RKNN_QUERY_CURRENT_OUTPUT_ATTR: Type = 15;
    pub const RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR: Type = 16;
    pub const RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR: Type = 17;
    pub const RKNN_QUERY_CMD_MAX: Type = 18;
}
pub use self::_rknn_query_cmd::Type as rknn_query_cmd;
pub mod _rknn_tensor_type {
    pub type Type = ::std::os::raw::c_uint;
    pub const RKNN_TENSOR_FLOAT32: Type = 0;
    pub const RKNN_TENSOR_FLOAT16: Type = 1;
    pub const RKNN_TENSOR_INT8: Type = 2;
    pub const RKNN_TENSOR_UINT8: Type = 3;
    pub const RKNN_TENSOR_INT16: Type = 4;
    pub const RKNN_TENSOR_UINT16: Type = 5;
    pub const RKNN_TENSOR_INT32: Type = 6;
    pub const RKNN_TENSOR_UINT32: Type = 7;
    pub const RKNN_TENSOR_INT64: Type = 8;
    pub const RKNN_TENSOR_BOOL: Type = 9;
    pub const RKNN_TENSOR_INT4: Type = 10;
    pub const RKNN_TENSOR_BFLOAT16: Type = 11;
    pub const RKNN_TENSOR_TYPE_MAX: Type = 12;
}
pub use self::_rknn_tensor_type::Type as rknn_tensor_type;
pub mod _rknn_tensor_qnt_type {
    pub type Type = ::std::os::raw::c_uint;
    pub const RKNN_TENSOR_QNT_NONE: Type = 0;
    pub const RKNN_TENSOR_QNT_DFP: Type = 1;
    pub const RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: Type = 2;
    pub const RKNN_TENSOR_QNT_MAX: Type = 3;
}
pub use self::_rknn_tensor_qnt_type::Type as rknn_tensor_qnt_type;
pub mod _rknn_tensor_format {
    pub type Type = ::std::os::raw::c_uint;
    pub const RKNN_TENSOR_NCHW: Type = 0;
    pub const RKNN_TENSOR_NHWC: Type = 1;
    pub const RKNN_TENSOR_NC1HWC2: Type = 2;
    pub const RKNN_TENSOR_UNDEFINED: Type = 3;
    pub const RKNN_TENSOR_FORMAT_MAX: Type = 4;
}
pub use self::_rknn_tensor_format::Type as rknn_tensor_format;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct _rknn_input_output_num {
    pub n_input: u32,
    pub n_output: u32,
}
pub type rknn_input_output_num = _rknn_input_output_num;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _rknn_tensor_attr {
    pub index: u32,
    pub n_dims: u32,
    pub dims: [u32; 16usize],
    pub name: [::std::os::raw::c_char; 256usize],
    pub n_elems: u32,
    pub size: u32,
    pub fmt: rknn_tensor_format,
    pub type_: rknn_tensor_type,
    pub qnt_type: rknn_tensor_qnt_type,
    pub fl: i8,
    pub zp: i32,
    pub scale: f32,
    pub w_stride: u32,
    pub size_with_stride: u32,
    pub pass_through: u8,
    pub h_stride: u32,
}
impl Default for _rknn_tensor_attr {
    fn default() -> Self {
        let mut s = ::std::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::std::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
pub type rknn_tensor_attr = _rknn_tensor_attr;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _rknn_sdk_version {
    pub api_version: [::std::os::raw::c_char; 256usize],
    pub drv_version: [::std::os::raw::c_char; 256usize],
}
impl Default for _rknn_sdk_version {
    fn default() -> Self {
        let mut s = ::std::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::std::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
pub type rknn_sdk_version = _rknn_sdk_version;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _rknn_input {
    pub index: u32,
    pub buf: *mut ::std::os::raw::c_void,
    pub size: u32,
    pub pass_through: u8,
    pub type_: rknn_tensor_type,
    pub fmt: rknn_tensor_format,
}
impl Default for _rknn_input {
    fn default() -> Self {
        let mut s = ::std::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::std::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
pub type rknn_input = _rknn_input;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _rknn_output {
    pub want_float: u8,
    pub is_prealloc: u8,
    pub index: u32,
    pub buf: *mut ::std::os::raw::c_void,
    pub size: u32,
}
impl Default for _rknn_output {
    fn default() -> Self {
        let mut s = ::std::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::std::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
pub type rknn_output = _rknn_output;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _rknn_init_extend {
    pub ctx: rknn_context,
    pub real_model_offset: i32,
    pub real_model_size: u32,
    pub model_buffer_fd: i32,
    pub model_buffer_flags: u32,
    pub reserved: [u8; 112usize],
}
impl Default for _rknn_init_extend {
    fn default() -> Self {
        let mut s = ::std::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::std::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
pub type rknn_init_extend = _rknn_init_extend;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct _rknn_run_extend {
    pub frame_id: u64,
    pub non_block: i32,
    pub timeout_ms: i32,
    pub fence_fd: i32,
}
pub type rknn_run_extend = _rknn_run_extend;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct _rknn_output_extend {
    pub frame_id: u64,
}
pub type rknn_output_extend = _rknn_output_extend;
unsafe extern "C" {
    pub fn rknn_init(
        context: *mut rknn_context,
        model: *mut ::std::os::raw::c_void,
        size: u32,
        flag: u32,
        extend: *mut rknn_init_extend,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn rknn_destroy(context: rknn_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn rknn_query(
        context: rknn_context,
        cmd: rknn_query_cmd,
        info: *mut ::std::os::raw::c_void,
        size: u32,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn rknn_inputs_set(
        context: rknn_context,
        n_inputs: u32,
        inputs: *mut rknn_input,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn rknn_run(context: rknn_context, extend: *mut rknn_run_extend) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn rknn_outputs_get(
        context: rknn_context,
        n_outputs: u32,
        outputs: *mut rknn_output,
        extend: *mut rknn_output_extend,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn rknn_outputs_release(
        context: rknn_context,
        n_ouputs: u32,
        outputs: *mut rknn_output,
    ) -> ::std::os::raw::c_int;
}
