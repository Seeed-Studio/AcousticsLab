//! Runtime-loaded FFI dispatch table: 7 RKNN symbols resolved eagerly so a
//! missing symbol fails at `load`, not mid-inference.

use crate::rknn_runtime::sys::{
    rknn_context, rknn_init_extend, rknn_input, rknn_output, rknn_output_extend, rknn_query_cmd,
    rknn_run_extend,
};
use libloading::Library;
use std::ffi::c_int;
use std::os::raw::c_void;
use std::path::Path;
use std::sync::Arc;

type FnInit =
    unsafe extern "C" fn(*mut rknn_context, *mut c_void, u32, u32, *mut rknn_init_extend) -> c_int;
type FnDestroy = unsafe extern "C" fn(rknn_context) -> c_int;
type FnQuery = unsafe extern "C" fn(rknn_context, rknn_query_cmd, *mut c_void, u32) -> c_int;
type FnInputsSet = unsafe extern "C" fn(rknn_context, u32, *mut rknn_input) -> c_int;
type FnRun = unsafe extern "C" fn(rknn_context, *mut rknn_run_extend) -> c_int;
type FnOutputsGet =
    unsafe extern "C" fn(rknn_context, u32, *mut rknn_output, *mut rknn_output_extend) -> c_int;
type FnOutputsRelease = unsafe extern "C" fn(rknn_context, u32, *mut rknn_output) -> c_int;

pub(crate) struct SymbolTable {
    // Must outlive every resolved fn pointer: dropping it unmaps the code pages
    // they point at. `Arc` lets a future pool share one `dlopen`.
    _lib: Arc<Library>,
    pub init: FnInit,
    pub destroy: FnDestroy,
    pub query: FnQuery,
    pub inputs_set: FnInputsSet,
    pub run: FnRun,
    pub outputs_get: FnOutputsGet,
    pub outputs_release: FnOutputsRelease,
}

impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolTable")
            .field("loaded", &true)
            .field("symbols", &7u32)
            .finish()
    }
}

impl SymbolTable {
    /// Load the library at `path` and eagerly resolve all 7 RKNN symbols.
    ///
    /// Safety: the loaded library is trusted to export the RKNN C ABI; signatures
    /// are not verified at runtime, so a mismatched ABI would be UB.
    pub(crate) fn load(path: &Path) -> Result<Self, LoadError> {
        // SAFETY: loading any .so runs its arbitrary init-code side effects; we
        // trust the vendored Rockchip library.
        let lib = unsafe { Library::new(path) }.map_err(|source| LoadError::Library {
            path: path.to_path_buf(),
            source,
        })?;

        // Copied fn pointer is valid only while `_lib` is held by the table.
        // `name_bytes` must end in \0 so libloading takes the no-alloc symbol path
        // (otherwise it copies and NUL-terminates); `name_str` is the human-readable
        // form for `LoadError::Symbol`.
        unsafe fn resolve<T: Copy>(
            lib: &Library,
            name_bytes: &[u8],
            name_str: &'static str,
        ) -> Result<T, LoadError> {
            let sym: libloading::Symbol<'_, T> =
                unsafe { lib.get(name_bytes) }.map_err(|source| LoadError::Symbol {
                    name: name_str,
                    source,
                })?;
            Ok(*sym)
        }

        let init: FnInit = unsafe { resolve(&lib, b"rknn_init\0", "rknn_init") }?;
        let destroy: FnDestroy = unsafe { resolve(&lib, b"rknn_destroy\0", "rknn_destroy") }?;
        let query: FnQuery = unsafe { resolve(&lib, b"rknn_query\0", "rknn_query") }?;
        let inputs_set: FnInputsSet =
            unsafe { resolve(&lib, b"rknn_inputs_set\0", "rknn_inputs_set") }?;
        let run: FnRun = unsafe { resolve(&lib, b"rknn_run\0", "rknn_run") }?;
        let outputs_get: FnOutputsGet =
            unsafe { resolve(&lib, b"rknn_outputs_get\0", "rknn_outputs_get") }?;
        let outputs_release: FnOutputsRelease =
            unsafe { resolve(&lib, b"rknn_outputs_release\0", "rknn_outputs_release") }?;

        Ok(SymbolTable {
            _lib: Arc::new(lib),
            init,
            destroy,
            query,
            inputs_set,
            run,
            outputs_get,
            outputs_release,
        })
    }
}

#[derive(Debug)]
pub(crate) enum LoadError {
    Library {
        path: std::path::PathBuf,
        source: libloading::Error,
    },
    Symbol {
        name: &'static str,
        source: libloading::Error,
    },
}
