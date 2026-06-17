/// <reference types="vite/client" />

// Vite inlines `import.meta.env.VITE_*` at build time.
interface ImportMetaEnv {
  // Absolute backend origin for HTTP/SSE/WebSocket; empty/unset => same-origin.
  readonly VITE_API_BASE?: string;
  // Reverse-proxy mount prefix; empty/unset => root mount. Feeds both kit.paths.base and the same-origin API/WS prefix.
  readonly VITE_BASE_PATH?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
