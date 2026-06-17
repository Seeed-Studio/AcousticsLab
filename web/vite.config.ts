import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, loadEnv } from 'vite';
import { normalizeBasePath } from './base-path.js';

// Dev-server proxy target -- the `host:port` that `vite dev` forwards
// `/api` + `/stream/*` to.  Configurable so the same-origin dev flow
// (relative paths, no CORS needed) can point at a daemon on another
// host/port without setting `VITE_API_BASE`.  Defaults to the
// daemon's launch.toml default bind.
//
// This is distinct from `VITE_API_BASE` (see `src/lib/api/base.ts`):
// set `VITE_DEV_PROXY_TARGET` to keep the SPA same-origin with the
// dev server while proxying to a custom daemon; set `VITE_API_BASE`
// to make the built/served SPA talk cross-origin straight to the
// daemon (which then needs a CORS allowlist).  `VITE_DEV_PROXY_TARGET`
// is read only here, in the dev server, never inlined into the bundle.
export default defineConfig(({ mode }) => {
  // `''` prefix loads every key (including the non-`VITE_`-exposed
  // proxy target) from `.env*` files and the process environment.
  const env = loadEnv(mode, process.cwd(), '');
  const daemon = env.VITE_DEV_PROXY_TARGET || '127.0.0.1:8787';

  // Mount prefix (see `base-path.js` / `svelte.config.js`).  When set,
  // the dev server serves the SPA under it and `base.ts` emits
  // `${base}/api` + `${base}/stream/*` requests, so the proxy must
  // match on the prefixed paths and strip the prefix before forwarding
  // -- exactly what the production Nginx gateway does (the daemon
  // serves `/api` + `/stream` at its own root).  Empty base => the
  // keys collapse to `/api` + `/stream/*` and `stripBase` is identity,
  // so dev behaves exactly as before.
  const base = normalizeBasePath(env.VITE_BASE_PATH);
  const stripBase = (path: string): string =>
    base && path.startsWith(base) ? path.slice(base.length) : path;

  return {
    plugins: [tailwindcss(), sveltekit()],
    server: {
      proxy: {
        [`${base}/api`]: { target: `http://${daemon}`, changeOrigin: false, rewrite: stripBase },
        [`${base}/stream/audio`]: {
          target: `ws://${daemon}`,
          ws: true,
          changeOrigin: false,
          rewrite: stripBase
        },
        [`${base}/stream/infer`]: {
          target: `ws://${daemon}`,
          ws: true,
          changeOrigin: false,
          rewrite: stripBase
        }
      }
    },
    worker: {
      format: 'es'
    }
  };
});
