// Build-time daemon origin (VITE_API_BASE set => cross-origin, daemon must CORS-allow the SPA
// origin for HTTP+SSE; unset => same-origin). Import-safe in main thread, worker, and SSR: only
// `import.meta.env` literals + lazy `self.location` (never `window`/`document`) and the
// VITE_BASE_PATH literal, not SvelteKit's runtime `base` (absent in the bundled worker).

const RAW = (import.meta.env.VITE_API_BASE ?? '').trim();

// Require an http(s) scheme so a typo warns and falls back to same-origin, not malformed URLs.
const IS_ABSOLUTE = /^https?:\/\//i.test(RAW);
if (RAW && !IS_ABSOLUTE) {
  console.warn(
    `VITE_API_BASE must be an absolute http(s) origin (e.g. http://host:8787); ` +
      `got ${JSON.stringify(RAW)} -- falling back to same-origin.`
  );
}

// Trailing slashes stripped so `API_BASE + '/api/...'` can't double-slash; `''` = same-origin.
export const API_BASE = IS_ABSOLUTE ? RAW.replace(/\/+$/, '') : '';

// Must stay byte-identical to `web/base-path.js` (duplicated, not imported, for the worker bundle).
function normalizeBasePath(value: string | undefined): string {
  const raw = (value ?? '').trim();
  if (raw === '' || raw === '/') return '';
  return ('/' + raw.replace(/^\/+/, '')).replace(/\/+$/, '');
}
// Same-origin reverse-proxy mount prefix (`''` = root); gateway strips it before forwarding so
// the daemon serves at root. Omitted cross-origin (daemon reached directly).
export const BASE_PATH = normalizeBasePath(import.meta.env.VITE_BASE_PATH);

// Anchored on the scheme's colon so `https` -> `wss` (not `wsS`) even for mixed-case input.
function toWsOrigin(origin: string): string {
  return origin.replace(/^https:/i, 'wss:').replace(/^http:/i, 'ws:');
}

export function apiUrl(path: string): string {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(path)) return path;
  if (API_BASE) return API_BASE + path;
  return BASE_PATH + path;
}

// `typeof self` guard keeps imports SSR/prerender-safe.
export function wsUrl(path: string): string {
  if (API_BASE) return toWsOrigin(API_BASE) + path;
  const origin = typeof self !== 'undefined' ? self.location.origin : '';
  return toWsOrigin(origin) + BASE_PATH + path;
}
