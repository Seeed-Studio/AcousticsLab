// Single source of truth for the reverse-proxy mount prefix (`VITE_BASE_PATH`),
// shared by the Node-side ESM config files (svelte.config.js, vite.config.ts).
// The browser/worker bundle (src/lib/api/base.ts) keeps a byte-identical mirror
// of normalizeBasePath because it cannot cleanly import a root-level `.js`; keep
// the two in sync. Rules match SvelteKit's kit.paths.base validation: empty, or
// starts with `/` and does NOT end with `/`.

/**
 * Canonicalize a raw `VITE_BASE_PATH` into a form `kit.paths.base` accepts.
 *
 * Empty / unset / `'/'` -> `''` (root mount, the default). Otherwise a single
 * leading slash is ensured and every trailing slash stripped, e.g.
 * `extension/acousticslab/` -> `/extension/acousticslab`.
 *
 * @param {string | undefined | null} value
 * @returns {string}
 */
export function normalizeBasePath(value) {
  const raw = (value ?? '').trim();
  if (raw === '' || raw === '/') return '';
  // Collapse leading slashes to exactly one, then strip trailing slashes.
  return ('/' + raw.replace(/^\/+/, '')).replace(/\/+$/, '');
}
