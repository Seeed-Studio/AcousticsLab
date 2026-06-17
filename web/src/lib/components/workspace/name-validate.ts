// Client-side mirror of the daemon's workspace-name rules (backend stays source of truth, keep
// constants in sync); i18n error strings let the inline alert re-render on locale switch.

import { m } from '$lib/i18n';

const MAX_BYTES = 128;
const ENCODER = new TextEncoder();

export function validateWorkspaceName(name: string): string | null {
  const t = m.validation.name;
  if (name.length === 0) return t.empty;
  // UTF-8 byte count, not String.length (UTF-16 units).
  if (ENCODER.encode(name).length > MAX_BYTES) {
    return t.max_bytes(MAX_BYTES);
  }
  if (name.includes('\0') || name.includes('/') || name.includes('\\')) {
    return t.slashes_or_nul;
  }
  // JS `\s` (not the daemon's Unicode White_Space) - diverges at the margins (e.g. U+FEFF rejected here but not by the backend); backend stays source of truth.
  if (/^\s/.test(name) || /\s$/.test(name)) {
    return t.starts_or_ends_whitespace;
  }
  // Reject C0/C1 control chars (Unicode Cc).
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x1f\x7f-\x9f]/.test(name)) {
    return t.control_chars;
  }
  return null;
}
