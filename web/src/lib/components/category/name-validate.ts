// Client mirror of the daemon's single-component AssetPath rule (backend stays source of truth):
// no leading `.` (bars `.`/`..`/`.hidden`), no leading `-` (else allowlisted `-` reads as a flag in
// unquoted shell-out), no leading `_` (frontend-only; reserves `_background_noise_`/`_unknown_` for
// Speech-Commands synthetic classes); 255-byte cap is NAME_MAX, not AssetPath's path-total/depth caps.

import { m } from '$lib/i18n';

const ALLOWED_RE = /^[A-Za-z0-9._-]+$/;
const MAX_BYTES = 255;
const ENCODER = new TextEncoder();

export function validateCategoryName(name: string): string | null {
  const t = m.validation.name;
  if (name.length === 0) return t.category_empty;
  if (name.startsWith('.')) {
    return t.starts_with_dot;
  }
  if (name.startsWith('_')) {
    return t.starts_with_underscore;
  }
  if (name.startsWith('-')) {
    return t.starts_with_hyphen;
  }
  if (!ALLOWED_RE.test(name)) {
    return t.bad_chars;
  }
  // UTF-8 byte count to match daemon `name.len()` (defence in depth; allowlisted ASCII is 1 byte/char).
  if (ENCODER.encode(name).length > MAX_BYTES) {
    return t.category_max_bytes(MAX_BYTES);
  }
  return null;
}

// Reject case-insensitive collisions: AssetPath is case-sensitive but HFS+/NTFS collapse `Cat`/`cat`.
export function findCaseInsensitiveDuplicate(
  candidate: string,
  existing: Iterable<string>
): string | null {
  const lower = candidate.toLowerCase();
  for (const name of existing) {
    if (name === candidate) return name; // caller decides re-add policy
    if (name.toLowerCase() === lower) return name;
  }
  return null;
}
