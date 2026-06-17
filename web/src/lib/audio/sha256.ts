// Canonical content-addressed slice identity (filename, IDB key, cache keys). MUST hash the
// exact bytes the daemon hashes - the WAV envelope (44-byte header + Int16 LE PCM), not raw
// PCM - so the daemon PUT-receipt `sha256` round-trips; a mismatch is upload corruption.
export async function sha256Hex(data: ArrayBuffer | ArrayBufferView): Promise<string> {
  // Cast is sound: no call site passes a SAB-backed buffer (the only BufferSource exclusion
  // `digest` accepts at runtime) and it avoids the otherwise-forced copy into a fresh buffer.
  const digest = await crypto.subtle.digest('SHA-256', data as BufferSource);
  return toHex(new Uint8Array(digest));
}

// Lowercase to match the daemon receipt `sha256`.
function toHex(bytes: Uint8Array): string {
  let out = '';
  for (const b of bytes) {
    out += (b < 16 ? '0' : '') + b.toString(16);
  }
  return out;
}
