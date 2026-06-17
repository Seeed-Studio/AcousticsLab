// Built on `crypto.getRandomValues()` rather than `crypto.randomUUID()`: the latter is
// secure-context-only (undefined/throws over plain `http://<LAN-IP>`), the former is the
// one Crypto member spec'd to work in an insecure context. Targets need only uniqueness,
// not RFC-4122 structure, so flat lowercase-hex suffices.
// Returns `2 * bytes` hex chars.
export function randomHex(bytes: number): string {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  let out = '';
  for (const b of buf) out += b.toString(16).padStart(2, '0');
  return out;
}
