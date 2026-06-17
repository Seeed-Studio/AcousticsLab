// Browser-native `.alpkg` ZIP packer (LFH + central directory + EOCD); Zip64 omitted
// since low-MiB payloads never reach 4 GiB and `> ZIP32_MAX` guards fail closed.
// `deflate-raw` emits the raw RFC-1951 stream ZIP method 8 needs; bare `'deflate'`
// prepends a zlib (RFC 1950) wrapper, yielding an unreadable archive.

export interface AlpkgEntry {
  /// Slash-separated, no leading slash, ASCII-clean: the UTF-8-filename ZIP flag is never set.
  path: string;
  bytes: Uint8Array;
}

export interface AlpkgManifest {
  /// Lets the importer reject a non-alpkg ZIP before reading entries.
  format: 'alpkg';
  /// Bumped only on a wire-incompatible change.
  version: 1;
  /// RFC3339 export wall-clock; distinct from the head's training `created_at`.
  exported_at: string;
}

// IEEE 802.3 polynomial, reflected. Lazy so central-directory-only importers pay nothing on load.
let CRC32_TABLE: Uint32Array | null = null;
function ensureCrcTable(): Uint32Array {
  if (CRC32_TABLE !== null) return CRC32_TABLE;
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) === 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    t[i] = c >>> 0;
  }
  CRC32_TABLE = t;
  return t;
}

/// CRC-32 (unsigned) over the uncompressed payload, as each ZIP entry records it regardless of method.
export function crc32(bytes: Uint8Array): number {
  const table = ensureCrcTable();
  let c = 0xffffffff;
  for (const b of bytes) {
    c = (table[(c ^ b) & 0xff] ^ (c >>> 8)) >>> 0;
  }
  return (c ^ 0xffffffff) >>> 0;
}

/// Raw DEFLATE bytes for ZIP method 8; narrow `Uint8Array<ArrayBuffer>` return satisfies
/// TS 5.7 `BlobPart` (admits only `ArrayBufferView<ArrayBuffer>`, not wide `ArrayBufferLike`).
async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array<ArrayBuffer>> {
  // Call sites always pass owned-buffer typed arrays, so the cast only narrows.
  const input = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream();
  const compressed = input.pipeThrough(new CompressionStream('deflate-raw'));
  const buf = await new Response(compressed).arrayBuffer();
  return new Uint8Array(buf);
}

// Magic numbers shared with the unpacker.
export const SIG_LFH = 0x04034b50;
export const SIG_CD = 0x02014b50;
export const SIG_EOCD = 0x06054b50;
// Packer always emits DEFLATE; STORE exists so the unpacker accepts other producers' archives.
export const METHOD_STORE = 0;
export const METHOD_DEFLATE = 8;
// Fixed at the DOS epoch so re-exports of the same head are byte-deterministic.
const DOS_TIME = 0x0000;
const DOS_DATE = 0x0021;
const ZIP32_MAX = 0xfffffffe;
const VERSION_NEEDED = 20; // 2.0 (DEFLATE)

function writeU16LE(dst: DataView, off: number, v: number): void {
  dst.setUint16(off, v, true);
}

function writeU32LE(dst: DataView, off: number, v: number): void {
  dst.setUint32(off, v >>> 0, true);
}

/// Local file header preceding each entry's payload: 30-byte fixed prefix + filename, no extras.
function buildLocalHeader(
  nameBytes: Uint8Array,
  crc: number,
  compressedSize: number,
  uncompressedSize: number
): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(30 + nameBytes.length);
  const dv = new DataView(out.buffer);
  writeU32LE(dv, 0, SIG_LFH);
  writeU16LE(dv, 4, VERSION_NEEDED);
  writeU16LE(dv, 6, 0); // general purpose bit flag
  writeU16LE(dv, 8, METHOD_DEFLATE);
  writeU16LE(dv, 10, DOS_TIME);
  writeU16LE(dv, 12, DOS_DATE);
  writeU32LE(dv, 14, crc);
  writeU32LE(dv, 18, compressedSize);
  writeU32LE(dv, 22, uncompressedSize);
  writeU16LE(dv, 26, nameBytes.length);
  writeU16LE(dv, 28, 0); // extra field length
  out.set(nameBytes, 30);
  return out;
}

/// Central-directory entry mirroring the local header plus the LFH offset: 46-byte prefix + filename.
function buildCentralEntry(
  nameBytes: Uint8Array,
  crc: number,
  compressedSize: number,
  uncompressedSize: number,
  localOffset: number
): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(46 + nameBytes.length);
  const dv = new DataView(out.buffer);
  writeU32LE(dv, 0, SIG_CD);
  writeU16LE(dv, 4, (3 << 8) | VERSION_NEEDED); // version made by: unix (3) + 2.0
  writeU16LE(dv, 6, VERSION_NEEDED);
  writeU16LE(dv, 8, 0); // general purpose bit flag
  writeU16LE(dv, 10, METHOD_DEFLATE);
  writeU16LE(dv, 12, DOS_TIME);
  writeU16LE(dv, 14, DOS_DATE);
  writeU32LE(dv, 16, crc);
  writeU32LE(dv, 20, compressedSize);
  writeU32LE(dv, 24, uncompressedSize);
  writeU16LE(dv, 28, nameBytes.length);
  writeU16LE(dv, 30, 0); // extra field length
  writeU16LE(dv, 32, 0); // file comment length
  writeU16LE(dv, 34, 0); // disk number start
  writeU16LE(dv, 36, 0); // internal file attributes
  writeU32LE(dv, 38, 0); // external file attributes
  writeU32LE(dv, 42, localOffset);
  out.set(nameBytes, 46);
  return out;
}

function buildEocd(entryCount: number, cdSize: number, cdOffset: number): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(22);
  const dv = new DataView(out.buffer);
  writeU32LE(dv, 0, SIG_EOCD);
  writeU16LE(dv, 4, 0); // disk number
  writeU16LE(dv, 6, 0); // CD start disk
  writeU16LE(dv, 8, entryCount); // CD entries this disk
  writeU16LE(dv, 10, entryCount); // CD entries total
  writeU32LE(dv, 12, cdSize);
  writeU32LE(dv, 16, cdOffset);
  writeU16LE(dv, 20, 0); // zip comment length
  return out;
}

/// Pack entries into a `.alpkg` blob (ZIP + DEFLATE) in the supplied order; callers put
/// `package.json` first so a streaming reader detects the kind before seeking the central
/// directory. Throws `RangeError` if any entry or the total exceeds the ZIP32 4 GiB ceiling.
export async function packAlpkg(entries: readonly AlpkgEntry[]): Promise<Blob> {
  if (entries.length === 0) {
    throw new Error('alpkg: must supply at least one entry');
  }

  const encoder = new TextEncoder();
  // Narrow buffer kind throughout so the final `Blob` accepts each push uncast.
  const parts: Uint8Array<ArrayBuffer>[] = [];
  const cdEntries: Uint8Array<ArrayBuffer>[] = [];
  let offset = 0;
  let entryCount = 0;

  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.path);
    if (nameBytes.length > 0xffff) {
      throw new RangeError(`alpkg: filename too long (${nameBytes.length} bytes): ${entry.path}`);
    }
    const uncompressedSize = entry.bytes.length;
    if (uncompressedSize > ZIP32_MAX) {
      throw new RangeError(
        `alpkg: entry ${entry.path} exceeds ZIP32 size cap (${uncompressedSize} bytes)`
      );
    }
    const crc = crc32(entry.bytes);
    const compressed = await deflateRaw(entry.bytes);
    if (compressed.length > ZIP32_MAX) {
      throw new RangeError(
        `alpkg: entry ${entry.path} compressed size exceeds ZIP32 cap (${compressed.length} bytes)`
      );
    }
    const lfh = buildLocalHeader(nameBytes, crc, compressed.length, uncompressedSize);
    parts.push(lfh, compressed);
    cdEntries.push(buildCentralEntry(nameBytes, crc, compressed.length, uncompressedSize, offset));
    offset += lfh.length + compressed.length;
    if (offset > ZIP32_MAX) {
      throw new RangeError(`alpkg: archive size exceeds ZIP32 cap at entry ${entry.path}`);
    }
    entryCount++;
  }

  const cdStart = offset;
  let cdSize = 0;
  for (const cd of cdEntries) cdSize += cd.length;
  if (cdStart + cdSize > ZIP32_MAX) {
    throw new RangeError('alpkg: central directory offset exceeds ZIP32 cap');
  }
  parts.push(...cdEntries, buildEocd(entryCount, cdSize, cdStart));

  // `application/zip` is the closest standard MIME; `<a download>` supplies `.alpkg`.
  return new Blob(parts, { type: 'application/zip' });
}

/// `now` is injectable so tests can pin it for byte-stable archives.
export function buildAlpkgManifest(now?: string): AlpkgManifest {
  return {
    format: 'alpkg',
    version: 1,
    exported_at: now ?? new Date().toISOString()
  };
}

/// Slug over the daemon's `[A-Za-z0-9._-]` allowlist: disallowed runs collapse to one
/// underscore, leading/trailing underscores strip, empty falls back to `fallback`.
export function safeFilenameSlug(s: string, fallback: string): string {
  const cleaned = s.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');
  return cleaned.length > 0 ? cleaned : fallback;
}
