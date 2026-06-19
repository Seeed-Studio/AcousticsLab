// Browser-native `.alpkg` unpacker: ZIP32 only (Zip64 fails closed, never truncated to low 32 bits),
// STORE/DEFLATE only, tolerates a non-zero EOCD comment. Reads via `Blob.slice` so peak memory stays
// the trailing EOCD scan + per-entry slices, not the whole archive.

import {
  crc32,
  METHOD_DEFLATE,
  METHOD_STORE,
  SIG_CD,
  SIG_LFH,
  type AlpkgEntry,
  type AlpkgManifest
} from './alpkg';

export type { AlpkgEntry, AlpkgManifest };

/// Subset of the daemon's `WorkspaceCore` (`workspace.json`) the import flow reads; the rest survive only in raw bytes.
export interface WorkspaceCoreShape {
  id: string;
  name: string;
  tags?: string[];
  created_at?: string;
  workspace_revision?: { id: number; at?: string };
  head_count?: number;
}

export interface AlpkgUnpackResult {
  manifest: AlpkgManifest;
  /// `null` for legacy archives that predate `workspace.json`.
  workspaceCore: WorkspaceCoreShape | null;
  /// Raw bytes so the verbatim metadata strip renders without round-tripping the typed shape.
  workspaceCoreBytes: Uint8Array | null;
  /// Excludes `package.json`/`workspace.json`, in central-directory order; re-sort by `path` for determinism.
  entries: AlpkgEntry[];
}

/// `kind` lets the dialog pick operator copy without parsing the message string.
export type AlpkgUnpackErrorKind =
  | 'not-a-zip'
  | 'eocd-not-found'
  | 'zip64-not-supported'
  | 'malformed-record'
  | 'unsupported-compression'
  | 'crc-mismatch'
  | 'missing-package-json'
  | 'package-json-malformed'
  | 'wrong-format'
  | 'unsupported-version'
  | 'too-many-entries'
  | 'entry-too-large'
  | 'archive-too-large';

// Zip-bomb bounds on declared CD sizes, checked pre-decompress: 1024 entries is ~10x the realistic
// ALPKG ceiling (<100); 256 MiB/entry clears any single artefact (TFJS shard ~50 MiB, head MPK <10 MiB);
// 512 MiB total stays generous for a multi-category export. An under-declaring archive (claims 1 byte,
// inflates to gigabytes) escapes these, so `inflateRaw` ALSO bounds the actual inflated size.
export const MAX_ZIP_ENTRY_COUNT = 1024;
export const MAX_ZIP_ENTRY_BYTES = 256 * 1024 * 1024;
export const MAX_ZIP_TOTAL_BYTES = 512 * 1024 * 1024;

function enforceZipCaps(entries: readonly CentralEntry[]): void {
  if (entries.length > MAX_ZIP_ENTRY_COUNT) {
    throw new AlpkgUnpackError(
      'too-many-entries',
      `Archive declares ${String(entries.length)} entries; the importer caps at ${String(MAX_ZIP_ENTRY_COUNT)}.`
    );
  }
  let total = 0;
  for (const e of entries) {
    if (e.uncompressedSize > MAX_ZIP_ENTRY_BYTES) {
      throw new AlpkgUnpackError(
        'entry-too-large',
        `Archive entry "${e.path}" declares ${formatBytesMiB(e.uncompressedSize)} uncompressed; the per-entry cap is ${formatBytesMiB(MAX_ZIP_ENTRY_BYTES)}.`
      );
    }
    total += e.uncompressedSize;
    if (total > MAX_ZIP_TOTAL_BYTES) {
      throw new AlpkgUnpackError(
        'archive-too-large',
        `Archive declares ${formatBytesMiB(total)} of uncompressed content; the cap is ${formatBytesMiB(MAX_ZIP_TOTAL_BYTES)}.`
      );
    }
  }
}

function formatBytesMiB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MiB`;
}

export class AlpkgUnpackError extends Error {
  readonly kind: AlpkgUnpackErrorKind;
  constructor(kind: AlpkgUnpackErrorKind, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'AlpkgUnpackError';
    this.kind = kind;
  }
}

// EOCD = 22-byte record + comment up to 65535; scan this trailing window backward (over-read past the archive start is harmless, the slice clamps).
const EOCD_SCAN_BYTES = 65_557; // 0xFFFF + 22

interface EocdView {
  entryCount: number;
  cdOffset: number;
  cdSize: number;
}

async function findEocd(blob: Blob): Promise<EocdView> {
  const totalSize = blob.size;
  if (totalSize < 22) {
    throw new AlpkgUnpackError(
      'not-a-zip',
      `File is too short to be a ZIP archive (${String(totalSize)} bytes).`
    );
  }
  const scanStart = Math.max(0, totalSize - EOCD_SCAN_BYTES);
  const tailBuf = await blob.slice(scanStart, totalSize).arrayBuffer();
  const tail = new Uint8Array(tailBuf);
  for (let i = tail.length - 22; i >= 0; i--) {
    if (tail[i] === 0x50 && tail[i + 1] === 0x4b && tail[i + 2] === 0x05 && tail[i + 3] === 0x06) {
      const dv = new DataView(tail.buffer, tail.byteOffset + i, 22);
      const entryCount = dv.getUint16(10, true);
      const cdSize = dv.getUint32(12, true);
      const cdOffset = dv.getUint32(16, true);
      const commentLen = dv.getUint16(20, true);
      // A comment can embed the EOCD signature, so the backward scan may hit a false match first.
      // The structural plausibility check (EOCD at cdOffset+cdSize, commentLen == trailing bytes)
      // MUST gate the Zip64 throw below, else a false signature with 0xffff slots aborts before the
      // real earlier EOCD.
      const eocdAbsPos = scanStart + i;
      const trailingBytes = totalSize - (eocdAbsPos + 22);
      if (cdOffset + cdSize !== eocdAbsPos) continue;
      if (commentLen !== trailingBytes) continue;
      // Zip64 sentinels: real figures live in an unparsed Zip64 EOCD; fail closed rather than truncate a > 4 GiB archive to its low 32 bits.
      if (entryCount === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
        throw new AlpkgUnpackError(
          'zip64-not-supported',
          'Archive uses Zip64 extensions; not supported by this importer.'
        );
      }
      return { entryCount, cdOffset, cdSize };
    }
  }
  throw new AlpkgUnpackError(
    'eocd-not-found',
    'Not a ZIP archive (End-of-Central-Directory record not found).'
  );
}

interface CentralEntry {
  path: string;
  method: number;
  crc32: number;
  compressedSize: number;
  uncompressedSize: number;
  localOffset: number;
}

async function readCentralDirectory(blob: Blob, eocd: EocdView): Promise<CentralEntry[]> {
  const cdBuf = await blob.slice(eocd.cdOffset, eocd.cdOffset + eocd.cdSize).arrayBuffer();
  const cd = new Uint8Array(cdBuf);
  const dv = new DataView(cd.buffer, cd.byteOffset, cd.byteLength);
  const decoder = new TextDecoder('utf-8', { fatal: false });
  const out: CentralEntry[] = [];
  let off = 0;
  for (let i = 0; i < eocd.entryCount; i++) {
    if (off + 46 > cd.length) {
      throw new AlpkgUnpackError(
        'malformed-record',
        `Central-directory entry ${String(i)} is truncated.`
      );
    }
    const sig = dv.getUint32(off, true);
    if (sig !== SIG_CD) {
      throw new AlpkgUnpackError(
        'malformed-record',
        `Central-directory entry ${String(i)} has wrong signature.`
      );
    }
    const method = dv.getUint16(off + 10, true);
    const crc = dv.getUint32(off + 16, true);
    const compressedSize = dv.getUint32(off + 20, true);
    const uncompressedSize = dv.getUint32(off + 24, true);
    const nameLen = dv.getUint16(off + 28, true);
    const extraLen = dv.getUint16(off + 30, true);
    const commentLen = dv.getUint16(off + 32, true);
    const localOffset = dv.getUint32(off + 42, true);
    if (
      compressedSize === 0xffffffff ||
      uncompressedSize === 0xffffffff ||
      localOffset === 0xffffffff
    ) {
      throw new AlpkgUnpackError(
        'zip64-not-supported',
        `Central-directory entry ${String(i)} uses Zip64 sentinels; not supported.`
      );
    }
    const nameStart = off + 46;
    if (nameStart + nameLen > cd.length) {
      throw new AlpkgUnpackError(
        'malformed-record',
        `Central-directory entry ${String(i)} filename overflows the record.`
      );
    }
    const path = decoder.decode(cd.subarray(nameStart, nameStart + nameLen));
    out.push({
      path,
      method,
      crc32: crc,
      compressedSize,
      uncompressedSize,
      localOffset
    });
    off = nameStart + nameLen + extraLen + commentLen;
  }
  return out;
}

async function readEntryBytes(blob: Blob, entry: CentralEntry): Promise<Uint8Array> {
  // Use the LFH's own name/extra lengths, not the CD's: the spec lets the two disagree.
  const hdrBuf = await blob.slice(entry.localOffset, entry.localOffset + 30).arrayBuffer();
  if (hdrBuf.byteLength < 30) {
    throw new AlpkgUnpackError(
      'malformed-record',
      `Local file header for "${entry.path}" is truncated.`
    );
  }
  const dv = new DataView(hdrBuf);
  const sig = dv.getUint32(0, true);
  if (sig !== SIG_LFH) {
    throw new AlpkgUnpackError(
      'malformed-record',
      `Local file header for "${entry.path}" has wrong signature.`
    );
  }
  const nameLen = dv.getUint16(26, true);
  const extraLen = dv.getUint16(28, true);
  const payloadOffset = entry.localOffset + 30 + nameLen + extraLen;
  const payloadEnd = payloadOffset + entry.compressedSize;
  const payloadBuf = await blob.slice(payloadOffset, payloadEnd).arrayBuffer();
  if (payloadBuf.byteLength !== entry.compressedSize) {
    throw new AlpkgUnpackError(
      'malformed-record',
      `Compressed payload for "${entry.path}" is truncated (${String(payloadBuf.byteLength)} of ${String(entry.compressedSize)} bytes).`
    );
  }
  const compressed = new Uint8Array(payloadBuf);
  let bytes: Uint8Array;
  if (entry.method === METHOD_STORE) {
    if (compressed.length !== entry.uncompressedSize) {
      throw new AlpkgUnpackError(
        'malformed-record',
        `STORE entry "${entry.path}" length mismatch (${String(compressed.length)} vs ${String(entry.uncompressedSize)}).`
      );
    }
    bytes = compressed;
  } else if (entry.method === METHOD_DEFLATE) {
    // `deflate-raw` consumes ZIP method 8's bare RFC-1951 stream (no zlib wrapper); needs Chrome 103+/FF 113+/Safari 16.4+.
    bytes = await inflateRaw(compressed, entry.uncompressedSize);
    if (bytes.length !== entry.uncompressedSize) {
      throw new AlpkgUnpackError(
        'malformed-record',
        `Decompressed entry "${entry.path}" has wrong size (${String(bytes.length)} vs ${String(entry.uncompressedSize)}).`
      );
    }
  } else {
    throw new AlpkgUnpackError(
      'unsupported-compression',
      `Entry "${entry.path}" uses unsupported ZIP method ${String(entry.method)}.`
    );
  }
  // CRC-32 over uncompressed bytes (ZIP spec) so corruption surfaces here, not as a wrong-sha head on the daemon.
  const observed = crc32(bytes);
  if (observed !== entry.crc32) {
    throw new AlpkgUnpackError(
      'crc-mismatch',
      `Entry "${entry.path}" failed CRC-32 (expected ${entry.crc32.toString(16)}, got ${observed.toString(16)}).`
    );
  }
  return bytes;
}

async function inflateRaw(compressed: Uint8Array, declaredSize: number): Promise<Uint8Array> {
  // Cast satisfies TS 5.7's narrower BlobPart shape; the buffer is always ArrayBuffer-backed.
  const input = new Blob([compressed as Uint8Array<ArrayBuffer>]).stream();
  const decompressed = input.pipeThrough(new DecompressionStream('deflate-raw'));
  // Bound the actual inflated size while streaming, since a bomb declaring `uncompressedSize = 1`
  // could expand to gigabytes: pre-allocate `cap` and copy chunks in (a concat-at-end holds ~2x
  // live), and cancel on the first overflowing chunk to stop the source inflating further.
  const cap = Math.min(declaredSize, MAX_ZIP_ENTRY_BYTES);
  const reader = decompressed.getReader();
  const out = new Uint8Array(cap);
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (total + value.byteLength > cap) {
        await reader.cancel().catch(() => undefined);
        throw new AlpkgUnpackError(
          'entry-too-large',
          `Decompressed content exceeds the per-entry cap of ${formatBytesMiB(cap)}; archive may be a zip bomb.`
        );
      }
      out.set(value, total);
      total += value.byteLength;
    }
  } finally {
    reader.releaseLock();
  }
  // Trim so a truncated entry fails the caller's length check rather than carrying zero padding.
  return total === out.length ? out : out.subarray(0, total);
}

function parsePackageJson(bytes: Uint8Array): AlpkgManifest {
  const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    throw new AlpkgUnpackError('package-json-malformed', 'package.json is not valid JSON.', {
      cause: e
    });
  }
  if (parsed === null || typeof parsed !== 'object') {
    throw new AlpkgUnpackError('package-json-malformed', 'package.json is not a JSON object.');
  }
  const obj = parsed as Record<string, unknown>;
  // Pre-narrow for interpolation: satisfies eslint `no-base-to-string`, renders a tampered `{}` as `""` not `[object Object]`.
  const formatLabel = typeof obj.format === 'string' ? obj.format : '';
  if (obj.format !== 'alpkg') {
    throw new AlpkgUnpackError(
      'wrong-format',
      `package.json declares format "${formatLabel}"; only "alpkg" is supported.`
    );
  }
  const versionLabel =
    typeof obj.version === 'number' || typeof obj.version === 'string' ? String(obj.version) : '';
  if (obj.version !== 1) {
    throw new AlpkgUnpackError(
      'unsupported-version',
      `Unsupported alpkg version ${versionLabel}; this importer reads version 1.`
    );
  }
  if (typeof obj.exported_at !== 'string') {
    throw new AlpkgUnpackError(
      'package-json-malformed',
      'package.json missing `exported_at` timestamp.'
    );
  }
  return { format: 'alpkg', version: 1, exported_at: obj.exported_at };
}

function parseWorkspaceCoreSafe(bytes: Uint8Array): WorkspaceCoreShape | null {
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    const parsed = JSON.parse(text) as unknown;
    if (parsed === null || typeof parsed !== 'object') return null;
    const obj = parsed as Record<string, unknown>;
    if (typeof obj.id !== 'string' || typeof obj.name !== 'string') return null;
    const core: WorkspaceCoreShape = { id: obj.id, name: obj.name };
    if (Array.isArray(obj.tags) && obj.tags.every((t): t is string => typeof t === 'string')) {
      core.tags = obj.tags;
    }
    if (typeof obj.created_at === 'string') core.created_at = obj.created_at;
    if (
      obj.workspace_revision !== null &&
      typeof obj.workspace_revision === 'object' &&
      typeof (obj.workspace_revision as Record<string, unknown>).id === 'number'
    ) {
      const rev = obj.workspace_revision as Record<string, unknown>;
      const revShape: { id: number; at?: string } = { id: rev.id as number };
      if (typeof rev.at === 'string') revShape.at = rev.at;
      core.workspace_revision = revShape;
    }
    if (typeof obj.head_count === 'number') core.head_count = obj.head_count;
    return core;
  } catch {
    // workspace.json is provenance only: a malformed copy returns null rather than blocking import.
    return null;
  }
}

/// Generic non-ALPKG ZIP entry (e.g. a TFJS bundle in a `.zip`); same CRC + CD discipline, no envelope assumptions.
export interface ExtractedZipEntry {
  /// Verbatim CD path (forward-slashes, no leading slash, may include subfolders); trailing-`/` dirs are filtered out.
  path: string;
  bytes: Uint8Array;
}

/// Read a ZIP's entries with no envelope assumption: `unpackAlpkg`'s machinery minus the `package.json`/workspace-core parses. Throws `AlpkgUnpackError`.
export async function extractZipEntries(blob: Blob): Promise<ExtractedZipEntry[]> {
  const eocd = await findEocd(blob);
  const central = await readCentralDirectory(blob, eocd);
  enforceZipCaps(central);
  const out: ExtractedZipEntry[] = [];
  for (const cd of central) {
    if (cd.path.endsWith('/')) continue;
    const bytes = await readEntryBytes(blob, cd);
    out.push({ path: cd.path, bytes });
  }
  return out;
}

/// Unpack a `.alpkg` into envelope + payload entries, CRC-verifying each and rejecting with a typed `AlpkgUnpackError` on the first defect.
export async function unpackAlpkg(blob: Blob): Promise<AlpkgUnpackResult> {
  const eocd = await findEocd(blob);
  const central = await readCentralDirectory(blob, eocd);
  // Caps before envelope rejects a valid-package.json/10-GiB-payload pairing pre-decode.
  enforceZipCaps(central);
  // Envelope before decompressing any payload so a non-alpkg ZIP fails fast.
  const pkgEntry = central.find((e) => e.path === 'package.json');
  if (!pkgEntry) {
    throw new AlpkgUnpackError(
      'missing-package-json',
      'Archive is not an .alpkg (missing package.json envelope).'
    );
  }
  const pkgBytes = await readEntryBytes(blob, pkgEntry);
  const manifest = parsePackageJson(pkgBytes);
  // workspace.json is optional (legacy archives predate it); best-effort so a name-less context strip still renders.
  let workspaceCore: WorkspaceCoreShape | null = null;
  let workspaceCoreBytes: Uint8Array | null = null;
  const wsEntry = central.find((e) => e.path === 'workspace.json');
  if (wsEntry) {
    workspaceCoreBytes = await readEntryBytes(blob, wsEntry);
    workspaceCore = parseWorkspaceCoreSafe(workspaceCoreBytes);
  }
  // Sequential read+verify: archives are small and deflate is CPU-bound, so concurrency would only complicate the abort path.
  const entries: AlpkgEntry[] = [];
  for (const cd of central) {
    if (cd.path === 'package.json' || cd.path === 'workspace.json') continue;
    // Skip directory placeholders, else an external `datasets/foo/`-shaped entry surfaces as a zero-byte spurious skip in classification.
    if (cd.path.endsWith('/')) continue;
    const bytes = await readEntryBytes(blob, cd);
    entries.push({ path: cd.path, bytes });
  }
  return { manifest, workspaceCore, workspaceCoreBytes, entries };
}

/// Bucketed view of an alpkg payload for the dataset/head selectors. Defective rows (orphan head half,
/// non-hex slice name) land in `errors` so the dialog shows a "skipped X" hint without failing the import.
export interface ClassifiedAlpkg {
  datasets: DatasetBucket[];
  /// One bucket per `<head_id>` with both halves resolved; orphan halves go to `errors`.
  heads: HeadBucket[];
  errors: AlpkgStructureError[];
}

export interface DatasetBucket {
  name: string;
  /// `filename` always `<sha256>.wav` (foreign-named files land in `errors`).
  slices: { filename: string; bytes: Uint8Array }[];
}

export interface HeadBucket {
  headId: string;
  weights: Uint8Array;
  /// Unparsed so a future daemon-side field rides through without a schema bump here.
  manifestBytes: Uint8Array;
  /// RFC3339; drives latest-first ordering + auto-select, `null` for legacy/malformed manifests.
  createdAt: string | null;
  /// Tie-breaks heads sharing a `createdAt` (a duplicate-publish edge legacy archives can carry).
  revisionId: number | null;
  labels: string[] | null;
  /// Trusted verbatim for display even if it disagrees with `labels.length`; the daemon validates at import.
  nClasses: number | null;
}

export interface AlpkgStructureError {
  path: string;
  message: string;
}

// Inlined rather than imported from the IDB layer to keep IDB off the import code path.
const SLICE_FILENAME_RE = /^[0-9a-f]{64}\.wav$/;
// Lowercase-only to match the daemon's `HeadId` Display (lowercase hex).
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/// Bucket each entry into its dataset category / head pair. Pure: no IO, stable for a given input.
export function classifyAlpkgEntries(entries: readonly AlpkgEntry[]): ClassifiedAlpkg {
  const datasetMap = new Map<string, { filename: string; bytes: Uint8Array }[]>();
  // Keyed by id so the .mpk/.json halves pair up and orphans surface as errors.
  const headHalves = new Map<string, { mpk?: Uint8Array; json?: Uint8Array }>();
  const errors: AlpkgStructureError[] = [];

  for (const entry of entries) {
    const path = entry.path;
    if (path.startsWith('datasets/')) {
      const rest = path.slice('datasets/'.length);
      const slashIdx = rest.indexOf('/');
      if (slashIdx <= 0 || rest.includes('/', slashIdx + 1)) {
        errors.push({
          path,
          message: 'Dataset path must be "datasets/<category>/<filename>".'
        });
        continue;
      }
      const category = rest.slice(0, slashIdx);
      const filename = rest.slice(slashIdx + 1);
      if (!SLICE_FILENAME_RE.test(filename)) {
        errors.push({
          path,
          message: 'Slice filename must be "<sha256>.wav" (lowercase hex).'
        });
        continue;
      }
      const list = datasetMap.get(category) ?? [];
      list.push({ filename, bytes: entry.bytes });
      datasetMap.set(category, list);
    } else if (path.startsWith('head/')) {
      // A lone half is an orphan: the daemon's convert pipeline needs both files to verify+publish.
      const rest = path.slice('head/'.length);
      let headId: string | null = null;
      let half: 'mpk' | 'json' | null = null;
      if (rest.endsWith('.mpk')) {
        headId = rest.slice(0, -4);
        half = 'mpk';
      } else if (rest.endsWith('.json')) {
        headId = rest.slice(0, -5);
        half = 'json';
      }
      if (headId === null || half === null || !UUID_RE.test(headId)) {
        errors.push({
          path,
          message: 'Model path must be "head/<head_id>.{mpk,json}" (UUID head id).'
        });
        continue;
      }
      const pair = headHalves.get(headId) ?? {};
      if (half === 'mpk') pair.mpk = entry.bytes;
      else pair.json = entry.bytes;
      headHalves.set(headId, pair);
    } else {
      errors.push({ path, message: 'Unrecognized entry; ignored.' });
    }
  }

  const datasets: DatasetBucket[] = [];
  // Sort category names for a stable summary-pane order across re-opens.
  for (const name of [...datasetMap.keys()].sort()) {
    const slices = datasetMap.get(name);
    if (slices === undefined || slices.length === 0) continue;
    slices.sort((a, b) => (a.filename < b.filename ? -1 : a.filename > b.filename ? 1 : 0));
    datasets.push({ name, slices });
  }

  const heads: HeadBucket[] = [];
  for (const [headId, pair] of headHalves) {
    if (pair.mpk === undefined) {
      errors.push({
        path: `head/${headId}.json`,
        message: 'Model manifest is present but the ".mpk" weights file is missing.'
      });
      continue;
    }
    if (pair.json === undefined) {
      errors.push({
        path: `head/${headId}.mpk`,
        message: 'Model weights are present but the ".json" manifest is missing.'
      });
      continue;
    }
    const meta = parseHeadManifestMetaSafe(pair.json);
    heads.push({
      headId,
      weights: pair.mpk,
      manifestBytes: pair.json,
      createdAt: meta.createdAt,
      revisionId: meta.revisionId,
      labels: meta.labels,
      nClasses: meta.nClasses
    });
  }
  // Newest-first auto-select by `createdAt`, then `revisionId`, then `headId`; no-`createdAt` buckets sort to the tail so they don't displace a real timestamp.
  heads.sort((a, b) => {
    if (a.createdAt !== null && b.createdAt !== null) {
      if (a.createdAt > b.createdAt) return -1;
      if (a.createdAt < b.createdAt) return 1;
    } else if (a.createdAt !== null) {
      return -1;
    } else if (b.createdAt !== null) {
      return 1;
    }
    const ar = a.revisionId ?? -1;
    const br = b.revisionId ?? -1;
    if (ar !== br) return br - ar;
    return a.headId < b.headId ? -1 : a.headId > b.headId ? 1 : 0;
  });

  return { datasets, heads, errors };
}

/// Never throws: a malformed/legacy manifest yields all-`null` so the head row still surfaces (its bytes are valid).
/// `labels` keeps strings only (tampered `[null,"foo"]` -> no null chip); `nClasses` falls back to `labels.length` when omitted.
function parseHeadManifestMetaSafe(bytes: Uint8Array): {
  createdAt: string | null;
  revisionId: number | null;
  labels: string[] | null;
  nClasses: number | null;
} {
  const empty = { createdAt: null, revisionId: null, labels: null, nClasses: null };
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    const parsed = JSON.parse(text) as unknown;
    if (parsed === null || typeof parsed !== 'object') return empty;
    const obj = parsed as Record<string, unknown>;
    const createdAt = typeof obj.created_at === 'string' ? obj.created_at : null;
    let revisionId: number | null = null;
    if (
      obj.workspace_revision !== null &&
      typeof obj.workspace_revision === 'object' &&
      typeof (obj.workspace_revision as Record<string, unknown>).id === 'number'
    ) {
      revisionId = (obj.workspace_revision as Record<string, unknown>).id as number;
    }
    let labels: string[] | null = null;
    if (Array.isArray(obj.labels)) {
      const filtered = obj.labels.filter((l): l is string => typeof l === 'string');
      labels = filtered;
    }
    let nClasses: number | null = null;
    if (typeof obj.n_classes === 'number' && Number.isFinite(obj.n_classes)) {
      nClasses = obj.n_classes;
    } else if (labels !== null) {
      nClasses = labels.length;
    }
    return { createdAt, revisionId, labels, nClasses };
  } catch {
    return empty;
  }
}
