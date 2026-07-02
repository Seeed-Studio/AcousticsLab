// `.alpkg` and TFJS imports share the upload/convert/SSE-wait/cleanup back half via
// runConvertPipeline. Failures are typed ImportError{phase} (UI picks copy without parsing); aborts
// re-throw AbortError to distinguish operator cancel from daemon failure. No store, no DOM.

import type { ConvertEvent, ConvertStartResp, HeadManifest, Uuid } from './types';
import { ApiError } from './http';
import { converters } from './endpoints';
import { trackJob } from './jobs';
import { enqueueDelete } from './delete-queue';
import { xhrPut } from './upload';
import { unpackAlpkg, type AlpkgEntry, type AlpkgUnpackResult } from '$lib/utils/alpkg-unpack';
import { sha256Hex } from '$lib/audio/sha256';

export type ImportPhase = 'preparing' | 'uploading' | 'starting' | 'converting' | 'cleaning';

export interface ImportProgress {
  phase: ImportPhase;
  /// Set during `uploading` only; convert has no normalised progress and rides `onlog`.
  ratio?: number;
  /// Known at `preparing` for `.alpkg` (embedded manifest) but only after `starting` for TFJS.
  headId?: Uuid;
}

/// Wrapped daemon `ApiError` lives on `cause` so `errorCopy(e.cause ?? e)` reads the daemon code + body.
export class ImportError extends Error {
  readonly phase: ImportPhase;
  constructor(phase: ImportPhase, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'ImportError';
    this.phase = phase;
  }
}

export interface ImportResult {
  /// `.alpkg`: embedded manifest's `head_id`; TFJS: fresh UUID the daemon allocates at dispatch.
  headId: Uuid;
  jobId: Uuid;
}

export interface ImportOptions {
  signal?: AbortSignal;
  onprogress?: (p: ImportProgress) => void;
  /// Fired per non-empty daemon `JobEvent.message` during `converting`.
  onlog?: (message: string) => void;
}

/// Summary the Import pane shows post-parse / pre-"Convert" so the operator can inspect and bail.
export interface AlpkgValidated {
  kind: 'alpkg';
  headId: Uuid;
  manifest: HeadManifest;
  /// Kept so the upload phase need not re-parse the archive.
  mpkBytes: Uint8Array;
  manifestBytes: Uint8Array;
  totalUploadBytes: number;
}

export interface TfjsValidated {
  kind: 'tfjs';
  /// Display-only; the daemon allocates the authoritative `head_id` at dispatch.
  displayName: string;
  /// Empty when `labelsFormat === 'tfjs_metadata'`: the daemon parses the metadata JSON itself.
  labels: string[];
  modelJsonFile: File;
  shardFiles: File[];
  labelsFile: File;
  labelsFormat: 'lines' | 'tfjs_metadata';
  totalUploadBytes: number;
}

export type Validated = AlpkgValidated | TfjsValidated;

/// Verifies `heads/<id>.mpk` length + SHA-256 against the embedded manifest; must complete before
/// `runAlpkgImport`, which re-reads the bytes validated here.
export async function validateAlpkgFile(file: File): Promise<AlpkgValidated> {
  let result: AlpkgUnpackResult;
  try {
    result = await unpackAlpkg(file);
  } catch (e) {
    throw new ImportError('preparing', describeError(e, 'Could not read .alpkg archive'), {
      cause: e
    });
  }

  // The .find() below would silently pick the first head, so reject multi-head / dataset-bearing
  // workspace exports here.
  const headMpks = result.entries.filter((e: AlpkgEntry) =>
    /^heads\/[A-Za-z0-9-]+\.mpk$/.test(e.path)
  );
  if (headMpks.length > 1) {
    throw new ImportError(
      'preparing',
      `This archive contains ${String(headMpks.length)} heads; the converter imports a single-head .alpkg. Use "Import workspace" for multi-head bundles.`
    );
  }
  if (result.entries.some((e: AlpkgEntry) => e.path.startsWith('datasets/'))) {
    throw new ImportError(
      'preparing',
      'This looks like a full workspace export (it contains datasets). Use "Import workspace" to import it.'
    );
  }

  // Look up by filename pattern, not position, so future packer additions don't break it.
  const headEntry =
    result.entries.find((e: AlpkgEntry) => /^heads\/[A-Za-z0-9-]+\.mpk$/.test(e.path)) ?? null;
  if (headEntry === null) {
    throw new ImportError('preparing', 'Archive is missing heads/<id>.mpk');
  }
  const manifestEntry =
    result.entries.find((e: AlpkgEntry) => /^heads\/[A-Za-z0-9-]+\.json$/.test(e.path)) ?? null;
  if (manifestEntry === null) {
    throw new ImportError('preparing', 'Archive is missing heads/<id>.json');
  }

  // Import side has no HeadRecord to cross-check against, so validate the manifest standalone.
  const manifestText = decodeUtf8OrThrow(manifestEntry.bytes, 'model manifest');
  // Cast is unsound until validateManifestStructure makes it true at runtime, below.
  const manifest = parseJsonOrThrow(manifestText, 'model manifest') as unknown as HeadManifest;
  validateManifestStructure(manifest);

  // Filenames must agree with the manifest's head_id; a mismatch points at a tampered archive.
  const expectedMpkName = `heads/${manifest.head_id}.mpk`;
  const expectedManifestName = `heads/${manifest.head_id}.json`;
  if (headEntry.path !== expectedMpkName) {
    throw new ImportError(
      'preparing',
      `Model weight filename mismatch (expected ${expectedMpkName}, got ${headEntry.path})`
    );
  }
  if (manifestEntry.path !== expectedManifestName) {
    throw new ImportError(
      'preparing',
      `Model manifest filename mismatch (expected ${expectedManifestName}, got ${manifestEntry.path})`
    );
  }

  // Size check before sha256: cheap failure before the expensive hash.
  const mpkBytes = headEntry.bytes;
  if (mpkBytes.byteLength !== manifest.size_bytes) {
    throw new ImportError(
      'preparing',
      `Model weight size mismatch (manifest declares ${String(manifest.size_bytes)} bytes, archive holds ${String(mpkBytes.byteLength)})`
    );
  }
  const observedSha = await sha256Hex(mpkBytes);
  if (observedSha !== manifest.sha256) {
    throw new ImportError(
      'preparing',
      'Model weight hash does not match the embedded manifest -- archive may be corrupt'
    );
  }

  return {
    kind: 'alpkg',
    headId: manifest.head_id,
    manifest,
    mpkBytes,
    manifestBytes: manifestEntry.bytes,
    totalUploadBytes: mpkBytes.byteLength + manifestEntry.bytes.byteLength
  };
}

/// Best-effort client-side check to catch obvious structural mistakes before wasting an upload +
/// convert; the daemon's converter does the real bundle validation.
export function validateTfjsFiles(files: readonly File[]): TfjsValidated {
  if (files.length === 0) {
    throw new ImportError('preparing', 'No files selected for TFJS import');
  }

  // Exact-name match: renaming model.json would break every TFJS consumer anyway.
  const modelJsonFile = files.find((f) => f.name === 'model.json') ?? null;
  if (modelJsonFile === null) {
    throw new ImportError(
      'preparing',
      'TFJS drop is missing model.json (the architecture descriptor)'
    );
  }

  // labelsFormat picks the daemon parser: metadata.json (SpeechCommands wordLabels) vs *.txt (`lines`).
  let labelsFile: File | null = null;
  let labelsFormat: 'lines' | 'tfjs_metadata' = 'lines';
  const metadataJson = files.find((f) => f.name === 'metadata.json') ?? null;
  if (metadataJson !== null) {
    labelsFile = metadataJson;
    labelsFormat = 'tfjs_metadata';
  } else {
    const labelsTxt =
      files.find((f) => f.name === 'labels.txt') ??
      files.find((f) => f.name.endsWith('.txt')) ??
      null;
    if (labelsTxt !== null) {
      labelsFile = labelsTxt;
      labelsFormat = 'lines';
    }
  }
  if (labelsFile === null) {
    throw new ImportError(
      'preparing',
      'TFJS drop is missing labels.txt or metadata.json (the class-label source)'
    );
  }

  // Any non-model/non-labels file is a shard (names are unconstrained); the daemon enforces the format.
  const shardFiles = files.filter((f) => f !== modelJsonFile && f !== labelsFile);
  if (shardFiles.length === 0) {
    throw new ImportError(
      'preparing',
      'TFJS drop is missing the weight shards (group1-shard1of...)'
    );
  }

  // Sub-paths are basenames, so duplicate names collide on one URL and the later serial PUT silently
  // overwrites the earlier (opaque daemon `source_malformed`); reject client-side first.
  const allNames = [modelJsonFile.name, labelsFile.name, ...shardFiles.map((f) => f.name)];
  if (new Set(allNames).size !== allNames.length) {
    const dup = allNames.find((n, i) => allNames.indexOf(n) !== i) ?? '';
    throw new ImportError(
      'preparing',
      `Two selected files share the name "${dup}"; rename or re-select so every file is unique.`
    );
  }

  const totalUploadBytes =
    modelJsonFile.size + labelsFile.size + shardFiles.reduce((acc, f) => acc + f.size, 0);

  return {
    kind: 'tfjs',
    displayName: modelJsonFile.name.replace(/\.json$/, ''),
    labels: [],
    modelJsonFile,
    shardFiles,
    labelsFile,
    labelsFormat,
    totalUploadBytes
  };
}

export async function runAlpkgImport(
  workspaceId: Uuid,
  validated: AlpkgValidated,
  opts: ImportOptions = {}
): Promise<ImportResult> {
  // One folder per head_id (a UUID, safe as an AssetPath segment unsanitised) so concurrent imports
  // coexist and cleanup wipes only this footprint.
  const subPath = `alpkg/${validated.headId}`;
  const mpkSubPath = `${subPath}/${validated.headId}.mpk`;
  const manifestSubPath = `${subPath}/${validated.headId}.json`;

  return runConvertPipeline({
    workspaceId,
    uploads: [
      {
        subPath: mpkSubPath,
        blob: new Blob([validated.mpkBytes as Uint8Array<ArrayBuffer>], {
          type: 'application/octet-stream'
        })
      },
      {
        subPath: manifestSubPath,
        blob: new Blob([validated.manifestBytes as Uint8Array<ArrayBuffer>], {
          type: 'application/json'
        })
      }
    ],
    totalUploadBytes: validated.totalUploadBytes,
    // Only the manifest path travels the wire; the daemon derives the sibling `<parent>/<head_id>.mpk`,
    // so the .mpk MUST sit in the same folder.
    convertReq: {
      converter_type: 'alpkg',
      manifest_path: manifestSubPath
    },
    cleanupSubPath: subPath,
    preAnnouncedHeadId: validated.headId,
    opts
  });
}

export async function runTfjsImport(
  workspaceId: Uuid,
  validated: TfjsValidated,
  opts: ImportOptions = {}
): Promise<ImportResult> {
  // Timestamped folder so re-dropping the same bundle doesn't collide; cleanup wipes only this sub-tree.
  const stamp = Date.now().toString(36);
  const subPath = `tfjs/${stamp}`;
  const modelSubPath = `${subPath}/${validated.modelJsonFile.name}`;
  const labelsSubPath = `${subPath}/${validated.labelsFile.name}`;
  const shardSubPaths = validated.shardFiles.map((f) => `${subPath}/${f.name}`);

  const uploads: PreparedUpload[] = [
    { subPath: modelSubPath, blob: validated.modelJsonFile },
    { subPath: labelsSubPath, blob: validated.labelsFile },
    ...validated.shardFiles.map((f, i) => ({
      subPath: shardSubPaths[i],
      blob: f
    }))
  ];

  // Only model.json + labels travel the wire; the daemon reads model.json's weightsManifest[].paths
  // and derives each shard as `<parent>/<shard>`, so all files MUST sit flat in one folder.
  return runConvertPipeline({
    workspaceId,
    uploads,
    totalUploadBytes: validated.totalUploadBytes,
    convertReq: {
      converter_type: 'tfjs',
      model_json_path: modelSubPath,
      labels_path: labelsSubPath,
      labels_format: validated.labelsFormat
    },
    cleanupSubPath: subPath,
    preAnnouncedHeadId: undefined,
    opts
  });
}

interface PreparedUpload {
  subPath: string;
  blob: Blob;
}

interface PipelineInput {
  workspaceId: Uuid;
  uploads: readonly PreparedUpload[];
  totalUploadBytes: number;
  convertReq: Parameters<typeof converters.start>[1];
  cleanupSubPath: string;
  preAnnouncedHeadId: Uuid | undefined;
  opts: ImportOptions;
}

async function runConvertPipeline(input: PipelineInput): Promise<ImportResult> {
  const {
    workspaceId,
    uploads,
    totalUploadBytes,
    convertReq,
    cleanupSubPath,
    preAnnouncedHeadId,
    opts
  } = input;
  const { signal, onprogress, onlog } = opts;

  // Serial PUTs (not Promise.all) keep byte progress monotonic and avoid racing workspace_revision bumps.
  emit(onprogress, { phase: 'uploading', ratio: 0, headId: preAnnouncedHeadId });
  let uploadedBytes = 0;
  for (const u of uploads) {
    throwIfAborted(signal, 'uploading');
    let perFileLoaded = 0;
    try {
      await xhrPut({
        url: converters.putAssetPath(workspaceId, u.subPath),
        body: u.blob,
        onProgress: (loaded) => {
          perFileLoaded = loaded;
          const ratio =
            totalUploadBytes > 0 ? (uploadedBytes + perFileLoaded) / totalUploadBytes : 0;
          emit(onprogress, {
            phase: 'uploading',
            ratio: Math.min(1, ratio),
            headId: preAnnouncedHeadId
          });
        },
        signal
      });
    } catch (e) {
      if (signal?.aborted === true) throw e; // honour caller abort
      throw new ImportError('uploading', describeError(e, `Could not upload ${u.subPath}`), {
        cause: e
      });
    }
    uploadedBytes += u.blob.size;
  }

  // A 409 (wire `conflict`) means another convert is in flight: the daemon's convert-semaphore is global.
  emit(onprogress, { phase: 'starting', headId: preAnnouncedHeadId });
  throwIfAborted(signal, 'starting');
  let startResp: ConvertStartResp;
  try {
    startResp = await converters.start(workspaceId, convertReq);
  } catch (e) {
    throw new ImportError('starting', describeError(e, 'Could not start convert job'), {
      cause: e
    });
  }
  const headId = startResp.head_id;
  const jobId = startResp.job_id;

  // trackJob (not awaitJobTerminal) for per-event hooks: fan every JobEvent.message into the log.
  emit(onprogress, { phase: 'converting', headId });
  await waitConvertTerminal(jobId, signal, onlog);

  // enqueueDelete serialises against the daemon's global max_delete_jobs=1 slot; failure is non-fatal.
  emit(onprogress, { phase: 'cleaning', headId });
  fireAndForgetCleanup(workspaceId, cleanupSubPath);

  return { headId, jobId };
}

function waitConvertTerminal(
  jobId: Uuid,
  signal: AbortSignal | undefined,
  onlog: ((message: string) => void) | undefined
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted === true) {
      reject(abortError(signal));
      return;
    }
    // Latch the typed job_failed so the failure terminal (message:None) can synthesise a rich banner;
    // the daemon emits JobFailed before consuming the JobHandle, so it's populated by then.
    let lastJobFailed: (ConvertEvent & { kind: 'job_failed' }) | null = null;
    const tracker = trackJob(jobId, {
      logs: true,
      onEvent: (ev) => {
        // Each typed ConvertEvent arrives as a JSON string in JobEvent.message.
        if (typeof ev.message !== 'string' || ev.message.length === 0) return;
        const parsed = parseConvertEventSafe(ev.message);
        if (parsed !== null && parsed.kind === 'job_failed') {
          lastJobFailed = parsed;
        }
        // The daemon appends a raw error after the typed JobFailed (duplicate), so suppress non-JSON
        // once failed; valid-JSON unknown kinds still flow.
        if (parsed === null && lastJobFailed !== null) return;
        const label = describeConvertEventMessage(ev.message);
        if (label !== null) onlog?.(label);
      },
      onTerminal: (ev) => {
        if (ev.state === 'succeeded') {
          resolve();
          return;
        }
        // Latched typed payload (rich stage + error) wins over ev.message over a synthesised line.
        let msg: string;
        if (lastJobFailed !== null) {
          msg = `convert failed at ${lastJobFailed.stage}: ${lastJobFailed.error}`;
        } else if (typeof ev.message === 'string' && ev.message.length > 0) {
          msg = ev.message;
        } else {
          msg = `convert ${ev.state ?? 'ended without success'}`;
        }
        reject(new ImportError('converting', msg));
      },
      onError: (reason) => {
        reject(new ImportError('converting', `event stream error: ${reason}`));
      }
    });
    // Operator-cancel only tears down the SSE and rejects AbortError; the daemon worker gets NO cancel
    // and runs to its terminal, so a head it publishes after the frontend gave up is never lost
    // (visible on next mount).
    if (signal !== undefined) {
      signal.addEventListener(
        'abort',
        () => {
          tracker.cancel();
          reject(abortError(signal));
        },
        { once: true }
      );
    }
  });
}

function fireAndForgetCleanup(workspaceId: Uuid, subPath: string): void {
  void enqueueDelete(async () => {
    // The ack suffices (disk reclaim is a background drain); don't await the SSE terminal.
    await converters.deletePath(workspaceId, subPath);
  }).catch((e: unknown) => {
    console.warn('[converter] cleanup failed for', subPath, e);
  });
}

function validateManifestStructure(manifest: HeadManifest): void {
  if (typeof manifest.head_id !== 'string' || manifest.head_id.length === 0) {
    throw new ImportError('preparing', 'Model manifest is missing the head id');
  }
  if (typeof manifest.workspace_id !== 'string' || manifest.workspace_id.length === 0) {
    throw new ImportError('preparing', 'Model manifest is missing the workspace id');
  }
  if (
    typeof manifest.n_classes !== 'number' ||
    !Number.isInteger(manifest.n_classes) ||
    manifest.n_classes < 1
  ) {
    throw new ImportError('preparing', 'Model manifest has a non-positive class count');
  }
  if (typeof manifest.size_bytes !== 'number' || manifest.size_bytes < 0) {
    throw new ImportError('preparing', 'Model manifest has an invalid size');
  }
  if (typeof manifest.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(manifest.sha256)) {
    throw new ImportError('preparing', 'Model manifest has an invalid sha256');
  }
  if (!Array.isArray(manifest.labels)) {
    throw new ImportError('preparing', 'Model manifest is missing the labels array');
  }
  if (manifest.labels.length !== manifest.n_classes) {
    throw new ImportError(
      'preparing',
      `Model manifest's label count (${String(manifest.labels.length)}) does not match n_classes (${String(manifest.n_classes)})`
    );
  }
  for (let i = 0; i < manifest.labels.length; i++) {
    const lbl = manifest.labels[i];
    if (typeof lbl !== 'string' || lbl.length === 0) {
      throw new ImportError(
        'preparing',
        `Model manifest's label at index ${String(i)} is empty or not a string`
      );
    }
  }
  // Guard the nested deref: a tampered manifest could omit workspace_revision.id.
  const rev = manifest.workspace_revision as unknown;
  if (rev === null || typeof rev !== 'object' || typeof (rev as { id?: unknown }).id !== 'number') {
    throw new ImportError('preparing', 'Model manifest workspace_revision is missing or malformed');
  }
}

function decodeUtf8OrThrow(bytes: Uint8Array, what: string): string {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch (e) {
    throw new ImportError('preparing', `${what} is not valid UTF-8`, { cause: e });
  }
}

function parseJsonOrThrow(text: string, what: string): Record<string, unknown> {
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch (e) {
    throw new ImportError('preparing', `${what} is not valid JSON`, { cause: e });
  }
}

function describeError(e: unknown, fallback: string): string {
  if (e instanceof ApiError) {
    // Operator-facing `error` verbatim; the ApiError itself stays on ImportError.cause for code lookup.
    return e.body.error || fallback;
  }
  if (e instanceof Error) return e.message || fallback;
  return fallback;
}

function throwIfAborted(signal: AbortSignal | undefined, phase: ImportPhase): void {
  if (signal?.aborted === true) {
    throw new ImportError(phase, abortReason(signal), { cause: signal.reason });
  }
}

function abortError(signal: AbortSignal): Error {
  const reason: unknown = signal.reason;
  if (reason instanceof Error) return reason;
  return new DOMException(typeof reason === 'string' ? reason : 'aborted', 'AbortError');
}

function abortReason(signal: AbortSignal): string {
  const reason: unknown = signal.reason;
  if (reason instanceof Error) return reason.message;
  if (typeof reason === 'string') return reason;
  return 'import aborted';
}

function emit(onprogress: ((p: ImportProgress) => void) | undefined, p: ImportProgress): void {
  if (onprogress) onprogress(p);
}

/// Null for malformed JSON or missing `kind` (the daemon's 8 KiB log-line cap can truncate the JSON
/// tail) so the SSE flow keeps moving.
function parseConvertEventSafe(message: string): ConvertEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(message);
  } catch {
    return null;
  }
  if (
    typeof parsed !== 'object' ||
    parsed === null ||
    !('kind' in parsed) ||
    typeof parsed.kind !== 'string'
  ) {
    return null;
  }
  // Trust the wire shape; unknown future `kind`s are handled at the consumer's default arm.
  return parsed as ConvertEvent;
}

/// One log line; null for events already shown by the phase indicator, or the raw message verbatim
/// for unparseable / unknown-kind payloads (forward-compat).
function describeConvertEventMessage(message: string): string | null {
  const parsed = parseConvertEventSafe(message);
  if (parsed === null) {
    // Surface the malformed/truncated payload rather than lose it.
    return message;
  }
  switch (parsed.kind) {
    case 'job_submitted':
    case 'job_running':
      return null;
    case 'stage_started':
      return `Stage: ${parsed.stage}`;
    case 'manifest_validated':
      return `Manifest validated (n_classes=${parsed.n_classes})`;
    case 'mpk_verified':
      return `.mpk verified (size=${parsed.size_bytes}B, sha=${parsed.sha256.slice(0, 12)}…)`;
    case 'weights_extracted':
      return `Weights extracted (n_classes=${parsed.n_classes}, in_dim=${parsed.in_dim})`;
    case 'labels_loaded':
      return `Labels loaded (n_labels=${parsed.n_labels})`;
    case 'head_published':
      return parsed.idempotent_skip
        ? `Model already present (idempotent skip)`
        : `Model published (n_classes=${parsed.n_classes})`;
    case 'job_completed':
      return `Convert completed`;
    case 'job_failed':
      return `Convert failed at ${parsed.stage}: ${parsed.error}`;
    default: {
      const unknownKind = (parsed as { kind: string }).kind;
      return `[${unknownKind}] ${message}`;
    }
  }
}
