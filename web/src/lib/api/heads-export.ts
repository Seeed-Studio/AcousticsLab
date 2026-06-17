// Trained-head export: fails closed by cross-checking each manifest vs its clicked HeadRecord and
// re-hashing weights vs manifest.sha256, throwing ExportError at the first mismatch.

import type { ApiErrorBody, HeadManifest, HeadRecord, Uuid } from './types';
import { apiUrl } from './base';
import { ApiError } from './http';
import { heads, workspaces } from './endpoints';
import { sha256Hex } from '$lib/audio/sha256';
import {
  buildAlpkgManifest,
  packAlpkg,
  safeFilenameSlug,
  type AlpkgEntry,
  type AlpkgManifest
} from '$lib/utils/alpkg';

export type ExportPhase =
  | 'fetching-workspace'
  | 'fetching-weights'
  | 'fetching-manifest'
  | 'validating'
  | 'packing'
  | 'downloading';

/// `phase` lets the UI pick copy without parsing the message; `headId` names the offending head in a multi-head walk; the daemon `ApiError` is preserved on `cause`.
export class ExportError extends Error {
  readonly phase: ExportPhase;
  readonly headId: Uuid | null;
  constructor(phase: ExportPhase, message: string, options?: { cause?: unknown; headId?: Uuid }) {
    super(message, options);
    this.name = 'ExportError';
    this.phase = phase;
    this.headId = options?.headId ?? null;
  }
}

export interface ExportHeadInput {
  workspaceId: Uuid;
  workspaceName: string;
  /// Clicked list-row, cross-validated against the daemon manifest so a row gone stale between list and click fails closed rather than packing an inconsistent artefact.
  head: HeadRecord;
}

export interface ExportHeadOptions {
  /// Threaded into the fetch chain so a workspace-swap mid-export tears down the pipeline.
  signal?: AbortSignal;
}

export interface ExportResult {
  filename: string;
  size_bytes: number;
}

export async function exportHead(
  input: ExportHeadInput,
  opts: ExportHeadOptions = {}
): Promise<ExportResult> {
  const { signal } = opts;
  const { workspaceId, workspaceName, head } = input;

  // Embed workspace.json so unzipping recovers source name/tags/revision without the API.
  const { entry: workspaceCoreEntry } = await fetchWorkspaceCoreEntry(workspaceId, signal);

  const pair = await fetchAndValidateOneHead(workspaceId, head, signal);

  // Entry order is load-bearing: package.json first so a streaming reader detects the alpkg kind before the central directory; workspace.json second mirrors the daemon's on-disk layout.
  const pkg: AlpkgManifest = buildAlpkgManifest();
  const pkgBytes = new TextEncoder().encode(stringifyAlpkgManifest(pkg));
  const entries: AlpkgEntry[] = [
    { path: 'package.json', bytes: pkgBytes },
    workspaceCoreEntry,
    ...pair
  ];

  throwIfAborted(signal, 'packing');
  const alpkgBlob = await packAlpkg(entries);

  throwIfAborted(signal, 'downloading');
  const filename = buildSingleHeadExportFilename(workspaceName, head.head_id);
  triggerDownload(alpkgBlob, filename);

  return { filename, size_bytes: alpkgBlob.size };
}

/// Fetch + validate every head (entries-only, no pack/download), returning pairs in input order;
/// `onHeadDone(done, total)` fires after each. Sequential to avoid holding many heads' bytes at once.
export async function buildHeadEntries(
  workspaceId: Uuid,
  headList: readonly HeadRecord[],
  signal: AbortSignal | undefined,
  onHeadDone?: (done: number, total: number) => void
): Promise<AlpkgEntry[]> {
  const entries: AlpkgEntry[] = [];
  const total = headList.length;
  let done = 0;
  for (const head of headList) {
    throwIfAborted(signal, 'fetching-weights');
    try {
      const pair = await fetchAndValidateOneHead(workspaceId, head, signal);
      entries.push(...pair);
    } catch (e) {
      // Stamp the offending head id so the dialog can name the failing head; an already-attributed
      // ExportError or any other throw propagates unchanged.
      if (e instanceof ExportError && e.headId === null) {
        throw new ExportError(e.phase, e.message, { cause: e.cause, headId: head.head_id });
      }
      throw e;
    }
    done++;
    onHeadDone?.(done, total);
  }
  return entries;
}

export interface WorkspaceCoreFetchResult {
  entry: AlpkgEntry;
  /// Lifted from the same verbatim bytes as `entry` so filename and payload can never disagree; the
  /// workspace-export filename embeds `rev_<N>` from it so different-revision exports stay distinct.
  workspaceRevisionId: number;
}

/// Returns the verbatim workspace.json bytes (never re-serialised, to avoid dropping future daemon
/// fields) plus the parsed `workspace_revision.id`, after cross-checking the parsed `id`.
export async function fetchWorkspaceCoreEntry(
  workspaceId: Uuid,
  signal: AbortSignal | undefined
): Promise<WorkspaceCoreFetchResult> {
  throwIfAborted(signal, 'fetching-workspace');
  const bytes = await fetchBinary(
    workspaces.workspaceCoreAssetPath(workspaceId),
    signal,
    'fetching-workspace'
  );
  // Workspace dir is named by its id, so an `id` mismatch means tampering/daemon bug: fail closed.
  const core = parseWorkspaceCoreJson(bytes);
  if (core.id !== workspaceId) {
    throw new ExportError(
      'fetching-workspace',
      `Workspace metadata reports a different workspace id (${core.id} vs ${workspaceId})`
    );
  }
  return {
    entry: { path: 'workspace.json', bytes },
    workspaceRevisionId: core.workspaceRevisionId
  };
}

interface WorkspaceCoreShape {
  id: Uuid;
  workspaceRevisionId: number;
}

/// Reads only the consumed fields; others stay verbatim in the embedded bytes and are deliberately
/// not re-validated so a future daemon field addition can't break the export.
function parseWorkspaceCoreJson(bytes: Uint8Array): WorkspaceCoreShape {
  const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    throw new ExportError('fetching-workspace', 'Workspace metadata is not valid JSON', {
      cause: e
    });
  }
  if (parsed === null || typeof parsed !== 'object') {
    throw new ExportError('fetching-workspace', 'Workspace metadata is not a JSON object');
  }
  const id = (parsed as { id?: unknown }).id;
  if (typeof id !== 'string' || id.length === 0) {
    throw new ExportError('fetching-workspace', 'Workspace metadata is missing the workspace id');
  }
  // Cast through `unknown` so a corrupt body fails the checks below instead of a raw TypeError that
  // would bypass the typed ExportError contract.
  const rev = (parsed as { workspace_revision?: unknown }).workspace_revision;
  if (rev === null || typeof rev !== 'object') {
    throw new ExportError(
      'fetching-workspace',
      'Workspace metadata is missing the workspace_revision object'
    );
  }
  const revId = (rev as { id?: unknown }).id;
  // A non-negative monotonic u64 well within 2^53: reject only fractional/negative/NaN/inf.
  if (typeof revId !== 'number' || !Number.isInteger(revId) || revId < 0) {
    throw new ExportError(
      'fetching-workspace',
      'Workspace metadata has an invalid workspace_revision.id'
    );
  }
  return { id, workspaceRevisionId: revId };
}

async function fetchAndValidateOneHead(
  workspaceId: Uuid,
  head: HeadRecord,
  signal: AbortSignal | undefined
): Promise<[AlpkgEntry, AlpkgEntry]> {
  // Asset GET streams the on-disk file verbatim, so this is byte-identical to what the daemon
  // hashed for HeadManifest.sha256.
  const weightsBytes = await fetchBinary(
    heads.weightsAssetPath(workspaceId, head.head_id),
    signal,
    'fetching-weights'
  );

  // Asset surface, not the /heads/{id} route, to stay symmetric with the weight fetch; that route's
  // orphan-index filter is redundant for a caller already holding a HeadRecord.
  const manifestRaw = await fetchBinary(
    heads.manifestAssetPath(workspaceId, head.head_id),
    signal,
    'fetching-manifest'
  );
  const manifest = parseManifestJson(manifestRaw);

  // Cheap structural + cross-checks before the expensive weight sha256.
  validateManifest(manifest, head, workspaceId);
  const weightsSha = await sha256Hex(weightsBytes);
  validateWeightsAgainstManifest(weightsBytes, weightsSha, manifest);

  // Reuse the raw manifest bytes verbatim: re-serialising would drop a future daemon field and break byte-for-byte reproducibility against the daemon's deterministic serde_json output.
  return [
    { path: `head/${manifest.head_id}.mpk`, bytes: weightsBytes },
    { path: `head/${manifest.head_id}.json`, bytes: manifestRaw }
  ];
}

async function fetchBinary(
  url: string,
  signal: AbortSignal | undefined,
  phase: ExportPhase
): Promise<Uint8Array> {
  let resp: Response;
  try {
    resp = await fetch(apiUrl(url), { signal });
  } catch (e) {
    if (signal?.aborted) throw e; // raw AbortError so callers distinguish abort from failure
    throw new ExportError(phase, `Network error fetching ${url}`, { cause: e });
  }
  if (!resp.ok) {
    // Parse the daemon's `{error, code}` envelope so `errorCopy` can pick fixed copy by code.
    let body: ApiErrorBody;
    try {
      body = (await resp.json()) as ApiErrorBody;
    } catch {
      body = { error: resp.statusText || `HTTP ${resp.status}`, code: 'unknown' };
    }
    throw new ExportError(phase, body.error || `HTTP ${String(resp.status)}`, {
      cause: new ApiError(resp.status, body)
    });
  }
  const buf = await resp.arrayBuffer();
  return new Uint8Array(buf);
}

function parseManifestJson(bytes: Uint8Array): HeadManifest {
  const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  try {
    return JSON.parse(text) as HeadManifest;
  } catch (e) {
    throw new ExportError('validating', 'Head metadata is not valid JSON', { cause: e });
  }
}

function validateManifest(manifest: HeadManifest, expected: HeadRecord, workspaceId: Uuid): void {
  // The HeadManifest cast is unsound until each field is runtime-checked here, so tampering surfaces
  // as a typed ExportError, not an undefined-method throw deep in the packer.
  if (typeof manifest.head_id !== 'string' || manifest.head_id.length === 0) {
    throw new ExportError('validating', 'Head metadata is missing the head id');
  }
  if (manifest.head_id !== expected.head_id) {
    throw new ExportError(
      'validating',
      `Head metadata reports a different head id (${manifest.head_id} vs ${expected.head_id})`
    );
  }
  if (typeof manifest.workspace_id !== 'string' || manifest.workspace_id !== workspaceId) {
    throw new ExportError(
      'validating',
      `Head metadata's workspace id (${manifest.workspace_id}) does not match the requested workspace (${workspaceId})`
    );
  }
  if (
    typeof manifest.n_classes !== 'number' ||
    !Number.isInteger(manifest.n_classes) ||
    manifest.n_classes < 1
  ) {
    throw new ExportError('validating', 'Head metadata has a non-positive class count');
  }
  if (typeof manifest.size_bytes !== 'number' || manifest.size_bytes < 0) {
    throw new ExportError('validating', 'Head metadata has an invalid size');
  }
  if (typeof manifest.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(manifest.sha256)) {
    throw new ExportError('validating', 'Head metadata has an invalid sha256');
  }
  if (!Array.isArray(manifest.labels)) {
    throw new ExportError('validating', 'Head metadata is missing the labels array');
  }
  if (manifest.labels.length !== manifest.n_classes) {
    throw new ExportError(
      'validating',
      `Head metadata's label count (${String(manifest.labels.length)}) does not match n_classes (${String(manifest.n_classes)})`
    );
  }
  for (let i = 0; i < manifest.labels.length; i++) {
    const lbl = manifest.labels[i];
    if (typeof lbl !== 'string' || lbl.length === 0) {
      throw new ExportError(
        'validating',
        `Head metadata's label at index ${String(i)} is empty or not a string`
      );
    }
  }
  // Cross-check vs the clicked HeadRecord: list rows come from the cached heads.json index, so a
  // divergence means a concurrent train/delete swapped the row out.
  if (manifest.sha256 !== expected.sha256) {
    throw new ExportError(
      'validating',
      `Head metadata sha256 does not match the workspace's index (${manifest.sha256} vs ${expected.sha256})`
    );
  }
  if (manifest.n_classes !== expected.n_classes) {
    throw new ExportError(
      'validating',
      `Head metadata class count (${String(manifest.n_classes)}) does not match the workspace's index (${String(expected.n_classes)})`
    );
  }
  if (manifest.size_bytes !== expected.size_bytes) {
    throw new ExportError(
      'validating',
      `Head metadata size (${String(manifest.size_bytes)}) does not match the workspace's index (${String(expected.size_bytes)})`
    );
  }
  // A revision mismatch means another producer published between list and click. Cast through
  // `unknown` and type-check before dereferencing `.id` so a tampered/omitted nested field yields a
  // typed ExportError, not a raw TypeError (the static type claims non-null, but JSON.parse doesn't).
  const rev = manifest.workspace_revision as unknown;
  if (rev === null || typeof rev !== 'object' || typeof (rev as { id?: unknown }).id !== 'number') {
    throw new ExportError('validating', 'Head metadata workspace_revision is missing or malformed');
  }
  const manifestRevId = (rev as { id: number }).id;
  if (manifestRevId !== expected.workspace_revision.id) {
    throw new ExportError(
      'validating',
      `Head metadata workspace revision (${String(manifestRevId)}) does not match the workspace's index (${String(expected.workspace_revision.id)})`
    );
  }
}

function validateWeightsAgainstManifest(
  bytes: Uint8Array,
  observedSha: string,
  manifest: HeadManifest
): void {
  if (bytes.byteLength !== manifest.size_bytes) {
    throw new ExportError(
      'validating',
      `Head weight size (${String(bytes.byteLength)}) does not match the metadata's size (${String(manifest.size_bytes)})`
    );
  }
  if (observedSha !== manifest.sha256) {
    throw new ExportError(
      'validating',
      'Head weight hash does not match the metadata; the download is corrupted or the head changed mid-export.'
    );
  }
}

function stringifyAlpkgManifest(pkg: AlpkgManifest): string {
  return JSON.stringify(pkg, null, 2) + '\n';
}

function buildSingleHeadExportFilename(workspaceName: string, headId: Uuid): string {
  const wsSlug = safeFilenameSlug(workspaceName, 'workspace');
  const headSlug = headId.replace(/-/g, '').slice(0, 8);
  return `${wsSlug}-head-${headSlug}.alpkg`;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.target = '_self';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Defer the revoke so the browser hands the download to the network stack before the blob URL dies.
  setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 30_000);
}

function throwIfAborted(signal: AbortSignal | undefined, phase: ExportPhase): void {
  if (signal?.aborted === true) {
    const reason: unknown = signal.reason;
    const reasonMsg =
      reason instanceof Error
        ? reason.message
        : typeof reason === 'string'
          ? reason
          : 'export aborted';
    throw new ExportError(phase, reasonMsg, { cause: reason });
  }
}
