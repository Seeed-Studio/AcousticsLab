// Workspace import orchestrator: per-item failures are first-failure-wins within a step and
// recorded on the summary rather than aborting remaining items; converter intake is best-effort
// deleted after every terminal (failure swallowed: the head landed and the daemon reaps orphans).

import type {
  AssetReceipt,
  ConvertEvent,
  ConvertStartResp,
  JobState,
  LabelsFormat,
  Uuid
} from './types';
import { apiUrl } from './base';
import { ApiError } from './http';
import { assets, heads as headsApi, jobs as jobsApi } from './endpoints';
import { converter } from './converter';
import { enqueueDelete } from './delete-queue';
import { awaitJobTerminal, isTerminal } from './jobs';
import { UploadPool, xhrPut } from './upload';
import { isNotFound } from '$lib/utils/error-copy';
import type { AlpkgUnpackResult, HeadBucket } from '$lib/utils/alpkg-unpack';
import type { ClassifiedTfjsBundle } from '$lib/utils/tfjs-classify';

export type WorkspaceImportPhase =
  | 'replacing-categories'
  | 'uploading-datasets'
  | 'importing-heads'
  | 'uploading-tfjs'
  | 'converting-tfjs';

export type HeadImportPhase =
  'uploading-files' | 'starting-convert' | 'awaiting-terminal' | 'cleaning-up' | 'done';

export interface WorkspaceImportProgress {
  phase: WorkspaceImportPhase;
  /// Global done/total for the overall strip; per-row counts are categoryUploaded/categoryFailed so
  /// two source rows merging into one target don't share a global denominator.
  itemsDone?: number;
  itemsTotal?: number;
  headIndex?: number;
  headPhase?: HeadImportPhase;
  /// Target category; NOT a per-row identifier (rename+merge maps several sources to one target).
  category?: string;
  /// Stable per-row identifier the dialog keys its run-state map by. Absent on phase-startup pulses.
  sourceCategory?: string;
  /// Per-row counters; monotonic within an import (contract). The pre-pulse reflects PRIOR items
  /// (this PUT not yet fired), the post-pulse folds in this item.
  categoryUploaded?: number;
  categoryFailed?: number;
}

export type HeadOutcome = 'imported' | 'skipped' | 'replaced' | 'failed';

export interface HeadOutcomeRecord {
  headId: Uuid;
  outcome: HeadOutcome;
  /// Null on `failed` and on the `skipped` no-op.
  publishedSha256: string | null;
  /// Non-null only when `outcome === 'failed'`.
  error: string | null;
  /// Set on a `head_id_collision` terminal that the operator did NOT pre-authorise replace for; lets
  /// the dialog render "Replace existing" without parsing the error.
  conflict?: {
    headId: Uuid;
    storedSha256: string;
    incomingSha256: string;
  };
}

export interface CategoryOutcomeRecord {
  /// On-disk target name; drives post-import slice refresh so it MUST match disk.
  category: string;
  /// Source name from the archive's `datasets/<name>/` path; kept separate to render the rename.
  sourceCategory: string;
  /// `'new'`/`'merge'` identical (additive PUT, split for telemetry only); `'replace'` had its
  /// target wiped in Phase 0. Content-addressed `<sha256>.wav` names make merge idempotent.
  mode: 'new' | 'merge' | 'replace';
  uploaded: number;
  /// Errored slice count; the import upload path does not retry.
  failed: number;
  /// First-failure message; null when failed === 0.
  error: string | null;
}

export interface WorkspaceImportSummary {
  categories: CategoryOutcomeRecord[];
  heads: HeadOutcomeRecord[];
  /// Highest `workspace_revision_id` across every receipt / `head_published`; lets the dialog drive
  /// refresh + the detail poller without an extra workspace GET.
  latestRevisionId: number | null;
}

export type WorkspaceImportErrorKind =
  /// Whole phase failed before any per-item outcome; per-item failures land on the summary instead.
  | 'phase-failed'
  /// Operator cancelled; pipeline tore down cleanly, no summary returned.
  | 'aborted'
  /// Input rejected at the boundary (empty selection, unreadable archive).
  | 'input-invalid';

export class WorkspaceImportError extends Error {
  readonly phase: WorkspaceImportPhase | null;
  readonly kind: WorkspaceImportErrorKind;
  constructor(
    kind: WorkspaceImportErrorKind,
    message: string,
    options?: { phase?: WorkspaceImportPhase; cause?: unknown }
  ) {
    super(message, options);
    this.name = 'WorkspaceImportError';
    this.kind = kind;
    this.phase = options?.phase ?? null;
  }
}

export interface DatasetImportRow {
  /// Matches a `name` in the unpack result's `datasets` buckets.
  sourceName: string;
  /// On-disk landing name (equals `sourceName` unless renamed). The dialog rejects two rows mapping
  /// to the same target (case-insensitive), so the orchestrator never sees that shape.
  targetName: string;
  /// `'new'`/`'merge'` identical (additive PUT, content-addressed names make merge idempotent);
  /// `'replace'` deletes the whole target tree before uploading, operator opts in per conflict row.
  mode: 'new' | 'merge' | 'replace';
}

export interface AlpkgImportSelection {
  datasets: readonly DatasetImportRow[];
  headIds: readonly Uuid[];
  /// Heads pre-authorised to replace a same-id existing head: a `head_id_collision` triggers a
  /// delete + convert retry instead of a manual-Replace conflict row on the summary.
  replaceHeadIds: ReadonlySet<Uuid>;
}

export interface ImportAlpkgInput {
  targetWorkspaceId: Uuid;
  archive: AlpkgUnpackResult;
  /// Threaded explicitly so the caller classifies once and shares it between the summary pane and
  /// the orchestrator.
  classified: {
    datasets: readonly { name: string; slices: { filename: string; bytes: Uint8Array }[] }[];
    heads: readonly HeadBucket[];
  };
  selection: AlpkgImportSelection;
}

export interface ImportAlpkgOptions {
  signal?: AbortSignal;
  onprogress?: (p: WorkspaceImportProgress) => void;
  onConvertEvent?: (headId: Uuid, ev: ConvertEvent) => void;
  /// Wipes a Phase-0 replace target before its archive counterpart uploads. A callback because the
  /// owning `categories` store (it also reaps IDB shadow + draft + slice-store state) lives in
  /// `stores/`, which `api/` cannot depend on. MUST resolve only after the daemon delete job reaches
  /// terminal; `not_found` is swallowed (already gone), any other error fails Phase 0 and aborts
  /// (else half-replaced). Omitting it falls back to a raw daemon delete the UI won't reconcile until
  /// its next refresh, so the dialog SHOULD always supply this hook.
  ondeleteCategory?: (name: string) => Promise<void>;
}

export async function importAlpkg(
  input: ImportAlpkgInput,
  opts: ImportAlpkgOptions = {}
): Promise<WorkspaceImportSummary> {
  const { targetWorkspaceId, classified, selection } = input;
  const { signal, onprogress, onConvertEvent, ondeleteCategory } = opts;

  if (selection.datasets.length === 0 && selection.headIds.length === 0) {
    throw new WorkspaceImportError(
      'input-invalid',
      'Pick at least one category or head to import.'
    );
  }

  const summary: WorkspaceImportSummary = {
    categories: [],
    heads: [],
    latestRevisionId: null
  };
  const revisionTracker = { latest: null as number | null };

  // Phase 0: replace pre-flight. Sequential + fail-fast before any upload so an abort mid-upload
  // can't leave some replace targets wiped+re-uploaded and others wiped+empty.
  const replaceRows = selection.datasets.filter((r) => r.mode === 'replace');
  if (replaceRows.length > 0) {
    await replaceCategories(targetWorkspaceId, replaceRows, signal, onprogress, ondeleteCategory);
  }

  // Phase 1: datasets. Safe now that every replace target is empty (or never existed).
  if (selection.datasets.length > 0) {
    const bucketBySource = new Map(classified.datasets.map((b) => [b.name, b]));
    const rows: UploadRow[] = [];
    for (const sel of selection.datasets) {
      const bucket = bucketBySource.get(sel.sourceName);
      // Dialog only passes in-archive sources, so an absent bucket is a bug; skip rather than throw.
      if (!bucket) continue;
      rows.push({
        sourceName: sel.sourceName,
        targetName: sel.targetName,
        mode: sel.mode,
        slices: bucket.slices
      });
    }
    if (rows.length > 0) {
      await uploadDatasets(targetWorkspaceId, rows, summary, revisionTracker, signal, onprogress);
    }
  }

  // Phase 2: heads. Sequential because the daemon serialises convert jobs (max_convert_jobs = 1) so
  // fan-out would just queue on the semaphore, and it keeps per-row progress honest.
  if (selection.headIds.length > 0) {
    const idSet = new Set(selection.headIds);
    const headBuckets = classified.heads.filter((b) => idSet.has(b.headId));
    if (headBuckets.length > 0) {
      await importHeads(
        targetWorkspaceId,
        headBuckets,
        selection.replaceHeadIds,
        summary,
        revisionTracker,
        signal,
        onprogress,
        onConvertEvent
      );
    }
  }

  summary.latestRevisionId = revisionTracker.latest;
  return summary;
}

export interface ImportTfjsInput {
  targetWorkspaceId: Uuid;
  bundle: ClassifiedTfjsBundle;
}

export interface ImportTfjsOptions {
  signal?: AbortSignal;
  onprogress?: (p: WorkspaceImportProgress) => void;
  onConvertEvent?: (ev: ConvertEvent) => void;
}

export async function importTfjs(
  input: ImportTfjsInput,
  opts: ImportTfjsOptions = {}
): Promise<WorkspaceImportSummary> {
  const { targetWorkspaceId, bundle } = input;
  const { signal, onprogress, onConvertEvent } = opts;

  if (!bundle.ready || !bundle.modelJson || !bundle.labels || !bundle.labelsFormat) {
    throw new WorkspaceImportError(
      'input-invalid',
      'TFJS bundle is incomplete; resolve diagnostics first.'
    );
  }

  const summary: WorkspaceImportSummary = {
    categories: [],
    heads: [],
    latestRevisionId: null
  };
  const revisionTracker = { latest: null as number | null };
  // Timestamped per-attempt sub-dir so re-imports of the same bundle don't collide in `converters/`.
  const sub = `converters/tfjs/${Date.now().toString(36)}`;
  const files: { name: string; file: File }[] = [
    { name: bundle.modelJson.name, file: bundle.modelJson },
    { name: bundle.labels.name, file: bundle.labels },
    ...bundle.shards.map((f) => ({ name: f.name, file: f }))
  ];

  // Upload phase. Any failure aborts the whole import: no point converting with missing shards.
  emit(onprogress, {
    phase: 'uploading-tfjs',
    itemsDone: 0,
    itemsTotal: files.length
  });
  throwIfAborted(signal);
  const pool = new UploadPool(3);
  let done = 0;
  let firstFailure: unknown = null;
  await Promise.all(
    files.map((f) =>
      pool.submit(async () => {
        if (firstFailure !== null || (signal?.aborted ?? false)) return;
        try {
          const receipt = await xhrPut<AssetReceipt>({
            url: assetPutUrl(targetWorkspaceId, `${sub}/${f.name}`),
            body: f.file,
            contentType: 'application/octet-stream',
            signal
          });
          bumpRevision(revisionTracker, receipt.workspace_revision_id);
          done++;
          emit(onprogress, {
            phase: 'uploading-tfjs',
            itemsDone: done,
            itemsTotal: files.length
          });
        } catch (e) {
          firstFailure ??= e;
        }
      })
    )
  );
  if (signal?.aborted ?? false) {
    await cleanupPath(targetWorkspaceId, sub);
    throw new WorkspaceImportError('aborted', 'Import aborted.', { phase: 'uploading-tfjs' });
  }
  if (firstFailure !== null) {
    await cleanupPath(targetWorkspaceId, sub);
    throw new WorkspaceImportError('phase-failed', errorMessage(firstFailure), {
      phase: 'uploading-tfjs',
      cause: firstFailure
    });
  }

  emit(onprogress, { phase: 'converting-tfjs' });
  throwIfAborted(signal);
  const labelsFormat: LabelsFormat = bundle.labelsFormat;
  let convertStart: ConvertStartResp;
  try {
    convertStart = await converter.startTfjs(targetWorkspaceId, {
      modelJsonPath: relConverterPath(sub, bundle.modelJson.name),
      labelsPath: relConverterPath(sub, bundle.labels.name),
      labelsFormat
    });
  } catch (e) {
    await cleanupPath(targetWorkspaceId, sub);
    throw new WorkspaceImportError('phase-failed', errorMessage(e), {
      phase: 'converting-tfjs',
      cause: e
    });
  }

  let terminal: ConvertTerminal;
  try {
    terminal = await awaitConvertTerminal(convertStart, signal, (ev) => onConvertEvent?.(ev));
    if (terminal.publishedRevisionId !== null) {
      bumpRevision(revisionTracker, terminal.publishedRevisionId);
    }
  } finally {
    await cleanupPath(targetWorkspaceId, sub);
  }

  const outcome = terminalToOutcome(convertStart.head_id, terminal);
  summary.heads.push(outcome);
  summary.latestRevisionId = revisionTracker.latest;
  if (outcome.outcome === 'failed') {
    throw new WorkspaceImportError('phase-failed', outcome.error ?? 'TFJS convert failed.', {
      phase: 'converting-tfjs'
    });
  }
  return summary;
}

/// Slice payload inlined so the upload pool need not keep the bucket map. `mode` is telemetry only:
/// all three modes share the upload path; `'replace'`'s pre-flight already ran in Phase 0.
interface UploadRow {
  sourceName: string;
  targetName: string;
  mode: 'new' | 'merge' | 'replace';
  slices: readonly { filename: string; bytes: Uint8Array }[];
}

async function replaceCategories(
  workspaceId: Uuid,
  rows: readonly DatasetImportRow[],
  signal: AbortSignal | undefined,
  onprogress: ((p: WorkspaceImportProgress) => void) | undefined,
  ondeleteCategory: ((name: string) => Promise<void>) | undefined
): Promise<void> {
  emit(onprogress, {
    phase: 'replacing-categories',
    itemsDone: 0,
    itemsTotal: rows.length
  });
  for (let i = 0; i < rows.length; i++) {
    throwIfAborted(signal);
    const row = rows[i];
    emit(onprogress, {
      phase: 'replacing-categories',
      itemsDone: i,
      itemsTotal: rows.length,
      category: row.targetName,
      sourceCategory: row.sourceName
    });
    try {
      if (ondeleteCategory) {
        await ondeleteCategory(row.targetName);
      } else {
        // No hook wired: raw daemon delete via the global queue; UI reconciles only on next poll.
        await enqueueDelete(async () => {
          const ack = await assets.deleteCategory(workspaceId, row.targetName);
          await awaitJobTerminal(ack.job_id);
        });
      }
    } catch (e) {
      // 404 == already wiped == success; anything else aborts (else workspace partly replaced).
      if (isNotFound(e)) continue;
      throw new WorkspaceImportError(
        'phase-failed',
        `Could not replace category "${row.targetName}": ${errorMessage(e)}`,
        { phase: 'replacing-categories', cause: e }
      );
    }
  }
  emit(onprogress, {
    phase: 'replacing-categories',
    itemsDone: rows.length,
    itemsTotal: rows.length
  });
}

async function uploadDatasets(
  workspaceId: Uuid,
  rows: readonly UploadRow[],
  summary: WorkspaceImportSummary,
  revisionTracker: { latest: number | null },
  signal: AbortSignal | undefined,
  onprogress: ((p: WorkspaceImportProgress) => void) | undefined
): Promise<void> {
  const totalSlices = rows.reduce((acc, r) => acc + r.slices.length, 0);
  // Keyed by source (unique per row) not target: several sources may merge into one target, but each
  // row keeps its own outcome record.
  const outcomesBySource = new Map<string, CategoryOutcomeRecord>();
  for (const r of rows) {
    outcomesBySource.set(r.sourceName, {
      category: r.targetName,
      sourceCategory: r.sourceName,
      mode: r.mode,
      uploaded: 0,
      failed: 0,
      error: null
    });
  }
  if (totalSlices === 0) {
    for (const r of outcomesBySource.values()) summary.categories.push(r);
    return;
  }

  emit(onprogress, {
    phase: 'uploading-datasets',
    itemsDone: 0,
    itemsTotal: totalSlices
  });

  // Flatten so the pool fans across categories evenly; each item carries source (outcome key) and
  // target (URL component) so a failure attributes without re-walking.
  interface Item {
    sourceName: string;
    targetName: string;
    filename: string;
    bytes: Uint8Array;
  }
  const items: Item[] = [];
  for (const r of rows) {
    for (const s of r.slices) {
      items.push({
        sourceName: r.sourceName,
        targetName: r.targetName,
        filename: s.filename,
        bytes: s.bytes
      });
    }
  }

  let done = 0;
  const pool = new UploadPool(3);
  await Promise.all(
    items.map((item) =>
      pool.submit(async () => {
        if (signal?.aborted ?? false) return;
        // Pre-emit: flips the row to 'uploading' as a slot opens; counters reflect PRIOR items in
        // this source (this one folds in at the post-finally emit).
        const preOut = outcomesBySource.get(item.sourceName);
        emit(onprogress, {
          phase: 'uploading-datasets',
          itemsDone: done,
          itemsTotal: totalSlices,
          category: item.targetName,
          sourceCategory: item.sourceName,
          categoryUploaded: preOut?.uploaded ?? 0,
          categoryFailed: preOut?.failed ?? 0
        });
        const url = assets.slicePutPath(workspaceId, item.targetName, item.filename);
        const body = new Blob([item.bytes as Uint8Array<ArrayBuffer>], { type: 'audio/wav' });
        try {
          const receipt = await xhrPut<AssetReceipt>({
            url,
            body,
            contentType: 'audio/wav',
            signal
          });
          bumpRevision(revisionTracker, receipt.workspace_revision_id);
          // Filenames are `<sha256>.wav`; a receipt sha256 != basename means transit corruption.
          const expectedSha = item.filename.replace(/\.wav$/, '');
          if (receipt.sha256 !== expectedSha) {
            recordSliceFailure(
              outcomesBySource,
              item.sourceName,
              `Daemon receipt sha256 (${receipt.sha256}) did not match slice id (${expectedSha}).`
            );
            return;
          }
          const out = outcomesBySource.get(item.sourceName);
          if (out) out.uploaded++;
        } catch (e) {
          if (signal?.aborted ?? false) return;
          recordSliceFailure(outcomesBySource, item.sourceName, errorMessage(e));
        } finally {
          done++;
          // Post-emit: re-read because both branches above mutated `outcomesBySource` synchronously,
          // so counters now include this item.
          const postOut = outcomesBySource.get(item.sourceName);
          emit(onprogress, {
            phase: 'uploading-datasets',
            itemsDone: done,
            itemsTotal: totalSlices,
            category: item.targetName,
            sourceCategory: item.sourceName,
            categoryUploaded: postOut?.uploaded ?? 0,
            categoryFailed: postOut?.failed ?? 0
          });
        }
      })
    )
  );

  if (signal?.aborted ?? false) {
    throw new WorkspaceImportError('aborted', 'Import aborted.', { phase: 'uploading-datasets' });
  }
  for (const r of outcomesBySource.values()) summary.categories.push(r);
}

function recordSliceFailure(
  outcomesBySource: Map<string, CategoryOutcomeRecord>,
  sourceName: string,
  message: string
): void {
  const r = outcomesBySource.get(sourceName);
  if (!r) return;
  r.failed++;
  r.error ??= message;
}

async function importHeads(
  workspaceId: Uuid,
  heads: readonly HeadBucket[],
  replaceIds: ReadonlySet<Uuid>,
  summary: WorkspaceImportSummary,
  revisionTracker: { latest: number | null },
  signal: AbortSignal | undefined,
  onprogress: ((p: WorkspaceImportProgress) => void) | undefined,
  onConvertEvent: ((headId: Uuid, ev: ConvertEvent) => void) | undefined
): Promise<void> {
  emit(onprogress, {
    phase: 'importing-heads',
    itemsDone: 0,
    itemsTotal: heads.length
  });

  for (let i = 0; i < heads.length; i++) {
    const head = heads[i];
    if (signal?.aborted ?? false) {
      throw new WorkspaceImportError('aborted', 'Import aborted.', { phase: 'importing-heads' });
    }
    const headId = head.headId;
    const subPath = `converters/alpkg/${headId}`;
    // Shared by the convert sub-phase ticks and the cleanup bracket so both render the same row.
    const emitPhase = (phase: HeadImportPhase): void => {
      emit(onprogress, {
        phase: 'importing-heads',
        itemsDone: i,
        itemsTotal: heads.length,
        headIndex: i,
        headPhase: phase
      });
    };
    let outcome: HeadOutcomeRecord;
    try {
      outcome = await runOneHead(
        workspaceId,
        head,
        subPath,
        replaceIds.has(headId),
        revisionTracker,
        signal,
        emitPhase,
        (ev) => onConvertEvent?.(headId, ev)
      );
    } catch (e) {
      // Per-head failures land on the summary and the walk continues; only an abort re-throws.
      if (e instanceof WorkspaceImportError && e.kind === 'aborted') throw e;
      outcome = {
        headId,
        outcome: 'failed',
        publishedSha256: null,
        error: errorMessage(e)
      };
    } finally {
      // Runs on every branch so orphan converter intake never lingers; emit before cleanup so the
      // phase label brackets the work.
      emitPhase('cleaning-up');
      await cleanupPath(workspaceId, subPath);
    }
    summary.heads.push(outcome);
    emit(onprogress, {
      phase: 'importing-heads',
      itemsDone: i + 1,
      itemsTotal: heads.length,
      headIndex: i,
      headPhase: 'done'
    });
  }
}

async function runOneHead(
  workspaceId: Uuid,
  head: HeadBucket,
  subPath: string,
  replaceOnConflict: boolean,
  revisionTracker: { latest: number | null },
  signal: AbortSignal | undefined,
  onPhase: (phase: HeadImportPhase) => void,
  onConvertEvent: (ev: ConvertEvent) => void
): Promise<HeadOutcomeRecord> {
  const headId = head.headId;
  const manifestPath = `${subPath}/${headId}.json`;
  const mpkPath = `${subPath}/${headId}.mpk`;

  // Manifest before weights so the convert worker's manifest parse can't race a still-streaming
  // `.mpk`.
  onPhase('uploading-files');
  throwIfAborted(signal);
  await putBytes(workspaceId, manifestPath, head.manifestBytes, 'application/json', signal);
  throwIfAborted(signal);
  await putBytes(workspaceId, mpkPath, head.weights, 'application/octet-stream', signal);

  onPhase('starting-convert');
  throwIfAborted(signal);
  let start = await converter.startAlpkg(workspaceId, {
    manifestPath: relConverterPath(subPath, `${headId}.json`)
  });

  onPhase('awaiting-terminal');
  let terminal = await awaitConvertTerminal(start, signal, onConvertEvent);
  let replaced = false;

  // Pre-authorised head_id_collision: delete the existing head and re-run; converter inputs are
  // still on disk at the same paths.
  if (terminal.conflict && replaceOnConflict) {
    onPhase('starting-convert');
    throwIfAborted(signal);
    await enqueueDelete(() => headsApi.delete(workspaceId, headId));
    start = await converter.startAlpkg(workspaceId, {
      manifestPath: relConverterPath(subPath, `${headId}.json`)
    });
    onPhase('awaiting-terminal');
    terminal = await awaitConvertTerminal(start, signal, onConvertEvent);
    replaced = true;
  }

  if (terminal.publishedRevisionId !== null) {
    bumpRevision(revisionTracker, terminal.publishedRevisionId);
  }
  // No `'cleaning-up'` emit here: the caller's finally brackets the cleanup work.
  return terminalToOutcome(headId, terminal, { replaced });
}

async function putBytes(
  workspaceId: Uuid,
  workspaceRootedPath: string,
  bytes: Uint8Array,
  contentType: string,
  signal: AbortSignal | undefined
): Promise<void> {
  const body = new Blob([bytes as Uint8Array<ArrayBuffer>], { type: contentType });
  await xhrPut<AssetReceipt>({
    url: assetPutUrl(workspaceId, workspaceRootedPath),
    body,
    contentType,
    signal
  });
}

/// Outcome of one convert job's SSE subscription, distilled from the typed `ConvertEvent` stream.
interface ConvertTerminal {
  /// `'skipped'` = alpkg idempotent no-op (same id+sha on disk); `'failed'` also covers a stream
  /// closing without a terminal.
  outcome: 'imported' | 'skipped' | 'failed';
  /// Lowercase-hex sha256 of the published head (success + skip); null on failure.
  publishedSha256: string | null;
  /// `workspace_revision.id` from `head_published`; null on failure.
  publishedRevisionId: number | null;
  error: string | null;
  /// Set on a `head_id_collision` terminal; drives the Replace branch (or a conflict row).
  conflict: {
    headId: Uuid;
    storedSha256: string;
    incomingSha256: string;
  } | null;
}

// EventSource stuck in CONNECTING auto-retries and our `error` handler ignores that state, so this
// escape hatch keeps a never-connecting stream (broken DNS, RST loop, captive portal) from pinning
// the dialog forever. Generous because convert is long-running; cleared on the first `open`.
const CONVERT_CONNECT_TIMEOUT_MS = 30_000;

async function awaitConvertTerminal(
  start: ConvertStartResp,
  signal: AbortSignal | undefined,
  onEvent: (ev: ConvertEvent) => void
): Promise<ConvertTerminal> {
  // Terminal classification keys off the last typed ConvertEvent, not the loose `JobEvent.message`:
  // `head_published` carries `idempotent_skip`, `job_failed` carries `category`.
  let lastConvertEvent: ConvertEvent | null = null;
  let publishedSha: string | null = null;
  let publishedRev: number | null = null;
  let idempotentSkip = false;
  return new Promise<ConvertTerminal>((resolve, reject) => {
    const url = jobsApi.eventsUrl(start.job_id, { logs: true });
    // `apiUrl` prefixes the backend origin for a cross-origin SPA (same-origin passes through).
    const source = new EventSource(apiUrl(url));
    let closed = false;
    let connectTimer: ReturnType<typeof setTimeout> | null = null;
    const clearConnectTimer = (): void => {
      if (connectTimer !== null) {
        clearTimeout(connectTimer);
        connectTimer = null;
      }
    };
    const closeAll = (): void => {
      if (closed) return;
      closed = true;
      clearConnectTimer();
      source.close();
    };
    const onAbort = (): void => {
      closeAll();
      reject(new WorkspaceImportError('aborted', 'Import aborted.'));
    };
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }
    source.addEventListener('open', clearConnectTimer);
    connectTimer = setTimeout(() => {
      if (closed) return;
      signal?.removeEventListener('abort', onAbort);
      closeAll();
      resolve({
        outcome: 'failed',
        publishedSha256: null,
        publishedRevisionId: null,
        error: 'Timed out connecting to the convert event stream.',
        conflict: null
      });
    }, CONVERT_CONNECT_TIMEOUT_MS);
    source.addEventListener('job', (e: MessageEvent) => {
      if (closed) return;
      const data = e.data as string;
      let envelope: { state?: string; message?: string };
      try {
        envelope = JSON.parse(data) as { state?: string; message?: string };
      } catch {
        return;
      }
      // `message` is a JSON-stringified ConvertEvent on log-line frames; non-log frames omit it.
      if (typeof envelope.message === 'string') {
        try {
          const parsed = JSON.parse(envelope.message) as ConvertEvent;
          if (typeof (parsed as { kind?: unknown }).kind === 'string') {
            lastConvertEvent = parsed;
            onEvent(parsed);
            if (parsed.kind === 'head_published') {
              publishedSha = parsed.head_sha256;
              idempotentSkip = parsed.idempotent_skip;
              publishedRev = parsed.workspace_revision.id;
            }
          }
        } catch {
          // Forward-compat: skip unknown/malformed payloads.
        }
      }
      const state = envelope.state;
      if (!isTerminal(state as JobState | undefined)) return;
      signal?.removeEventListener('abort', onAbort);
      closeAll();
      if (state === 'succeeded') {
        resolve({
          outcome: idempotentSkip ? 'skipped' : 'imported',
          publishedSha256: publishedSha,
          publishedRevisionId: publishedRev,
          error: null,
          conflict: null
        });
        return;
      }
      // Terminal failure: head_id_collision gets a conflict payload, else plain failed.
      if (
        lastConvertEvent?.kind === 'job_failed' &&
        lastConvertEvent.category === 'head_id_collision'
      ) {
        resolve({
          outcome: 'failed',
          publishedSha256: null,
          publishedRevisionId: null,
          error: lastConvertEvent.error,
          conflict: {
            headId: lastConvertEvent.head_id,
            storedSha256: lastConvertEvent.stored_sha256,
            incomingSha256: lastConvertEvent.got_sha256
          }
        });
        return;
      }
      const message =
        lastConvertEvent?.kind === 'job_failed'
          ? lastConvertEvent.error
          : (envelope.message ?? `convert ${state ?? 'ended without success'}`);
      resolve({
        outcome: 'failed',
        publishedSha256: null,
        publishedRevisionId: null,
        error: message,
        conflict: null
      });
    });
    source.addEventListener('error', () => {
      if (closed) return;
      if (source.readyState === EventSource.CLOSED) {
        signal?.removeEventListener('abort', onAbort);
        closeAll();
        resolve({
          outcome: 'failed',
          publishedSha256: null,
          publishedRevisionId: null,
          error: 'Event stream closed before terminal state.',
          conflict: null
        });
      }
    });
  });
}

/// `replaced` flips an `imported` outcome to `replaced` (conflict branch deleted the existing head).
function terminalToOutcome(
  headId: Uuid,
  terminal: ConvertTerminal,
  options: { replaced?: boolean } = {}
): HeadOutcomeRecord {
  if (terminal.outcome === 'imported') {
    return {
      headId,
      outcome: options.replaced === true ? 'replaced' : 'imported',
      publishedSha256: terminal.publishedSha256,
      error: null
    };
  }
  if (terminal.outcome === 'skipped') {
    return {
      headId,
      outcome: 'skipped',
      publishedSha256: terminal.publishedSha256,
      error: null
    };
  }
  const base: HeadOutcomeRecord = {
    headId,
    outcome: 'failed',
    publishedSha256: null,
    error: terminal.error
  };
  if (terminal.conflict) base.conflict = terminal.conflict;
  return base;
}

function bumpRevision(tracker: { latest: number | null }, observed: number): void {
  if (tracker.latest === null || observed > tracker.latest) tracker.latest = observed;
}

function assetPutUrl(workspaceId: Uuid, workspaceRootedPath: string): string {
  const encoded = workspaceRootedPath
    .split('/')
    .map((seg) => encodeURIComponent(seg))
    .join('/');
  return `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/${encoded}`;
}

function relConverterPath(workspaceRootedDir: string, filename: string): string {
  // Convert request bodies want converter-rooted paths; our paths keep the `converters/` prefix for
  // the asset PUT URL, so strip it here.
  const stripped = workspaceRootedDir.replace(/^converters\//, '');
  return `${stripped}/${filename}`;
}

async function cleanupPath(workspaceId: Uuid, workspaceRootedPath: string): Promise<void> {
  // Via the global queue because the daemon's delete slot is single-tenant (parallel deletes 409).
  // Failures swallowed: the head has landed and daemon housekeeping reaps orphan inputs.
  try {
    await enqueueDelete(async () => {
      const ack = await assets.deletePath(workspaceId, workspaceRootedPath);
      // Fire-and-forget: don't await the SSE terminal; a later sweep reaps straggler bytes.
      void ack;
    });
  } catch {
    // intentional swallow
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted ?? false) {
    throw new WorkspaceImportError('aborted', 'Import aborted.');
  }
}

function emit(
  onprogress: ((p: WorkspaceImportProgress) => void) | undefined,
  p: WorkspaceImportProgress
): void {
  if (onprogress) onprogress(p);
}

function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message || `HTTP ${String(e.status)}`;
  if (e instanceof Error) return e.message;
  if (typeof e === 'string') return e;
  return 'Import failed.';
}
