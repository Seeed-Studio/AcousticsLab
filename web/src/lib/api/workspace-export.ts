// Workspace `.alpkg` export: package.json + always-verbatim workspace.json + datasets/<cat>/<sha256>.wav + heads/<id>.{mpk,json}; requires >=1 of categories/heads (either section may be empty), workspace.json fetched once first.

import type { HeadRecord, Uuid } from './types';
import {
  buildDatasetEntries,
  DatasetEntriesError,
  type DatasetEntriesProgress
} from './datasets-export';
import { buildHeadEntries, ExportError, fetchWorkspaceCoreEntry } from './heads-export';
import {
  buildAlpkgManifest,
  packAlpkg,
  safeFilenameSlug,
  type AlpkgEntry,
  type AlpkgManifest
} from '$lib/utils/alpkg';

export type WorkspaceExportPhase =
  'preparing-workspace' | 'preparing-datasets' | 'preparing-heads' | 'packing' | 'downloading';

export interface WorkspaceExportProgress {
  phase: WorkspaceExportPhase;
  /// Set only during `preparing-datasets` (fetching subphase) and `preparing-heads`.
  itemsTotal?: number;
  itemsDone?: number;
  /// `preparing-datasets` only: listing vs fetching subphase, for copy.
  subphase?: 'listing' | 'fetching';
}

/// One shape the dialog switches on for banner copy, wrapping dataset/heads errors.
export class WorkspaceExportError extends Error {
  readonly phase: WorkspaceExportPhase;
  /// Category that triggered a dataset-side failure; null for head-side/workspace-wide.
  readonly category: string | null;
  readonly headId: Uuid | null;
  constructor(
    phase: WorkspaceExportPhase,
    message: string,
    options?: { cause?: unknown; category?: string; headId?: Uuid }
  ) {
    super(message, options);
    this.name = 'WorkspaceExportError';
    this.phase = phase;
    this.category = options?.category ?? null;
    this.headId = options?.headId ?? null;
  }
}

export interface WorkspaceExportInput {
  workspaceId: Uuid;
  workspaceName: string;
  categories: readonly string[];
  heads: readonly HeadRecord[];
}

export interface WorkspaceExportOptions {
  signal?: AbortSignal;
  onprogress?: (p: WorkspaceExportProgress) => void;
}

export interface WorkspaceExportResult {
  filename: string;
  size_bytes: number;
  /// Categories that actually landed; below `input.categories.length` when one was empty on disk.
  categories_count: number;
  slices_count: number;
  heads_count: number;
}

export async function exportWorkspace(
  input: WorkspaceExportInput,
  opts: WorkspaceExportOptions = {}
): Promise<WorkspaceExportResult> {
  const { signal, onprogress } = opts;
  const { workspaceId, workspaceName, categories, heads: headList } = input;

  if (categories.length === 0 && headList.length === 0) {
    // Backstops the dialog's Export-button guard for non-UI callers.
    throw new WorkspaceExportError('preparing-datasets', 'Pick at least one item to export.');
  }

  // Phase 0: embed verbatim workspace.json so an importer recovers id/name/tags/revision/head_count without `GET /workspaces`; fetched first so a 404 attributes to a gone workspace, not a stale dataset/heads row.
  emit(onprogress, { phase: 'preparing-workspace' });
  let workspaceCoreEntry: AlpkgEntry;
  // Read from the same workspace.json parse as the embedded copy so the filename's `rev_<N>` can't drift from the archived payload (a caller-passed revision could).
  let workspaceRevisionId: number;
  try {
    const result = await fetchWorkspaceCoreEntry(workspaceId, signal);
    workspaceCoreEntry = result.entry;
    workspaceRevisionId = result.workspaceRevisionId;
  } catch (e) {
    if (e instanceof ExportError) {
      throw new WorkspaceExportError('preparing-workspace', e.message, { cause: e });
    }
    throw e;
  }

  // Phase 1: skipped for heads-only to avoid a spurious phase transition + emit.
  let datasetEntries: AlpkgEntry[] = [];
  let resolvedCategoriesCount = 0;
  if (categories.length > 0) {
    try {
      datasetEntries = await buildDatasetEntries(
        workspaceId,
        categories,
        signal,
        (p: DatasetEntriesProgress) => {
          emit(onprogress, {
            phase: 'preparing-datasets',
            subphase: p.phase,
            itemsTotal: p.itemsTotal,
            itemsDone: p.itemsDone
          });
        }
      );
    } catch (e) {
      if (e instanceof DatasetEntriesError) {
        throw new WorkspaceExportError('preparing-datasets', e.message, {
          cause: e,
          category: e.category ?? undefined
        });
      }
      throw e;
    }
    // Helper silently drops empty/404 categories; recover survivors from entry paths `datasets/<cat>/<filename>` (hence `split('/')[1]`).
    const seen = new Set<string>();
    for (const entry of datasetEntries) {
      seen.add(entry.path.split('/')[1]);
    }
    resolvedCategoriesCount = seen.size;
  }

  // Phase 2: skipped when no heads selected; adapts buildHeadEntries' (done, total) callback to the workspace progress shape.
  let headEntries: AlpkgEntry[] = [];
  if (headList.length > 0) {
    emit(onprogress, {
      phase: 'preparing-heads',
      itemsTotal: headList.length,
      itemsDone: 0
    });
    try {
      headEntries = await buildHeadEntries(workspaceId, headList, signal, (done, total) => {
        emit(onprogress, {
          phase: 'preparing-heads',
          itemsTotal: total,
          itemsDone: done
        });
      });
    } catch (e) {
      if (e instanceof ExportError) {
        throw new WorkspaceExportError('preparing-heads', e.message, {
          cause: e,
          headId: e.headId ?? undefined
        });
      }
      throw e;
    }
  }

  // Both helpers drop empty inputs silently; refuse rather than ship a metadata-only archive.
  if (datasetEntries.length === 0 && headEntries.length === 0) {
    throw new WorkspaceExportError(
      'preparing-datasets',
      categories.length > 0 && headList.length === 0
        ? 'The selected categories have no slices to export.'
        : 'Nothing to export.'
    );
  }

  // Phase 3: order is contractual -- package.json first so a streaming reader detects kind without seeking the central directory, then workspace.json, datasets (helper-sorted), heads (input order); re-exporting unchanged state is byte-identical modulo `exported_at`.
  emit(onprogress, { phase: 'packing' });
  throwIfAborted(signal, 'packing');
  const pkg: AlpkgManifest = buildAlpkgManifest();
  const pkgBytes = new TextEncoder().encode(stringifyAlpkgManifest(pkg));
  const entries: AlpkgEntry[] = [
    { path: 'package.json', bytes: pkgBytes },
    workspaceCoreEntry,
    ...datasetEntries,
    ...headEntries
  ];

  let alpkgBlob: Blob;
  try {
    alpkgBlob = await packAlpkg(entries);
  } catch (e) {
    throw new WorkspaceExportError('packing', "Couldn't pack the workspace archive.", {
      cause: e
    });
  }

  // Phase 4: runs on the operator's gesture chain so popup blockers cooperate.
  emit(onprogress, { phase: 'downloading' });
  throwIfAborted(signal, 'downloading');
  const filename = buildExportFilename(workspaceName, workspaceId, workspaceRevisionId);
  triggerDownload(alpkgBlob, filename);

  return {
    filename,
    size_bytes: alpkgBlob.size,
    categories_count: resolvedCategoriesCount,
    slices_count: datasetEntries.length,
    // From the selection, not `headEntries.length / 2`, so a per-head sidecar can't skew it.
    heads_count: headList.length
  };
}

function stringifyAlpkgManifest(pkg: AlpkgManifest): string {
  // Two-space + trailing newline matches the per-row head export for identical unzipped bytes.
  return JSON.stringify(pkg, null, 2) + '\n';
}

function buildExportFilename(workspaceName: string, workspaceId: Uuid, revisionId: number): string {
  // `<ws-slug>-<wsId8>-rev_<N>.alpkg`: wsId8 (first 8 UUID hex) keeps recreated same-named workspaces distinct, rev_<N> gives different-revision exports distinct download entries instead of OS " (1)" suffixing; wsId8 skips `safeFilenameSlug` (daemon UUID is already hex-clean).
  const wsSlug = safeFilenameSlug(workspaceName, 'workspace');
  const wsIdSlug = workspaceId.replace(/-/g, '').slice(0, 8);
  return `${wsSlug}-${wsIdSlug}-rev_${revisionId}.alpkg`;
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
  // Defer revoke so the download starts before the blob is GC'd.
  setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 30_000);
}

function throwIfAborted(signal: AbortSignal | undefined, phase: WorkspaceExportPhase): void {
  if (signal?.aborted === true) {
    const reason: unknown = signal.reason;
    const reasonMsg =
      reason instanceof Error
        ? reason.message
        : typeof reason === 'string'
          ? reason
          : 'export aborted';
    throw new WorkspaceExportError(phase, reasonMsg, { cause: reason });
  }
}

function emit(
  onprogress: ((p: WorkspaceExportProgress) => void) | undefined,
  p: WorkspaceExportProgress
): void {
  if (onprogress) onprogress(p);
}
