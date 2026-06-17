<script lang="ts">
  import { onDestroy, tick, untrack } from 'svelte';
  import { SvelteMap, SvelteSet } from 'svelte/reactivity';
  import { fade, slide } from 'svelte/transition';
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import UploadIcon from '$lib/components/ui/UploadIcon.svelte';
  import { workspaces } from '$lib/stores/workspaces.svelte';
  import { categories } from '$lib/stores/categories.svelte';
  import { slices as slicesStore } from '$lib/stores/slices.svelte';
  import { config as configStore } from '$lib/stores/config.svelte';
  import { heads as headsApi } from '$lib/api/endpoints';
  import {
    importAlpkg,
    importTfjs,
    WorkspaceImportError,
    type DatasetImportRow,
    type HeadImportPhase,
    type HeadOutcome,
    type HeadOutcomeRecord,
    type WorkspaceImportProgress,
    type WorkspaceImportSummary
  } from '$lib/api/workspace-import';
  import {
    classifyAlpkgEntries,
    unpackAlpkg,
    extractZipEntries,
    AlpkgUnpackError,
    type AlpkgUnpackResult,
    type ClassifiedAlpkg,
    type DatasetBucket,
    type ExtractedZipEntry,
    type HeadBucket
  } from '$lib/utils/alpkg-unpack';
  import {
    classifyTfjsBundle,
    parseTfjsLabels,
    type ClassifiedTfjsBundle
  } from '$lib/utils/tfjs-classify';
  import { prettyCategoryName } from '$lib/components/category/labels';
  import { validateCategoryName } from '$lib/components/category/name-validate';
  import { errorCopy } from '$lib/utils/error-copy';
  import { formatBytes } from '$lib/utils/format';
  import { formatAbsolute, formatRelative } from '$lib/utils/time';
  import { m } from '$lib/i18n';
  import { validateWorkspaceName } from './name-validate';
  import { inputClass } from '$lib/components/ui/inputClass';
  import type {
    ConvertEvent,
    ConvertStage,
    HeadRecord,
    Uuid,
    WorkspaceMutationResp
  } from '$lib/api/types';

  // Step machine: pick-file -> (pick-target?) -> summary -> running -> done. `into-current` locks the
  // target to the prop and skips pick-target; a mid-pipeline failure rolls back to `summary` with the
  // selection preserved so the operator retries without re-dropping the file.
  type Mode = 'into-current' | 'pick-target';
  type Step = 'pick-file' | 'pick-target' | 'summary' | 'running' | 'done';
  type Branch = 'alpkg' | 'tfjs';
  /// `'skip'` reaches the orchestrator by omission. `'new'`/`'merge'` are identical at the daemon
  /// (additive PUT); the split is display telemetry only.
  type DatasetMode = 'new' | 'merge' | 'replace' | 'skip';

  interface Props {
    open: boolean;
    mode: Mode;
    /// Required in `into-current`; unused in `pick-target` (resolved via the picker pane).
    workspaceId?: Uuid;
    /// Title-strip copy in `into-current`.
    workspaceName?: string;
    onclose: () => void;
    onimported?: (workspaceId: Uuid) => void;
  }
  let {
    open,
    mode,
    workspaceId: lockedWorkspaceId,
    workspaceName: lockedWorkspaceName,
    onclose,
    onimported
  }: Props = $props();

  let step = $state<Step>('pick-file');
  let branch = $state<Branch | null>(null);

  let dragOver = $state(false);
  let fileError = $state<string | null>(null);
  let parsingFile = $state(false);

  // Drop/pick caps catch bulk-drag mistakes and tab-locking shapes (a multi-GiB `.alpkg` materialised
  // via `blob.arrayBuffer()`) before the daemon would; the byte caps mirror the unpacker's zip-bomb
  // caps (256 MiB/entry, 512 MiB total) so an accepted drop is never rejected later, while the 100-file
  // cap is a stricter bulk-drag guard than the unpacker's 1024-entry ceiling.
  const MAX_FILES_PER_DROP = 100;
  const MAX_FILE_BYTES = 256 * 1024 * 1024;
  const MAX_TOTAL_BYTES = 512 * 1024 * 1024;

  let alpkg = $state<AlpkgUnpackResult | null>(null);
  let alpkgClassified = $state<ClassifiedAlpkg | null>(null);
  let tfjs = $state<ClassifiedTfjsBundle | null>(null);

  // TFJS bundles often arrive across multiple drops (model.json, shards, labels in separate folders),
  // so accumulate File handles and re-classify the merged set every drop, advancing past step 1 only
  // once `classifyTfjsBundle` reports `ready: true`. Dedupe key `(name, size, lastModified)` catches
  // re-dragging the same folder without colliding distinct same-name files. An ALPKG drop or Clear
  // wipes this (single-archive flow).
  let tfjsStagedFiles = $state<File[]>([]);

  type TfjsStagedRole = 'model' | 'shard' | 'labels' | 'unknown';
  interface TfjsStagedFileRow {
    key: string;
    file: File;
    role: TfjsStagedRole;
  }

  function tfjsStagedKey(f: File): string {
    return `${f.name}::${String(f.size)}::${String(f.lastModified)}`;
  }

  function mergeTfjsFiles(existing: readonly File[], incoming: readonly File[]): File[] {
    // Map insertion order keeps existing files in place and appends new ones (stable staging panel
    // across repeat drops). Native Map (not SvelteMap): pure dedup aid, no reactive reader.
    // eslint-disable-next-line svelte/prefer-svelte-reactivity
    const seen = new Map<string, File>();
    for (const f of existing) seen.set(tfjsStagedKey(f), f);
    for (const f of incoming) seen.set(tfjsStagedKey(f), f);
    return Array.from(seen.values());
  }

  /// Merge incoming files into the staged set, enforcing CUMULATIVE caps (the per-drop check bounds
  /// only each incoming batch, so two 90-file/256 MiB drops would otherwise overflow). Returns
  /// `false` on rejection, staging UNCHANGED and `fileError` set.
  function tryStageMergedTfjsFiles(incoming: readonly File[]): boolean {
    const merged = mergeTfjsFiles(tfjsStagedFiles, incoming);
    if (merged.length > MAX_FILES_PER_DROP) {
      fileError = m.workspace.import_dialog.pick_file.error_tfjs_merged_file_count(
        merged.length,
        MAX_FILES_PER_DROP
      );
      return false;
    }
    let mergedBytes = 0;
    for (const f of merged) mergedBytes += f.size;
    if (mergedBytes > MAX_TOTAL_BYTES) {
      fileError = m.workspace.import_dialog.pick_file.error_tfjs_merged_bytes(
        formatBytes(mergedBytes),
        formatBytes(MAX_TOTAL_BYTES)
      );
      return false;
    }
    tfjsStagedFiles = merged;
    return true;
  }

  // Group rows by role (model -> shards -> labels -> unknown); all-`'unknown'` while `tfjs` is null
  // (pre-classification) so the row count never lags the file count.
  const tfjsStagedFileRows = $derived.by((): TfjsStagedFileRow[] => {
    if (tfjsStagedFiles.length === 0) return [];
    if (!tfjs) {
      return tfjsStagedFiles.map((f) => ({
        key: tfjsStagedKey(f),
        file: f,
        role: 'unknown'
      }));
    }
    const rows: TfjsStagedFileRow[] = [];
    // Native Set (not SvelteSet): local membership check, no reactive reader.
    // eslint-disable-next-line svelte/prefer-svelte-reactivity
    const accountedFor = new Set<File>();
    if (tfjs.modelJson) {
      rows.push({ key: tfjsStagedKey(tfjs.modelJson), file: tfjs.modelJson, role: 'model' });
      accountedFor.add(tfjs.modelJson);
    }
    for (const f of tfjs.shards) {
      rows.push({ key: tfjsStagedKey(f), file: f, role: 'shard' });
      accountedFor.add(f);
    }
    if (tfjs.labels) {
      rows.push({ key: tfjsStagedKey(tfjs.labels), file: tfjs.labels, role: 'labels' });
      accountedFor.add(tfjs.labels);
    }
    for (const f of tfjs.unknown) {
      rows.push({ key: tfjsStagedKey(f), file: f, role: 'unknown' });
      accountedFor.add(f);
    }
    // A staged file the classifier didn't return (e.g. a duplicate `model.json` flagged as a blocker
    // without surfacing in `modelJson`) shows as `unknown` so every staged byte appears.
    for (const f of tfjsStagedFiles) {
      if (!accountedFor.has(f)) {
        rows.push({ key: tfjsStagedKey(f), file: f, role: 'unknown' });
      }
    }
    return rows;
  });

  function clearTfjsStaging(): void {
    tfjsStagedFiles = [];
    tfjs = null;
    if (branch === 'tfjs') branch = null;
    fileError = null;
  }

  // ZIP entries -> `File` handles for the classifier. Basename becomes `File.name` (classifier keys
  // off lowercase basename, so wrapper folders are transparent); `lastModified` 0 keeps the dedupe key
  // stable across re-extracts; empty entries kept so the classifier can flag a zero-byte file.
  function zipEntriesToTfjsFiles(entries: readonly ExtractedZipEntry[]): File[] {
    return entries.map((e) => {
      const basename = e.path.split('/').pop() ?? e.path;
      return new File([e.bytes as BlobPart], basename, { lastModified: 0 });
    });
  }

  let targetMode = $state<'use-existing' | 'create-new'>('use-existing');
  let pickedExistingId = $state<Uuid | null>(null);
  let newWorkspaceName = $state('');
  let creatingTarget = $state(false);
  let createTargetError = $state<string | null>(null);

  let pickedListEl = $state<HTMLUListElement | undefined>();

  // Resolved on leaving pick-target, or eagerly in `into-current`; `null` = upstream of resolution.
  let resolvedTargetId = $state<Uuid | null>(null);
  let resolvedTargetName = $state<string>('');

  // Bring the default-picked row into view on list mount (prefill can sit below the `max-h-56` fold).
  // Tracks only `pickedListEl` so it fires on mount but NOT on click-driven `pickedExistingId` changes
  // (untracked read) -- a click never yanks the scroll. Already-visible rows are left alone.
  $effect(() => {
    const el = pickedListEl;
    if (!el) return;
    untrack(() => {
      const id = pickedExistingId;
      if (id === null) return;
      void tick().then(() => {
        const node = el.querySelector<HTMLElement>(`[data-workspace-id="${id}"]`);
        if (!node) return;
        const containerRect = el.getBoundingClientRect();
        const nodeRect = node.getBoundingClientRect();
        const fullyVisible =
          nodeRect.top >= containerRect.top && nodeRect.bottom <= containerRect.bottom;
        if (fullyVisible) return;
        node.scrollIntoView({ block: 'center' });
      });
    });
  });

  // Per-bucket state keyed by SOURCE name (stable across target renames). source -> target rename:
  const datasetTargetNames = new SvelteMap<string, string>();
  // source -> chosen mode; storing only EXPLICIT choices keeps untouched rows' defaults reactive to
  // collision changes. `effectiveDatasetMode` degrades a chosen mode as the operator types.
  const datasetModes = new SvelteMap<string, DatasetMode>();
  // Open-rename-popover flag (at most one entry); a Set so reactivity composes with `.has(name)`.
  const editingRenameSources = new SvelteSet<string>();
  // Per-row DRAFT target (verify-before-save), seeded from the committed target on open and committed
  // back to `datasetTargetNames` only on `saveRenameEdit`, so invalid drafts never reach the committed
  // map. Cancel/Escape/re-clicking the pencil discards it.
  const datasetTargetDrafts = new SvelteMap<string, string>();

  // Open mode-dropdown source (single `string | null`, not a Set, to enforce single-popover discipline
  // structurally). The 120 ms close timer absorbs the cursor crossing the trigger->menu gap.
  let openModeDropdownSource = $state<string | null>(null);
  let modeDropdownCloseTimer: ReturnType<typeof setTimeout> | null = null;

  // Popovers use `position: fixed` (viewport coords) to escape the modal's `overflow: auto` clip.
  // Fixed elements don't scroll with their containing block, so the shared scroll/resize `$effect`
  // re-runs `refreshPopupPositions` while open; each anchor is a plain `let` so the refresher re-reads
  // its rect without subscribing as a reactive dependency.
  interface PopupPos {
    top: number;
    left?: number;
    right?: number;
    width?: number;
  }
  let modeMenuPos = $state<PopupPos | null>(null);
  let renamePopoverPos = $state<PopupPos | null>(null);
  let headPopoverPos = $state<PopupPos | null>(null);
  let tfjsLabelsPopoverPos = $state<PopupPos | null>(null);
  let modeMenuAnchor: HTMLElement | null = null;
  let renamePopoverAnchor: HTMLElement | null = null;
  let headPopoverAnchor: HTMLElement | null = null;
  let tfjsLabelsPopoverAnchor: HTMLElement | null = null;

  // Height estimates (px) used only to pick open direction; CSS produces the real height (slop is
  // harmless -- the "neither side fits" fallback picks the larger side).
  const MODE_MENU_HEIGHT_ESTIMATE = 108;
  const RENAME_POPOVER_HEIGHT_ESTIMATE = 260;
  const HEAD_POPOVER_HEIGHT_ESTIMATE = 180;
  const TFJS_LABELS_POPOVER_HEIGHT_ESTIMATE = 172;
  // Trigger-to-popover gap for ADJACENT popovers (mode menu, head info); overlay popovers have none.
  const POPOVER_TRIGGER_GAP = 4;
  // Min popover-edge-to-viewport-edge gap so the drop-shadow doesn't kiss the screen edge.
  const POPOVER_EDGE_PAD = 8;

  const selectedHeadIds = new SvelteSet<Uuid>();
  // Per-head opt-in to overwrite an existing same-id head, set by the done pane's "Replace existing"
  // affordance on a conflict row before retrying.
  const replaceHeadIds = new SvelteSet<Uuid>();

  // Two-channel per-head "model card" popover: hover=peek, focus=pin (survives the cursor leaving).
  // Hovering a row clears any OTHER row's focus-pin so both never render at once; both cleared in
  // `resetAll` since a mid-hover close may unmount the button before its leave event fires.
  let popoverHoveredHeadId = $state<Uuid | null>(null);
  let popoverFocusedHeadId = $state<Uuid | null>(null);

  // Frontend mirror of the daemon's per-workspace head cap (no API exposes it -- bump in lockstep).
  // Selection ceiling = `cap - active_in_target` (active head pinned, rotation won't displace it); the
  // displacement warning fires when `existing + new > cap` since rotation silently drops the oldest
  // non-active heads.
  const HEAD_HISTORY_CAP = 3;

  let targetExistingHeads = $state<HeadRecord[] | null>(null);
  let targetLoading = $state(false);
  let targetLoadError = $state<string | null>(null);
  // Categories drive the target-collision derived (conflict UI + "skip on collision" default), so a
  // "checking…" line shows while the refresh is in flight.
  let targetCategoriesLoading = $state(false);
  let targetCategoriesLoadError = $state<string | null>(null);

  let abortController = $state<AbortController | null>(null);
  let progress = $state<WorkspaceImportProgress | null>(null);
  let pipelineError = $state<string | null>(null);
  let summary = $state<WorkspaceImportSummary | null>(null);

  // Running + done panes share one per-row layout. `datasetRunStates` keyed by SOURCE name (two
  // sources -> one target stay distinct); `headRunStates`/`headRunLogs` keyed by `headId` (`onprogress`
  // maps `headIndex` -> headId via the ordered selection); `expandedHeadId` is single-disclosure,
  // auto-set to the running head.
  type DatasetRunPhase = 'pending' | 'replacing' | 'uploading' | 'done' | 'failed';
  interface DatasetRunState {
    source: string;
    target: string;
    mode: 'new' | 'merge' | 'replace';
    total: number;
    uploaded: number;
    failed: number;
    phase: DatasetRunPhase;
    error: string | null;
  }
  type HeadRunPhase = 'queued' | HeadImportPhase | 'failed';
  interface HeadRunState {
    headId: Uuid;
    phase: HeadRunPhase;
    outcome: HeadOutcome | null;
    error: string | null;
    conflict: HeadOutcomeRecord['conflict'] | null;
  }
  interface ConvertLogLine {
    timestampMs: number;
    message: string;
  }
  const datasetRunStates = new SvelteMap<string, DatasetRunState>();
  const headRunStates = new SvelteMap<Uuid, HeadRunState>();
  const headRunLogs = new SvelteMap<Uuid, ConvertLogLine[]>();
  let expandedHeadId = $state<Uuid | null>(null);

  // TFJS has no real head_id pre-publish (convert worker assigns one mid-import); the running/done row
  // keys on this nil UUID (never collides a real UUIDv4) and the display swaps the id chip for a "TFJS
  // bundle" label so the sentinel never leaks.
  const TFJS_HEAD_SENTINEL_ID = '00000000-0000-0000-0000-000000000000';

  // Auto-tail: stick to the log floor while the operator is pinned there, release on scroll-up. Single
  // shared state (at most one log panel mounted).
  const LOG_STICK_PX = 4;
  let logScrollEl = $state<HTMLDivElement | undefined>();
  let logStuckToBottom = $state(true);

  function onLogScroll(): void {
    const el = logScrollEl;
    if (!el) return;
    const distance = el.scrollHeight - el.clientHeight - el.scrollTop;
    logStuckToBottom = distance <= LOG_STICK_PX;
  }

  $effect(() => {
    // Reset stuck state on expanded-head change so a freshly-opened log tails on first paint; separate
    // effect from the tail below so the reset always wins on transition.
    void expandedHeadId;
    logStuckToBottom = true;
  });

  $effect(() => {
    // Scroll to floor when new lines arrive AND the operator is pinned at the bottom. Depends on log
    // length; `tick()` waits for the new line to mount before reading `scrollHeight`.
    if (expandedHeadId === null) return;
    const logs = headRunLogs.get(expandedHeadId);
    void logs?.length;
    if (!logStuckToBottom) return;
    const el = logScrollEl;
    if (!el) return;
    void tick().then(() => {
      el.scrollTop = el.scrollHeight;
    });
  });

  let lastOpenSeen = false;
  $effect(() => {
    if (open && !lastOpenSeen) {
      lastOpenSeen = true;
      resetAll();
      void workspaces.refresh();
      if (mode === 'into-current' && lockedWorkspaceId) {
        resolvedTargetId = lockedWorkspaceId;
        resolvedTargetName = lockedWorkspaceName ?? '';
      }
    } else if (!open && lastOpenSeen) {
      lastOpenSeen = false;
      if (abortController) {
        abortController.abort();
        abortController = null;
      }
    }
  });

  function resetAll(): void {
    step = 'pick-file';
    branch = null;
    fileError = null;
    parsingFile = false;
    alpkg = null;
    alpkgClassified = null;
    tfjs = null;
    tfjsStagedFiles = [];
    targetMode = 'use-existing';
    pickedExistingId = null;
    newWorkspaceName = '';
    creatingTarget = false;
    createTargetError = null;
    resolvedTargetId = null;
    resolvedTargetName = '';
    datasetTargetNames.clear();
    datasetModes.clear();
    editingRenameSources.clear();
    datasetTargetDrafts.clear();
    closeModeDropdown();
    selectedHeadIds.clear();
    replaceHeadIds.clear();
    popoverHoveredHeadId = null;
    popoverFocusedHeadId = null;
    // Clear popover caches so a re-opened dialog doesn't flash a stale popover at prior coords.
    modeMenuAnchor = null;
    renamePopoverAnchor = null;
    headPopoverAnchor = null;
    tfjsLabelsPopoverAnchor = null;
    modeMenuPos = null;
    renamePopoverPos = null;
    headPopoverPos = null;
    tfjsLabelsPopoverPos = null;
    tfjsLabelsHovered = false;
    tfjsLabelsFocused = false;
    tfjsLabels = null;
    datasetRunStates.clear();
    headRunStates.clear();
    headRunLogs.clear();
    expandedHeadId = null;
    targetExistingHeads = null;
    targetLoading = false;
    targetLoadError = null;
    targetCategoriesLoading = false;
    targetCategoriesLoadError = null;
    progress = null;
    pipelineError = null;
    summary = null;
    abortController = null;
  }

  // Single-file `.alpkg` -> ALPKG branch; multi-file or non-`.alpkg` -> TFJS branch. Hard-reject >1
  // `.alpkg` or an `.alpkg` mixed with non-archive files: two self-contained archives can't
  // disambiguate a target, and the mix is likely a TFJS selection with an archive slipped in.
  async function handleFiles(files: File[]): Promise<void> {
    // Re-entry guard: the drop label (unlike the `<input>`) can't disable on `parsingFile`, and a
    // second drop mid unpack/classify would race the state machine. Dropped silently.
    if (parsingFile) return;
    if (files.length === 0) {
      fileError = m.workspace.import_dialog.pick_file.error_empty_drop;
      return;
    }
    const alpkgFiles = files.filter((f) => /\.alpkg$/i.test(f.name));
    if (alpkgFiles.length > 1) {
      fileError = m.workspace.import_dialog.pick_file.error_multi_alpkg(alpkgFiles.length);
      return;
    }
    if (alpkgFiles.length === 1 && files.length > 1) {
      fileError = m.workspace.import_dialog.pick_file.error_mixed_archive;
      return;
    }
    // Per-drop caps bound this INCOMING batch only; cumulative caps re-checked post-merge in
    // `tryStageMergedTfjsFiles`, archive contents bounded separately during unpack/extract.
    if (files.length > MAX_FILES_PER_DROP) {
      fileError = m.workspace.import_dialog.pick_file.error_file_count_cap(
        MAX_FILES_PER_DROP,
        files.length
      );
      return;
    }
    const oversized = files.find((f) => f.size > MAX_FILE_BYTES);
    if (oversized) {
      fileError = m.workspace.import_dialog.pick_file.error_single_too_large(
        oversized.name,
        formatBytes(oversized.size),
        formatBytes(MAX_FILE_BYTES)
      );
      return;
    }
    const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
    if (totalBytes > MAX_TOTAL_BYTES) {
      fileError = m.workspace.import_dialog.pick_file.error_total_too_large(
        formatBytes(totalBytes),
        formatBytes(MAX_TOTAL_BYTES)
      );
      return;
    }
    fileError = null;
    parsingFile = true;
    try {
      const single = files.length === 1 ? files[0] : null;
      const isAlpkgExt = single !== null && /\.alpkg$/i.test(single.name);
      const isZipExt = single !== null && /\.zip$/i.test(single.name);
      if (isAlpkgExt || isZipExt) {
        // Single-archive flow. `.zip` is ambiguous (ALPKG-renamed or a pre-zipped TFJS bundle): try
        // ALPKG first, fall through to TFJS extraction on `missing-package-json` ("valid ZIP, not
        // ours"); any other ALPKG error is a real structural failure surfaced as is.
        try {
          // Wipe staged TFJS state (switching branches); restored on the fall-through paths below.
          const priorStaged = tfjsStagedFiles;
          const priorBundle = tfjs;
          tfjsStagedFiles = [];
          tfjs = null;
          try {
            const result = await unpackAlpkg(single);
            alpkg = result;
            alpkgClassified = classifyAlpkgEntries(result.entries);
            branch = 'alpkg';
          } catch (e) {
            const zipButNotAlpkg =
              isZipExt && e instanceof AlpkgUnpackError && e.kind === 'missing-package-json';
            if (zipButNotAlpkg) {
              // Restore staging so the zip's contents merge into the prior accumulation.
              tfjsStagedFiles = priorStaged;
              tfjs = priorBundle;
              const zipEntries = await extractZipEntries(single);
              const tfjsFiles = zipEntriesToTfjsFiles(zipEntries);
              if (!tryStageMergedTfjsFiles(tfjsFiles)) {
                if (priorStaged.length > 0) branch = 'tfjs';
                return;
              }
              const bundle = await classifyTfjsBundle(tfjsStagedFiles);
              tfjs = bundle;
              branch = 'tfjs';
              if (!tfjs.ready) return;
            } else if (e instanceof AlpkgUnpackError) {
              // Real ALPKG failure: restore the staging the attempt cleared.
              tfjsStagedFiles = priorStaged;
              tfjs = priorBundle;
              if (priorStaged.length > 0) branch = 'tfjs';
              fileError = e.message;
              return;
            } else {
              tfjsStagedFiles = priorStaged;
              tfjs = priorBundle;
              if (priorStaged.length > 0) branch = 'tfjs';
              fileError =
                e instanceof Error
                  ? e.message
                  : m.workspace.import_dialog.pick_file.error_could_not_read_archive;
              return;
            }
          }
        } catch (e) {
          fileError =
            e instanceof Error
              ? e.message
              : m.workspace.import_dialog.pick_file.error_could_not_read_file;
          return;
        }
      } else {
        // TFJS path: loose multi-file drop. Accumulate and re-classify; the advance below gates on
        // `bundle.ready`, so a not-yet-ready set stays on step 1 with the staging panel.
        if (!tryStageMergedTfjsFiles(files)) return;
        try {
          const bundle = await classifyTfjsBundle(tfjsStagedFiles);
          tfjs = bundle;
          branch = 'tfjs';
        } catch (e) {
          fileError =
            e instanceof Error
              ? e.message
              : m.workspace.import_dialog.pick_file.error_could_not_read_picked_files;
          return;
        }
        if (!tfjs.ready) {
          return;
        }
      }
      if (mode === 'pick-target') {
        primePickTargetFromArchive();
        step = 'pick-target';
      } else {
        seedSelectionFromTarget();
        step = 'summary';
      }
    } finally {
      parsingFile = false;
    }
  }

  function primePickTargetFromArchive(): void {
    if (branch === 'tfjs') {
      // TFJS bundles carry no source identity, so require an explicit pick.
      targetMode = 'use-existing';
      pickedExistingId = null;
      newWorkspaceName = '';
      return;
    }
    const sourceId = alpkg?.workspaceCore?.id ?? null;
    const sourceName = alpkg?.workspaceCore?.name ?? '';
    const matched = sourceId !== null ? workspaces.entries.find((w) => w.id === sourceId) : null;
    if (matched) {
      targetMode = 'use-existing';
      pickedExistingId = matched.id;
      newWorkspaceName = '';
    } else {
      targetMode = 'create-new';
      pickedExistingId = null;
      newWorkspaceName = sourceName.length > 0 ? sourceName : 'imported';
    }
  }

  function onFilePickerChange(e: Event): void {
    const input = e.currentTarget as HTMLInputElement;
    const list = input.files ? Array.from(input.files) : [];
    void handleFiles(list);
    // Clear so re-picking the same file still fires `change`.
    input.value = '';
  }

  function onDrop(e: DragEvent): void {
    e.preventDefault();
    dragOver = false;
    const files = e.dataTransfer?.files ? Array.from(e.dataTransfer.files) : [];
    void handleFiles(files);
  }

  function onDragOver(e: DragEvent): void {
    e.preventDefault();
    dragOver = true;
  }

  function onDragLeave(e: DragEvent): void {
    const next = e.relatedTarget as Node | null;
    if (next && (e.currentTarget as Node).contains(next)) return;
    dragOver = false;
  }

  const tfjsBundleTotalBytes = $derived.by((): number => {
    if (!tfjs) return 0;
    let total = 0;
    if (tfjs.modelJson) total += tfjs.modelJson.size;
    for (const s of tfjs.shards) total += s.size;
    if (tfjs.labels) total += tfjs.labels.size;
    return total;
  });

  // Parsed class labels, populated async when the bundle has a recognised labels file (the classifier
  // identifies but doesn't open it, staying pure/re-runnable per drop). Labels-only deliberately:
  // `metadata.json`'s extra fields would vary the popover surface unpredictably; labels are the stable
  // intersection.
  let tfjsLabels = $state<string[] | null>(null);
  $effect(() => {
    const bundle = tfjs;
    if (!bundle?.labels || !bundle.labelsFormat) {
      tfjsLabels = null;
      // Close before the icon unmounts: no pointerleave fires when the icon vanishes under the cursor,
      // so its channels would stay live and a later bundle would render at stale coords.
      closeTfjsLabelsPopover();
      return;
    }
    // Cancellation flag: a mid-parse bundle change must not let stale results overwrite newer ones.
    const file = bundle.labels;
    const format = bundle.labelsFormat;
    let cancelled = false;
    void parseTfjsLabels(file, format).then((labels) => {
      if (cancelled) return;
      tfjsLabels = labels;
    });
    return () => {
      cancelled = true;
      // Close on re-fire so the popover resets even when the new bundle still has labels.
      closeTfjsLabelsPopover();
    };
  });

  // Two-channel TFJS labels popover (hover=peek / focus=pin, same split as the head info popover).
  let tfjsLabelsHovered = $state(false);
  let tfjsLabelsFocused = $state(false);
  const tfjsLabelsPopoverOpen = $derived(tfjsLabelsHovered || tfjsLabelsFocused);

  // Walks up to the `data-tfjs-card` marker so the popover anchors to the CARD's width (`align:
  // 'span'`), not the icon's 12 px button.
  function openTfjsLabelsPopover(triggerEl: HTMLElement): void {
    const cardEl = triggerEl.closest<HTMLElement>('[data-tfjs-card]');
    if (!cardEl) return;
    tfjsLabelsPopoverAnchor = cardEl;
    tfjsLabelsPopoverPos = computePopupPosition(cardEl, {
      align: 'span',
      overlay: false,
      triggerGap: POPOVER_TRIGGER_GAP,
      popupHeight: TFJS_LABELS_POPOVER_HEIGHT_ESTIMATE
    });
  }

  function closeTfjsLabelsPopover(): void {
    tfjsLabelsHovered = false;
    tfjsLabelsFocused = false;
    tfjsLabelsPopoverAnchor = null;
    tfjsLabelsPopoverPos = null;
  }

  const newWorkspaceNameTrimmed = $derived(newWorkspaceName.trim());
  const newWorkspaceNameError = $derived(
    targetMode === 'create-new' && newWorkspaceNameTrimmed.length > 0
      ? validateWorkspaceName(newWorkspaceNameTrimmed)
      : null
  );
  const canConfirmTarget = $derived(
    targetMode === 'use-existing'
      ? pickedExistingId !== null
      : newWorkspaceNameTrimmed.length > 0 && !newWorkspaceNameError && !creatingTarget
  );

  async function confirmTarget(): Promise<void> {
    if (!canConfirmTarget) return;
    if (targetMode === 'use-existing') {
      const id = pickedExistingId;
      if (!id) return;
      const entry = workspaces.entries.find((w) => w.id === id);
      resolvedTargetId = id;
      resolvedTargetName = entry?.name ?? '';
    } else {
      creatingTarget = true;
      createTargetError = null;
      try {
        const tags = alpkg?.workspaceCore?.tags ?? [];
        const resp: WorkspaceMutationResp = await workspaces.create({
          name: newWorkspaceNameTrimmed,
          tags
        });
        resolvedTargetId = resp.id;
        resolvedTargetName = resp.name;
      } catch (e) {
        createTargetError = errorCopy(e);
        return;
      } finally {
        creatingTarget = false;
      }
    }
    seedSelectionFromTarget();
    step = 'summary';
  }

  // Clear selection, populate ALPKG-only seeds (dataset target-name map + category load), and ALWAYS
  // fetch target heads (ALPKG: collision + cap math; TFJS: displacement notice). Dataset modes are NOT
  // seeded; `effectiveDatasetMode` derives the default once the category refresh settles.
  function seedSelectionFromTarget(): void {
    datasetTargetNames.clear();
    datasetModes.clear();
    selectedHeadIds.clear();
    replaceHeadIds.clear();
    if (!resolvedTargetId) return;
    if (branch === 'alpkg' && alpkgClassified) {
      for (const bucket of alpkgClassified.datasets) {
        datasetTargetNames.set(bucket.name, bucket.name);
      }
      // Head selection stays empty until `loadTargetHeads` resolves: auto-pre-select needs the target's
      // existing ids + slot math; seeding early would flicker checkboxes or silently include rows that
      // should be deselected (triggering an eviction at import).
      void loadTargetCategories();
    }
    // TFJS skips `loadTargetCategories` (no per-category collision for a single-head bundle).
    void loadTargetHeads();
  }

  async function loadTargetCategories(): Promise<void> {
    if (!resolvedTargetId) return;
    targetCategoriesLoading = true;
    targetCategoriesLoadError = null;
    try {
      // Force a fresh GET so the collision check reflects post-mutation reality, not a stale slice;
      // `effectiveDatasetMode` reads it reactively, so a row's default flips "new" -> "skip" the moment
      // a collision lands.
      await categories.refresh(resolvedTargetId, true);
    } catch (e) {
      targetCategoriesLoadError = errorCopy(e);
    } finally {
      targetCategoriesLoading = false;
    }
  }

  async function loadTargetHeads(): Promise<void> {
    if (!resolvedTargetId) return;
    // Capture the fetched id so a superseded response (Back then re-confirm a DIFFERENT target
    // mid-flight) can't write the old target's heads; `headsApi.list` has no abort/request-id, so every
    // write below guards on `resolvedTargetId === reqId`.
    const reqId = resolvedTargetId;
    targetLoading = true;
    targetLoadError = null;
    try {
      const list = await headsApi.list(reqId);
      if (resolvedTargetId !== reqId) return;
      targetExistingHeads = list;
      // One-shot auto-pre-select of the newest non-colliding archive heads, capped at both empty slots
      // (`cap - existing`) and the rotation ceiling (`cap - active`); a full target pre-selects nothing
      // (operator must opt into displace/replace). Collision-id rows are never ticked (would 409 or
      // silently overwrite) -- they go through retry-with-replace.
      const existingHeadIds = new Set(list.map((h) => h.head_id));
      if (alpkgClassified) {
        const ceiling = Math.max(0, HEAD_HISTORY_CAP - activeInTarget);
        const emptySlots = Math.max(0, HEAD_HISTORY_CAP - list.length);
        const budget = Math.min(emptySlots, ceiling);
        let taken = 0;
        for (const bucket of alpkgClassified.heads) {
          if (taken >= budget) break;
          if (existingHeadIds.has(bucket.headId)) continue;
          selectedHeadIds.add(bucket.headId);
          taken += 1;
        }
      }
    } catch (e) {
      if (resolvedTargetId === reqId) targetLoadError = errorCopy(e);
    } finally {
      if (resolvedTargetId === reqId) targetLoading = false;
    }
  }

  function setDatasetMode(sourceName: string, mode: DatasetMode): void {
    datasetModes.set(sourceName, mode);
  }

  // Mode-dropdown lifecycle, shared close-timer pattern: "stay open" paths call
  // `cancelCloseModeDropdown` first so a pending close doesn't fire after hover re-establishes;
  // "leaving" paths schedule a delayed close absorbing cursor traversal across the trigger->menu gap.
  function openModeDropdown(sourceName: string): void {
    cancelCloseModeDropdown();
    // `activeModeBtnEl` set by the trigger's `pointerenter`; capture it as the anchor.
    if (activeModeBtnEl) {
      modeMenuAnchor = activeModeBtnEl;
      modeMenuPos = computePopupPosition(activeModeBtnEl, {
        align: 'right',
        overlay: false,
        triggerGap: POPOVER_TRIGGER_GAP,
        popupHeight: MODE_MENU_HEIGHT_ESTIMATE
      });
    }
    openModeDropdownSource = sourceName;
  }

  /// Viewport-space position for a `position: fixed` popover. `align`: 'right' pins the right edge to
  /// the anchor's (grows leftward), 'span' matches anchor width. `overlay`: true covers the anchor
  /// (needs `height - anchorHeight` past its edge), false adjacent (needs `height + gap`). `triggerGap`
  /// applies only to adjacent popovers; `popupHeight` drives the up/down direction choice.
  function computePopupPosition(
    anchorEl: HTMLElement,
    config: {
      align: 'right' | 'span';
      overlay: boolean;
      triggerGap?: number;
      popupHeight: number;
    }
  ): PopupPos {
    const r = anchorEl.getBoundingClientRect();
    const gap = config.triggerGap ?? 0;
    // Pick 'down'/'up' for fit; if neither fits (tiny viewport/extreme zoom) pick the larger side.
    const spaceBelow = window.innerHeight - r.bottom - POPOVER_EDGE_PAD;
    const spaceAbove = r.top - POPOVER_EDGE_PAD;
    const need = config.overlay
      ? Math.max(0, config.popupHeight - r.height)
      : config.popupHeight + gap;
    const direction: 'down' | 'up' =
      spaceBelow >= need
        ? 'down'
        : spaceAbove >= need
          ? 'up'
          : spaceBelow >= spaceAbove
            ? 'down'
            : 'up';
    const top = config.overlay
      ? direction === 'down'
        ? r.top
        : r.bottom - config.popupHeight
      : direction === 'down'
        ? r.bottom + gap
        : r.top - gap - config.popupHeight;
    if (config.align === 'span') {
      return { top, left: r.left, width: r.width };
    }
    return { top, right: window.innerWidth - r.right };
  }

  // Re-compute position for every open popover when its trigger may have moved (scroll/resize).
  function refreshPopupPositions(): void {
    if (openModeDropdownSource !== null && modeMenuAnchor) {
      modeMenuPos = computePopupPosition(modeMenuAnchor, {
        align: 'right',
        overlay: false,
        triggerGap: POPOVER_TRIGGER_GAP,
        popupHeight: MODE_MENU_HEIGHT_ESTIMATE
      });
    }
    if (editingRenameSources.size > 0 && renamePopoverAnchor) {
      renamePopoverPos = computePopupPosition(renamePopoverAnchor, {
        align: 'span',
        overlay: true,
        popupHeight: RENAME_POPOVER_HEIGHT_ESTIMATE
      });
    }
    if ((popoverHoveredHeadId !== null || popoverFocusedHeadId !== null) && headPopoverAnchor) {
      headPopoverPos = computePopupPosition(headPopoverAnchor, {
        align: 'span',
        overlay: false,
        triggerGap: POPOVER_TRIGGER_GAP,
        popupHeight: HEAD_POPOVER_HEIGHT_ESTIMATE
      });
    }
    if (tfjsLabelsPopoverOpen && tfjsLabelsPopoverAnchor) {
      tfjsLabelsPopoverPos = computePopupPosition(tfjsLabelsPopoverAnchor, {
        align: 'span',
        overlay: false,
        triggerGap: POPOVER_TRIGGER_GAP,
        popupHeight: TFJS_LABELS_POPOVER_HEIGHT_ESTIMATE
      });
    }
  }

  // Keep open popovers attached on scroll/resize. Listens on the WINDOW with `capture: true` to catch
  // scroll from any ancestor without finding the specific scroller. Each getBoundingClientRect forces a
  // layout flush, so rAF coalesces scroll bursts to one re-position per paint.
  $effect(() => {
    const anyOpen =
      openModeDropdownSource !== null ||
      editingRenameSources.size > 0 ||
      popoverHoveredHeadId !== null ||
      popoverFocusedHeadId !== null ||
      tfjsLabelsPopoverOpen;
    if (!anyOpen) return;
    let pendingFrame: number | null = null;
    const handler = (): void => {
      if (pendingFrame !== null) return;
      pendingFrame = requestAnimationFrame(() => {
        pendingFrame = null;
        refreshPopupPositions();
      });
    };
    window.addEventListener('scroll', handler, { capture: true, passive: true });
    window.addEventListener('resize', handler);
    return () => {
      if (pendingFrame !== null) {
        cancelAnimationFrame(pendingFrame);
        pendingFrame = null;
      }
      window.removeEventListener('scroll', handler, { capture: true });
      window.removeEventListener('resize', handler);
    };
  });

  function openHeadPopoverFromTarget(target: EventTarget | null): void {
    const rowEl = (target as HTMLElement | null)?.closest('li') ?? null;
    if (!rowEl) return;
    headPopoverAnchor = rowEl;
    headPopoverPos = computePopupPosition(rowEl, {
      align: 'span',
      overlay: false,
      triggerGap: POPOVER_TRIGGER_GAP,
      popupHeight: HEAD_POPOVER_HEIGHT_ESTIMATE
    });
  }

  function scheduleCloseModeDropdown(sourceName: string): void {
    cancelCloseModeDropdown();
    modeDropdownCloseTimer = setTimeout(() => {
      // Only clear if the scheduled row is still open (may race an `openModeDropdown` on another row).
      if (openModeDropdownSource === sourceName) openModeDropdownSource = null;
      modeDropdownCloseTimer = null;
    }, 120);
  }

  function cancelCloseModeDropdown(): void {
    if (modeDropdownCloseTimer !== null) {
      clearTimeout(modeDropdownCloseTimer);
      modeDropdownCloseTimer = null;
    }
  }

  function toggleModeDropdown(sourceName: string): void {
    if (openModeDropdownSource === sourceName) {
      cancelCloseModeDropdown();
      openModeDropdownSource = null;
    } else {
      openModeDropdown(sourceName);
    }
  }

  function closeModeDropdown(): void {
    cancelCloseModeDropdown();
    openModeDropdownSource = null;
  }

  function toggleRenameEdit(sourceName: string, rowEl?: HTMLElement | null): void {
    if (editingRenameSources.has(sourceName)) {
      // Re-clicking the pencil backs out: discard the draft, leave the committed name.
      datasetTargetDrafts.delete(sourceName);
      editingRenameSources.delete(sourceName);
    } else {
      // Single-popover discipline: close any other open rename UI first.
      editingRenameSources.clear();
      datasetTargetDrafts.clear();
      // Seed the draft from the committed target so the first keystroke type-overs it (paired with
      // `autoFocusInput`'s `.select()`).
      datasetTargetDrafts.set(sourceName, datasetTargetNames.get(sourceName) ?? sourceName);
      editingRenameSources.add(sourceName);
      // Callers pass the row as anchor when they have it; the clickOutside dismiss path doesn't.
      if (rowEl) {
        renamePopoverAnchor = rowEl;
        renamePopoverPos = computePopupPosition(rowEl, {
          align: 'span',
          overlay: true,
          popupHeight: RENAME_POPOVER_HEIGHT_ESTIMATE
        });
      }
    }
  }

  /// Commit the draft to the committed map; silently refuses an invalid draft (defence-in-depth, since
  /// callers already gate on `editingDraftError`).
  function saveRenameEdit(sourceName: string): void {
    if (editingDraftError !== null) return;
    const draft = datasetTargetDrafts.get(sourceName);
    if (draft !== undefined) {
      datasetTargetNames.set(sourceName, draft.trim());
    }
    datasetTargetDrafts.delete(sourceName);
    editingRenameSources.delete(sourceName);
  }

  function autoFocusInput(node: HTMLInputElement): void {
    node.focus();
    node.select();
  }

  /// Fire `ondismiss` (cancel semantics, discards the draft) on a pointerdown outside both the bound
  /// element and the optional `ignore` element (the opening trigger -- without the gate, clicking it to
  /// close would fire dismiss AND its toggle, re-opening immediately). `pointerdown` capture, not
  /// `click`, so a drag-release-outside doesn't dismiss and capture beats descendant `stopPropagation`.
  function clickOutside(
    node: HTMLElement,
    options: { ignore?: HTMLElement | null; ondismiss: () => void }
  ): {
    update(o: { ignore?: HTMLElement | null; ondismiss: () => void }): void;
    destroy(): void;
  } {
    let current = options;
    function handle(e: PointerEvent): void {
      const target = e.target as Node | null;
      if (target === null) return;
      if (node.contains(target)) return;
      if (current.ignore?.contains(target)) return;
      current.ondismiss();
    }
    document.addEventListener('pointerdown', handle, true);
    return {
      update(newOptions: { ignore?: HTMLElement | null; ondismiss: () => void }): void {
        current = newOptions;
      },
      destroy(): void {
        document.removeEventListener('pointerdown', handle, true);
      }
    };
  }

  /// The pencil that opened the current popover, so its `clickOutside` can exempt it. Read only while a
  /// popover is mounted, so a stale value when closed is harmless.
  let activePencilEl = $state<HTMLButtonElement | null>(null);

  /// Same for the mode-dropdown trigger, set on `pointerenter` (before `pointerdown`) so the
  /// document-capture `clickOutside` sees the right ignore target. `bind:this` would NOT work in the
  /// `{#each}` (one writable, last-mounted trigger wins, so a tap on any other row would
  /// dismiss-then-re-open into a stuck menu).
  let activeModeBtnEl = $state<HTMLButtonElement | null>(null);

  // Parameter is `mode` (not `m`) to avoid shadowing the imported `m` i18n proxy; same for
  // `modeTooltip`/`modeDisabledReason` below.
  function modeLabel(mode: DatasetMode): string {
    return m.workspace.import_dialog.modes[mode];
  }

  function modeTooltip(mode: DatasetMode): string {
    return m.workspace.import_dialog.mode_tooltips[mode];
  }

  /// Tooltip for why a mode pill is disabled (doesn't fit the current collision state); 'skip' is never
  /// disabled and returns ''.
  function modeDisabledReason(mode: DatasetMode): string {
    if (mode === 'skip') return '';
    const reasons = m.workspace.import_dialog.mode_disabled_reasons;
    if (mode === 'new') return reasons.new_exists;
    if (mode === 'merge') return reasons.merge_missing;
    return reasons.replace_missing;
  }

  /// Trigger-badge classes, four-colour encoding row state: `new` emerald (safe), `merge` amber
  /// (collision, additive), `replace` rose (destructive wipe), `skip` zinc (inactive).
  function modeBadgeClass(m: DatasetMode): string {
    const base =
      'inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2';
    switch (m) {
      case 'new':
        return `${base} bg-success-soft text-success-soft-fg hover:bg-success-soft focus-visible:ring-success-line`;
      case 'merge':
        return `${base} bg-warning-soft text-warning-soft-fg hover:bg-warning-soft focus-visible:ring-warning-line`;
      case 'replace':
        return `${base} bg-danger-soft text-danger-soft-fg hover:bg-danger-soft focus-visible:ring-danger-line`;
      case 'skip':
        return `${base} bg-surface-2 text-fg-secondary hover:bg-surface-2 focus-visible:ring-line`;
    }
  }

  /// Dropdown menu OPTION classes: active option gets a bold tint + check glyph; disabled items fade
  /// with no hover.
  function modeMenuItemClass(isActive: boolean, isApplicable: boolean): string {
    // `py-1` (denser than elsewhere) trims a 4-item menu to ~106 px so it fits either side of any
    // trigger within `max-h-[90vh]`.
    const base =
      'flex w-full items-center justify-between gap-4 rounded-sm px-2 py-1 text-left text-xs transition';
    if (!isApplicable) {
      return `${base} cursor-not-allowed text-fg-subtle hover:bg-transparent`;
    }
    if (isActive) return `${base} bg-surface-2 font-semibold text-fg`;
    return `${base} text-fg-secondary hover:bg-surface-2`;
  }

  /// Stable display order so pill positions never depend on the dynamic `applicableDatasetModes` set
  /// (which drives only the disabled flag).
  const ALL_DATASET_MODES: readonly DatasetMode[] = ['new', 'merge', 'replace', 'skip'];

  function toggleHead(id: Uuid): void {
    if (selectedHeadIds.has(id)) {
      // Untick also clears the replace authorisation so a promoted row doesn't keep a stale flag.
      selectedHeadIds.delete(id);
      replaceHeadIds.delete(id);
      return;
    }
    // Forbid ticking an id already in the target (would 409 or silently overwrite); the sanctioned path
    // is `retryWithReplace`, which adds programmatically with an explicit replace authorisation.
    if (targetExistingHeadIds.has(id)) return;
    // Ceiling guard: a tick beyond `cap - active` can't land (rotation won't displace the pinned active
    // head). `selectedNewHeadCount` (not `.size`) since a replace-preselected row is net-zero.
    if (selectedNewHeadCount >= headSelectionCeiling) return;
    selectedHeadIds.add(id);
  }

  const alpkgDatasetRows = $derived<DatasetBucket[]>(alpkgClassified?.datasets ?? []);
  const alpkgHeadRows = $derived<HeadBucket[]>(alpkgClassified?.heads ?? []);
  const targetExistingHeadIds = $derived(
    new Set((targetExistingHeads ?? []).map((h) => h.head_id))
  );

  // Selection ceiling is `cap - activeInTarget` (rotation evicts non-active heads to fit, only the
  // active one can't move), independent of `existingCount`. `activeInTarget` reads the global
  // `configStore.active` to skip a redundant `GET /active`.
  const existingHeadCount = $derived(targetExistingHeads?.length ?? 0);
  const activeInTarget = $derived(
    resolvedTargetId !== null &&
      configStore.active?.origin === 'head' &&
      configStore.active.source_workspace_id === resolvedTargetId
      ? 1
      : 0
  );
  const headSelectionCeiling = $derived(Math.max(0, HEAD_HISTORY_CAP - activeInTarget));
  // Selected heads whose id isn't in the target -- the "new publishes" consuming a slot (a matching-id
  // selection is a post-conflict replace, net zero on slots).
  const selectedNewHeadCount = $derived.by((): number => {
    let n = 0;
    for (const id of selectedHeadIds) if (!targetExistingHeadIds.has(id)) n += 1;
    return n;
  });
  // Heads the rotation will silently evict (tail-most non-pinned) to honour the cap, surfaced
  // pre-Import. Incoming = ALPKG's `selectedNewHeadCount`, or +1 for TFJS (one fresh-UUID head).
  const headsDisplacedByImport = $derived.by((): number => {
    if (targetExistingHeads === null) return 0;
    const incoming = branch === 'tfjs' ? 1 : selectedNewHeadCount;
    return Math.max(0, existingHeadCount + incoming - HEAD_HISTORY_CAP);
  });

  const targetCategoryNamesLower = $derived(
    resolvedTargetId
      ? new Set(categories.for(resolvedTargetId).entries.map((e) => e.name.toLowerCase()))
      : new Set<string>()
  );

  // "Reuse existing" chips in the rename UI: clicking a chip sets the row's target to that category,
  // intentionally driving a merge/replace.
  const existingTargetCategoryEntries = $derived(
    resolvedTargetId ? categories.for(resolvedTargetId).entries : []
  );

  function resolvedTargetFor(sourceName: string): string {
    return (datasetTargetNames.get(sourceName) ?? sourceName).trim();
  }

  const editingSource = $derived.by((): string | null => {
    for (const name of editingRenameSources) return name;
    return null;
  });

  const editingDraft = $derived.by((): string => {
    if (editingSource === null) return '';
    return datasetTargetDrafts.get(editingSource) ?? editingSource;
  });

  // Live draft validation (Enter-to-save and chip-click gate on it). Skipped when the draft matches an
  // existing target category (intentional reuse), else `validateCategoryName`. Cross-row collisions
  // (two sources -> one target) are NOT errors: the orchestrator handles them safely (404-tolerant
  // wipes, additive source-keyed uploads).
  const editingDraftError = $derived.by((): string | null => {
    const source = editingSource;
    if (source === null) return null;
    const draft = editingDraft.trim();
    if (targetCategoryNamesLower.has(draft.toLowerCase())) return null;
    return validateCategoryName(draft);
  });

  // Source names whose RESOLVED target collides with an existing target category; drives the
  // applicable-modes set + per-row conflict hue.
  const datasetTargetCollisions = $derived.by((): ReadonlySet<string> => {
    const result = new SvelteSet<string>();
    if (branch !== 'alpkg' || !alpkgClassified) return result;
    for (const bucket of alpkgClassified.datasets) {
      const target = resolvedTargetFor(bucket.name).toLowerCase();
      if (target.length === 0) continue;
      if (targetCategoryNamesLower.has(target)) result.add(bucket.name);
    }
    return result;
  });

  // Per-row effective mode from the operator's explicit choice + collision state. Default: no collision
  // -> 'new', collision -> 'skip'. Degradation preserves intent (import vs skip), not the label:
  // 'new'/'merge' follow collision state; 'replace' -> 'new' when no collision but holds for any
  // collision (store accepts a `force: true` wipe); 'skip' always honoured.
  const effectiveDatasetMode = $derived.by((): ReadonlyMap<string, DatasetMode> => {
    const out = new SvelteMap<string, DatasetMode>();
    if (!alpkgClassified) return out;
    for (const bucket of alpkgClassified.datasets) {
      const name = bucket.name;
      const collides = datasetTargetCollisions.has(name);
      const chosen = datasetModes.get(name);
      let mode: DatasetMode;
      if (chosen === undefined) {
        mode = collides ? 'skip' : 'new';
      } else if (chosen === 'skip') {
        mode = 'skip';
      } else if (chosen === 'replace') {
        // Replace holds for ANY collision, including mandatory `_background_noise_`.
        if (!collides) mode = 'new';
        else mode = 'replace';
      } else {
        mode = collides ? 'merge' : 'new';
      }
      out.set(name, mode);
    }
    return out;
  });

  // Source names with effective mode != 'skip'; drives validation / canStart / the counter.
  const selectedDatasetNames = $derived.by((): ReadonlySet<string> => {
    const out = new SvelteSet<string>();
    for (const [name, mode] of effectiveDatasetMode) {
      if (mode !== 'skip') out.add(name);
    }
    return out;
  });

  // Selectable modes per row: no collision -> [New, Skip], collision -> [Merge, Replace, Skip].
  // Mandatory categories always collide, so share the collision treatment.
  function applicableDatasetModes(sourceName: string): DatasetMode[] {
    const collides = datasetTargetCollisions.has(sourceName);
    if (collides) return ['merge', 'replace', 'skip'];
    return ['new', 'skip'];
  }

  // Per-row validation on the COMMITTED target (drafts validated separately via `editingDraftError`).
  // Two sources sharing one target are intentionally allowed. Skipped when the target matches an
  // existing category (intentional reuse; also lets a bucket rename to `_background_noise_` to merge the
  // reserved class), else `validateCategoryName`.
  const datasetValidationErrors = $derived.by((): ReadonlyMap<string, string> => {
    const errors = new SvelteMap<string, string>();
    if (branch !== 'alpkg' || !alpkgClassified) return errors;
    for (const bucket of alpkgClassified.datasets) {
      const target = resolvedTargetFor(bucket.name);
      if (targetCategoryNamesLower.has(target.toLowerCase())) continue;
      const valErr = validateCategoryName(target);
      if (valErr !== null) errors.set(bucket.name, valErr);
    }
    return errors;
  });

  const canStart = $derived.by((): boolean => {
    if (branch === 'alpkg') {
      if (selectedDatasetNames.size === 0 && selectedHeadIds.size === 0) return false;
      // A validation error on any non-skip row blocks import; skipped rows' errors are ignored.
      for (const source of selectedDatasetNames) {
        if (datasetValidationErrors.has(source)) return false;
      }
      return true;
    }
    if (branch === 'tfjs') {
      return tfjs?.ready ?? false;
    }
    return false;
  });

  // Orchestrator selection: only non-skip rows, with the resolved target name and the effective
  // (smart-degraded) mode, so the orchestrator never sees a 'replace' on a non-existent target
  // ('replace' IS emitted for a mandatory-category collision, force-wiped).
  function buildDatasetSelection(): DatasetImportRow[] {
    const out: DatasetImportRow[] = [];
    if (!alpkgClassified) return out;
    for (const bucket of alpkgClassified.datasets) {
      const mode = effectiveDatasetMode.get(bucket.name);
      if (mode === undefined || mode === 'skip') continue;
      out.push({
        sourceName: bucket.name,
        targetName: resolvedTargetFor(bucket.name),
        mode
      });
    }
    return out;
  }

  // Seed the run-state maps from the selection before the first progress event, so the full row list
  // shows with "queued"/"pending" status on entry.
  function initRunStates(): void {
    datasetRunStates.clear();
    headRunStates.clear();
    headRunLogs.clear();
    expandedHeadId = null;
    if (branch === 'tfjs') {
      // Single sentinel-keyed row; TFJS progress maps onto the ALPKG head-phase vocabulary so the badge
      // UI is shared.
      headRunStates.set(TFJS_HEAD_SENTINEL_ID, {
        headId: TFJS_HEAD_SENTINEL_ID,
        phase: 'queued',
        outcome: null,
        error: null,
        conflict: null
      });
      headRunLogs.set(TFJS_HEAD_SENTINEL_ID, []);
      return;
    }
    if (branch !== 'alpkg' || !alpkgClassified) return;
    // One row per imported bucket, source-keyed; pre-fills `total` from the slice count so the "X / Y"
    // denominator shows before the first upload event.
    for (const row of buildDatasetSelection()) {
      const bucket = alpkgClassified.datasets.find((b) => b.name === row.sourceName);
      datasetRunStates.set(row.sourceName, {
        source: row.sourceName,
        target: row.targetName,
        mode: row.mode,
        total: bucket?.slices.length ?? 0,
        uploaded: 0,
        failed: 0,
        phase: 'pending',
        error: null
      });
    }
    // One head row per selected id, in the orchestrator's iteration order (NOT
    // `Array.from(selectedHeadIds)`, whose click order would route `headIndex` events to wrong rows
    // after a manual untick/re-tick).
    for (const headId of orderedSelectedHeadIds()) {
      headRunStates.set(headId, {
        headId,
        phase: 'queued',
        outcome: null,
        error: null,
        conflict: null
      });
      headRunLogs.set(headId, []);
    }
  }

  /// Selected head ids in the orchestrator's iteration order; MUST match its own filter-by-selected
  /// since progress events emit `headIndex = i` against that filtered array, read back here for row
  /// attribution. Empty on TFJS / pre-classification.
  function orderedSelectedHeadIds(): Uuid[] {
    if (!alpkgClassified) return [];
    return alpkgClassified.heads.filter((b) => selectedHeadIds.has(b.headId)).map((b) => b.headId);
  }

  // Update one run-state entry from a progress event. Dataset phases route by SOURCE (not target) so
  // two rows sharing a target flip in turn, with per-source counters (the global `itemsDone`/`itemsTotal`
  // are the overall strip's and would overwrite both shared-target rows with the same number). TFJS
  // phases auto-pin the disclosure on first tick; ALPKG does NOT (N serial heads would constantly remap it).
  function applyProgressToRunStates(p: WorkspaceImportProgress): void {
    switch (p.phase) {
      case 'replacing-categories': {
        if (!p.sourceCategory) return;
        const ds = datasetRunStates.get(p.sourceCategory);
        if (!ds) return;
        datasetRunStates.set(p.sourceCategory, { ...ds, phase: 'replacing' });
        return;
      }
      case 'uploading-datasets': {
        if (!p.sourceCategory) return;
        const ds = datasetRunStates.get(p.sourceCategory);
        if (!ds) return;
        datasetRunStates.set(p.sourceCategory, {
          ...ds,
          phase: 'uploading',
          uploaded: p.categoryUploaded ?? ds.uploaded,
          failed: p.categoryFailed ?? ds.failed
        });
        return;
      }
      case 'importing-heads': {
        if (p.headIndex === undefined || !p.headPhase) return;
        const orderedHeadIds = orderedSelectedHeadIds();
        const headId = orderedHeadIds[p.headIndex];
        if (!headId) return;
        const hs = headRunStates.get(headId);
        if (!hs) return;
        headRunStates.set(headId, { ...hs, phase: p.headPhase });
        return;
      }
      // TFJS upload/convert map onto the ALPKG `HeadRunPhase` vocabulary so badge + spinner reuse
      // `prettyHeadPhase`.
      case 'uploading-tfjs': {
        const hs = headRunStates.get(TFJS_HEAD_SENTINEL_ID);
        if (!hs) return;
        headRunStates.set(TFJS_HEAD_SENTINEL_ID, { ...hs, phase: 'uploading-files' });
        expandedHeadId ??= TFJS_HEAD_SENTINEL_ID;
        return;
      }
      case 'converting-tfjs': {
        const hs = headRunStates.get(TFJS_HEAD_SENTINEL_ID);
        if (!hs) return;
        headRunStates.set(TFJS_HEAD_SENTINEL_ID, { ...hs, phase: 'awaiting-terminal' });
        expandedHeadId ??= TFJS_HEAD_SENTINEL_ID;
        return;
      }
    }
  }

  // Reconcile per-row state with the final summary: datasets get final upload/fail counts; heads get
  // terminal outcome + any conflict record (so the row can offer "Replace & retry" without re-deriving
  // it).
  function applySummaryToRunStates(s: WorkspaceImportSummary): void {
    // Both per-category records and the run-state map are source-keyed, so the join is direct.
    for (const cat of s.categories) {
      const ds = datasetRunStates.get(cat.sourceCategory);
      if (!ds) continue;
      datasetRunStates.set(cat.sourceCategory, {
        ...ds,
        uploaded: cat.uploaded,
        failed: cat.failed,
        error: cat.error,
        phase: cat.failed > 0 || cat.error !== null ? 'failed' : 'done'
      });
    }
    if (branch === 'tfjs') {
      // The summary's first (and only, by current daemon contract) head record maps onto the sentinel.
      // Handled apart from the ALPKG loop so a future >1-head TFJS import doesn't overwrite the sentinel
      // once per record.
      if (s.heads.length === 0) return;
      const h = s.heads[0];
      const hs = headRunStates.get(TFJS_HEAD_SENTINEL_ID);
      if (!hs) return;
      headRunStates.set(TFJS_HEAD_SENTINEL_ID, {
        ...hs,
        phase: h.outcome === 'failed' ? 'failed' : hs.phase,
        outcome: h.outcome,
        error: h.error,
        conflict: h.conflict ?? null
      });
      return;
    }
    for (const h of s.heads) {
      const hs = headRunStates.get(h.headId);
      if (!hs) continue;
      headRunStates.set(h.headId, {
        ...hs,
        phase: h.outcome === 'failed' ? 'failed' : hs.phase,
        outcome: h.outcome,
        error: h.error,
        conflict: h.conflict ?? null
      });
    }
  }

  function formatLogTime(ms: number): string {
    const d = new Date(ms);
    if (Number.isNaN(d.getTime())) return '--:--:--';
    return d.toLocaleTimeString([], { hour12: false });
  }

  function prettyHeadPhase(phase: HeadRunPhase): string {
    const t = m.workspace.import_dialog.head_phase;
    switch (phase) {
      case 'queued':
        return t.queued;
      case 'uploading-files':
        return t.uploading_files;
      case 'starting-convert':
        return t.starting_convert;
      case 'awaiting-terminal':
        return t.converting;
      case 'cleaning-up':
        return t.cleaning_up;
      case 'done':
        return t.done;
      case 'failed':
        return t.failed;
    }
  }

  function prettyHeadOutcome(o: HeadOutcome): string {
    return m.workspace.import_dialog.head_outcome[o];
  }

  async function runImport(): Promise<void> {
    if (!resolvedTargetId || !canStart) return;
    pipelineError = null;
    summary = null;
    progress = null;
    initRunStates();
    const controller = new AbortController();
    abortController = controller;
    step = 'running';
    const targetId = resolvedTargetId;
    try {
      let result: WorkspaceImportSummary;
      if (branch === 'alpkg' && alpkg && alpkgClassified) {
        result = await importAlpkg(
          {
            targetWorkspaceId: targetId,
            archive: alpkg,
            classified: alpkgClassified,
            selection: {
              datasets: buildDatasetSelection(),
              headIds: Array.from(selectedHeadIds),
              replaceHeadIds: new Set(replaceHeadIds)
            }
          },
          {
            signal: controller.signal,
            onprogress: (p) => {
              progress = p;
              applyProgressToRunStates(p);
            },
            onConvertEvent: (headId, ev) => recordConvertEvent(ev, headId),
            // Delegate the wipe to the store so IDB shadow + draft + slice-store cleanup mirror the
            // daemon delete. `force: true` bypasses the mandatory-protect guard and tolerates a 404
            // (nothing on disk = success for "wipe and re-import").
            ondeleteCategory: (name) => categories.delete(targetId, name, { force: true })
          }
        );
      } else if (branch === 'tfjs' && tfjs) {
        result = await importTfjs(
          {
            targetWorkspaceId: targetId,
            bundle: tfjs
          },
          {
            signal: controller.signal,
            onprogress: (p) => {
              progress = p;
              applyProgressToRunStates(p);
            },
            onConvertEvent: (ev) => recordConvertEvent(ev)
          }
        );
      } else {
        throw new Error(m.workspace.import_dialog.error_invalid_state);
      }
      summary = result;
      applySummaryToRunStates(result);
      // Reconcile target stores so the UI reflects the imports without waiting for a poll tick.
      void categories.refresh(targetId, true);
      if (result.latestRevisionId !== null) {
        // Dedupe target names so two source rows -> one target don't refresh twice.
        const catNames = Array.from(
          new Set(result.categories.filter((c) => c.uploaded > 0).map((c) => c.category))
        );
        if (catNames.length > 0) {
          void slicesStore.refreshForWorkspace(targetId, catNames, result.latestRevisionId);
        }
      }
      step = 'done';
      // Collapse the TFJS sentinel row's auto-expanded log on done (focus shifts to the outcome badge,
      // log one click away). ALPKG stays expanded -- its auto-pin was on the last-importing row, the
      // most recent outcome the operator is likely to review.
      if (branch === 'tfjs' && expandedHeadId === TFJS_HEAD_SENTINEL_ID) {
        expandedHeadId = null;
      }
    } catch (e) {
      if (controller.signal.aborted) {
        // Operator cancelled; silent rollback to summary with selection preserved for retry.
        step = 'summary';
        progress = null;
        return;
      }
      pipelineError = e instanceof WorkspaceImportError ? e.message : errorCopy(e);
      step = 'summary';
    } finally {
      if (abortController === controller) abortController = null;
    }
  }

  function recordConvertEvent(ev: ConvertEvent, headId?: Uuid): void {
    const message = convertEventLine(ev);
    // ALPKG passes `headId`; TFJS has no per-event id, so route to the sentinel.
    const logKey: Uuid | null = headId ?? (branch === 'tfjs' ? TFJS_HEAD_SENTINEL_ID : null);
    if (logKey === null) return;
    const lines = headRunLogs.get(logKey) ?? [];
    const next = lines.concat({
      timestampMs: Date.now(),
      message
    });
    // Cap 200 lines/head: bounds memory while keeping the full story for typical converts.
    headRunLogs.set(logKey, next.length > 200 ? next.slice(-200) : next);
  }

  // Falls back to the raw stage string for forward-compat (`ConvertStage` is non-exhaustive).
  function convertStageLabel(stage: ConvertStage): string {
    const catalog = m.workspace.import_dialog.convert_stage as Readonly<
      Partial<Record<ConvertStage, string>>
    >;
    return catalog[stage] ?? stage;
  }

  function convertEventLine(ev: ConvertEvent): string {
    const t = m.workspace.import_dialog.convert_event;
    switch (ev.kind) {
      case 'job_submitted':
        return t.job_submitted(ev.converter);
      case 'job_running':
        return t.job_running;
      case 'stage_started':
        return t.phase(convertStageLabel(ev.stage));
      case 'manifest_validated':
        return t.manifest_validated(ev.n_classes);
      case 'mpk_verified':
        return t.mpk_verified(formatBytes(ev.size_bytes));
      case 'weights_extracted':
        return t.weights_extracted(ev.n_classes, ev.in_dim);
      case 'labels_loaded':
        return t.labels_loaded(ev.n_labels);
      case 'head_published':
        return t.head_published(ev.idempotent_skip);
      case 'job_completed':
        return t.job_completed(ev.result.n_classes);
      case 'job_failed':
        return t.job_failed(ev.category, ev.error);
    }
  }

  function cancelRunning(): void {
    if (step !== 'running') return;
    abortController?.abort();
  }

  // Parent-driven unmount safety net: a parent unmount (route change, workspace bulk-delete) tears the
  // dialog down without the open->closed transition's abort, leaving the import (incl. post-unmount
  // `ondeleteCategory` IDB writes) to run to completion.
  onDestroy(() => {
    abortController?.abort();
    // Cancel the pending close timer so its callback can't touch a destroyed component.
    cancelCloseModeDropdown();
  });

  function retryWithReplace(failedHead: HeadOutcomeRecord): void {
    // Pre-authorise the replace and roll back to summary; the orchestrator DELETEs the existing head and
    // retries (later heads/categories re-run but daemon idempotent-skip avoids duplicate work).
    if (!failedHead.conflict) return;
    replaceHeadIds.add(failedHead.headId);
    selectedHeadIds.add(failedHead.headId);
    summary = null;
    pipelineError = null;
    step = 'summary';
  }

  const progressCopy = $derived.by((): string => {
    if (!progress) return '';
    const t = m.workspace.import_dialog.running;
    switch (progress.phase) {
      case 'replacing-categories': {
        const cat = progress.category ? prettyCategoryName(progress.category) : null;
        const total = progress.itemsTotal ?? 0;
        const done = progress.itemsDone ?? 0;
        return t.progress_replacing_categories(cat, done, total);
      }
      case 'uploading-datasets': {
        const cat = progress.category ? prettyCategoryName(progress.category) : null;
        return t.progress_uploading_datasets(
          cat,
          progress.itemsDone ?? null,
          progress.itemsTotal ?? null
        );
      }
      case 'importing-heads': {
        const idx = progress.headIndex ?? 0;
        const total = progress.itemsTotal ?? 0;
        // Soften the daemon's hyphenated sub-phase to spaces; per-row badges carry the localized form.
        const sub = progress.headPhase ? progress.headPhase.replace(/-/g, ' ') : null;
        return t.progress_importing_heads(idx + 1, total, sub);
      }
      case 'uploading-tfjs':
        return t.progress_uploading_tfjs(progress.itemsDone ?? 0, progress.itemsTotal ?? 0);
      case 'converting-tfjs':
        return t.progress_converting_tfjs;
    }
  });

  const progressFraction = $derived.by((): number => {
    if (!progress) return 0;
    const total = progress.itemsTotal ?? 0;
    const done = progress.itemsDone ?? 0;
    if (total <= 0) return 0;
    return Math.min(1, done / total);
  });

  function closeAndFire(): void {
    const id = resolvedTargetId;
    onclose();
    if (summary && id) onimported?.(id);
  }

  function backToSummary(): void {
    pipelineError = null;
    step = 'summary';
  }

  function backToFile(): void {
    // ALPKG is single-archive: wipe it for a fresh drop. TFJS is incremental: preserve it so a corrected
    // shard/labels file can be dropped without re-assembling the bundle.
    alpkg = null;
    alpkgClassified = null;
    if (branch !== 'tfjs') {
      tfjs = null;
      tfjsStagedFiles = [];
      branch = null;
    }
    fileError = null;
    datasetTargetNames.clear();
    datasetModes.clear();
    editingRenameSources.clear();
    datasetTargetDrafts.clear();
    closeModeDropdown();
    selectedHeadIds.clear();
    replaceHeadIds.clear();
    popoverHoveredHeadId = null;
    popoverFocusedHeadId = null;
    // Clear popover caches so the re-opened dialog doesn't flash a stale popover at prior coords.
    modeMenuAnchor = null;
    renamePopoverAnchor = null;
    headPopoverAnchor = null;
    modeMenuPos = null;
    renamePopoverPos = null;
    headPopoverPos = null;
    // The TFJS labels popover's render gate has no step guard, so a focus-pinned one would otherwise
    // survive Back and float over the file-pick step at stale coords.
    closeTfjsLabelsPopover();
    datasetRunStates.clear();
    headRunStates.clear();
    headRunLogs.clear();
    expandedHeadId = null;
    targetExistingHeads = null;
    targetCategoriesLoading = false;
    targetCategoriesLoadError = null;
    if (mode === 'pick-target') resolvedTargetId = null;
    step = 'pick-file';
  }

  const dialogTitle = $derived.by((): string => {
    // `into-current` reads the locked name eagerly; `pick-target` populates `resolvedTargetName` on
    // confirm. Generic "Import" fallback until the target is known.
    const name =
      mode === 'into-current' ? (lockedWorkspaceName ?? resolvedTargetName) : resolvedTargetName;
    return name
      ? m.workspace.import_dialog.title_into(name)
      : m.workspace.import_dialog.title_fallback;
  });

  // `into-current` is two steps, `pick-target` three; terminal panes (`running`/`done`) drop the
  // indicator since post-commit the counter is just noise.
  const totalSteps = $derived(mode === 'into-current' ? 2 : 3);
  const currentStep = $derived.by((): number | null => {
    switch (step) {
      case 'pick-file':
        return 1;
      case 'pick-target':
        return 2;
      case 'summary':
        return mode === 'into-current' ? 2 : 3;
      case 'running':
      case 'done':
      default:
        return null;
    }
  });
</script>

<!-- `pick-file` carries a lone Cancel so dismissal is reachable without keyboard or a precise backdrop click. -->
{#snippet footerContent()}
  {#if step === 'pick-file'}
    <Button variant="secondary" onclick={onclose}>{m.workspace.import_dialog.footer.cancel}</Button>
  {:else if step === 'pick-target'}
    <Button variant="secondary" onclick={backToFile}>{m.workspace.import_dialog.footer.back}</Button
    >
    <Button
      onclick={() => void confirmTarget()}
      disabled={!canConfirmTarget}
      loading={creatingTarget}
    >
      {m.workspace.import_dialog.footer.next}
    </Button>
  {:else if step === 'summary'}
    <Button variant="secondary" onclick={backToFile}>{m.workspace.import_dialog.footer.back}</Button
    >
    <Button onclick={() => void runImport()} disabled={!canStart}>
      <UploadIcon />
      {m.workspace.import_dialog.footer.import}
    </Button>
  {:else if step === 'running'}
    <Button variant="secondary" onclick={cancelRunning}
      >{m.workspace.import_dialog.footer.cancel}</Button
    >
    <Button disabled loading>{m.workspace.import_dialog.footer.importing}</Button>
  {:else if step === 'done'}
    {#if summary?.heads.some((h) => h.outcome === 'failed' && h.conflict)}
      <Button variant="secondary" onclick={backToSummary}>
        {m.workspace.import_dialog.footer.back_to_selection}
      </Button>
    {/if}
    <Button onclick={closeAndFire}>{m.workspace.import_dialog.footer.done}</Button>
  {/if}
{/snippet}

<Modal
  {open}
  title={dialogTitle}
  onclose={() => {
    if (step === 'running') cancelRunning();
    onclose();
  }}
  closeOnBackdrop={step !== 'running'}
  footer={footerContent}
  class="max-w-lg"
>
  {#snippet headerRight()}
    {#if currentStep !== null}
      <span class="shrink-0 text-[11px] tabular-nums text-fg-muted">
        {m.workspace.import_dialog.step_indicator(currentStep, totalSteps)}
      </span>
    {/if}
  {/snippet}
  {#if pipelineError !== null}
    <div
      class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
      role="alert"
    >
      <p class="font-medium">{m.workspace.import_dialog.pipeline_error_title}</p>
      <p class="mt-0.5">{pipelineError}</p>
    </div>
  {/if}

  <!-- Step 1: pick file. Two-element drop zone (wrapper paints the gray surface, inner `<label>` carries
       the dashed border + tints) so the opacity hover wash composites as a subtle darkening over the
       wrapper's gray rather than rendering LIGHTER than idle on an opaque-grey label. -->
  {#if step === 'pick-file'}
    {@const tfjsStaging = branch === 'tfjs' && tfjsStagedFiles.length > 0}
    <!-- Height collapses `min-h-56` -> `min-h-32` mid-staging to make room for the staging panel. -->
    <div class="rounded-md bg-surface-2">
      <label
        ondragover={onDragOver}
        ondragleave={onDragLeave}
        ondrop={onDrop}
        class="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-md border-2 border-dashed border-line-strong px-6 text-center text-fg-muted transition hover:border-line-strong hover:bg-surface-2/40 {tfjsStaging
          ? 'min-h-32 py-6'
          : 'min-h-56 py-10'}"
        class:border-accent-hover={dragOver}
        class:bg-accent-soft={dragOver}
        title={m.workspace.import_dialog.pick_file.drop_zone_title_attr}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="h-7 w-7 text-fg-subtle"
          aria-hidden="true"
        >
          <path d="M12 4v12" />
          <path d="M6 10l6-6 6 6" />
          <path d="M4 20h16" />
        </svg>
        <span class="text-sm text-fg-secondary">
          {#if parsingFile}
            {m.workspace.import_dialog.pick_file.reading}
          {:else if tfjsStaging}
            {m.workspace.import_dialog.pick_file.drop_zone_tfjs_staging}
          {:else}
            {m.workspace.import_dialog.pick_file.drop_zone_idle}
          {/if}
        </span>
        <span
          class="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-0.5 text-[11px] font-medium text-fg-secondary"
        >
          {m.workspace.import_dialog.pick_file.browse_button}
        </span>
        <input
          type="file"
          multiple
          onchange={onFilePickerChange}
          class="sr-only"
          disabled={parsingFile}
        />
      </label>
    </div>

    {#if tfjsStaging && tfjs}
      <!-- TFJS staging panel, shown only while a non-ready bundle is mid-collection (unmounts once
           `tfjs.ready`). Diagnostics rendered verbatim (classifier sentence-cases them): blockers rose,
           warnings amber. -->
      <div class="overflow-hidden rounded-md border border-line bg-surface">
        <header
          class="flex items-baseline justify-between border-b border-line bg-surface-2 px-3 py-1.5"
        >
          <span class="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
            {m.workspace.import_dialog.pick_file.staged_files_heading}
          </span>
          <div class="flex items-baseline gap-3">
            <span class="font-mono text-[10px] text-fg-subtle tabular-nums">
              {m.workspace.import_dialog.pick_file.staged_files_count(tfjsStagedFileRows.length)}
            </span>
            <button
              type="button"
              onclick={clearTfjsStaging}
              class="text-[11px] font-medium text-fg-secondary transition hover:text-danger-soft-fg"
            >
              {m.workspace.import_dialog.pick_file.clear_button}
            </button>
          </div>
        </header>
        <ul class="flex flex-col divide-y divide-line-subtle">
          {#each tfjsStagedFileRows as row (row.key)}
            {@const roleLabel =
              row.role === 'model'
                ? 'model'
                : row.role === 'shard'
                  ? 'shard'
                  : row.role === 'labels'
                    ? 'labels'
                    : 'other'}
            {@const roleClass =
              row.role === 'unknown'
                ? 'bg-warning-soft text-warning-soft-fg'
                : 'bg-surface-2 text-fg-secondary'}
            <li class="flex items-center gap-2 px-3 py-1.5 text-xs">
              <span
                class="inline-flex w-12 shrink-0 justify-center rounded px-1 py-0.5 text-[10px] font-medium tracking-wider uppercase {roleClass}"
              >
                {roleLabel}
              </span>
              <span class="flex-1 truncate font-mono text-fg-secondary" title={row.file.name}>
                {row.file.name}
              </span>
              <span class="shrink-0 font-mono text-[10px] text-fg-subtle tabular-nums">
                {formatBytes(row.file.size)}
              </span>
            </li>
          {/each}
        </ul>
        {#if tfjs.diagnostics.length > 0}
          <div class="flex flex-col gap-1 border-t border-line px-3 py-2">
            {#each tfjs.diagnostics as d, i (i)}
              <p
                class="text-[11px] {d.severity === 'blocker'
                  ? 'text-danger-soft-fg'
                  : 'text-warning-soft-fg'}"
                role={d.severity === 'blocker' ? 'alert' : undefined}
              >
                {d.message}
              </p>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    {#if fileError !== null}
      <div
        class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
        role="alert"
      >
        {fileError}
      </div>
    {/if}
  {/if}

  <!-- Step 2: pick target (pick-target mode only). A per-branch identity card above a shared target
       picker: ALPKG shows the source workspace card; TFJS a bundle card (no UUID pre-import). -->
  {#if step === 'pick-target'}
    {#if branch === 'alpkg'}
      {@const source = alpkg?.workspaceCore}
      {#if source}
        <!-- Read-only source context bar carrying the SOURCE workspace's own facts (not the archive's)
             so the operator can verify the export-time head-of-train state. -->
        <div class="rounded-md border border-line bg-surface-2 px-3 py-2">
          <p class="truncate text-sm font-semibold text-fg" title={`${source.name} (${source.id})`}>
            {source.name}
          </p>
          <!-- `·` separators are literal `{#if}`-gated segments, shown only when their preceding datum exists. -->
          <p class="mt-0.5 text-[11px] text-fg-muted">
            {#if source.created_at}
              <span title={formatAbsolute(source.created_at)}>
                {m.workspace.import_dialog.pick_target.alpkg_source_created_label(
                  formatRelative(source.created_at)
                )}
              </span>
            {/if}
            {#if source.created_at && source.workspace_revision}
              ·
            {/if}
            {#if source.workspace_revision}
              {m.workspace.import_dialog.pick_target.alpkg_source_rev_label(
                source.workspace_revision.id
              )}{#if source.workspace_revision.at}
                · <span title={formatAbsolute(source.workspace_revision.at)}
                  >{m.workspace.import_dialog.pick_target.alpkg_source_modified_label(
                    formatRelative(source.workspace_revision.at)
                  )}</span
                >{/if}
            {/if}
          </p>
        </div>
      {/if}
    {:else if branch === 'tfjs' && tfjs}
      <!-- TFJS bundle card. Generic headline (no UUID pre-publish); info icon appears once labels parse,
           with the class-count segment absent while the parse is in flight/failed. `data-tfjs-card`
           lets the icon's handlers find the card via `closest()` to anchor the card-width popover. -->
      <div data-tfjs-card class="rounded-md border border-line bg-surface-2 px-3 py-2">
        <div class="flex items-baseline gap-1.5">
          <p class="truncate text-sm font-semibold text-fg">
            {m.workspace.import_dialog.pick_target.tfjs_bundle_card_title}
          </p>
          {#if tfjsLabels && tfjsLabels.length > 0}
            <!-- Info icon, two channels: pointerenter/leave gated to mouse (hover-peek), click toggles
                 the focus pin, onfocus gated on `:focus-visible` so only keyboard Tab auto-pins (a touch
                 tap fires focus + a trailing click that would otherwise unpin in the same tap). -->
            <button
              type="button"
              aria-label={m.workspace.import_dialog.pick_target.tfjs_show_labels_aria}
              aria-expanded={tfjsLabelsPopoverOpen}
              onclick={(e) => {
                e.preventDefault();
                if (tfjsLabelsFocused) {
                  closeTfjsLabelsPopover();
                } else {
                  tfjsLabelsFocused = true;
                  openTfjsLabelsPopover(e.currentTarget);
                }
              }}
              onpointerenter={(e) => {
                if (e.pointerType !== 'mouse') return;
                tfjsLabelsHovered = true;
                openTfjsLabelsPopover(e.currentTarget);
              }}
              onpointerleave={(e) => {
                if (e.pointerType !== 'mouse') return;
                tfjsLabelsHovered = false;
              }}
              onfocus={(e) => {
                if (!(e.currentTarget as HTMLElement).matches(':focus-visible')) return;
                tfjsLabelsFocused = true;
                openTfjsLabelsPopover(e.currentTarget);
              }}
              onblur={() => {
                tfjsLabelsFocused = false;
              }}
              class="inline-flex h-3 w-3 shrink-0 translate-y-px items-center justify-center rounded-full text-fg-subtle transition hover:text-fg-secondary focus-visible:text-fg-secondary focus-visible:ring-2 focus-visible:ring-accent-line focus-visible:outline-none"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2.25"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="h-3 w-3"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M12 16v-4" />
                <path d="M12 8h.01" />
              </svg>
            </button>
          {/if}
        </div>
        <p class="mt-0.5 text-[11px] text-fg-muted">
          {m.workspace.import_dialog.pick_target.tfjs_meta_strip(
            formatBytes(tfjsBundleTotalBytes),
            tfjs.shards.length,
            tfjsLabels && tfjsLabels.length > 0 ? tfjsLabels.length : null,
            tfjs.labels ? tfjs.labels.name : null
          )}
        </p>
      </div>
    {/if}

    <!-- Target picker: two modes (use-existing / create-new) sharing a caption + segmented control;
         body swaps a workspace list for a name input. -->
    <div class="flex flex-col gap-2">
      <div class="flex items-center justify-between gap-2">
        <p class="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
          {m.workspace.import_dialog.pick_target.section_label}
        </p>
        <!-- Mode toggle. Active-button elevation is intentionally SUBTLER than the picker rows' blue ring
             so hierarchy matches semantics (blue ring = THE target workspace, load-bearing; toggle =
             just a view switch). Inset ring stays inside the border-box so it doesn't clip against
             `p-0.5`; `min-w-20` equalises widths so the toggle doesn't shift as modes swap. -->
        <div
          class="flex shrink-0 rounded-md border border-line bg-surface-2 p-0.5 text-[11px] font-medium"
          role="radiogroup"
          aria-label={m.workspace.import_dialog.pick_target.mode_radio_aria}
        >
          <button
            type="button"
            role="radio"
            aria-checked={targetMode === 'use-existing'}
            onclick={() => (targetMode = 'use-existing')}
            class="min-w-20 rounded px-2 py-1 transition-colors {targetMode === 'use-existing'
              ? 'bg-surface text-fg shadow-card ring-1 ring-inset ring-line-strong'
              : 'text-fg-muted hover:bg-surface/50 hover:text-fg-secondary'}"
          >
            {m.workspace.import_dialog.pick_target.mode_use_existing}
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={targetMode === 'create-new'}
            onclick={() => (targetMode = 'create-new')}
            class="min-w-20 rounded px-2 py-1 transition-colors {targetMode === 'create-new'
              ? 'bg-surface text-fg shadow-card ring-1 ring-inset ring-line-strong'
              : 'text-fg-muted hover:bg-surface/50 hover:text-fg-secondary'}"
          >
            {m.workspace.import_dialog.pick_target.mode_create_new}
          </button>
        </div>
      </div>

      {#if targetMode === 'use-existing'}
        {#if workspaces.entries.length === 0}
          <!-- Zero-state: distinct copy so the operator flips to "Create new" rather than awaiting a list. -->
          <p
            class="rounded-md border border-dashed border-line-strong bg-surface-2 px-3 py-3 text-center text-[11px] text-fg-muted"
          >
            {m.workspace.import_dialog.pick_target.no_workspaces_prefix}<span
              class="font-medium text-fg-secondary"
              >{m.workspace.import_dialog.pick_target.no_workspaces_link_label}</span
            >{m.workspace.import_dialog.pick_target.no_workspaces_suffix}
          </p>
        {:else}
          <!-- Uniform `p-1` (4 px) all sides so the picked row's surround reads symmetrically;
               `block: 'center'` clamps scrollTop to reveal the last row's bottom pad. -->
          <ul
            bind:this={pickedListEl}
            class="flex max-h-56 flex-col gap-1 overflow-auto rounded-md border border-line bg-surface p-1"
            role="listbox"
            aria-label={m.workspace.import_dialog.pick_target.workspace_list_aria}
          >
            {#each workspaces.entries as w (w.id)}
              {@const isPicked = pickedExistingId === w.id}
              <li>
                <button
                  type="button"
                  role="option"
                  aria-selected={isPicked}
                  data-workspace-id={w.id}
                  onclick={() => (pickedExistingId = w.id)}
                  class="flex w-full items-center gap-3 rounded-md px-2.5 py-1.5 text-left transition-colors {isPicked
                    ? 'bg-accent-soft ring-1 ring-inset ring-accent'
                    : 'hover:bg-surface-2'}"
                >
                  <span
                    class="min-w-0 flex-1 truncate text-sm font-medium {isPicked
                      ? 'text-accent-soft-fg'
                      : 'text-fg'}"
                    title={`${w.name} (${w.id})`}
                  >
                    {w.name}
                  </span>
                  <span
                    class="shrink-0 text-[11px] text-fg-muted"
                    title={formatAbsolute(w.created_at)}
                  >
                    {m.workspace.import_dialog.pick_target.workspace_created_label(
                      formatRelative(w.created_at)
                    )}
                  </span>
                </button>
              </li>
            {/each}
          </ul>
        {/if}
      {:else}
        <div class="flex flex-col gap-1">
          <input
            type="text"
            bind:value={newWorkspaceName}
            disabled={creatingTarget}
            placeholder={m.workspace.import_dialog.pick_target.create_name_placeholder}
            maxlength="128"
            aria-invalid={!!newWorkspaceNameError}
            class={inputClass(!!newWorkspaceNameError)}
          />
          {#if newWorkspaceNameError}
            <p class="text-[11px] text-danger-soft-fg" role="alert">{newWorkspaceNameError}</p>
          {/if}
          {#if branch === 'alpkg' && alpkg?.workspaceCore?.tags?.length}
            <p class="text-[11px] text-fg-muted">
              {m.workspace.import_dialog.pick_target.create_will_carry_tags(
                alpkg.workspaceCore.tags.join(', ')
              )}
            </p>
          {/if}
        </div>
      {/if}
    </div>

    {#if createTargetError !== null}
      <div
        class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
        role="alert"
      >
        {createTargetError}
      </div>
    {/if}
  {/if}

  <!-- Step 3: summary (alpkg). Target workspace is in the dialog title, not repeated inline. -->
  {#if step === 'summary' && branch === 'alpkg' && alpkgClassified}
    {@const errors = alpkgClassified.errors}

    {#if alpkgDatasetRows.length > 0}
      <section class="flex flex-col gap-2">
        <!-- Counter uses `selectedDatasetNames` (effective mode != 'skip') so it reflects what will land,
             not the row count. -->
        <div class="flex items-baseline justify-between gap-2">
          <h3 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
            {m.workspace.import_dialog.summary.datasets_heading}
          </h3>
          <span class="text-[10px] tabular-nums text-fg-muted">
            {m.workspace.import_dialog.summary.datasets_counter(
              selectedDatasetNames.size,
              alpkgDatasetRows.length
            )}
          </span>
        </div>
        {#if targetCategoriesLoading}
          <p class="text-[11px] text-fg-muted">
            {m.workspace.import_dialog.summary.checking_categories}
          </p>
        {/if}
        {#if targetCategoriesLoadError}
          <p class="text-[11px] text-danger-soft-fg" role="alert">{targetCategoriesLoadError}</p>
        {/if}
        <ul class="flex flex-col gap-1.5">
          {#each alpkgDatasetRows as row (row.name)}
            {@const validationError = datasetValidationErrors.get(row.name) ?? null}
            {@const targetVal = datasetTargetNames.get(row.name) ?? row.name}
            {@const editing = editingRenameSources.has(row.name)}
            {@const renamed = targetVal !== row.name}
            {@const effective = effectiveDatasetMode.get(row.name) ?? 'skip'}
            {@const applicableSet = new SvelteSet(applicableDatasetModes(row.name))}
            {@const active = effective !== 'skip'}
            {@const modeDropdownOpen = openModeDropdownSource === row.name}
            <!-- `relative` anchors the rename popover overlay. Skip rows fade border + tint body (clarify
                 which land) but keep constant dimensions across mode/rename changes. All rows share the
                 affordance set; the pretty source name is the mandatory-class cue. -->
            <li
              class="group/row relative rounded-md border bg-surface transition-colors {active
                ? 'border-line'
                : 'border-line/60 bg-surface-2/40'}"
            >
              <!-- `inert` while editing disables every inner control (the workflow gate). Only the left
                   cluster (`min-w-0 flex-1`) narrows on a constrained viewport; the right stays intrinsic
                   to protect the mode dropdown. Collision info folds into the badge hue. -->
              <div class="flex items-center gap-2 px-3 py-1.5" inert={editing}>
                <div class="flex min-w-0 flex-1 items-center gap-1.5 text-xs">
                  <!-- `min-w-0` (outer + each span) un-pins the flex content-size floor so source/arrow/target truncate proportionally as the row narrows. -->
                  <div class="flex min-w-0 items-center gap-1.5">
                    <span
                      class="min-w-0 truncate font-medium {active ? 'text-fg' : 'text-fg-muted'}"
                      title={row.name}
                    >
                      {prettyCategoryName(row.name)}
                    </span>
                    {#if renamed}
                      <span class="shrink-0 font-medium text-fg-subtle">-></span>
                      <span
                        class="min-w-0 truncate font-medium {active ? 'text-fg' : 'text-fg-muted'}"
                        title={targetVal}
                      >
                        {prettyCategoryName(targetVal)}
                      </span>
                    {/if}
                  </div>

                  <!-- Rename pencil. Hidden on precise-pointer devices, revealed on row hover/focus;
                       always visible on touch (`pointer-coarse`, no hover to discover through) and in
                       editing/validation-error states. `p-0.5` = 16 px touch target. -->
                  <button
                    type="button"
                    onclick={(e) => {
                      activePencilEl = e.currentTarget;
                      // Pass the row element so `toggleRenameEdit` picks the open direction by available side.
                      toggleRenameEdit(row.name, (e.currentTarget as HTMLElement).closest('li'));
                    }}
                    aria-expanded={editing}
                    aria-haspopup="dialog"
                    class="shrink-0 rounded p-0.5 transition duration-200 ease-out {editing
                      ? 'bg-surface-2 text-fg-secondary'
                      : validationError !== null
                        ? 'text-danger-dot hover:bg-danger-soft'
                        : 'pointer-events-none text-fg-subtle opacity-0 group-hover/row:pointer-events-auto group-hover/row:opacity-100 focus-visible:pointer-events-auto focus-visible:opacity-100 hover:bg-surface-2 hover:text-fg-secondary pointer-coarse:pointer-events-auto pointer-coarse:opacity-100'}"
                    title={validationError ??
                      m.workspace.import_dialog.summary.rename_button_title_default}
                    aria-label={m.workspace.import_dialog.summary.rename_button_aria}
                  >
                    <svg
                      viewBox="0 0 16 16"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="1.75"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      class="h-3 w-3"
                      aria-hidden="true"
                    >
                      <path d="M11.5 1.5l3 3-9 9H2.5v-3l9-9z" />
                      <path d="M10 3l3 3" />
                    </svg>
                  </button>
                </div>

                <!-- Right cluster (slice count + mode dropdown), `shrink-0` so the names absorb the squeeze. -->
                <div class="flex shrink-0 items-center gap-2">
                  <span class="font-mono text-[10px] tabular-nums text-fg-muted">
                    {m.workspace.import_dialog.summary.slice_count(row.slices.length)}
                  </span>

                  <!-- Mode-dropdown trigger + menu: mouse hover owns open/close, touch/keyboard use click
                       (per-handler gating below). `relative inline-flex` (not `inline-block`): an
                       inline-block wrapper's line-box would drop the badge ~2.5 px below centre; a flex
                       wrapper has no line-box and centres cleanly while staying inline-level. -->
                  <div class="relative inline-flex">
                    <button
                      type="button"
                      class={modeBadgeClass(effective)}
                      onpointerenter={(e) => {
                        // Capture on EVERY entry (not just mouse): pointerenter fires before pointerdown,
                        // so the menu's document-capture clickOutside sees the right ignore target and a
                        // second touch tap doesn't leave the menu stuck open.
                        activeModeBtnEl = e.currentTarget;
                        if (e.pointerType !== 'mouse') return;
                        openModeDropdown(row.name);
                      }}
                      onpointerleave={(e) => {
                        if (e.pointerType !== 'mouse') return;
                        scheduleCloseModeDropdown(row.name);
                      }}
                      onfocus={(e) => {
                        if (!(e.currentTarget as HTMLElement).matches(':focus-visible')) return;
                        openModeDropdown(row.name);
                      }}
                      onblur={() => scheduleCloseModeDropdown(row.name)}
                      onclick={(e) => {
                        if (e instanceof PointerEvent && e.pointerType === 'mouse') return;
                        toggleModeDropdown(row.name);
                      }}
                      aria-haspopup="listbox"
                      aria-expanded={modeDropdownOpen}
                      aria-label={m.workspace.import_dialog.summary.mode_aria(modeLabel(effective))}
                      title={modeTooltip(effective)}
                    >
                      <span>{modeLabel(effective)}</span>
                      <svg
                        viewBox="0 0 16 16"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="1.75"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        class="h-2.5 w-2.5 transition-transform duration-150 {modeDropdownOpen
                          ? 'rotate-180'
                          : ''}"
                        aria-hidden="true"
                      >
                        <path d="M4 6l4 4 4-4" />
                      </svg>
                    </button>

                    {#if modeDropdownOpen && modeMenuPos}
                      <!-- `position: fixed` escapes the modal overflow clip; `right` aligns the menu's
                           right edge with the trigger's (grows leftward); `min-w-32` holds "Replace". -->
                      <div
                        use:clickOutside={{
                          ignore: activeModeBtnEl,
                          ondismiss: closeModeDropdown
                        }}
                        onpointerenter={(e) => {
                          // Mirror the trigger's mouse gating so a touch/pen synthetic pointerenter doesn't
                          // extend the open lifetime past what the tap implies.
                          if (e.pointerType !== 'mouse') return;
                          cancelCloseModeDropdown();
                        }}
                        onpointerleave={(e) => {
                          if (e.pointerType !== 'mouse') return;
                          scheduleCloseModeDropdown(row.name);
                        }}
                        class="fixed z-50 min-w-32 rounded-md border border-line bg-surface p-1 shadow-popover"
                        style:top="{modeMenuPos.top}px"
                        style:right="{modeMenuPos.right}px"
                        role="listbox"
                        tabindex="-1"
                        aria-label={m.workspace.import_dialog.summary.mode_menu_aria(row.name)}
                      >
                        {#each ALL_DATASET_MODES as m (m)}
                          {@const isMenuActive = effective === m}
                          {@const isApplicable = applicableSet.has(m)}
                          <button
                            type="button"
                            role="option"
                            aria-selected={isMenuActive}
                            disabled={!isApplicable}
                            onclick={() => {
                              if (!isApplicable) return;
                              setDatasetMode(row.name, m);
                              closeModeDropdown();
                            }}
                            class={modeMenuItemClass(isMenuActive, isApplicable)}
                            title={isApplicable ? modeTooltip(m) : modeDisabledReason(m)}
                          >
                            <span>{modeLabel(m)}</span>
                            {#if isMenuActive}
                              <!-- Check glyph redundant with the bold + tint so colour-blind operators don't rely on the bg tint alone. -->
                              <svg
                                viewBox="0 0 16 16"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                class="h-3 w-3"
                                aria-hidden="true"
                              >
                                <path d="M3 8l3 3 7-7" />
                              </svg>
                            {/if}
                          </button>
                        {/each}
                      </div>
                    {/if}
                  </div>
                </div>
              </div>

              <!-- Anchored to the `<li>` so it covers the row (makes the workflow gate obvious: other
                   controls are physically hidden, not "off but visible"; the row's `inert` backs this for
                   keyboard/SR). `{#if editing}` removes it from the DOM at rest (not opacity-0): an
                   absolute descendant past the scroll container's clientHeight inflates `scrollHeight`
                   into a phantom scrollbar. -->
              {#if editing && renamePopoverPos}
                <div
                  use:clickOutside={{
                    ignore: activePencilEl,
                    ondismiss: () => toggleRenameEdit(row.name)
                  }}
                  class="fixed z-50 rounded-md border border-line bg-surface shadow-popover ring-1 ring-black/5"
                  style:top="{renamePopoverPos.top}px"
                  style:left="{renamePopoverPos.left}px"
                  style:width="{renamePopoverPos.width}px"
                  role="dialog"
                  aria-label={m.workspace.import_dialog.summary.rename_popover_aria(row.name)}
                >
                  <div class="flex flex-col gap-2 p-3">
                    <label
                      for="rename-input-{row.name}"
                      class="text-[10px] font-medium tracking-wider text-fg-muted uppercase"
                    >
                      {m.workspace.import_dialog.summary.rename_popover_heading}
                    </label>
                    <div class="flex flex-col gap-1">
                      <!-- Input displays the prettified draft but stores it RAW (e.g. `_background_noise_`);
                           prettifying only transforms reserved-form names, so the display falls back to
                           the literal value as soon as the operator edits. -->
                      <input
                        id="rename-input-{row.name}"
                        use:autoFocusInput
                        type="text"
                        value={prettyCategoryName(editingDraft)}
                        oninput={(e) => datasetTargetDrafts.set(row.name, e.currentTarget.value)}
                        onkeydown={(e) => {
                          if (e.key === 'Enter') {
                            e.stopPropagation();
                            e.preventDefault();
                            if (editingDraftError === null) saveRenameEdit(row.name);
                          } else if (e.key === 'Escape') {
                            e.stopPropagation();
                            e.preventDefault();
                            toggleRenameEdit(row.name);
                          }
                        }}
                        maxlength={255}
                        spellcheck={false}
                        placeholder={prettyCategoryName(row.name)}
                        aria-invalid={editingDraftError !== null}
                        class="block w-full rounded border bg-surface px-2 py-1 text-xs font-medium text-fg-secondary transition-colors {editingDraftError !==
                        null
                          ? 'border-danger-line hover:border-danger-line'
                          : 'border-line hover:border-line-strong'}"
                      />
                      {#if editingDraftError !== null}
                        <p class="text-[11px] text-danger-soft-fg" role="alert">
                          {editingDraftError}
                        </p>
                      {/if}
                    </div>

                    {#if existingTargetCategoryEntries.length > 0}
                      <div class="flex flex-col gap-1">
                        <span
                          class="text-[10px] font-medium tracking-wider text-fg-muted uppercase"
                        >
                          {m.workspace.import_dialog.summary.rename_chips_heading}
                        </span>
                        <!-- Chip click commits + closes in one tap; `saveRenameEdit` still guards on
                             `editingDraftError` so an invalid pick keeps the popover open. -->
                        <div class="flex max-h-32 flex-wrap gap-1 overflow-auto">
                          {#each existingTargetCategoryEntries as cat (cat.name)}
                            {@const isCurrent = cat.name === editingDraft}
                            <button
                              type="button"
                              onclick={() => {
                                datasetTargetDrafts.set(row.name, cat.name);
                                saveRenameEdit(row.name);
                              }}
                              disabled={isCurrent}
                              class="rounded border px-1.5 py-0.5 text-[11px] font-medium transition-colors {isCurrent
                                ? 'cursor-default border-accent bg-accent-soft text-accent-soft-fg'
                                : 'border-line text-fg-secondary hover:border-line-strong hover:bg-surface-2'}"
                              title={cat.name}
                            >
                              {prettyCategoryName(cat.name)}
                            </button>
                          {/each}
                        </div>
                      </div>
                    {/if}
                  </div>
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      </section>
    {/if}

    {#if alpkgHeadRows.length > 0}
      {@const ceilingReached = selectedNewHeadCount >= headSelectionCeiling}
      {@const loaded = targetExistingHeads !== null}
      <section class="flex flex-col gap-2">
        <!-- Slot-usage chip (X / cap + "active pinned" badge) hidden until `targetExistingHeads` lands so
             it never lies about the count during the heads-list round-trip; active-pinned reads
             `configStore.active`, known before the list returns. -->
        <div class="flex items-baseline justify-between gap-2">
          <h3 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
            {m.workspace.import_dialog.summary.heads_heading}
          </h3>
          {#if loaded}
            <span
              class="text-[10px] tabular-nums text-fg-muted"
              title={m.workspace.import_dialog.summary.heads_cap_tooltip(HEAD_HISTORY_CAP)}
            >
              {m.workspace.import_dialog.summary.heads_counter(
                selectedHeadIds.size,
                existingHeadCount,
                HEAD_HISTORY_CAP,
                activeInTarget
              )}
            </span>
          {/if}
        </div>

        {#if targetLoading}
          <p class="text-[11px] text-fg-muted">
            {m.workspace.import_dialog.summary.checking_heads}
          </p>
        {/if}
        {#if targetLoadError}
          <p class="text-[11px] text-danger-soft-fg" role="alert">{targetLoadError}</p>
        {/if}

        <!-- Displacement banner: the rotation evicts tail-most non-pinned heads without
             confirmation, so this pre-Import count is the only chance to bail. -->
        {#if loaded && headsDisplacedByImport > 0}
          <p
            class="rounded-md border border-warning-line bg-warning-soft px-2 py-1 text-[11px] text-warning-soft-fg"
            role="status"
          >
            {m.workspace.import_dialog.summary.displacement_warning(
              headsDisplacedByImport,
              HEAD_HISTORY_CAP
            )}
          </p>
        {/if}

        <ul class="flex flex-col gap-1.5">
          {#each alpkgHeadRows as row (row.headId)}
            {@const exists = targetExistingHeadIds.has(row.headId)}
            {@const checked = selectedHeadIds.has(row.headId)}
            <!-- Native `disabled` is load-bearing: a disabled checkbox never fires `onchange`, so the DOM
                 check state can't drift from `selectedHeadIds`. Disabled when (1) list still loading,
                 (2) id already in target AND unchecked (forbid rule; a checked exists-row is a
                 `retryWithReplace` promotion, left enabled to untick), or (3) ceiling reached AND
                 unchecked (would overflow the rotation cap). -->
            {@const disabled = !loaded || (exists && !checked) || (!checked && ceilingReached)}
            {@const disabledReason = !loaded
              ? m.workspace.import_dialog.summary.head_disabled_reasons.loading
              : exists && !checked
                ? m.workspace.import_dialog.summary.head_disabled_reasons.exists
                : !checked && ceilingReached
                  ? m.workspace.import_dialog.summary.head_disabled_reasons.ceiling
                  : null}
            {@const popoverId = `head-popover-${row.headId}`}
            {@const isPopoverOpen =
              popoverHoveredHeadId === row.headId || popoverFocusedHeadId === row.headId}
            <!-- The popover MUST live outside the `<label>`: a `role="tooltip"` isn't excluded by the
                 label-name calculation, so its descendants (UUID, meta, chips) would otherwise be
                 slurped into the checkbox's accessible name. -->
            <li class="group/headrow @container/headrow relative">
              <label
                class="flex items-center gap-3 rounded-md border border-line px-3 py-1.5 text-xs transition-colors {disabled
                  ? 'cursor-not-allowed bg-surface-2/40 opacity-60'
                  : 'cursor-pointer hover:bg-surface-2'}"
                title={disabledReason}
              >
                <!-- Left cluster = identity (checkbox, short id, Exist badge, info icon); data weight
                     lives in the right cluster so the eye can sweep that column independently. -->
                <span class="flex min-w-0 items-center gap-2">
                  <input
                    type="checkbox"
                    {checked}
                    {disabled}
                    onchange={() => toggleHead(row.headId)}
                    class="h-3.5 w-3.5 shrink-0 cursor-pointer accent-blue-500 disabled:cursor-not-allowed"
                  />
                  <span
                    class="shrink-0 rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-fg-secondary"
                  >
                    {row.headId.replace(/-/g, '').slice(0, 8)}
                  </span>
                  {#if exists}
                    <span
                      class="shrink-0 rounded-full bg-warning-soft px-1.5 py-0.5 text-[10px] font-medium text-warning-soft-fg"
                      title={m.workspace.import_dialog.summary.head_exists_badge_title}
                    >
                      {m.workspace.import_dialog.summary.head_exists_badge}
                    </span>
                  {/if}

                  <!-- Info icon, hover-revealed drill-down. `focus:` (NOT `focus-visible:`) keeps it visible
                       while a click-pinned popover stays open after the cursor leaves; `pointer-coarse:`
                       keeps it tap-reachable. The 10 px hit target is below WCAG's 24 px but adequate for
                       a soft-discovery affordance. Channels `popoverHoveredHeadId` (hover) /
                       `popoverFocusedHeadId` (pin); handlers gate the same touch/mouse/keyboard discipline
                       as the TFJS labels icon, with a defensive `preventDefault` for Safari's
                       button-inside-label propagation. -->
                  <button
                    type="button"
                    class="pointer-events-none inline-flex h-2.5 w-2.5 shrink-0 items-center justify-center rounded-full text-fg-subtle opacity-0 transition duration-200 ease-out group-hover/headrow:pointer-events-auto group-hover/headrow:opacity-100 focus:pointer-events-auto focus:opacity-100 hover:text-fg-secondary focus-visible:text-fg-secondary focus-visible:ring-2 focus-visible:ring-accent-line focus-visible:outline-none pointer-coarse:pointer-events-auto pointer-coarse:opacity-100"
                    onclick={(e) => {
                      e.preventDefault();
                      // Single-popover discipline is implicit: one focused-id value, so pinning row B overwrites A.
                      if (popoverFocusedHeadId === row.headId) {
                        popoverFocusedHeadId = null;
                      } else {
                        popoverFocusedHeadId = row.headId;
                        openHeadPopoverFromTarget(e.currentTarget);
                      }
                    }}
                    onpointerenter={(e) => {
                      if (e.pointerType !== 'mouse') return;
                      popoverHoveredHeadId = row.headId;
                      // Hovering a new row clears any other row's pin so only one popover shows.
                      if (popoverFocusedHeadId !== null && popoverFocusedHeadId !== row.headId) {
                        popoverFocusedHeadId = null;
                      }
                      openHeadPopoverFromTarget(e.currentTarget);
                    }}
                    onpointerleave={(e) => {
                      if (e.pointerType !== 'mouse') return;
                      if (popoverHoveredHeadId === row.headId) popoverHoveredHeadId = null;
                    }}
                    onfocus={(e) => {
                      if (!(e.currentTarget as HTMLElement).matches(':focus-visible')) return;
                      popoverFocusedHeadId = row.headId;
                      openHeadPopoverFromTarget(e.currentTarget);
                    }}
                    onblur={() => {
                      if (popoverFocusedHeadId === row.headId) popoverFocusedHeadId = null;
                    }}
                    aria-label={m.workspace.import_dialog.summary.head_show_details_aria}
                    aria-describedby={isPopoverOpen ? popoverId : undefined}
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="2.25"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      class="h-2.5 w-2.5"
                      aria-hidden="true"
                    >
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 16v-4" />
                      <path d="M12 8h.01" />
                    </svg>
                  </button>
                </span>

                <!-- Right cluster: data weight, `ml-auto` so the left cluster's variable width (Exist
                     badge) doesn't scramble the byte-count column. "· N classes" collapses below
                     `@xs/headrow` (320 px) where the row crowds; recoverable from the info popover. -->
                <span class="ml-auto shrink-0 font-mono text-[10px] tabular-nums text-fg-muted">
                  {formatBytes(row.weights.byteLength)}
                  {#if row.nClasses !== null}
                    <span class="hidden @xs/headrow:inline">
                      <span aria-hidden="true" class="text-fg-subtle">·</span>
                      {m.workspace.import_dialog.summary.head_class_count(row.nClasses)}
                    </span>
                  {/if}
                </span>
              </label>

              <!-- Model-card popover (preview, not a deep-dive). `{#if isPopoverOpen}` removes it from the
                   DOM at rest (not opacity-0), else an off-screen descendant inflates `scrollHeight` into
                   a phantom scrollbar. `pointer-events-none` lets the pointer pass through so hover ends
                   when the cursor leaves the icon. -->
              {#if isPopoverOpen && headPopoverPos}
                <div
                  id={popoverId}
                  role="tooltip"
                  transition:fade|local={{ duration: 150 }}
                  class="pointer-events-none fixed z-50 rounded-md border border-line bg-surface shadow-popover ring-1 ring-black/5"
                  style:top="{headPopoverPos.top}px"
                  style:left="{headPopoverPos.left}px"
                  style:width="{headPopoverPos.width}px"
                >
                  <div class="flex flex-col gap-2 px-3 py-2">
                    <!-- Identity strip; deploy-state badges dropped (not knowable from an archive). -->
                    <div class="flex flex-col gap-0.5">
                      <code class="font-mono text-[11px] font-semibold break-all text-fg">
                        {row.headId}
                      </code>
                      <p class="text-[10px] text-fg-muted">
                        {m.workspace.import_dialog.summary.head_info_metadata(
                          formatBytes(row.weights.byteLength),
                          row.nClasses,
                          row.revisionId,
                          row.createdAt !== null ? formatAbsolute(row.createdAt) : null,
                          row.createdAt !== null ? formatRelative(row.createdAt) : null
                        )}
                      </p>
                    </div>

                    <!-- Class chips capped `max-h-20 overflow-hidden` so hundreds of labels don't make the popover a half-screen wall. -->
                    {#if row.labels !== null && row.labels.length > 0}
                      <div class="flex flex-col gap-1">
                        <span
                          class="text-[10px] font-medium tracking-wider text-fg-muted uppercase"
                        >
                          {m.workspace.import_dialog.summary.head_classes_heading}
                        </span>
                        <ul
                          class="flex max-h-20 flex-wrap gap-1 overflow-hidden"
                          aria-label={m.workspace.import_dialog.summary.head_class_labels_aria}
                        >
                          {#each row.labels as label, idx (`${idx}-${label}`)}
                            <li
                              class="inline-flex max-w-full items-center rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] wrap-break-word text-fg-secondary"
                            >
                              {prettyCategoryName(label)}
                            </li>
                          {/each}
                        </ul>
                      </div>
                    {/if}
                  </div>
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      </section>
    {/if}

    {#if errors.length > 0}
      <details class="text-[11px] text-fg-muted">
        <summary class="cursor-pointer">
          {m.workspace.import_dialog.summary.archive_errors_summary(errors.length)}
        </summary>
        <ul class="mt-1 ml-4 list-disc">
          {#each errors as err, i (i)}
            <li>
              <code>{err.path}</code> - {err.message}
            </li>
          {/each}
        </ul>
      </details>
    {/if}
  {/if}

  <!-- Step 3: summary (tfjs). Identity card matches the pick-target bundle card; target workspace is in
       the title, diagnostics + ignored-files below. -->
  {#if step === 'summary' && branch === 'tfjs' && tfjs}
    <!-- `data-tfjs-card` matches the pick-target card site so the popover anchor lookup finds the right
         element (step gates are mutually exclusive, only one card mounts at a time). -->
    <div data-tfjs-card class="rounded-md border border-line bg-surface-2 px-3 py-2">
      <div class="flex items-baseline gap-1.5">
        <p class="truncate text-sm font-semibold text-fg">
          {m.workspace.import_dialog.pick_target.tfjs_bundle_card_title}
        </p>
        {#if tfjsLabels && tfjsLabels.length > 0}
          <button
            type="button"
            aria-label={m.workspace.import_dialog.pick_target.tfjs_show_labels_aria}
            aria-expanded={tfjsLabelsPopoverOpen}
            onclick={(e) => {
              e.preventDefault();
              if (tfjsLabelsFocused) {
                closeTfjsLabelsPopover();
              } else {
                tfjsLabelsFocused = true;
                openTfjsLabelsPopover(e.currentTarget);
              }
            }}
            onpointerenter={(e) => {
              if (e.pointerType !== 'mouse') return;
              tfjsLabelsHovered = true;
              openTfjsLabelsPopover(e.currentTarget);
            }}
            onpointerleave={(e) => {
              if (e.pointerType !== 'mouse') return;
              tfjsLabelsHovered = false;
            }}
            onfocus={(e) => {
              if (!(e.currentTarget as HTMLElement).matches(':focus-visible')) return;
              tfjsLabelsFocused = true;
              openTfjsLabelsPopover(e.currentTarget);
            }}
            onblur={() => {
              tfjsLabelsFocused = false;
            }}
            class="inline-flex h-3 w-3 shrink-0 translate-y-px items-center justify-center rounded-full text-fg-subtle transition hover:text-fg-secondary focus-visible:text-fg-secondary focus-visible:ring-2 focus-visible:ring-accent-line focus-visible:outline-none"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.25"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="h-3 w-3"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M12 16v-4" />
              <path d="M12 8h.01" />
            </svg>
          </button>
        {/if}
      </div>
      <p class="mt-0.5 text-[11px] text-fg-muted">
        {m.workspace.import_dialog.pick_target.tfjs_meta_strip(
          formatBytes(tfjsBundleTotalBytes),
          tfjs.shards.length,
          tfjsLabels && tfjsLabels.length > 0 ? tfjsLabels.length : null,
          tfjs.labels ? tfjs.labels.name : null
        )}
      </p>
    </div>

    <!-- Target-heads load indicators + cap-displacement banner. A TFJS import always adds exactly one
         head, so displacement risk is binary (1 if the target is at/over the cap); the warning still
         lets the operator bail before the silent drop. -->
    {#if targetLoading}
      <p class="text-[11px] text-fg-muted">
        {m.workspace.import_dialog.summary.checking_heads}
      </p>
    {/if}
    {#if targetLoadError !== null}
      <p class="text-[11px] text-danger-soft-fg" role="alert">{targetLoadError}</p>
    {/if}
    {#if targetExistingHeads !== null && headsDisplacedByImport > 0}
      <p
        class="rounded-md border border-warning-line bg-warning-soft px-2 py-1 text-[11px] text-warning-soft-fg"
        role="status"
      >
        {m.workspace.import_dialog.summary.displacement_warning(
          headsDisplacedByImport,
          HEAD_HISTORY_CAP
        )}
      </p>
    {/if}

    {#if tfjs.diagnostics.length > 0}
      <div class="flex flex-col gap-1">
        {#each tfjs.diagnostics as d, i (i)}
          <p
            class="text-[11px] {d.severity === 'blocker'
              ? 'text-danger-soft-fg'
              : 'text-warning-soft-fg'}"
            role={d.severity === 'blocker' ? 'alert' : undefined}
          >
            {d.message}
          </p>
        {/each}
      </div>
    {/if}
    {#if tfjs.unknown.length > 0}
      <p class="text-[11px] text-fg-muted">
        {m.workspace.import_dialog.summary.tfjs_ignored_unknown(
          tfjs.unknown.length,
          tfjs.unknown.map((f) => f.name).join(', ')
        )}
      </p>
    {/if}
  {/if}

  <!-- Steps 4 & 5: running/done reuse the selection pane's row layout, each row reflecting its own
       status with per-head log expansion (auto-pinned to the running head). ALPKG renders datasets +
       heads; TFJS renders one sentinel-keyed head row in the shared heads section. -->

  {#if step === 'running' || step === 'done'}
    {#snippet runningSpinner()}
      <!-- Defined inside the running/done branch so it isn't read as a snippet prop of `<Modal>`. -->
      <svg
        class="h-2.5 w-2.5 shrink-0 animate-spin"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="3"
        stroke-linecap="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="9" opacity="0.25" />
        <path d="M21 12 a9 9 0 0 0 -9 -9" />
      </svg>
    {/snippet}
    {#if step === 'running'}
      <div class="flex flex-col gap-1">
        <p class="text-xs text-fg-secondary" aria-live="polite">{progressCopy}</p>
        <div class="h-1 overflow-hidden rounded-full bg-surface-2">
          <div
            class="h-full bg-accent transition-[width] duration-200"
            style="width: {Math.round(progressFraction * 100)}%"
            aria-hidden="true"
          ></div>
        </div>
      </div>
    {/if}

    <!-- Datasets section (ALPKG only), one row per imported bucket. The step-2 mode badge is dropped here
         -- the mode is decided, and the status badge carries the outcome. -->
    {#if branch === 'alpkg' && datasetRunStates.size > 0}
      <section class="flex flex-col gap-2">
        <h3 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
          {m.workspace.import_dialog.summary.datasets_heading}
        </h3>
        <ul class="flex flex-col gap-1.5">
          {#each [...datasetRunStates.values()] as ds (ds.source)}
            {@const renamed = ds.target !== ds.source}
            <li
              class="flex items-center gap-2 rounded-md border border-line bg-surface px-3 py-1.5 text-xs"
            >
              <div class="flex min-w-0 flex-1 items-center gap-1.5">
                <span class="min-w-0 truncate font-medium text-fg" title={ds.source}>
                  {prettyCategoryName(ds.source)}
                </span>
                {#if renamed}
                  <span class="shrink-0 font-medium text-fg-subtle">-></span>
                  <span class="min-w-0 truncate font-medium text-fg" title={ds.target}>
                    {prettyCategoryName(ds.target)}
                  </span>
                {/if}
              </div>
              {#if ds.phase === 'pending'}
                <span
                  class="shrink-0 rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary"
                >
                  {m.workspace.import_dialog.running.ds_pending}
                </span>
              {:else if ds.phase === 'replacing'}
                <span
                  class="inline-flex shrink-0 items-center gap-1 rounded-full bg-warning-soft px-1.5 py-0.5 text-[10px] font-medium text-warning-soft-fg"
                >
                  <!-- no-confusing-void-expression misreads the `{@render}` directive as a void call. -->
                  <!-- eslint-disable-next-line @typescript-eslint/no-confusing-void-expression -->
                  {@render runningSpinner()}
                  {m.workspace.import_dialog.running.ds_replacing}
                </span>
              {:else if ds.phase === 'uploading'}
                <span
                  class="inline-flex shrink-0 items-center gap-1 rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-accent-soft-fg"
                >
                  <!-- eslint-disable-next-line @typescript-eslint/no-confusing-void-expression -->
                  {@render runningSpinner()}
                  {m.workspace.import_dialog.running.ds_uploading_counter(ds.uploaded, ds.total)}
                </span>
              {:else if ds.phase === 'done'}
                <span
                  class="inline-flex shrink-0 items-center gap-1 rounded-full bg-success-soft px-1.5 py-0.5 text-[10px] font-medium text-success-soft-fg"
                >
                  <svg
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2.25"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    class="h-2.5 w-2.5"
                    aria-hidden="true"
                  >
                    <path d="M3 8l3 3 7-7" />
                  </svg>
                  {m.workspace.import_dialog.running.ds_done_uploaded(ds.uploaded)}
                </span>
              {:else if ds.phase === 'failed'}
                <span
                  class="inline-flex shrink-0 items-center gap-1 rounded-full bg-danger-soft px-1.5 py-0.5 text-[10px] font-medium text-danger-soft-fg"
                  title={ds.error ??
                    m.workspace.import_dialog.running.ds_failed_title_count(ds.failed)}
                >
                  <svg
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2.25"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    class="h-2.5 w-2.5"
                    aria-hidden="true"
                  >
                    <path d="M4 4l8 8" />
                    <path d="M12 4l-8 8" />
                  </svg>
                  {ds.failed > 0
                    ? m.workspace.import_dialog.running.ds_failed_count(ds.failed)
                    : m.workspace.import_dialog.running.ds_failed_label}
                </span>
              {/if}
            </li>
          {/each}
        </ul>
      </section>
    {/if}

    <!-- Heads section. Each row is single-disclosure (at most one open, auto-pinned to the running head
         so live log lines render). ALPKG keys one entry per archive head id; TFJS uses one sentinel-keyed
         row. Both share the chrome; only the identity chip swaps (real id vs "TFJS bundle" literal). -->
    {#if headRunStates.size > 0}
      <section class="flex flex-col gap-2">
        <h3 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
          {m.workspace.import_dialog.summary.heads_heading}
        </h3>
        <ul class="flex flex-col gap-1.5">
          {#each [...headRunStates.values()] as hs (hs.headId)}
            {@const isTfjsSentinel = hs.headId === TFJS_HEAD_SENTINEL_ID}
            {@const exists = !isTfjsSentinel && targetExistingHeadIds.has(hs.headId)}
            {@const logs = headRunLogs.get(hs.headId) ?? []}
            {@const expanded = expandedHeadId === hs.headId}
            {@const inFlight =
              hs.phase !== 'queued' &&
              hs.phase !== 'failed' &&
              hs.phase !== 'done' &&
              hs.outcome === null}
            <li class="overflow-hidden rounded-md border border-line bg-surface">
              <button
                type="button"
                onclick={() => {
                  expandedHeadId = expanded ? null : hs.headId;
                }}
                aria-expanded={expanded}
                class="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-surface-2"
              >
                <!-- Disclosure chevron; collapsed `translate-y-px` corrects an optical misalignment between the chevron's path centroid and the text baseline. -->
                <svg
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  aria-hidden="true"
                  class="h-3.5 w-3.5 shrink-0 text-fg-subtle transition-transform duration-200"
                  class:translate-y-px={!expanded}
                  class:rotate-90={expanded}
                >
                  <path
                    fill-rule="evenodd"
                    d="M7.21 5.23a.75.75 0 011.06.02L12 9l-3.73 3.71a.75.75 0 11-1.06-1.06L9.94 9 7.19 6.29a.75.75 0 01.02-1.06z"
                    clip-rule="evenodd"
                  />
                </svg>
                {#if isTfjsSentinel}
                  <span
                    class="shrink-0 rounded bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary"
                  >
                    {m.workspace.import_dialog.pick_target.tfjs_bundle_card_title}
                  </span>
                {:else}
                  <span
                    class="shrink-0 rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-fg-secondary"
                  >
                    {hs.headId.replace(/-/g, '').slice(0, 8)}
                  </span>
                {/if}
                {#if exists}
                  <span
                    class="shrink-0 rounded-full bg-warning-soft px-1.5 py-0.5 text-[10px] font-medium text-warning-soft-fg"
                    title={m.workspace.import_dialog.summary.head_exists_badge_title}
                  >
                    {m.workspace.import_dialog.summary.head_exists_badge}
                  </span>
                {/if}
                {#if logs.length > 0}
                  <span class="shrink-0 font-mono text-[10px] tabular-nums text-fg-subtle">
                    {m.workspace.import_dialog.running.log_count(logs.length)}
                  </span>
                {/if}
                <!-- Status badge, `ml-auto` to a stable outcome column. Reads outcome over phase when both
                     are set (outcome is the terminal post-summary truth); in-flight phases collapse into
                     one "active" pill. -->
                <div class="ml-auto shrink-0">
                  {#if hs.phase === 'queued'}
                    <span
                      class="rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary"
                    >
                      {m.workspace.import_dialog.running.head_queued}
                    </span>
                  {:else if hs.outcome === 'imported' || hs.outcome === 'replaced'}
                    <span
                      class="inline-flex items-center gap-1 rounded-full bg-success-soft px-1.5 py-0.5 text-[10px] font-medium text-success-soft-fg"
                    >
                      <svg
                        viewBox="0 0 16 16"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2.25"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        class="h-2.5 w-2.5"
                        aria-hidden="true"
                      >
                        <path d="M3 8l3 3 7-7" />
                      </svg>
                      {prettyHeadOutcome(hs.outcome)}
                    </span>
                  {:else if hs.outcome === 'skipped'}
                    <span
                      class="rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary"
                      title={m.workspace.import_dialog.running.head_skipped_badge_title}
                    >
                      {m.workspace.import_dialog.head_outcome.skipped}
                    </span>
                  {:else if hs.phase === 'failed' || hs.outcome === 'failed'}
                    <span
                      class="inline-flex items-center gap-1 rounded-full bg-danger-soft px-1.5 py-0.5 text-[10px] font-medium text-danger-soft-fg"
                      title={hs.error ?? undefined}
                    >
                      <svg
                        viewBox="0 0 16 16"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2.25"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        class="h-2.5 w-2.5"
                        aria-hidden="true"
                      >
                        <path d="M4 4l8 8" />
                        <path d="M12 4l-8 8" />
                      </svg>
                      {m.workspace.import_dialog.head_outcome.failed}
                    </span>
                  {:else if hs.phase === 'done'}
                    <!-- Convert done but outcome not yet landed (orchestrator populates outcomes once after
                         all heads resolve). STATIC pill (no spinner) so a finished head doesn't spin on
                         "Done" while later heads convert. -->
                    <span
                      class="rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary"
                    >
                      {prettyHeadPhase(hs.phase)}
                    </span>
                  {:else if inFlight}
                    <span
                      class="inline-flex items-center gap-1 rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-medium text-accent-soft-fg"
                    >
                      <!-- eslint-disable-next-line @typescript-eslint/no-confusing-void-expression -->
                      {@render runningSpinner()}
                      {prettyHeadPhase(hs.phase)}
                    </span>
                  {/if}
                </div>
              </button>
              {#if expanded}
                <!-- Expanded log body; auto-tail via `logScrollEl` + `onLogScroll` + the log `$effect`. -->
                <div
                  transition:slide|local={{ duration: 150 }}
                  class="border-t border-line bg-surface-2"
                >
                  {#if logs.length === 0}
                    <p class="px-3 py-2 text-[11px] text-fg-muted">
                      {#if hs.phase === 'queued'}
                        {m.workspace.import_dialog.running.head_per_log_not_started}
                      {:else}
                        {m.workspace.import_dialog.running.head_per_log_no_events}
                      {/if}
                    </p>
                  {:else}
                    <div
                      bind:this={logScrollEl}
                      onscroll={onLogScroll}
                      class="overflow-y-auto px-3 py-2 text-[11px]"
                      style="height: 144px;"
                      role="log"
                      aria-live={inFlight ? 'polite' : 'off'}
                      aria-relevant="additions"
                    >
                      <ol class="flex flex-col gap-0.5 font-mono leading-snug">
                        {#each logs as line, i (i)}
                          <li class="flex gap-2 text-fg-secondary">
                            <span class="shrink-0 text-fg-subtle tabular-nums">
                              {formatLogTime(line.timestampMs)}
                            </span>
                            <span class="wrap-break-word whitespace-pre-wrap text-fg-secondary">
                              {line.message}
                            </span>
                          </li>
                        {/each}
                      </ol>
                    </div>
                  {/if}
                  <!-- Error + retry (done step only): reconstructs a `HeadOutcomeRecord` from per-row state for `retryWithReplace`. -->
                  {#if step === 'done' && hs.error !== null}
                    <div class="border-t border-line px-3 py-2">
                      <p class="text-[11px] text-danger-soft-fg">{hs.error}</p>
                      {#if hs.conflict}
                        <p class="mt-1 text-[11px] text-fg-secondary">
                          {m.workspace.import_dialog.done.conflict_detail(
                            hs.conflict.storedSha256.slice(0, 8),
                            hs.conflict.incomingSha256.slice(0, 8)
                          )}
                        </p>
                        <div class="mt-2">
                          <Button
                            size="sm"
                            variant="destructive"
                            onclick={() =>
                              retryWithReplace({
                                headId: hs.headId,
                                outcome: 'failed',
                                publishedSha256: null,
                                error: hs.error,
                                conflict: hs.conflict ?? undefined
                              })}
                          >
                            {m.workspace.import_dialog.done.retry_button}
                          </Button>
                        </div>
                      {/if}
                    </div>
                  {/if}
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      </section>
    {/if}
  {/if}

  <!-- TFJS labels popover, one shared surface for both card sites (only one mounts at a time), rendered
       at modal-body end to stack above the card via DOM order + z-50. `position: fixed` escapes the
       modal overflow; `pointer-events-none` lets the cursor pass through so hover lives solely on the
       icon. `max-h-32 overflow-hidden` shows ~30 of a 100+-label bundle, the convert worker being
       authority for full validation. -->
  {#if (step === 'pick-target' || step === 'summary') && tfjsLabelsPopoverOpen && tfjsLabelsPopoverPos && tfjsLabels && tfjsLabels.length > 0}
    <div
      role="tooltip"
      transition:fade|local={{ duration: 150 }}
      class="pointer-events-none fixed z-50 rounded-md border border-line bg-surface shadow-popover ring-1 ring-black/5"
      style:top="{tfjsLabelsPopoverPos.top}px"
      style:left={tfjsLabelsPopoverPos.left !== undefined ? `${tfjsLabelsPopoverPos.left}px` : null}
      style:width={tfjsLabelsPopoverPos.width !== undefined
        ? `${tfjsLabelsPopoverPos.width}px`
        : null}
    >
      <div class="flex flex-col gap-1.5 px-3 py-2">
        <span class="text-[10px] font-medium tracking-wider text-fg-muted uppercase">
          {m.workspace.import_dialog.summary.tfjs_classes_popover_heading(tfjsLabels.length)}
        </span>
        <ul
          class="flex max-h-32 flex-wrap gap-1 overflow-hidden"
          aria-label={m.workspace.import_dialog.summary.tfjs_classes_popover_aria}
        >
          {#each tfjsLabels as label, idx (`${idx}-${label}`)}
            <li
              class="inline-flex max-w-full items-center rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] wrap-break-word text-fg-secondary"
            >
              {prettyCategoryName(label)}
            </li>
          {/each}
        </ul>
      </div>
    </div>
  {/if}
</Modal>
