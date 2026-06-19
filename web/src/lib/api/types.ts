// Mirrors the acousticsd REST contract; the Rust modules are authoritative.

export type Uuid = string;
export type Rfc3339 = string;

export interface WorkspaceRevision {
  id: number;
  at: Rfc3339;
}

// GET /api/v1/workspaces row; omits tags/revision/heads (cached `workspace.json` read, no asset walk).
export interface WorkspaceListEntry {
  id: Uuid;
  name: string;
  created_at: Rfc3339;
}

// GET /api/v1/workspaces/{id} summary; `tags` absent (only POST/PATCH carry it).
export interface WorkspaceDetail {
  id: Uuid;
  name: string;
  created_at: Rfc3339;
  workspace_revision: WorkspaceRevision;
  heads: HeadRecord[];
}

// POST/PATCH /api/v1/workspaces/{id}; the only response carrying `tags`.
export interface WorkspaceMutationResp {
  id: Uuid;
  name: string;
  tags: string[];
  created_at: Rfc3339;
  workspace_revision: WorkspaceRevision;
}

export interface WorkspaceCreateReq {
  name: string;
  tags?: string[];
}

export interface WorkspacePatchReq {
  name?: string;
  tags?: string[];
}

// 202 ack for DELETE workspace/asset; terminal state via GET /jobs/{job_id}/events.
export interface AsyncJobAck {
  job_id: Uuid;
}

// .../datasets/{name}/rename is synchronous (`rename(2)`): returns 200 + revision, not an `AsyncJobAck`.
export interface RenameCategoryResp {
  workspace_revision_id: number;
}

export type HeadStatus = 'current' | 'stale';

// List rows add derived `status`; the manifest streams the disk file verbatim so omits `status` (a derived-on-read field would lie on stale reads).
interface HeadCore {
  head_id: Uuid;
  workspace_revision: WorkspaceRevision;
  sha256: string;
  n_classes: number;
  size_bytes: number;
  created_at: Rfc3339;
}

export interface HeadRecord extends HeadCore {
  // Daemon-derived freshness (current/stale by revision id). Kept for the wire contract but NOT
  // rendered: the deploy UI derives freshness from `liveRevision`, which leads the daemon's basis
  // during the optimistic post-upload window. See HeadsTable.svelte.
  status: HeadStatus;
}

// GET .../heads/{id} manifest. `workspace_id` provenance feeds alpkg cross-workspace integrity checks; `labels` is omitted from list rows to keep enumeration cheap.
export interface HeadManifest extends HeadCore {
  workspace_id: Uuid;
  labels: string[];
}

export type ActiveOrigin = 'head' | 'default';

interface ActiveBase {
  sha256: string;
  labels_sha256: string;
  runtime_head_id: Uuid;
  n_classes: number;
  labels: string[];
  activated_at: Rfc3339;
  activation_id: Uuid;
}

// Revision is `workspace_revision`, NOT `source_workspace_revision` (a `source_`-prefixed name reads undefined `.id` and throws on activation). `source_workspace_alive` is GET-only (POST drops it via `skip_serializing_if`, GET re-derives from live `is_dir()`): treat `undefined` as alive and test `=== false` for orphaned, else a truthy `!alive` flips just-activated heads into "deleted" until the next GET.
export type ActiveResp =
  | (ActiveBase & {
      origin: 'head';
      source_workspace_id: Uuid;
      workspace_revision: WorkspaceRevision;
      source_head_id: Uuid;
      source_workspace_alive?: boolean;
    })
  | (ActiveBase & { origin: 'default' });

// POST .../train body. `seed: null` = daemon entropy; `validation_split: 0` disables the stratified split and publishes the last-epoch head (else best epoch wins, accuracy-first then val-loss).
export interface TrainingCfg {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  seed?: number | null;
  validation_split?: number;
}

// `head_id` is pre-allocated so a running job matches the eventual head record.
export interface TrainStartResp {
  head_id: Uuid;
  job_id: Uuid;
}

// Pipeline stage for failure attribution. `save` = atomic local `.mpk` write; `publish` = rotation into `<workspace>/heads/`.
export type Stage = 'prepare' | 'dataset_scan' | 'feature_extract' | 'train' | 'save' | 'publish';

// Present only when `progress.phase === 'train'`; every `val*` is `null` when `validation_split === 0` (daemon serialises NaN to JSON null).
export interface EpochMetrics {
  epoch: number;
  epochs: number;
  train_loss: number;
  train_acc: number;
  val_acc: number | null;
  best_val_acc: number | null;
  // Mean validation cross-entropy; secondary selection key (accuracy-first, loss-second).
  val_loss: number | null;
  // Best published snapshot's val loss; `null` before the first best lands (daemon `+inf`).
  best_val_loss: number | null;
}

// GET .../training/{job} returns this verbatim to recover a running job after reload; live progress reconstructs it client-side from SSE.
export interface TrainingProgress {
  phase: Stage;
  current: number;
  total: number;
  message: string;
  metrics?: EpochMetrics;
}

// `final_val_acc` null when `validation_split === 0` (NaN serialised to JSON null).
export interface TrainingResult {
  head_id: Uuid;
  head_sha256: string;
  n_classes: number;
  classes: string[];
  final_train_acc: number;
  final_val_acc: number | null;
}

// No `queued`: the producer transitions to `running` synchronously at admission.
export type TrainingJobState = 'running' | 'completed' | 'failed' | 'cancelled';

// GET .../training[/{job}]. Unlike `JobSnapshot`, carries phase + per-epoch metrics (unified `/jobs` stores only `JobProgress { done, total? }`).
export interface TrainingJobView {
  job_id: Uuid;
  workspace_id: Uuid;
  state: TrainingJobState;
  progress: TrainingProgress;
  result?: TrainingResult | null;
  error?: string | null;
  started_at: Rfc3339;
  finished_at?: Rfc3339 | null;
}

export interface TrainingListResp {
  jobs: TrainingJobView[];
}

// DELETE .../training/{job} ack: cancel flag set synchronously; the worker exits `state: cancelled` at its next checkpoint.
export interface CancelResp {
  ok: true;
}

// DELETE .../heads/{head_id} response; synchronous, no `AsyncJobAck` job machinery.
export interface DeleteHeadResp {
  deleted_head_id: Uuid;
}

// One JSONL line: only `seq`/`at` fixed, rest producer-defined (`serde(flatten)`). Train/converter logs share this envelope; narrow on `kind` (`TrainEvent`/`ConvertEvent`) and skip unknown for forward-compat.
export interface LogEvent {
  seq: number;
  at: Rfc3339;
  [key: string]: unknown;
}

export interface ClassCount {
  name: string;
  n_samples: number;
}

// `operator_fixable` = user-fixable dataset/upload (amber card); `internal` = daemon panic/IO/corruption, retry-only (red card).
export type Severity = 'operator_fixable' | 'internal';

// `operator` = DELETE .../training/{job}; `shutdown` = daemon pre-drain hook.
export type CancelReason = 'operator' | 'shutdown';

// `kind: 'job_failed'` payload, discriminated on `category`; per-variant fields feed hint copy without re-parsing the free-form `error` string.
export type FailPayload =
  | { category: 'bad_dataset'; path: string; reason: string }
  | { category: 'dataset_read'; path: string; reason: string }
  | {
      category: 'empty_class';
      class: string;
      per_class_kept: readonly (readonly [string, number])[];
    }
  // Post-preproc sibling of `empty_class`: survived scan but every example failed decode/resample/finiteness.
  | {
      category: 'empty_class_after_extract';
      class: string;
      per_class_kept: readonly (readonly [string, number])[];
      per_class_dropped: readonly (readonly [string, number])[];
    }
  | {
      category: 'drop_ratio_exceeded';
      dropped: number;
      total: number;
      threshold: number;
      per_class_kept: readonly (readonly [string, number])[];
      per_class_dropped: readonly (readonly [string, number])[];
    }
  // One class crossed the per-class drop cap while the aggregate ratio stayed under `drop_ratio_exceeded`'s ceiling.
  | {
      category: 'per_class_drop_exceeded';
      class: string;
      dropped: number;
      total: number;
      threshold: number;
      per_class_kept: readonly (readonly [string, number])[];
      per_class_dropped: readonly (readonly [string, number])[];
    }
  | {
      category: 'stratified_split_impossible';
      class: string;
      per_class_kept: readonly (readonly [string, number])[];
      val_split: number;
    }
  | { category: 'invalid_config'; detail: string }
  | { category: 'model_error'; detail: string }
  // Mid-training NaN/+Inf loss; pre-step abort leaves the prior best-epoch head usable.
  | { category: 'numeric_failure'; epoch: number; batch_index: number; kind: string; value: number }
  // `io` not `io_error`: serde snake_case of the single-word `Io` variant; renaming to `IoError` must update this literal in lock-step.
  | { category: 'io'; path: string; detail: string }
  | { category: 'panic'; detail: string }
  | { category: 'internal'; detail: string };

// Typed training events, discriminated on snake_case `kind`; `seq`/`at` come from `TrainLogLine`, not per variant. Transports: JSONL as `LogEvent`; SSE as `JobEvent` whose `message` is JSON to `JSON.parse` then `kind`-narrow. Skip unknown `kind`.
export type TrainEvent =
  | {
      kind: 'job_submitted';
      head_id: Uuid;
      cfg: TrainingCfg;
      backbone: string;
    }
  | { kind: 'job_running' }
  | { kind: 'phase_started'; phase: Stage }
  | {
      kind: 'dataset_scanned';
      n_classes: number;
      classes: ClassCount[];
      n_examples_total: number;
    }
  | {
      kind: 'feature_extract_completed';
      kept: number;
      dropped_nan: number;
      dropped_io: number;
      elapsed_ms: number;
    }
  | { kind: 'train_split'; train_n: number; val_n: number }
  | {
      kind: 'epoch_completed';
      epoch: number;
      epochs: number;
      train_loss: number;
      train_acc: number;
      val_acc: number | null;
      best_val_acc: number | null;
      val_loss: number | null;
      best_val_loss: number | null;
      lr: number;
      elapsed_ms: number;
    }
  | {
      kind: 'train_completed';
      epochs_run: number;
      total_elapsed_ms: number;
      best_val_epoch?: number;
      best_val_acc?: number | null;
      best_val_loss?: number | null;
    }
  | {
      kind: 'head_published';
      head_id: Uuid;
      head_sha256: string;
      size_bytes: number;
      n_classes: number;
      classes: string[];
      workspace_revision: WorkspaceRevision;
    }
  | { kind: 'job_completed'; result: TrainingResult }
  | ({
      kind: 'job_failed';
      stage: Stage;
      severity: Severity;
      error: string;
    } & FailPayload)
  | { kind: 'job_cancelled'; stage: Stage; reason: CancelReason };

// Intersection, not `interface extends`, because interfaces cannot extend a discriminated union.
export type TrainLogLine = TrainEvent & {
  seq: number;
  at: Rfc3339;
};

// `ConvertEvent['job_submitted'].converter` and POST /convert `converter_type`.
export type ConverterType = 'tfjs' | 'alpkg';

export type ConvertStage =
  // Shared.
  | 'prepare'
  | 'publish_head'
  // Alpkg-only.
  | 'read_manifest'
  | 'validate_manifest'
  | 'verify_mpk'
  | 'stage_mpk'
  // Tfjs-only.
  | 'read_model_json'
  | 'stage_shards'
  | 'extract_weights'
  | 'read_labels'
  | 'stage_head_mpk';

// `kind: 'job_failed'` payload, discriminated on `category`, feeding hint-card copy. The daemon collapses all TFJS parse failures to `source_malformed` (advice is identical: re-export).
export type ConvertFailPayload =
  | { category: 'source_malformed'; detail: string }
  | { category: 'limit_exceeded'; what: string; value: number; max: number }
  | { category: 'bad_class_count'; got: number; max: number }
  | { category: 'labels'; detail: string }
  | { category: 'alpkg_manifest_schema'; reason: string }
  | { category: 'alpkg_size_mismatch'; expected: number; observed: number }
  | { category: 'alpkg_hash_mismatch'; expected: string; observed: string }
  | {
      category: 'head_id_collision';
      head_id: string;
      got_sha256: string;
      stored_sha256: string;
    }
  | { category: 'internal'; detail: string };

// `kind: 'job_completed'` summary; self-sufficient so JSONL hydration on tab-refresh re-renders the banner without a `/heads/{head_id}` fetch.
export interface ConvertResult {
  head_id: Uuid;
  head_sha256: string;
  n_classes: number;
  classes: string[];
}

// Same envelope/SSE transport as `TrainEvent`. Skip unknown `kind`. Convert jobs are not cancellable (no cancellation variant).
export type ConvertEvent =
  | { kind: 'job_submitted'; head_id: Uuid; converter: ConverterType }
  | { kind: 'job_running' }
  | { kind: 'stage_started'; stage: ConvertStage }
  | { kind: 'manifest_validated'; n_classes: number; sha256: string }
  | { kind: 'mpk_verified'; size_bytes: number; sha256: string }
  | { kind: 'weights_extracted'; n_classes: number; in_dim: number }
  | { kind: 'labels_loaded'; n_labels: number }
  | {
      kind: 'head_published';
      head_id: Uuid;
      head_sha256: string;
      size_bytes: number;
      n_classes: number;
      classes: string[];
      workspace_revision: WorkspaceRevision;
      idempotent_skip: boolean;
    }
  | { kind: 'job_completed'; result: ConvertResult }
  | ({
      kind: 'job_failed';
      stage: ConvertStage;
      severity: Severity;
      error: string;
    } & ConvertFailPayload);

// JSONL paging for GET .../assets/{*path}. `next_after_seq` = last yielded `seq`, or the caller's `after_seq` on an empty page, so `next_after_seq === after_seq` means caught up.
export interface LogPageResp {
  events: LogEvent[];
  next_after_seq: number;
}

export type JobState = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type JobType =
  | 'train'
  | 'convert'
  | 'dataset_delete'
  | 'converter_delete'
  | 'workspace_delete'
  | 'training_logs_delete'
  | 'converter_logs_delete';

export interface JobProgress {
  done: number;
  // Backend `Option<u64>` lacks `skip_serializing_if`: None is explicit `null` (not absent), so `| null`.
  total?: number | null;
}

// GET /jobs[/{job_id}]. Only `target_path` is skipped-when-None (asset-delete jobs); `workspace_id`/`progress`/`result` serialise None as explicit `null` - `workspace_id`/`progress` are typed `T | null`, `result` as plain `unknown` (which already subsumes null).
export interface JobSnapshot {
  job_id: Uuid;
  job_type: JobType;
  workspace_id?: Uuid | null;
  target_path?: string;
  state: JobState;
  progress?: JobProgress | null;
  result?: unknown;
  last_seq: number;
  updated_at: Rfc3339;
}

// SSE over GET /jobs/{job_id}/events: any mix of state transition, progress tick, log line; react to whichever fields are present.
export interface JobEvent {
  seq: number;
  at: Rfc3339;
  state?: JobState;
  progress?: JobProgress;
  message?: string;
}

export interface SubsystemHealth {
  healthy: boolean;
  detail?: string;
  degraded_reason?: string;
  age_ms?: number;
  stale: boolean;
}

// Flat /status; all counters at the root. Runtime head + labels are NOT here -- query /active.
export interface StatusSnapshot {
  cpu_pct: number;
  mem_rss_kb: number;
  disk_free_kb: number;
  metrics_age_ms: number;
  metrics_stale: boolean;
  uptime_s: number;
  subsystems: Record<string, SubsystemHealth>;
  broadcast_audio_messages_dropped: number;
  broadcast_inference_messages_dropped: number;
  workspace: Record<string, number>;
  // `serde(default)` lets an older daemon omit it.
  config_reload?: ConfigReloadSnapshot;
}

// `rejected` bumps when the watcher discards a reload on parse/validate/callback failure.
export interface ConfigReloadSnapshot {
  reloads_succeeded_total: number;
  reloads_rejected_total: number;
}

export interface InferenceCfg {
  hop_samples: number;
  top_k: number;
}

// `alsa` negotiates the actual rate at open time, hence no `sample_rate` here (unlike `mock`'s static one).
export interface AlsaMicSource {
  kind: 'alsa';
  hw_spec: string;
  period_size: number;
  buffer_size: number;
}

export interface MockMicSource {
  kind: 'mock';
  sample_rate: number;
  period_size: number;
  waveforms: unknown[];
}

export type MicSource = AlsaMicSource | MockMicSource;

export interface MicCandidate {
  id: string;
  source: MicSource;
  channels: number[];
}

export interface MicCatalogue {
  candidates: MicCandidate[];
}

export type MicPolicyMic = { kind: 'first_available' } | { kind: 'fixed'; id: string };
export type MicPolicyChannel = { kind: 'auto' } | { kind: 'fixed'; channel: number };

export interface MicPolicy {
  mic: MicPolicyMic;
  channel: MicPolicyChannel;
}

export interface MicState {
  catalogue: MicCatalogue;
  policy: MicPolicy;
  version: number;
}

// One direct child; `size_bytes` null on directories (listing never walks). Literal is `'directory'` NOT `'dir'`: a `=== 'dir'` filter silently matches nothing.
export interface AssetEntry {
  name: string;
  kind: 'file' | 'directory';
  size_bytes: number | null;
  mtime: Rfc3339;
}

// Directory response for GET .../assets[/{*path}] (file reads return raw bytes). `offset`/`limit` echo the request (defaults 0/100, max 1000).
export interface DatasetListing {
  entries: AssetEntry[];
  total: number;
  offset: number;
  limit: number;
}

// `'lines'` = `labels.txt` one-per-line; `'tfjs_metadata'` = TFJS/Teachable-Machine `metadata.json` (labels under `wordLabels`/`words`).
export type LabelsFormat = 'lines' | 'tfjs_metadata';

// TFJS bundle convert body; daemon rejects unknown keys (`deny_unknown_fields`). Shards are derived from `model.json`'s `weightsManifest[].paths` (rooted at its parent), not named here. Paths below are converter-rooted (no leading slash).
export interface TfjsConvertParams {
  converter_type: 'tfjs';
  model_json_path: string;
  labels_path: string;
  labels_format: LabelsFormat;
}

// `.alpkg` head-import body; daemon rejects unknown keys (`deny_unknown_fields`). Operator names only the `.json` manifest; daemon derives sibling `<parent>/<head_id>.mpk`, validates sha256/n_classes/labels, publishes into `<workspace>/heads/`. Idempotency: same head_id+sha256 is a no-op; same head_id+different sha256 is 409 (delete before re-importing a divergent head).
export interface AlpkgConvertParams {
  converter_type: 'alpkg';
  // Converter-rooted (no leading slash); must end `.json` so sibling `.mpk` derivation is well-defined.
  manifest_path: string;
}

export type ConvertRequest = TfjsConvertParams | AlpkgConvertParams;

// `head_id` is pre-allocated so the UI can match the published head before the SSE replay catches up.
export interface ConvertStartResp {
  head_id: Uuid;
  job_id: Uuid;
}

export interface AssetReceipt {
  path: string;
  sha256: string;
  size_bytes: number;
  workspace_revision_id: number;
}

export interface ApiErrorBody {
  error: string;
  code: string;
  oldest_seq?: number;
  latest_seq?: number;
}
