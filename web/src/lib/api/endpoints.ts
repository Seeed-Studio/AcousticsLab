import { api, ApiError } from './http';
import type {
  ActiveResp,
  AsyncJobAck,
  CancelResp,
  ConvertRequest,
  ConvertStartResp,
  DatasetListing,
  DeleteHeadResp,
  HeadManifest,
  HeadRecord,
  InferenceCfg,
  JobSnapshot,
  LogPageResp,
  MicPolicy,
  MicState,
  RenameCategoryResp,
  StatusSnapshot,
  TrainingCfg,
  TrainingJobView,
  TrainingListResp,
  TrainStartResp,
  Uuid,
  WorkspaceCreateReq,
  WorkspaceDetail,
  WorkspaceListEntry,
  WorkspaceMutationResp,
  WorkspacePatchReq
} from './types';

export const status = {
  get: () => api.get<StatusSnapshot>('/api/v1/status')
};

export const mic = {
  get: (minVersion?: number) => {
    const q = minVersion !== undefined ? `?min_version=${minVersion}` : '';
    return api.get<MicState>(`/api/v1/mic${q}`);
  },
  set: async (policy: MicPolicy): Promise<MicState> => {
    const fresh = await api.post<MicState>('/api/v1/mic', { policy });
    return readYourWrites(fresh.version);
  }
};

// Read-your-writes: re-fetch at min_version, retrying 425 (daemon not yet caught up).
async function readYourWrites(minVersion: number, attempts = 0): Promise<MicState> {
  try {
    return await mic.get(minVersion);
  } catch (err) {
    if (err instanceof ApiError && err.status === 425 && attempts < 3) {
      await sleep(50 * 2 ** attempts);
      return readYourWrites(minVersion, attempts + 1);
    }
    throw err;
  }
}

const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

// Defence-in-depth (the daemon validator is the real security boundary): encodeURIComponent
// leaves `.` intact so a `.`/`..` segment would reach the wire as traversal -- reject it client-side
// for a clear error. Empty segments dropped for leading/trailing-slash tolerance.
function encodeAssetSubPath(subPath: string): string {
  return subPath
    .split('/')
    .filter((seg) => seg.length > 0)
    .map((seg) => {
      if (seg === '.' || seg === '..') {
        throw new Error(`invalid asset path segment: ${JSON.stringify(seg)}`);
      }
      return encodeURIComponent(seg);
    })
    .join('/');
}

export const inference = {
  get: () => api.get<{ cfg: InferenceCfg }>('/api/v1/inference').then((r) => r.cfg),
  set: (cfg: Partial<InferenceCfg>) =>
    api.post<{ cfg: InferenceCfg }>('/api/v1/inference', cfg).then((r) => r.cfg)
};

export const active = {
  get: () => api.get<ActiveResp>('/api/v1/active'),
  setHead: (workspace_id: Uuid, head_id: Uuid) =>
    api.post<ActiveResp>('/api/v1/active', { workspace_id, head_id }),
  setDefault: () => api.post<ActiveResp>('/api/v1/active', { default: true })
};

// `delete` is async: the 202 ack carries a job_id drained via the jobs SSE stream.
export const workspaces = {
  list: () =>
    api.get<{ workspaces: WorkspaceListEntry[] }>('/api/v1/workspaces').then((r) => r.workspaces),
  get: (id: Uuid) => api.get<WorkspaceDetail>(`/api/v1/workspaces/${encodeURIComponent(id)}`),
  create: (req: WorkspaceCreateReq) => api.post<WorkspaceMutationResp>('/api/v1/workspaces', req),
  patch: (id: Uuid, req: WorkspacePatchReq) =>
    api.patch<WorkspaceMutationResp>(`/api/v1/workspaces/${encodeURIComponent(id)}`, req),
  delete: (id: Uuid) => api.delete<AsyncJobAck>(`/api/v1/workspaces/${encodeURIComponent(id)}`),
  // Byte-verbatim `/assets` GET so alpkg export embeds source provenance.
  workspaceCoreAssetPath: (id: Uuid): string =>
    `/api/v1/workspaces/${encodeURIComponent(id)}/assets/workspace.json`
};

function buildPaging(opts: { offset?: number; limit?: number }): string {
  const params = new URLSearchParams();
  if (opts.offset !== undefined) params.set('offset', String(opts.offset));
  if (opts.limit !== undefined) params.set('limit', String(opts.limit));
  const q = params.toString();
  return q ? `?${q}` : '';
}

// Stand-alone so the `assets` aliases survive destructuring without `this`.
export function sliceAssetPath(workspaceId: Uuid, category: string, filename: string): string {
  return `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/datasets/${encodeURIComponent(category)}/${encodeURIComponent(filename)}`;
}

// Every async DELETE ack MUST flow through `enqueueDelete`: the delete-family slot is global
// single-tenant (max_delete_jobs = 1), so parallel deletes 409.
export const assets = {
  listRoot: (workspaceId: Uuid, opts: { offset?: number; limit?: number } = {}) =>
    api.get<DatasetListing>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets${buildPaging(opts)}`
    ),
  // Only categories with >=1 slice on disk; empty-on-disk ones are absent here (the categories
  // store synthesises them from IDB).
  listDatasets: (workspaceId: Uuid, opts: { offset?: number; limit?: number } = {}) =>
    api.get<DatasetListing>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/datasets${buildPaging(opts)}`
    ),
  listCategory: (
    workspaceId: Uuid,
    category: string,
    opts: { offset?: number; limit?: number } = {}
  ) =>
    api.get<DatasetListing>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/datasets/${encodeURIComponent(category)}${buildPaging(opts)}`
    ),
  deleteCategory: (workspaceId: Uuid, category: string) =>
    api.delete<AsyncJobAck>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/datasets/${encodeURIComponent(category)}`
    ),
  // SYNCHRONOUS (atomic rename(2), no job) on `/datasets/.../rename`, not `/assets`; returns the
  // new workspace_revision_id (current on a same-name no-op).
  renameCategory: (workspaceId: Uuid, category: string, toName: string) =>
    api.post<RenameCategoryResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/datasets/${encodeURIComponent(category)}/rename`,
      { to_name: toName }
    ),
  deleteSlice: (workspaceId: Uuid, category: string, filename: string) =>
    api.delete<AsyncJobAck>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/datasets/${encodeURIComponent(category)}/${encodeURIComponent(filename)}`
    ),
  // Generic asset DELETE for a workspace-rooted path; daemon allowlists `datasets/`,
  // `converters/`, `*_logs/`. Helper strips edge slashes against empty/ambiguous segments.
  deletePath: (workspaceId: Uuid, workspaceRootedPath: string) => {
    const encoded = encodeAssetSubPath(workspaceRootedPath);
    return api.delete<AsyncJobAck>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/${encoded}`
    );
  },
  sliceAssetPath,
  slicePutPath: sliceAssetPath
};

// Unified `/jobs` family over every producer. `eventsUrl` only builds the path; SSE consumers own
// the EventSource lifecycle. Train events arrive as a JSON-stringified TrainEvent in
// JobEvent.message; the workspace-scoped `training[*]` endpoints coexist for per-workspace lists +
// `started_at` snapshot recovery.
export const jobs = {
  get: (id: Uuid) => api.get<JobSnapshot>(`/api/v1/jobs/${encodeURIComponent(id)}`),
  eventsUrl: (id: Uuid, opts: { afterSeq?: number; logs?: boolean } = {}): string => {
    const params = new URLSearchParams();
    if (opts.afterSeq !== undefined) params.set('after_seq', String(opts.afterSeq));
    if (opts.logs !== undefined) params.set('logs', String(opts.logs));
    const q = params.toString();
    return `/api/v1/jobs/${encodeURIComponent(id)}/events${q ? `?${q}` : ''}`;
  }
};

// `start` returns a pre-allocated head id + job id; the head record commits only on successful
// publish. Progress streams SSE on `/jobs/{id}/events`; `get` is a recovery snapshot. `cancel` is
// idempotent only while Running (re-sets a worker-polled flag, so the `cancelled` terminal lands at
// the next checkpoint); once the job leaves Running it 409s JobNotCancellable.
export const training = {
  start: (workspaceId: Uuid, cfg: TrainingCfg) =>
    api.post<TrainStartResp>(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/train`, cfg),
  list: (workspaceId: Uuid) =>
    api
      .get<TrainingListResp>(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/training`)
      .then((r) => r.jobs),
  get: (workspaceId: Uuid, jobId: Uuid) =>
    api.get<TrainingJobView>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/training/${encodeURIComponent(jobId)}`
    ),
  cancel: (workspaceId: Uuid, jobId: Uuid) =>
    api.delete<CancelResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/training/${encodeURIComponent(jobId)}`
    ),
  // Single-file delete (not bare-tree, avoids global-slot collision). 409 if a train producer holds
  // the tree, 404 if the keep-last-10 reaper pruned it, 400 if malformed. Call sites MUST
  // `enqueueDelete` then `awaitJobTerminal` before treating the row as gone.
  deleteLog: (workspaceId: Uuid, jobId: Uuid) =>
    api.delete<AsyncJobAck>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/training_logs/${encodeURIComponent(jobId)}.jsonl`
    ),
  // Durable JSONL backstop, sparse (only `started` + one terminal event): surfaces a terminal
  // reason after reload past in-memory retention.
  logPath: (workspaceId: Uuid, jobId: Uuid): string =>
    `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/training_logs/${encodeURIComponent(jobId)}.jsonl`,
  // `after_seq` is exclusive: yields `seq > after_seq` (default 0, limit 200).
  readLogPage: (
    workspaceId: Uuid,
    jobId: Uuid,
    opts: { afterSeq?: number; limit?: number } = {}
  ) => {
    const params = new URLSearchParams();
    if (opts.afterSeq !== undefined) params.set('after_seq', String(opts.afterSeq));
    if (opts.limit !== undefined) params.set('limit', String(opts.limit));
    const q = params.toString();
    return api.get<LogPageResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/training_logs/${encodeURIComponent(jobId)}.jsonl${q ? `?${q}` : ''}`
    );
  },
  // Hydrates history past in-memory retention. Never-trained workspace yields empty `entries`
  // (synthesised, not 404); missing workspace 404s. Server-sorts by name (= jobId), so consumers
  // MUST re-sort by mtime for chronology. limit clamps 1000; default 100.
  listLogs: (workspaceId: Uuid, opts: { offset?: number; limit?: number } = {}) =>
    api.get<DatasetListing>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/training_logs${buildPaging(opts)}`
    )
};

// `list` omits `labels` (built from the cached index, no per-head manifest) -- use `manifest()`.
// `delete` is synchronous, 409s if the head is the active generation's source.
// `weightsAssetPath`/`manifestAssetPath` use the verbatim-bytes `/assets` GET, not `/heads/{id}`,
// because alpkg export content-addresses payloads.
export const heads = {
  list: (workspaceId: Uuid) =>
    api
      .get<{ heads: HeadRecord[] }>(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/heads`)
      .then((r) => r.heads),
  manifest: (workspaceId: Uuid, headId: Uuid) =>
    api.get<HeadManifest>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/heads/${encodeURIComponent(headId)}`
    ),
  delete: (workspaceId: Uuid, headId: Uuid) =>
    api.delete<DeleteHeadResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/heads/${encodeURIComponent(headId)}`
    ),
  weightsAssetPath: (workspaceId: Uuid, headId: Uuid): string =>
    `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/heads/${encodeURIComponent(headId)}.mpk`,
  manifestAssetPath: (workspaceId: Uuid, headId: Uuid): string =>
    `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/heads/${encodeURIComponent(headId)}.json`
};

// `start` returns `{ head_id, job_id }`; head_id is pre-allocated (TFJS) or echoed from the
// embedded manifest (alpkg-head) so the UI can match the published head before SSE replay catches
// up; progress streams on `/jobs/{job_id}/events`. The upload handler rejects a bare `converters`
// (uploads require >=1 child), hence we always namespace under `alpkg/`/`tfjs/`; the AssetPath
// validator separately caps per-component `[A-Za-z0-9._-]` <=255B, total <=256B, depth <=8.
// `deleteAll` wipes the tree and MUST flow through `enqueueDelete`.
export const converters = {
  start: (workspaceId: Uuid, req: ConvertRequest) =>
    api.post<ConvertStartResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/convert`,
      req
    ),
  // `subPath` splices in untrusted operator-chosen File.name (TFJS).
  putAssetPath: (workspaceId: Uuid, subPath: string): string =>
    `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/converters/${encodeAssetSubPath(subPath)}`,
  deleteAll: (workspaceId: Uuid) =>
    api.delete<AsyncJobAck>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/converters`
    ),
  // Single sub-tree delete: post-convert cleanup prunes only the just-uploaded tree so concurrent
  // imports elsewhere aren't disturbed.
  deletePath: (workspaceId: Uuid, subPath: string) =>
    api.delete<AsyncJobAck>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/converters/${encodeAssetSubPath(subPath)}`
    ),
  // `after_seq` is exclusive: yields `seq > after_seq` (default 0, limit 200).
  readLogPage: (
    workspaceId: Uuid,
    jobId: Uuid,
    opts: { afterSeq?: number; limit?: number } = {}
  ) => {
    const params = new URLSearchParams();
    if (opts.afterSeq !== undefined) params.set('after_seq', String(opts.afterSeq));
    if (opts.limit !== undefined) params.set('limit', String(opts.limit));
    const q = params.toString();
    return api.get<LogPageResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/converter_logs/${encodeURIComponent(jobId)}.jsonl${q ? `?${q}` : ''}`
    );
  },
  logPath: (workspaceId: Uuid, jobId: Uuid): string =>
    `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/converter_logs/${encodeURIComponent(jobId)}.jsonl`,
  // Hydrates import history past in-memory pane state. Never-converted workspace yields empty
  // `entries` (synthesised, not 404); missing workspace 404s. limit clamps 1000; default 100.
  listLogs: (workspaceId: Uuid, opts: { offset?: number; limit?: number } = {}) =>
    api.get<DatasetListing>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/converter_logs${buildPaging(opts)}`
    ),
  // Async single-file delete. 409 if a convert producer is active, 404 if the keep-last-10 reaper
  // pruned it, 400 if malformed. Call sites MUST `enqueueDelete` then `awaitJobTerminal` first.
  deleteLog: (workspaceId: Uuid, jobId: Uuid) =>
    api.delete<AsyncJobAck>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/assets/converter_logs/${encodeURIComponent(jobId)}.jsonl`
    )
};
