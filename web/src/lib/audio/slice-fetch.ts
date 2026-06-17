import { apiUrl } from '$lib/api/base';
import { sliceAssetPath } from '$lib/api/endpoints';
import { ApiError } from '$lib/api/http';
import { UploadPool } from '$lib/api/upload';
import { sliceFilename } from '$lib/idb/db';
import { sha256Hex } from './sha256';
import type { ApiErrorBody } from '$lib/api/types';
import type { SliceRecord } from '$lib/idb/db';

// Per-slice WAV-blob fetch + cache. Local slices carry `slice.blob`; others lazy-fetch from the
// daemon. Keyed by sha256 (= `slice.id`) so identical bytes across categories share one blob;
// an in-memory Map that grows for the process lifetime and is discarded only on a full page reload
// (`resetDB` deletes the on-disk IndexedDB but never touches this Map).

const MAX_CONCURRENT_DOWNLOADS = 6;
const downloadPool = new UploadPool(MAX_CONCURRENT_DOWNLOADS);

const blobCache = new Map<string, Blob>();
const inflight = new Map<string, Promise<Blob>>();

export function sliceAssetUrl(
  slice: Pick<SliceRecord, 'workspace_id' | 'category_name' | 'id'>
): string {
  return sliceAssetPath(slice.workspace_id, slice.category_name, sliceFilename(slice.id));
}

// Concurrent calls for one hash dedup via `inflight` so simultaneous mounts share a round-trip.
export async function getSliceBlob(slice: SliceRecord): Promise<Blob> {
  if (slice.blob && slice.blob.size > 0) return slice.blob;
  const cached = blobCache.get(slice.id);
  if (cached) return cached;
  const pending = inflight.get(slice.id);
  if (pending) return pending;
  const work = (async (): Promise<Blob> => {
    try {
      const blob = await downloadPool.submit(() => fetchSliceBlob(slice));
      blobCache.set(slice.id, blob);
      return blob;
    } finally {
      inflight.delete(slice.id);
    }
  })();
  inflight.set(slice.id, work);
  return work;
}

async function fetchSliceBlob(slice: SliceRecord): Promise<Blob> {
  const resp = await fetch(apiUrl(sliceAssetUrl(slice)));
  if (!resp.ok) {
    let body: ApiErrorBody;
    try {
      const parsed: unknown = await resp.json();
      body =
        parsed && typeof parsed === 'object' && 'error' in parsed && 'code' in parsed
          ? (parsed as ApiErrorBody)
          : { error: resp.statusText || `HTTP ${resp.status}`, code: 'unknown' };
    } catch {
      body = { error: resp.statusText || `HTTP ${resp.status}`, code: 'unknown' };
    }
    throw new ApiError(resp.status, body);
  }
  const blob = await resp.blob();
  // The slice id is the sha256 of its WAV bytes, so re-hashing the download catches
  // corruption/tampering/mis-targeting; mismatch throws (caller leaves cache unpopulated).
  const buf = await blob.arrayBuffer();
  const observed = await sha256Hex(buf);
  if (observed !== slice.id) {
    throw new Error(
      `Slice ${slice.id} content mismatch: daemon returned bytes hashing to ${observed}`
    );
  }
  return blob;
}
