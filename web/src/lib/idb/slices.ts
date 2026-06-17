import { getDB, STORE_SLICES, type SliceKey, type SliceRecord } from './db';

// Primary key `[workspace_id, category_name, id]` with `id` = WAV-bytes sha256 hex: identical
// content coexists across categories but dedupes by overwrite within one (workspace, category).

function byCreatedAsc(a: SliceRecord, b: SliceRecord): number {
  if (a.created_at === b.created_at) return 0;
  return a.created_at < b.created_at ? -1 : 1;
}

export function sliceKey(record: SliceRecord): SliceKey {
  return [record.workspace_id, record.category_name, record.id];
}

export async function listSlicesForCategory(
  workspaceId: string,
  categoryName: string
): Promise<SliceRecord[]> {
  const db = await getDB();
  const rows = await db.getAllFromIndex(
    STORE_SLICES,
    'by-workspace-category',
    IDBKeyRange.only([workspaceId, categoryName])
  );
  return rows.sort(byCreatedAsc);
}

export async function putSlice(record: SliceRecord): Promise<void> {
  const db = await getDB();
  await db.put(STORE_SLICES, record);
}

export async function bulkPutSlices(records: readonly SliceRecord[]): Promise<void> {
  if (records.length === 0) return;
  const db = await getDB();
  const tx = db.transaction(STORE_SLICES, 'readwrite');
  // Await each per-put promise, not just tx.done: a failing put aborts the tx, and subscribing to
  // every per-op promise surfaces it as a rejection instead of a browser unhandled-rejection.
  await Promise.all([...records.map((r) => tx.store.put(r)), tx.done]);
}

export async function bulkDeleteSlices(keys: readonly SliceKey[]): Promise<void> {
  if (keys.length === 0) return;
  const db = await getDB();
  const tx = db.transaction(STORE_SLICES, 'readwrite');
  await Promise.all([...keys.map((k) => tx.store.delete(k)), tx.done]);
}

export async function deleteSlice(
  workspaceId: string,
  categoryName: string,
  id: string
): Promise<void> {
  const db = await getDB();
  await db.delete(STORE_SLICES, [workspaceId, categoryName, id]);
}

export async function deleteSlicesForCategory(
  workspaceId: string,
  categoryName: string
): Promise<number> {
  const db = await getDB();
  const tx = db.transaction(STORE_SLICES, 'readwrite');
  const index = tx.store.index('by-workspace-category');
  let deleted = 0;
  let cursor = await index.openCursor(IDBKeyRange.only([workspaceId, categoryName]));
  while (cursor) {
    await cursor.delete();
    deleted++;
    cursor = await cursor.continue();
  }
  await tx.done;
  return deleted;
}
