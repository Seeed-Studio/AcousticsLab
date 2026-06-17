import { getDB, STORE_WORKSPACE_SYNC, type WorkspaceSyncRecord } from './db';

// One row per workspace holding the highest daemon revision id provably synced; an id >= the live
// revision on mount short-circuits per-category index reconcile. Written only when synced state is
// provably complete: a full successful reconcile, or a single committed upload that is the sole
// mutation in the +1 gap (markCommitted). A missing id re-triggers reconcile and self-heals; a
// stale id skips it and leaks drift.

export async function getWorkspaceSync(
  workspaceId: string
): Promise<WorkspaceSyncRecord | undefined> {
  const db = await getDB();
  return db.get(STORE_WORKSPACE_SYNC, workspaceId);
}

export async function putWorkspaceSync(record: WorkspaceSyncRecord): Promise<void> {
  const db = await getDB();
  await db.put(STORE_WORKSPACE_SYNC, record);
}

export async function deleteWorkspaceSync(workspaceId: string): Promise<void> {
  const db = await getDB();
  await db.delete(STORE_WORKSPACE_SYNC, workspaceId);
}
