import { getDB, STORE_DRAFTS, type DraftRecord } from './db';

// Single-slot per `(workspace_id, category_name)` key: a `put` overwrites in place, keeping only the most recent clip.

export async function getDraft(
  workspaceId: string,
  categoryName: string
): Promise<DraftRecord | undefined> {
  const db = await getDB();
  return db.get(STORE_DRAFTS, [workspaceId, categoryName]);
}

export async function putDraft(record: DraftRecord): Promise<void> {
  const db = await getDB();
  await db.put(STORE_DRAFTS, record);
}

export async function deleteDraft(workspaceId: string, categoryName: string): Promise<void> {
  const db = await getDB();
  await db.delete(STORE_DRAFTS, [workspaceId, categoryName]);
}
