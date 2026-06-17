import { getDB, STORE_CATEGORIES, type CategoryRecord } from './db';

// Operator-added categories not yet materialised on the daemon (no slice uploaded, so the
// server directory doesn't exist). After upload the row is a redundant passive copy left in
// place: the name-keyed merge across IDB + server dedups it, so GC is deferred until
// measurement proves it matters.

function byCreatedDesc(a: CategoryRecord, b: CategoryRecord): number {
  if (a.created_at === b.created_at) return 0;
  return a.created_at < b.created_at ? 1 : -1;
}

export async function listCategoriesForWorkspace(workspaceId: string): Promise<CategoryRecord[]> {
  const db = await getDB();
  const rows = await db.getAllFromIndex(STORE_CATEGORIES, 'by-workspace', workspaceId);
  return rows.sort(byCreatedDesc);
}

export async function putCategoryRecord(entry: CategoryRecord): Promise<void> {
  const db = await getDB();
  await db.put(STORE_CATEGORIES, entry);
}

export async function deleteCategoryRecord(workspaceId: string, name: string): Promise<void> {
  const db = await getDB();
  await db.delete(STORE_CATEGORIES, [workspaceId, name]);
}
