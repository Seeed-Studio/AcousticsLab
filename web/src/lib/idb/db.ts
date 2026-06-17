import { deleteDB, openDB, type DBSchema, type IDBPDatabase } from 'idb';

// Single per-origin DB (reset = one atomic deleteDatabase). Slices are content-addressed: WAV-bytes
// sha256 is the id, daemon filename `<sha>.wav`, and cache key at once, so re-upload is idempotent.
// Spectrogram PNGs bake palette per theme, so each hash is cached per-mode (light/dark separate stores).

export const DB_NAME = 'acoustics-lab';
export const DB_VERSION = 8;

export const STORE_CATEGORIES = 'categories' as const;
export const STORE_DRAFTS = 'drafts' as const;
export const STORE_SLICES = 'slices' as const;
export const STORE_WORKSPACE_SYNC = 'workspace_sync' as const;
export const STORE_SPECTROGRAMS = 'spectrograms' as const;
export const STORE_SPECTROGRAMS_DARK = 'spectrograms_dark' as const;

export interface CategoryRecord {
  workspace_id: string;
  name: string;
  created_at: string;
}

export type DraftSource = 'recorded' | 'imported';

export interface DraftRecord {
  workspace_id: string;
  category_name: string;
  blob: Blob;
  duration_ms: number;
  sample_rate: number;
  size_bytes: number;
  source: DraftSource;
  created_at: string;
  original_name?: string;
  trim_start_samples?: number;
  trim_end_samples?: number;
}

// `committed` means the daemon acked and the IDB blob may be dropped (canonical copy now on disk).
export type SliceState = 'local' | 'uploading' | 'committed' | 'failed';

export function sliceFilename(id: string): string {
  return `${id}.wav`;
}

// Reconcile skips foreign-named files (not lowercase-hex sha256): they'd fail the content-addressed
// integrity check anyway, rendering a permanently-broken card. A violation is a setup bug, not input.
const SHA256_HEX = /^[0-9a-f]{64}$/;

export function sliceIdFromFilename(filename: string): string | null {
  if (!filename.endsWith('.wav')) return null;
  const id = filename.slice(0, -'.wav'.length);
  return SHA256_HEX.test(id) ? id : null;
}

export interface SliceRecord {
  id: string;
  workspace_id: string;
  category_name: string;
  blob: Blob | null; // bytes for in-flight rows; null after commit (canonical copy on daemon)
  state: SliceState;
  upload_progress?: number;
  workspace_revision_id?: number;
  last_error?: string;
  created_at: string;
}

// On mount, last_synced_revision_id >= freshly-fetched workspace_revision.id skips every per-category
// dataset GET. last_synced_at is debug-only.
export interface WorkspaceSyncRecord {
  workspace_id: string;
  last_synced_revision_id: number;
  last_synced_at: string;
}

// A PNG is valid only for its render mode (light PNG on a dark page paints a bright rectangle).
export type SpectrogramTheme = 'light' | 'dark';

export function spectrogramStoreFor(
  theme: SpectrogramTheme
): typeof STORE_SPECTROGRAMS | typeof STORE_SPECTROGRAMS_DARK {
  return theme === 'dark' ? STORE_SPECTROGRAMS_DARK : STORE_SPECTROGRAMS;
}

// No per-row eviction (a scoped delete can't prove the hash unreferenced -- another slice may share
// it), so resetDB is the only reset; PNGs are shared across categories/workspaces.
export interface SpectrogramRecord {
  sha256: string;
  png: Blob;
  created_at: string;
}

// Category is in the key because identical content can live in two categories at once (trainer labels
// by directory name); a content-only key would flip category_name on the second put.
export type SliceKey = [workspace_id: string, category_name: string, id: string];

interface AcousticsLabDB extends DBSchema {
  [STORE_CATEGORIES]: {
    key: [string, string];
    value: CategoryRecord;
    indexes: {
      'by-workspace': string;
    };
  };
  [STORE_DRAFTS]: {
    key: [string, string];
    value: DraftRecord;
    indexes: {
      'by-workspace': string;
    };
  };
  [STORE_SLICES]: {
    key: SliceKey;
    value: SliceRecord;
    indexes: {
      'by-workspace': string;
      'by-workspace-category': [string, string];
    };
  };
  [STORE_WORKSPACE_SYNC]: {
    key: string;
    value: WorkspaceSyncRecord;
  };
  [STORE_SPECTROGRAMS]: {
    key: string;
    value: SpectrogramRecord;
  };
  [STORE_SPECTROGRAMS_DARK]: {
    key: string;
    value: SpectrogramRecord;
  };
}

export type AppDB = IDBPDatabase<AcousticsLabDB>;

let dbPromise: Promise<AppDB> | null = null;

export function getDB(): Promise<AppDB> {
  // Cache the in-flight/resolved promise; the .catch clears it on rejection so the next call retries
  // instead of pinning a stale rejection for the tab's lifetime.
  dbPromise ??= openDB<AcousticsLabDB>(DB_NAME, DB_VERSION, {
    // oldVersion is 0 on first install; gating each step on oldVersion < N runs each branch at most
    // once even across a multi-step bump (pre-v6 -> v8 runs both in one transaction).
    upgrade(db, oldVersion) {
      if (oldVersion < 6) {
        // Drop+re-cut all stores: v6 was lock-stepped with an operator-side workspace wipe (daemon
        // slice filenames changed too), so carried-over IDB content would mis-key.
        for (const name of Array.from(db.objectStoreNames)) {
          db.deleteObjectStore(name);
        }
        const categories = db.createObjectStore(STORE_CATEGORIES, {
          keyPath: ['workspace_id', 'name']
        });
        categories.createIndex('by-workspace', 'workspace_id', { unique: false });

        const drafts = db.createObjectStore(STORE_DRAFTS, {
          keyPath: ['workspace_id', 'category_name']
        });
        drafts.createIndex('by-workspace', 'workspace_id', { unique: false });

        const slices = db.createObjectStore(STORE_SLICES, {
          keyPath: ['workspace_id', 'category_name', 'id']
        });
        slices.createIndex('by-workspace', 'workspace_id', { unique: false });
        slices.createIndex('by-workspace-category', ['workspace_id', 'category_name'], {
          unique: false
        });

        db.createObjectStore(STORE_WORKSPACE_SYNC, { keyPath: 'workspace_id' });
        db.createObjectStore(STORE_SPECTROGRAMS, { keyPath: 'sha256' });
      }
      if (oldVersion < 8) {
        // Add the dark-mode sibling store; v6 stores survive untouched. contains guards idempotency --
        // a leftover store from an unreleased migration would else throw ConstraintError.
        if (!db.objectStoreNames.contains(STORE_SPECTROGRAMS_DARK)) {
          db.createObjectStore(STORE_SPECTROGRAMS_DARK, { keyPath: 'sha256' });
        }
      }
    },
    blocked() {
      console.warn('[idb] upgrade blocked by another tab');
    },
    blocking() {
      console.warn(
        '[idb] another tab requested upgrade -- closing this connection so it can proceed'
      );
      // Close the held connection or the upgrading tab's upgradeneeded hangs forever (its open promise
      // never resolves, wedging every IDB feature there); drop the cache so this tab re-opens lazily.
      void dbPromise?.then((db) => db.close()).catch(() => undefined);
      dbPromise = null;
    },
    terminated() {
      dbPromise = null;
    }
  }).catch((err: unknown) => {
    dbPromise = null;
    throw err;
  });
  return dbPromise;
}

export async function resetDB(): Promise<void> {
  if (dbPromise) {
    // Swallow a rejecting in-flight open and still delete on disk: a reset must not abort just because
    // the prior open failed.
    const db = await dbPromise.catch(() => null);
    db?.close();
    dbPromise = null;
  }
  await deleteDB(DB_NAME);
}

// Cascade-delete a workspace across categories/drafts/slices/workspace_sync in ONE readwrite tx so a
// mid-cascade page close can't leave orphan rows a same-id recreation would inherit. Spectrogram
// caches are untouched -- their content-addressed PNGs are shared across workspaces.
export async function deleteAllForWorkspace(workspaceId: string): Promise<void> {
  const db = await getDB();
  const tx = db.transaction(
    [STORE_CATEGORIES, STORE_DRAFTS, STORE_SLICES, STORE_WORKSPACE_SYNC],
    'readwrite'
  );

  const cascadeStore = async (
    storeName: typeof STORE_CATEGORIES | typeof STORE_DRAFTS | typeof STORE_SLICES,
    indexName: 'by-workspace'
  ): Promise<void> => {
    const index = tx.objectStore(storeName).index(indexName);
    let cursor = await index.openCursor(IDBKeyRange.only(workspaceId));
    while (cursor) {
      await cursor.delete();
      cursor = await cursor.continue();
    }
  };

  await cascadeStore(STORE_CATEGORIES, 'by-workspace');
  await cascadeStore(STORE_DRAFTS, 'by-workspace');
  await cascadeStore(STORE_SLICES, 'by-workspace');
  await tx.objectStore(STORE_WORKSPACE_SYNC).delete(workspaceId);

  await tx.done;
}

// Re-key a category across the three name-bearing stores in ONE readwrite tx. IDB keys are immutable,
// so rename is delete-at-old + put-at-new; one tx so a mid-rename page close aborts atomically instead
// of orphaning drafts/slices under the old name.
export async function renameCategoryForWorkspace(
  workspaceId: string,
  oldName: string,
  newName: string
): Promise<void> {
  if (oldName === newName) return;
  const db = await getDB();
  const tx = db.transaction([STORE_CATEGORIES, STORE_DRAFTS, STORE_SLICES], 'readwrite');

  // Category row may be absent (server-only, no IDB shadow) -- skip it but still re-key any drafts/
  // slices below, which may exist under the old name from an upload.
  const catStore = tx.objectStore(STORE_CATEGORIES);
  const cat = await catStore.get([workspaceId, oldName]);
  if (cat) {
    await catStore.delete([workspaceId, oldName]);
    await catStore.put({ ...cat, name: newName });
  }

  const draftStore = tx.objectStore(STORE_DRAFTS);
  const draft = await draftStore.get([workspaceId, oldName]);
  if (draft) {
    await draftStore.delete([workspaceId, oldName]);
    await draftStore.put({ ...draft, category_name: newName });
  }

  // Collect-then-put: the cursor walk only deletes (puts deferred until exhausted) so it never meets a
  // row it just rewrote; Blob handles move by reference, not copied.
  const sliceStore = tx.objectStore(STORE_SLICES);
  const sliceIndex = sliceStore.index('by-workspace-category');
  const moved: SliceRecord[] = [];
  let cursor = await sliceIndex.openCursor(IDBKeyRange.only([workspaceId, oldName]));
  while (cursor) {
    moved.push({ ...cursor.value, category_name: newName });
    await cursor.delete();
    cursor = await cursor.continue();
  }
  for (const row of moved) {
    await sliceStore.put(row);
  }

  await tx.done;
}
