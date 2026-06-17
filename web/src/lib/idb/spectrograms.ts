import { getDB, spectrogramStoreFor, type SpectrogramRecord, type SpectrogramTheme } from './db';

// `theme` selects one of two stores both keyed by `keyPath: 'sha256'`. The PNG is a
// deterministic function of (WAV bytes, palette mode), so each (hash, theme) row is valid
// forever and byte-identical slices across categories share one row; persisting it (vs a
// `blob:` URL) avoids re-running WAV decode + FFT + colour-map per visible card on refresh,
// route swap, or device sleep. Content-addressing makes a delete unable to prove a hash
// unreferenced, so there is no per-row eviction - the cache only grows; `resetDB` is the
// single reset point.

export async function getSpectrogramRecord(
  sha256: string,
  theme: SpectrogramTheme
): Promise<SpectrogramRecord | undefined> {
  const db = await getDB();
  return db.get(spectrogramStoreFor(theme), sha256);
}

export async function putSpectrogramRecord(
  record: SpectrogramRecord,
  theme: SpectrogramTheme
): Promise<void> {
  const db = await getDB();
  await db.put(spectrogramStoreFor(theme), record);
}
