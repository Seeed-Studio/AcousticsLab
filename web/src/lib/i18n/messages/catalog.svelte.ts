// Lazy catalog registry: en is bundled (default + fallback); zh/ja/de are dynamic-import chunks
// fetched on demand. catalogFor returns en until the resolved locale's chunk lands, then the registry
// swap re-renders. ensureCatalog is idempotent and awaitable — the layout and switcher await it to
// preload before flipping, avoiding an en flash.

import type { Messages } from '../types';
import { type LocaleCode } from '../locales';
import { en } from './en';

// Literal import() so Vite splits each catalog into a chunk; en is absent (static). The satisfies
// shape makes a missing non-default loader a compile error.
const LOADERS = {
  'zh-CN': () => import('./zh').then((m) => m.zh),
  ja: () => import('./ja').then((m) => m.ja),
  de: () => import('./de').then((m) => m.de)
} satisfies Record<Exclude<LocaleCode, 'en'>, () => Promise<Messages>>;

// $state.raw, not $state: the registry swaps wholesale on load, so map-level reactivity suffices —
// deep-proxying read-only catalogs would tax every m.* read with pointless per-leaf tracking.
let loaded: Partial<Record<LocaleCode, Messages>> = $state.raw({ en });

// Internal load bookkeeping — plain objects (never read in a tracked/render context, so not
// SvelteMaps): `inflight` dedupes concurrent loads and clears on settle; `failedAt` records the last
// failure so catalogFor backs its auto-retry off to once per cooldown instead of once per render.
const inflight: Partial<Record<LocaleCode, Promise<void>>> = {};
const failedAt: Partial<Record<LocaleCode, number>> = {};
const RETRY_COOLDOWN_MS = 30_000;

/** Idempotent. Resolves once the locale's catalog is in `loaded` (immediately for en / already-loaded). */
export function ensureCatalog(code: LocaleCode): Promise<void> {
  if (loaded[code]) return Promise.resolve();
  const existing = inflight[code];
  if (existing) return existing;
  // en is in the initial map (never reaches here); the null branch only satisfies Exclude<…, 'en'>.
  const load = code === 'en' ? null : LOADERS[code];
  if (!load) return Promise.resolve();
  const p = load()
    .then((cat) => {
      loaded = { ...loaded, [code]: cat };
    })
    .catch(() => {
      // Keep the en fallback; record the failure so catalogFor backs off its auto-retry.
      failedAt[code] = Date.now();
    })
    .finally(() => {
      inflight[code] = undefined;
    });
  inflight[code] = p;
  return p;
}

/**
 * Reactive accessor: returns the resolved locale's catalog, or `en` while its chunk loads (kicking
 * off that load). Reading `loaded[code]` registers the dependency, so the import's write re-renders.
 */
export function catalogFor(code: LocaleCode): Messages {
  const cat = loaded[code];
  if (cat) return cat;
  // Auto-load on demand, but skip during a failure cooldown so a broken chunk can't refetch every
  // render; an explicit ensureCatalog (switcher / layout load) always retries regardless.
  const failed = failedAt[code];
  if (failed === undefined || Date.now() - failed >= RETRY_COOLDOWN_MS) void ensureCatalog(code);
  return en;
}
