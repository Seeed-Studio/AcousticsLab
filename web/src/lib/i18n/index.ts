// Read full key paths inline (`{m.x.y}`): the Proxy only intercepts top-level gets, so aliasing a
// subtree (`const { x } = m`) snapshots it at init and breaks reactivity.

import { locale } from '$lib/stores/locale.svelte';
import type { Messages } from './types';
import { catalogFor } from './messages/catalog.svelte';

// Reads `locale.resolved` so readers re-render on switch; `catalogFor` handles lazy-load + en fallback.
function current(): Messages {
  return catalogFor(locale.resolved);
}

/** Reactive message accessor; read full key paths inline. */
export const m: Messages = new Proxy({} as Messages, {
  get(_target, prop: string): Messages[keyof Messages] {
    return current()[prop as keyof Messages];
  }
});

export {
  SUPPORTED_LOCALES,
  DEFAULT_LOCALE,
  LOCALE_LABELS,
  LOCALE_CHIPS,
  isLocaleCode,
  type LocaleCode
} from './locales';
export { ensureCatalog } from './messages/catalog.svelte';
export { pluralCategory, type PluralCategory } from './plural';
export type { Messages } from './types';
