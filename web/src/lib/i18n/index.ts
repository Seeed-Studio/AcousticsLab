// Read full key paths inline (`{m.x.y}`): the Proxy only intercepts top-level gets, so aliasing a
// subtree (`const { x } = m`) snapshots it at init and breaks reactivity.

import { locale } from '$lib/stores/locale.svelte';
import { type LocaleCode } from './locales';
import type { Messages } from './types';
import { en } from './messages/en';
import { zh } from './messages/zh';

const CATALOGS: Readonly<Record<LocaleCode, Messages>> = {
  en,
  'zh-CN': zh
};

// Reads `locale.resolved` so callers re-render on switch; no fallback (the store's guard keeps the key in-set).
function current(): Messages {
  return CATALOGS[locale.resolved];
}

/** Reactive message accessor; read inline only (see the alias trap above). */
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
export { pluralCategory, type PluralCategory } from './plural';
export type { Messages } from './types';
