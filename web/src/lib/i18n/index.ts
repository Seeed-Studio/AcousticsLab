// Read full key paths inline (`{m.workspace.delete_title}`): the Proxy intercepts only top-level
// namespace gets, so aliasing (`const { workspace } = m`) snapshots the subtree at script-init and
// breaks reactivity (sub-key access then bypasses the reactive `locale.resolved` read).

import { locale } from '$lib/stores/locale.svelte';
import { type LocaleCode } from './locales';
import type { Messages } from './types';
import { en } from './messages/en';

const CATALOGS: Readonly<Record<LocaleCode, Messages>> = {
  en
};

// Reads `locale.resolved` so callers re-evaluate on locale switch. No fallback by design: the
// store's `isLocaleCode` guard keeps the key in-set; an out-of-set code fails fast as `undefined`.
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
