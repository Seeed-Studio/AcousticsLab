// `detected` is snapshotted once at construction (no `languagechange` listener by design, since a
// mid-session locale flip would be jarring), so a new system locale lands only on reload. An app.html FOUC IIFE duplicates these same
// resolution rules to stamp `<html lang>` pre-paint (it can't import this module and must beat
// hydration); this store's first effect re-applies it (no-op cold) and owns every change after.

import {
  DEFAULT_LOCALE,
  SUPPORTED_LOCALES,
  isLocaleCode,
  type LocaleCode
} from '$lib/i18n/locales';

export type LocaleMode = 'auto' | LocaleCode;

const STORAGE_KEY = 'acousticslab-locale';

// 'auto' for anything but a strictly-supported stored tag (incl. a localStorage throw in Safari
// private browsing), so the detected value takes over.
function readInitialMode(): LocaleMode {
  if (typeof localStorage === 'undefined') return 'auto';
  try {
    const m = localStorage.getItem(STORAGE_KEY);
    if (isLocaleCode(m)) return m;
    return 'auto';
  } catch {
    return 'auto';
  }
}

// The one supported locale whose primary subtag is `base`, else undefined when 0 or 2+ share it
// (an ambiguous subtag, e.g. a future zh-CN + zh-TW, needs an exact tag).
function localeForBase(base: string): LocaleCode | undefined {
  const matches = SUPPORTED_LOCALES.filter((c) => c.split('-')[0].toLowerCase() === base);
  return matches.length === 1 ? matches[0] : undefined;
}

// Preference order: exact tag, else primary subtag -> its sole locale ('de-DE'->'de', 'zh-TW'->'zh-CN').
// Else DEFAULT_LOCALE.
function readInitialDetected(): LocaleCode {
  if (typeof navigator === 'undefined') return DEFAULT_LOCALE;
  // Array.isArray (guarding a missing navigator.languages) widens to any[]; the cast restores string.
  const list: readonly string[] =
    Array.isArray(navigator.languages) && navigator.languages.length > 0
      ? (navigator.languages as readonly string[])
      : navigator.language
        ? [navigator.language]
        : [];
  for (const tag of list) {
    if (isLocaleCode(tag)) return tag;
    const mapped = localeForBase(tag.split('-')[0].toLowerCase());
    if (mapped) return mapped;
  }
  return DEFAULT_LOCALE;
}

class LocaleStore {
  mode = $state<LocaleMode>(readInitialMode());
  detected = $state<LocaleCode>(readInitialDetected());

  // Plain getter, not $derived: Svelte 5 tracks the reads, so callers re-run on either input change.
  get resolved(): LocaleCode {
    return this.mode === 'auto' ? this.detected : this.mode;
  }

  private disposeEffects: (() => void) | null = null;

  // Cross-tab sync; `e.key === null` (a `localStorage.clear()`) and a removed key both reset to 'auto'.
  private onStorage = (e: StorageEvent): void => {
    if (e.key !== null && e.key !== STORAGE_KEY) return;
    if (e.key === null) {
      this.mode = 'auto';
      return;
    }
    const v = e.newValue;
    if (isLocaleCode(v)) {
      this.mode = v;
    } else if (v === null) {
      this.mode = 'auto';
    }
  };

  start(): void {
    if (this.disposeEffects !== null) return;
    if (typeof document === 'undefined') return;

    if (typeof window !== 'undefined') {
      window.addEventListener('storage', this.onStorage);
    }

    this.disposeEffects = $effect.root(() => {
      // `lang` drives screen-reader pronunciation, the translate-prompt, and CSS `:lang()`.
      $effect(() => {
        document.documentElement.lang = this.resolved;
      });

      // 'auto' persists as key removal so it round-trips through "no stored value".
      $effect(() => {
        const m = this.mode;
        try {
          if (m === 'auto') localStorage.removeItem(STORAGE_KEY);
          else localStorage.setItem(STORAGE_KEY, m);
        } catch {
          /* private-browsing / quota */
        }
      });
    });
  }

  stop(): void {
    if (typeof window !== 'undefined') {
      window.removeEventListener('storage', this.onStorage);
    }
    if (this.disposeEffects !== null) {
      this.disposeEffects();
      this.disposeEffects = null;
    }
  }

  setMode(m: LocaleMode): void {
    this.mode = m;
  }
}

export const locale = new LocaleStore();
export { SUPPORTED_LOCALES, DEFAULT_LOCALE, type LocaleCode };
