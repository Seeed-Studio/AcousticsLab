// Dependency-free so the store and i18n proxy can both import it without an init-order cycle. A new
// locale's code must also be duplicated in app.html's FOUC IIFE, which beats hydration so can't import this.

export const SUPPORTED_LOCALES = ['en'] as const;
export type LocaleCode = (typeof SUPPORTED_LOCALES)[number];

export const DEFAULT_LOCALE: LocaleCode = 'en';

// Endonyms ('English' / '中文'), not the current-locale name, so operators scan by their own term.
export const LOCALE_LABELS: Readonly<Record<LocaleCode, string>> = {
  en: 'English'
};

// Header switcher chip: ISO 639-1 uppercase, but region-tagged locales (zh-CN/zh-TW) need a script-distinguishing short form to fit the 30px pill.
export const LOCALE_CHIPS: Readonly<Record<LocaleCode, string>> = {
  en: 'EN'
};

export function isLocaleCode(s: unknown): s is LocaleCode {
  return typeof s === 'string' && (SUPPORTED_LOCALES as readonly string[]).includes(s);
}
