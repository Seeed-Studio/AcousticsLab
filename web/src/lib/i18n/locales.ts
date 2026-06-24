// Dependency-free (the store and i18n proxy both import it without an init cycle). A new locale's
// code must also be added to app.html's FOUC IIFE, which beats hydration and can't import this.

export const SUPPORTED_LOCALES = ['en', 'zh-CN', 'ja', 'de'] as const;
export type LocaleCode = (typeof SUPPORTED_LOCALES)[number];

export const DEFAULT_LOCALE: LocaleCode = 'en';

// Endonyms ('English' / '中文') so operators scan by their own term.
export const LOCALE_LABELS: Readonly<Record<LocaleCode, string>> = {
  en: 'English',
  'zh-CN': '简体中文',
  ja: '日本語',
  de: 'Deutsch'
};

// Switcher chip: short, script-distinguishing form (zh-CN vs zh-TW) that fits the pill.
export const LOCALE_CHIPS: Readonly<Record<LocaleCode, string>> = {
  en: 'EN',
  'zh-CN': '简',
  ja: '日',
  de: 'DE'
};

export function isLocaleCode(s: unknown): s is LocaleCode {
  return typeof s === 'string' && (SUPPORTED_LOCALES as readonly string[]).includes(s);
}
