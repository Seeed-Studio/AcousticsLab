// Category set is locale-dependent (en: one|other; ar: all six; zh: other only). Cached per
// locale because `Intl.PluralRules` construction is costly and the same locale repeats per session.
const CACHE = new Map<string, Intl.PluralRules>();

function rules(locale: string): Intl.PluralRules {
  let r = CACHE.get(locale);
  if (!r) {
    r = new Intl.PluralRules(locale);
    CACHE.set(locale, r);
  }
  return r;
}

export type PluralCategory = Intl.LDMLPluralRule;

export function pluralCategory(n: number, locale: string): PluralCategory {
  return rules(locale).select(n);
}
