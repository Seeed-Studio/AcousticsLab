// formatRelative/formatAbsolute read `locale.resolved` per call so a template expression
// tracks both $state reads (locale + timestamp) and re-renders on either change;
// formatRelativeShort is English-only and locale-independent. `localeOverride` forces a
// non-operator locale (rare, debug only).

import { locale as localeStore } from '$lib/stores/locale.svelte';

const RTF_CACHE = new Map<string, Intl.RelativeTimeFormat>();
const ABS_CACHE = new Map<string, Intl.DateTimeFormat>();

function rtf(locale: string): Intl.RelativeTimeFormat {
  let f = RTF_CACHE.get(locale);
  if (!f) {
    f = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
    RTF_CACHE.set(locale, f);
  }
  return f;
}

function abs(locale: string): Intl.DateTimeFormat {
  let f = ABS_CACHE.get(locale);
  if (!f) {
    f = new Intl.DateTimeFormat(locale, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
    ABS_CACHE.set(locale, f);
  }
  return f;
}

// Space a digit from a following Han glyph so Intl's glued CJK count ("2小时前") matches the catalog's
// "${n} 秒" spacing, reading "开始于 2 小时前"; a no-op for Latin ("2 hours ago"). Relative branch only:
// the absolute format binds digits to date units ("2026年6月23日"), where a space would be wrong.
const CJK_COUNT_RE = /(\d)(\p{Script=Han})/gu;
function spaceCjkCount(formatted: string): string {
  return formatted.replace(CJK_COUNT_RE, '$1 $2');
}

// Relative within 24 h, absolute beyond; returns the raw input on parse failure to avoid `Invalid Date`.
export function formatRelative(
  rfc3339: string,
  now: Date = new Date(),
  localeOverride?: string
): string {
  const t = Date.parse(rfc3339);
  if (Number.isNaN(t)) return rfc3339;
  const loc = localeOverride ?? localeStore.resolved;
  const deltaMs = t - now.getTime();
  const absMs = Math.abs(deltaMs);
  const ONE_DAY = 24 * 60 * 60 * 1000;
  if (absMs > ONE_DAY) return abs(loc).format(new Date(t));
  // Largest unit with |n| >= 1, so "just now" doesn't lose to "0 hours ago".
  const sec = deltaMs / 1000;
  const min = sec / 60;
  const hr = min / 60;
  if (Math.abs(hr) >= 1) return spaceCjkCount(rtf(loc).format(Math.round(hr), 'hour'));
  if (Math.abs(min) >= 1) return spaceCjkCount(rtf(loc).format(Math.round(min), 'minute'));
  return spaceCjkCount(rtf(loc).format(Math.round(sec), 'second'));
}

export function formatAbsolute(rfc3339: string, localeOverride?: string): string {
  const t = Date.parse(rfc3339);
  if (Number.isNaN(t)) return rfc3339;
  const loc = localeOverride ?? localeStore.resolved;
  return abs(loc).format(new Date(t));
}

// Tight English-only past-only relative time for stat tiles: hand-rolled abbreviations
// avoid `Intl` 'short' style's en-US period ("5 min."); future clamps to "just now"
// (callers feed past timestamps barring clock skew); no 24 h absolute cliff (the tile
// wants "how long ago", not a wall-clock moment).
export function formatRelativeShort(rfc3339: string, now: Date = new Date()): string {
  const t = Date.parse(rfc3339);
  if (Number.isNaN(t)) return rfc3339;
  const elapsedSec = Math.max(0, (now.getTime() - t) / 1000);
  if (elapsedSec < 60) return 'just now';
  const min = Math.floor(elapsedSec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.floor(elapsedSec / 3600);
  if (hr < 24) return `${hr} h ago`;
  const day = Math.floor(elapsedSec / 86400);
  return `${day} d ago`;
}
