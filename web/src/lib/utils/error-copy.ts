import { ApiError, isApiError } from '$lib/api/http';
import { m } from '$lib/i18n';

// Generic codes omitted so they fall through to the daemon's `error` text; `m.error` read per-call to follow live locale.
function fixedCopy(code: string): string | undefined {
  const t = m.error;
  switch (code) {
    case 'another_train_running':
      return t.another_train_running;
    // `another_convert_running`/`job_conflict` omitted: daemon maps both to Conflict -> `conflict` passthrough.
    case 'event_gap':
      return t.event_gap;
    case 'too_early':
      return t.too_early;
    case 'unavailable':
      return t.unavailable;
    case 'internal':
      return t.internal;
    case 'unknown':
      return t.unknown;
    default:
      return undefined;
  }
}

// Codes whose backend `error` text is operator-friendly enough to pass straight through `finish`.
const PASSTHROUGH_CODES: ReadonlySet<string> = new Set([
  'bad_request',
  'not_found',
  'conflict',
  'method_not_allowed',
  // `bad_dataset` omitted (maps to `bad_request`); `head_id_collision` passthrough keeps the daemon diagnostic (both sha256 hashes + recovery).
  'head_id_collision'
]);

export function errorCopy(err: unknown): string {
  if (!isApiError(err)) {
    if (err instanceof Error) return finish(err.message);
    return finish(String(err));
  }
  const fixed = fixedCopy(err.code);
  if (fixed) return fixed;
  if (PASSTHROUGH_CODES.has(err.code)) return finish(err.body.error || err.message);
  return finish(err.body.error || err.message || m.error.request_failed(err.code));
}

// Daemon errors are thiserror `"<layer>: <msg>"`; PREFIX_RE strips exactly one leading known prefix (nested kept).
const DAEMON_LAYER_PREFIXES = [
  'fs',
  'file',
  'config',
  'mic',
  'head load',
  'head swap',
  'convert',
  'training',
  'activation',
  'invalid identifier',
  'invalid request',
  'internal'
];
const PREFIX_RE = new RegExp(`^(?:${DAEMON_LAYER_PREFIXES.join('|')}):\\s*`, 'i');

function stripLayerPrefix(s: string): string {
  return s.replace(PREFIX_RE, '');
}

function finish(s: string): string {
  return capFirst(stripLayerPrefix(s));
}

// Sentence-case + trailing period, NO prefix stripping (could lop a legit leading word) for already-untagged messages like SSE `message`; blank -> `fallback` or generic.
export function capFirst(s: string, fallback?: string): string {
  const t = s.trim();
  if (!t) return fallback ?? m.error.something_went_wrong;
  const head = t[0].toUpperCase() + t.slice(1);
  return /[.!?…]$/.test(head) ? head : `${head}.`;
}

// "Resource gone" via EITHER 404 status OR `not_found` code; both are checked because they can diverge, e.g. a proxy/gateway 404 with a non-JSON body parses to code `unknown`, so status alone signals it.
export function isNotFound(err: unknown): boolean {
  return isApiError(err) && (err.status === 404 || err.code === 'not_found');
}

export function isConflict(err: unknown): boolean {
  return isApiError(err) && err.status === 409;
}

export { ApiError };
