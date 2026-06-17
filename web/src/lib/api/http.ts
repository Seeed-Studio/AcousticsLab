import { apiUrl } from './base';
import type { ApiErrorBody } from './types';

const DEFAULT_TIMEOUT_MS = 5_000;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly body: ApiErrorBody;

  constructor(status: number, body: ApiErrorBody) {
    super(body.error || `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.code = body.code || 'unknown';
    this.body = body;
  }
}

export interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

async function parseError(resp: Response): Promise<ApiError> {
  let body: ApiErrorBody;
  try {
    body = (await resp.json()) as ApiErrorBody;
  } catch {
    body = { error: resp.statusText || `HTTP ${resp.status}`, code: 'unknown' };
  }
  return new ApiError(resp.status, body);
}

function withTimeout(opts: RequestOptions): { signal: AbortSignal; cancel: () => void } {
  const ctrl = new AbortController();
  const timeout = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer =
    timeout > 0
      ? setTimeout(() => {
          ctrl.abort(new Error('request timeout'));
        }, timeout)
      : null;
  const userSignal = opts.signal;
  // Track the bridge listener so cancel() can detach it; a long-lived caller signal would
  // otherwise leak one listener + one ctrl per request.
  let userAbortListener: (() => void) | null = null;
  if (userSignal) {
    if (userSignal.aborted) {
      ctrl.abort(userSignal.reason);
    } else {
      userAbortListener = (): void => {
        ctrl.abort(userSignal.reason);
      };
      userSignal.addEventListener('abort', userAbortListener, { once: true });
    }
  }
  return {
    signal: ctrl.signal,
    cancel: () => {
      if (timer) clearTimeout(timer);
      if (userAbortListener && userSignal) {
        userSignal.removeEventListener('abort', userAbortListener);
      }
    }
  };
}

async function request<T>(
  method: string,
  path: string,
  body: unknown,
  opts: RequestOptions
): Promise<T> {
  const { signal, cancel } = withTimeout(opts);
  const init: RequestInit = { method, signal, headers: { ...(opts.headers ?? {}) } };
  if (body !== undefined) {
    (init.headers as Record<string, string>)['content-type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  try {
    const resp = await fetch(apiUrl(path), init);
    if (!resp.ok) throw await parseError(resp);
    if (resp.status === 204 || resp.headers.get('content-length') === '0') return undefined as T;
    const ct = resp.headers.get('content-type') ?? '';
    if (ct.includes('application/json')) return (await resp.json()) as T;
    // A non-JSON 2xx body (proxy/CDN interstitial or login HTML returned with a success status) cannot satisfy the typed <T>;
    // fail fast naming the content-type instead of `text() as T`, which would throw far from
    // the cause at the first field access.
    throw new Error(
      `unexpected non-JSON success response (content-type: ${ct || 'none'}, status: ${resp.status})`
    );
  } finally {
    cancel();
  }
}

export const api = {
  get: <T>(path: string, opts: RequestOptions = {}) => request<T>('GET', path, undefined, opts),
  post: <T>(path: string, body?: unknown, opts: RequestOptions = {}) =>
    request<T>('POST', path, body, opts),
  put: <T>(path: string, body?: unknown, opts: RequestOptions = {}) =>
    request<T>('PUT', path, body, opts),
  patch: <T>(path: string, body?: unknown, opts: RequestOptions = {}) =>
    request<T>('PATCH', path, body, opts),
  delete: <T>(path: string, opts: RequestOptions = {}) =>
    request<T>('DELETE', path, undefined, opts)
};

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}
