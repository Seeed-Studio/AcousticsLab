import { apiUrl } from './base';
import { ApiError } from './http';
import type { ApiErrorBody } from './types';

// Retry-worthy: network blips (plain `Error`) and daemon 5xx/429; permanent: other 4xx. Aborts are out of scope - caller must check `signal.aborted` first.
export function isTransientUploadError(e: unknown): boolean {
  if (e instanceof ApiError) {
    return e.status >= 500 || e.status === 429;
  }
  return e instanceof Error && !(e instanceof DOMException);
}

// Rejects `DOMException('AbortError')` on abort - same shape as xhrPut so call-sites can check `signal.aborted` regardless of which phase rejected.
export function sleepAbortable(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('aborted', 'AbortError'));
      return;
    }
    const onAbort = (): void => {
      clearTimeout(t);
      reject(new DOMException('aborted', 'AbortError'));
    };
    const t = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

// `fetch` lacks `upload.onprogress`, so byte-level progress requires XHR.
export interface XhrUploadOptions {
  url: string;
  body: Blob;
  contentType?: string;
  // Fires only when `lengthComputable` (always true for Blob bodies; guard defends a future chunked-stream caller).
  onProgress?: (loaded: number, total: number) => void;
  signal?: AbortSignal;
}

export function xhrPut<T>(opts: XhrUploadOptions): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', apiUrl(opts.url), true);
    // Daemon writes JSON on both success and error paths, so let XHR parse.
    xhr.responseType = 'json';
    xhr.setRequestHeader('content-type', opts.contentType ?? 'application/octet-stream');

    if (opts.onProgress) {
      xhr.upload.onprogress = (e: ProgressEvent): void => {
        if (e.lengthComputable) opts.onProgress?.(e.loaded, e.total);
      };
    }

    // Detach the abort listener on terminal completion, else a long-lived caller signal leaks one listener + xhr per request (abort branch self-cleans: sync returns pre-add, async is `{ once: true }`).
    const userSignal = opts.signal;
    let userAbortListener: (() => void) | null = null;
    const cleanupSignal = (): void => {
      if (userAbortListener && userSignal) {
        userSignal.removeEventListener('abort', userAbortListener);
        userAbortListener = null;
      }
    };

    xhr.onload = (): void => {
      cleanupSignal();
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as T);
        return;
      }
      // Fall back to status text when the `{error,code}` envelope didn't parse (e.g. malformed JSON).
      const body: ApiErrorBody = (xhr.response as ApiErrorBody | null) ?? {
        error: xhr.statusText || `HTTP ${xhr.status}`,
        code: 'unknown'
      };
      reject(new ApiError(xhr.status, body));
    };
    xhr.onerror = (): void => {
      cleanupSignal();
      // Plain Error (not ApiError) so callers distinguish transport failure from server rejection.
      reject(new Error('Network error during upload.'));
    };
    // No `xhr.timeout`: it would kill slow-but-progressing multi-MB uploads; cancellation relies solely on the AbortSignal, so a black-holed connection hangs until the platform TCP timeout absent an abort.

    if (userSignal) {
      // `signal.reason` is `any`; coerce to Error (non-Error reasons -> native-abort shape) so call sites narrow under `catch (e: unknown)`.
      const abortError = (reason: unknown): Error =>
        reason instanceof Error ? reason : new DOMException('aborted', 'AbortError');
      if (userSignal.aborted) {
        xhr.abort();
        reject(abortError(userSignal.reason));
        return;
      }
      userAbortListener = (): void => {
        xhr.abort();
        reject(abortError(userSignal.reason));
      };
      userSignal.addEventListener('abort', userAbortListener, { once: true });
    }

    xhr.send(opts.body);
  });
}

// Bounded-concurrency pool: FIFO drain so progress follows submit order.
export class UploadPool {
  private active = 0;
  private readonly waiting: (() => void)[] = [];

  constructor(private readonly max: number) {
    // A non-positive/fractional cap deadlocks every submit forever, so fail loudly rather than hang.
    if (!Number.isInteger(max) || max < 1) {
      throw new RangeError('UploadPool max must be a positive integer');
    }
  }

  // Returns `task`'s own promise so the pool never swallows failures.
  async submit<T>(task: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await task();
    } finally {
      this.release();
    }
  }

  get pending(): number {
    return this.waiting.length;
  }
  get inflight(): number {
    return this.active;
  }

  private acquire(): Promise<void> {
    if (this.active < this.max) {
      this.active++;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.waiting.push(() => {
        this.active++;
        resolve();
      });
    });
  }

  private release(): void {
    this.active--;
    const next = this.waiting.shift();
    if (next) next();
  }
}
