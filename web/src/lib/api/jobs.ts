import { apiUrl } from './base';
import { jobs as jobsApi } from './endpoints';
import type { JobEvent, JobState, Uuid } from './types';

const TERMINAL_STATES: ReadonlySet<JobState> = new Set(['succeeded', 'failed', 'cancelled']);

export function isTerminal(state: JobState | undefined): boolean {
  return state !== undefined && TERMINAL_STATES.has(state);
}

export interface TrackJobOptions {
  // Cursor; only events with `seq > afterSeq` emit. 0 = fresh subscription.
  afterSeq?: number;
  // Default false omits log-line events, keeping SSE traffic to state+progress for delete tracking.
  logs?: boolean;
  onEvent?: (ev: JobEvent) => void;
  onTerminal?: (ev: JobEvent) => void;
  onError?: (reason: string) => void;
  // Cap on time to first `open`; 0 disables.
  connectTimeoutMs?: number;
}

const DEFAULT_CONNECT_TIMEOUT_MS = 30_000;

export interface JobTracker {
  // Idempotent; after the first call no further callbacks fire.
  cancel(): void;
}

// SSE subscriber for `GET /api/v1/jobs/{job_id}/events`. Scoped to short jobs (delete tracking +
// converter cleanup) that rarely overflow the event ring, so 409 `event_gap` JSONL-backfill recovery
// is intentionally omitted.
export function trackJob(jobId: Uuid, opts: TrackJobOptions = {}): JobTracker {
  const url = jobsApi.eventsUrl(jobId, {
    afterSeq: opts.afterSeq,
    logs: opts.logs ?? false
  });
  // `apiUrl` prefixes the VITE_API_BASE origin when cross-origin (else the same-origin BASE_PATH mount); cross-origin SSE also needs the daemon's CORS allowlist to admit this origin.
  const source = new EventSource(apiUrl(url));
  let closed = false;
  let connectTimer: ReturnType<typeof setTimeout> | null = null;

  const close = (): void => {
    if (closed) return;
    closed = true;
    if (connectTimer !== null) {
      clearTimeout(connectTimer);
      connectTimer = null;
    }
    source.close();
  };

  // A never-connecting stream (DNS broken, TCP RST loop, captive portal) emits no `open` and hangs the
  // caller, and `error` can't distinguish it from a transient retry; timer cleared on first `open`.
  const connectTimeoutMs = opts.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS;
  if (connectTimeoutMs > 0) {
    connectTimer = setTimeout(() => {
      connectTimer = null;
      if (closed) return;
      if (source.readyState === EventSource.OPEN) return;
      opts.onError?.(`event stream did not connect within ${connectTimeoutMs}ms`);
      close();
    }, connectTimeoutMs);
  }

  source.addEventListener('open', () => {
    // Disarm: later CONNECTING transitions are mid-stream auto-retry, not a hang.
    if (connectTimer !== null) {
      clearTimeout(connectTimer);
      connectTimer = null;
    }
  });

  // Daemon names its SSE frames `event: job`; listen specifically so future frame types don't route here.
  source.addEventListener('job', (e: MessageEvent) => {
    if (closed) return;
    let ev: JobEvent;
    try {
      ev = JSON.parse(e.data as string) as JobEvent;
    } catch (parseErr) {
      opts.onError?.(`malformed SSE payload: ${String(parseErr)}`);
      close();
      return;
    }
    opts.onEvent?.(ev);
    if (isTerminal(ev.state)) {
      opts.onTerminal?.(ev);
      close();
    }
  });

  // EventSource fires `error` for both transport failure and normal close; readyState disambiguates.
  // `closed` is already set after a terminal frame, so CLOSED here means the stream died pre-terminal;
  // CONNECTING is left to the browser's auto-reconnect (connect timeout covers never-connects).
  source.addEventListener('error', () => {
    if (closed) return;
    if (source.readyState === EventSource.CLOSED) {
      opts.onError?.('event stream closed before terminal state');
      close();
    }
  });

  return {
    cancel: close
  };
}

// Resolve on `succeeded`, reject otherwise; `operation` flavours the fallback message when the frame has none.
export function awaitJobTerminal(jobId: Uuid, operation = 'delete'): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    trackJob(jobId, {
      onTerminal: (ev) => {
        if (ev.state === 'succeeded') resolve();
        else reject(new Error(ev.message ?? `${operation} ${ev.state ?? 'ended without success'}`));
      },
      onError: (reason) => reject(new Error(reason))
    });
  });
}
