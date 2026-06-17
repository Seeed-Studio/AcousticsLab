// Live SSE subscription to a training job. `JobEvent.message` carries the JSON-stringified typed
// `TrainEvent`, parsed and stamped with the envelope's `seq`/`at` into a `TrainLogLine` matching
// the JSONL backfill shape so store paths are source-agnostic. Relies on EventSource auto-reconnect
// (deduped on `seq`) not `Last-Event-ID`; pre-empting it races in-flight `error` events.

import { apiUrl } from './base';
import { jobs as jobsApi } from './endpoints';
import type {
  JobEvent,
  JobProgress,
  JobState,
  Rfc3339,
  TrainEvent,
  TrainLogLine,
  Uuid
} from './types';

// Inlined rather than imported from the jobs helper to avoid a circular dependency.
const TERMINAL_JOB_STATES: ReadonlySet<JobState> = new Set(['succeeded', 'failed', 'cancelled']);

// Gap-recovery bounds from a `code: event_gap` 409 (cursor fell outside the per-job ring):
// backfill JSONL over `[oldest_seq, latest_seq]` inclusive, then resubscribe at `latest_seq`.
export interface TrainingSubscriberGap {
  oldest_seq: number;
  latest_seq: number;
}

export interface TrainingSubscriberOptions {
  // Default 0 (ring beginning); gap recovery passes `latest_seq` to resume after backfill.
  afterSeq?: number;
  onEvent?: (event: TrainLogLine) => void;
  // The daemon emits `state` only on terminal transitions; mid-run state is implicitly 'running'.
  onStateTransition?: (state: JobState, at: Rfc3339, seq: number) => void;
  // Daemon-throttled to 4 Hz, carrying flat `{done, total}` (no phase/metrics).
  onProgress?: (progress: JobProgress, at: Rfc3339, seq: number) => void;
  // Fired on a `code: event_gap` 409, detected via the post-close diagnostic fetch.
  onGap?: (gap: TrainingSubscriberGap) => void;
  // Permanent CLOSE not classifiable as a gap; transient CONNECTING is left to auto-reconnect.
  onError?: (reason: string) => void;
}

export class TrainingSubscriber {
  private source: EventSource | null = null;
  private jobId: Uuid | null = null;
  private afterSeq = 0;
  private opts: TrainingSubscriberOptions = {};
  // Latches on a terminal `state` so the ensuing CLOSE reads as clean shutdown, not failure.
  private terminalObserved = false;
  // CLOSE while false = likely connection-time 4xx (diagnose); while true = mid-stream failure.
  private hasReceivedAnyEvent = false;
  private diagnosticCtrl: AbortController | null = null;

  // Idempotent.
  start(jobId: Uuid, opts: TrainingSubscriberOptions = {}): void {
    this.stop();
    this.jobId = jobId;
    this.afterSeq = opts.afterSeq ?? 0;
    this.opts = opts;
    this.terminalObserved = false;
    this.hasReceivedAnyEvent = false;

    const url = jobsApi.eventsUrl(jobId, { afterSeq: this.afterSeq, logs: true });
    const source = new EventSource(apiUrl(url));
    this.source = source;

    // Named `job` frame only, so future frame types (heartbeats, status) don't route here.
    source.addEventListener('job', (e: MessageEvent<string>) => {
      // Stale-source guard: a re-start() swapped the EventSource between bind and fire.
      if (this.source !== source) return;
      if (typeof e.data !== 'string') return;
      this.hasReceivedAnyEvent = true;
      let envelope: JobEvent;
      try {
        envelope = JSON.parse(e.data) as JobEvent;
      } catch (err) {
        opts.onError?.(`malformed SSE envelope: ${String(err)}`);
        return;
      }
      this.dispatch(envelope);
    });

    source.addEventListener('error', () => {
      if (this.source !== source) return;
      // Act only on permanent CLOSE; CONNECTING is the browser's retry loop (deduped on seq).
      if (source.readyState !== EventSource.CLOSED) return;
      if (this.terminalObserved) return;
      if (this.hasReceivedAnyEvent) {
        opts.onError?.('event stream closed mid-stream');
        return;
      }
      void this.diagnoseFailedSubscribe();
    });
  }

  stop(): void {
    this.jobId = null;
    if (this.source !== null) {
      this.source.close();
      this.source = null;
    }
    // Abort the in-flight diagnostic fetch; the post-await `jobId` guard only drops stale callbacks.
    this.diagnosticCtrl?.abort();
    this.diagnosticCtrl = null;
  }

  get running(): boolean {
    return this.source !== null && this.source.readyState !== EventSource.CLOSED;
  }

  // A JobEvent carries any subset of {state, progress, message}; the daemon batches state+progress on
  // a terminal frame while message always arrives in its own frame, so each branch fires independently.
  private dispatch(ev: JobEvent): void {
    if (ev.state !== undefined && TERMINAL_JOB_STATES.has(ev.state)) {
      this.terminalObserved = true;
    }
    if (ev.message !== undefined) {
      // Report-and-fall-through, never early-return: a malformed message must not skip a frame's
      // progress/state branches below; the fall-through is harmless, only the typed-event emit is dropped.
      let payload: TrainEvent | null = null;
      try {
        payload = JSON.parse(ev.message) as TrainEvent;
      } catch (err) {
        this.opts.onError?.(`malformed TrainEvent payload: ${String(err)}`);
      }
      if (payload !== null) {
        // Spread first so the envelope's `seq`/`at` (source of truth) win over payload fields.
        const line: TrainLogLine = { ...payload, seq: ev.seq, at: ev.at };
        this.opts.onEvent?.(line);
      }
    }
    if (ev.progress !== undefined) {
      this.opts.onProgress?.(ev.progress, ev.at, ev.seq);
    }
    if (ev.state !== undefined) {
      this.opts.onStateTransition?.(ev.state, ev.at, ev.seq);
    }
  }

  // EventSource hides HTTP status/body, so re-fetch the same URL (same `afterSeq`, so the daemon
  // re-evaluates the gap predicate identically) to classify a pre-event close: 409+`event_gap` ->
  // onGap, else onError; 5s timeout -> onError.
  private async diagnoseFailedSubscribe(): Promise<void> {
    const jobId = this.jobId;
    if (jobId === null) return;
    const url = jobsApi.eventsUrl(jobId, { afterSeq: this.afterSeq, logs: true });
    const ctrl = new AbortController();
    this.diagnosticCtrl = ctrl;
    const clearCtrl = (): void => {
      if (this.diagnosticCtrl === ctrl) this.diagnosticCtrl = null;
    };
    const timer = setTimeout(() => {
      ctrl.abort();
    }, 5_000);
    let resp: Response;
    try {
      resp = await fetch(apiUrl(url), {
        headers: { Accept: 'text/event-stream' },
        signal: ctrl.signal
      });
    } catch (e) {
      clearTimeout(timer);
      clearCtrl();
      // stop() clears `jobId` before aborting, so a teardown abort lands here stale and is dropped;
      // a timeout abort or real network error keeps the binding and surfaces.
      if (this.jobId !== jobId) return;
      this.opts.onError?.(`gap-diagnostic fetch failed: ${String(e)}`);
      return;
    }
    clearTimeout(timer);
    clearCtrl();
    // Bindings may have swapped while the fetch was in flight; drop and release the body.
    if (this.jobId !== jobId) {
      void resp.body?.cancel();
      return;
    }
    if (resp.status === 200) {
      void resp.body?.cancel();
      this.opts.onError?.('event stream closed before terminal state');
      return;
    }
    if (resp.status === 409) {
      let body: { code?: string; oldest_seq?: number; latest_seq?: number };
      try {
        body = (await resp.json()) as typeof body;
      } catch {
        this.opts.onError?.('409 response had non-JSON body');
        return;
      }
      // Re-check after the body-read suspension: a stop()/start(otherJob) here must not deliver this
      // old job's gap bounds to the now-live onGap (wrong-cursor backfill).
      if (this.jobId !== jobId) return;
      if (
        body.code === 'event_gap' &&
        typeof body.oldest_seq === 'number' &&
        typeof body.latest_seq === 'number'
      ) {
        this.opts.onGap?.({ oldest_seq: body.oldest_seq, latest_seq: body.latest_seq });
        return;
      }
      this.opts.onError?.(`409 with unexpected body code: ${body.code ?? 'unknown'}`);
      return;
    }
    if (resp.status === 404) {
      void resp.body?.cancel();
      this.opts.onError?.('job not found at daemon');
      return;
    }
    void resp.body?.cancel();
    this.opts.onError?.(`unexpected diagnostic status: ${resp.status}`);
  }
}
