<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import { Recorder, type RecorderResult } from '$lib/audio/recorder.svelte';
  import { drafts } from '$lib/stores/drafts.svelte';
  import { encodeWavPcm16, SLICE_SAMPLES, WAV_SAMPLE_RATE } from '$lib/audio/wav';
  import { decodeAudioFile, encodeWavFromChunks, encodeWavFromFloat32 } from '$lib/audio/resample';
  import { readWavMagic, decodeCanonicalWav } from '$lib/audio/wav-decode';
  import { chunkPcmToSlices, sliceCountFor } from '$lib/audio/slicer';
  import { sha256Hex } from '$lib/audio/sha256';
  import { slices } from '$lib/stores/slices.svelte';
  import { SvelteSet } from 'svelte/reactivity';
  import { streams } from '$lib/stores/streams.svelte';
  import {
    formatRecordingClock,
    formatDuration,
    formatDurationHuman,
    formatBytes
  } from '$lib/utils/format';
  import { SLICE_BATCH_WARN_THRESHOLD, prettyCategoryName } from './labels';
  import LiveRecorderWaveform from './LiveRecorderWaveform.svelte';
  import EnvelopeWaveform from '$lib/components/EnvelopeWaveform.svelte';
  import TrimWaveform from './TrimWaveform.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Tips from '$lib/components/ui/Tips.svelte';
  import Spinner from '$lib/components/Spinner.svelte';
  import { socketLabel } from '$lib/components/dashboard/socketPill';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';
  import type { DraftRecord } from '$lib/idb/db';

  interface Props {
    workspaceId: Uuid;
    categoryName: string;
    workspaceName: string;
    maxDurationMs?: number;
  }
  // Must equal the recorder's default (~50 min, under the import cap) else the stream-stop hint disagrees.
  let { workspaceId, categoryName, workspaceName, maxDurationMs = 3_000_000 }: Props = $props();

  // Checked on `file.size` before decode since decode+resample peaks past ~2 GiB transient and OOMs low-end machines.
  const MAX_IMPORT_BYTES = 256 * 1024 * 1024;

  type Op = 'recording' | 'streaming' | 'finalizing' | 'importing' | null;

  // Distinct from `recorder.state` so a no-mic import/stream op doesn't touch the mic graph.
  let op = $state<Op>(null);
  let maxReached = $state(false);
  let error = $state<string | null>(null);
  // The `streams` ring only holds the visualizer's ~10 s lookback, so a PCM tap accumulates every frame
  // to honor `maxDurationMs`; the RAF counter freezes in a hidden tab but the setTimeout auto-stop fires.
  let streamStartedAtMs = $state(0);
  let streamDurationMs = $state(0);
  let streamRafId = 0;
  let streamAutoStopTimer: ReturnType<typeof setTimeout> | null = null;
  // Plain `let` (NOT `$state`): tap pushes ~50 Hz; only `stopStream`'s synchronous snapshot reads these.
  // A future `$derived` over them must convert BOTH to `$state` (a half-reactive read snapshots once + silently drifts).
  let streamChunks: Float32Array[] = [];
  let streamCapturedSamples = 0;
  let streamTapDispose: (() => void) | null = null;
  const streamMaxDurationMs = $derived(maxDurationMs);
  const isStreaming = $derived(op === 'streaming'); // hoisted above its $effect/$derived users (TDZ)
  let draftPcm = $state<Float32Array | null>(null);
  let decodingDraft = $state(false);

  const initialMaxDurationMs = untrack(() => maxDurationMs);
  const recorder = new Recorder({
    maxDurationMs: initialMaxDurationMs,
    onMaxDurationReached: () => {
      maxReached = true;
      op = 'finalizing';
    },
    // Persist the cap-finalized WAV via the same path a manual Stop uses, else it's discarded.
    onAutoStop: (result) => {
      void persistAutoStopRecording(result);
    }
  });

  // Input source = one localStorage key: `:stream`/`:mic` sentinels, a bare mic id, or
  // `{id,label,groupId}` JSON re-attaching across the per-session `deviceId` rotation (Firefox/Safari).
  // DEFAULT (absent key) is the opus stream so slices share the daemon's inference DSP. INVARIANT: '' is
  // never written (every `''`/`!k` arm is dead defense).
  type InputSource = { kind: 'mic'; deviceId: string } | { kind: 'stream' };
  interface SavedSource {
    id: string;
    label?: string;
    groupId?: string;
  }

  const SOURCE_STORAGE_KEY = 'acousticslab:input-device-id';
  const STREAM_KEY = ':stream';
  const DEFAULT_MIC_KEY = ':mic'; // distinct from an absent entry (-> stream default)
  const DEFAULT_SOURCE_KEY = STREAM_KEY;

  function readSaved(): SavedSource {
    if (typeof window === 'undefined') return { id: '' };
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(SOURCE_STORAGE_KEY);
    } catch {
      return { id: '' }; // private-mode / quota: treat as no preference
    }
    if (!raw) return { id: '' };
    if (!raw.startsWith('{')) return { id: raw }; // bare deviceId / sentinel
    try {
      const p = JSON.parse(raw) as Partial<SavedSource>;
      return {
        id: typeof p.id === 'string' ? p.id : '',
        ...(typeof p.label === 'string' && p.label ? { label: p.label } : {}),
        ...(typeof p.groupId === 'string' && p.groupId ? { groupId: p.groupId } : {})
      };
    } catch {
      return { id: '' };
    }
  }

  function asSource(key: string): InputSource {
    if (key === STREAM_KEY) return { kind: 'stream' };
    // `:mic` -> empty deviceId, passed to `getUserMedia` as `deviceId || undefined` -> system default.
    if (key === DEFAULT_MIC_KEY) return { kind: 'mic', deviceId: '' };
    return { kind: 'mic', deviceId: key };
  }

  // Recovery target from storage, nulled only on an explicit pick / `refreshDevices`' no-target branch.
  // Reactive so the "(remembered) …" phantom re-renders when nulled.
  let initialSaved = $state<SavedSource | null>(readSaved());

  let audioInputs = $state<MediaDeviceInfo[]>([]);
  // Explicit `=== ''` (not `??`/`||`) since the stream-default fallback must fire on the empty string.
  let selectedKey = $state<string>(
    untrack(() => {
      const savedId = initialSaved?.id ?? '';
      return savedId === '' ? DEFAULT_SOURCE_KEY : savedId;
    })
  );
  const selectedSource = $derived<InputSource>(asSource(selectedKey));

  // Refcount the streams worker while the opus stream is selected (intent-, not mount-driven).
  // `$effect.pre` so `streamOptionLabel` reads fresh `audioStatus` on the flip, not stale 'closed' a frame.
  $effect.pre(() => {
    if (selectedSource.kind === 'stream') return streams.acquire();
  });

  // Persist the selection (bare id -> metadata JSON once labeled). Guards: never overwrite JSON with a
  // bare id for the same id (strips metadata); never clobber storage mid-recovery (loses replug entry).
  $effect(() => {
    const k = selectedKey;
    if (typeof window === 'undefined') return;
    try {
      if (k === DEFAULT_MIC_KEY || !k) {
        // Write `:mic` only when recovery isn't in flight (absent key reloads as stream).
        if (!initialSaved) {
          localStorage.setItem(SOURCE_STORAGE_KEY, DEFAULT_MIC_KEY);
        }
        return;
      }
      if (k === STREAM_KEY) {
        localStorage.setItem(SOURCE_STORAGE_KEY, STREAM_KEY);
        return;
      }
      const d = audioInputs.find((x) => x.deviceId === k);
      const next =
        d && (d.label || d.groupId)
          ? JSON.stringify({
              id: k,
              ...(d.label ? { label: d.label } : {}),
              ...(d.groupId ? { groupId: d.groupId } : {})
            })
          : k;
      const current = localStorage.getItem(SOURCE_STORAGE_KEY);
      if (current === next) return;
      if (current?.startsWith('{') && !next.startsWith('{')) {
        const parsed = JSON.parse(current) as Partial<SavedSource>;
        if (parsed.id === k) return; // would strip metadata for the same id
      }
      localStorage.setItem(SOURCE_STORAGE_KEY, next);
    } catch {
      /* best-effort: ignore quota / private-mode / parse failures */
    }
  });

  // Bails silently on missing `mediaDevices` (insecure context / old Safari). Reconciliation is gated on
  // a LABELED enumeration: pre-permission, the placeholder list would wipe the preference every visit.
  async function refreshDevices(): Promise<void> {
    const md = navigator.mediaDevices as MediaDevices | undefined;
    if (!md) {
      audioInputs = [];
      return;
    }
    try {
      const all = await md.enumerateDevices();
      audioInputs = all.filter((d) => d.kind === 'audioinput');
    } catch {
      audioInputs = [];
      return;
    }
    // Not yet labeled: bail and let the post-`recorder.start()` refresh reconcile.
    if (!audioInputs.some((d) => d.label)) return;
    const saved = initialSaved;
    // No recovery target (no preference / a sentinel): drop the tombstone to unblock the default write.
    if (!saved?.id || saved.id === STREAM_KEY || saved.id === DEFAULT_MIC_KEY) {
      if (saved) initialSaved = null;
      return;
    }
    // Match in stability order deviceId -> groupId (per physical device) -> label, surviving id
    // rotation. `.label`/`.groupId` are empty pre-permission, so they auto-skip.
    let match: MediaDeviceInfo | undefined = audioInputs.find((d) => d.deviceId === saved.id);
    if (!match && saved.groupId) {
      const want = saved.groupId;
      match = audioInputs.find((d) => d.groupId === want);
    }
    if (!match && saved.label) {
      const want = saved.label.toLowerCase();
      match = audioInputs.find((d) => d.label.toLowerCase() === want);
    }
    if (match) {
      // `initialSaved` kept alive (nulled only on an explicit pick) so a later replug recovers.
      if (selectedKey !== match.deviceId) selectedKey = match.deviceId;
    } else {
      // Saved device disconnected: surface `:mic`; `initialSaved` stays non-null so the saved entry
      // survives for a future replug.
      if (selectedKey !== DEFAULT_MIC_KEY) selectedKey = DEFAULT_MIC_KEY;
    }
  }

  $effect(() => {
    void refreshDevices();
    const md = navigator.mediaDevices as MediaDevices | undefined;
    if (!md) return;
    const onDeviceChange = (): void => {
      void refreshDevices();
    };
    md.addEventListener('devicechange', onDeviceChange);
    return () => {
      md.removeEventListener('devicechange', onDeviceChange);
    };
  });

  // `untrack` the refresh: `drafts.refresh` reads the slice it writes, which would re-fire this effect.
  $effect(() => {
    const id = workspaceId;
    const name = categoryName;
    untrack(() => {
      void drafts.refresh(id, name);
    });
  });

  const draftSlice = $derived(drafts.for(workspaceId, categoryName));
  const draft = $derived(draftSlice.draft);

  // `decodeCanonicalWav` is cheap (header skip + Int16->Float32, no AudioContext) since the blob is canonical.
  $effect(() => {
    const current = draft;
    if (!current) {
      draftPcm = null;
      decodingDraft = false;
      return;
    }
    decodingDraft = true;
    let cancelled = false;
    void decodeCanonicalWav(current.blob)
      .then(({ pcm }) => {
        if (cancelled) return;
        draftPcm = pcm;
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        error = e instanceof Error ? e.message : m.category.input_pane.error_could_not_decode_draft;
        draftPcm = null;
      })
      .finally(() => {
        if (cancelled) return;
        decodingDraft = false;
      });
    return () => {
      cancelled = true;
    };
  });

  // Local mirror of persisted `draft.trim_*_samples`: TrimWaveform writes per pointermove, persists once on pointerup.
  let trimStart = $state(0);
  let trimEnd = $state(0);

  // Clamp via locals, writing trimStart/trimEnd once: reading them in the conditions would add them to
  // the dep set, so a drag would re-fire this effect and reset the value. Reads only `draft`/`draftPcm`.
  $effect(() => {
    const current = draft;
    const pcm = draftPcm;
    sliceNote = null; // write-only (no re-fire) before the early return so a stale note doesn't survive
    if (!current || !pcm) return;
    let ns = current.trim_start_samples ?? 0;
    let ne = current.trim_end_samples ?? pcm.length;
    if (ns < 0) ns = 0;
    if (ne > pcm.length) ne = pcm.length;
    if (ne - ns < SLICE_SAMPLES) {
      // Persisted range collapsed below the slicer minimum; reset to full clip to allow a re-trim.
      ns = 0;
      ne = pcm.length;
    }
    trimStart = ns;
    trimEnd = ne;
  });

  function onTrimChange(start: number, end: number): void {
    trimStart = start;
    trimEnd = end;
    sliceNote = null;
  }
  function onTrimCommit(start: number, end: number): void {
    if (!draft) return;
    void drafts.patchTrim(workspaceId, categoryName, start, end);
  }

  let slicing = $state(false); // declared before `canSlice` $derived to dodge a TDZ error

  const trimRangeSamples = $derived(Math.max(0, trimEnd - trimStart));
  const trimRangeMs = $derived(Math.round((trimRangeSamples / WAV_SAMPLE_RATE) * 1000));
  const projectedSliceCount = $derived(sliceCountFor(trimStart, trimEnd));
  // Samples past the last full 1 s slice (slicer floor-divides), telegraphed so the operator can reclaim them.
  const unusedSamples = $derived(
    Math.max(0, trimRangeSamples - projectedSliceCount * SLICE_SAMPLES)
  );
  const unusedMs = $derived(Math.round((unusedSamples / WAV_SAMPLE_RATE) * 1000));
  // Post-slice summary in the status hint (NOT the `error` banner); cleared on later trim
  // drag / draft. Reports sha256-dedupe collapse (routine for silence recordings).
  let sliceNote = $state<string | null>(null);
  // No cumulative cap (daemon has none): amber-but-enabled.
  const largeBatch = $derived(projectedSliceCount > SLICE_BATCH_WARN_THRESHOLD);
  const canSlice = $derived(!!draftPcm && !slicing && trimRangeSamples >= SLICE_SAMPLES);

  // Selection playback + cursor: shared AudioContext + AudioBuffer (invalidated on `draftPcm` change),
  // each play a fresh one-shot source. Seek mutes during drag (`seeking`) + restarts once on pointerup.
  let playAudioCtx: AudioContext | null = null;
  let playAudioBuffer: AudioBuffer | null = null;
  let activeSource: AudioBufferSourceNode | null = null;
  let playing = $state(false);
  let playbackSample = $state<number | null>(null);
  let playbackStartCtxTime = 0;
  let playbackStartOffset = 0;
  let playbackRaf = 0;
  // True between first drag tick and pointerup; audio muted + RAF cursor paused while set.
  let seeking = $state(false);
  // An AnalyserNode tees off the playback graph so the bar reflects audible (post-mix) loudness, not a
  // pre-computed envelope. Lazy on first play, reused across plays + seek-restarts.
  let playAnalyser: AnalyserNode | null = null;
  let playLevelBuf: Float32Array | null = null;
  let playbackLevel = $state(0);

  $effect(() => {
    void draftPcm; // track to invalidate the cached AudioBuffer on PCM change
    playAudioBuffer = null;
  });

  function tickPlayback(): void {
    playbackRaf = 0;
    if (!activeSource || !playAudioCtx || seeking) return;
    const elapsedSec = playAudioCtx.currentTime - playbackStartCtxTime;
    const pos = playbackStartOffset + Math.floor(elapsedSec * WAV_SAMPLE_RATE);
    // RMS, smoothed (50/50) like `recorder.level` so the bar reads identically for mic or playback.
    if (playAnalyser) {
      const n = playAnalyser.fftSize;
      if (playLevelBuf?.length !== n) {
        playLevelBuf = new Float32Array(n);
      }
      playAnalyser.getFloatTimeDomainData(playLevelBuf as Float32Array<ArrayBuffer>);
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = playLevelBuf[i];
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / n);
      playbackLevel = playbackLevel * 0.5 + rms * 0.5;
    }
    if (pos >= trimEnd) {
      playbackSample = trimEnd;
      return; // source.onended will fire imminently
    }
    playbackSample = pos;
    playbackRaf = requestAnimationFrame(tickPlayback);
  }

  function stopActiveSource(): void {
    if (activeSource) {
      activeSource.onended = null;
      try {
        activeSource.stop();
      } catch {
        /* source may already be stopped */
      }
      activeSource = null;
    }
    if (playbackRaf !== 0) {
      cancelAnimationFrame(playbackRaf);
      playbackRaf = 0;
    }
    playbackLevel = 0; // no stale tail on next play
  }

  async function startPlayback(fromSample: number): Promise<void> {
    if (!draftPcm) return;
    const safeFrom = Math.max(trimStart, Math.min(trimEnd - 1, fromSample));
    const remainingSamples = trimEnd - safeFrom;
    if (remainingSamples <= 0) return;
    playAudioCtx ??= new AudioContext();
    if (playAudioCtx.state === 'suspended') {
      await playAudioCtx.resume();
      // Re-check across the suspension point: unmount/discardDraft may null `playAudioCtx`/`draftPcm`
      // mid-resume (TS's flow analysis can't see it), else the post-await code starts a zombie source.
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      if (!playAudioCtx || !draftPcm) return;
    }
    if (!playAudioBuffer) {
      const buf = playAudioCtx.createBuffer(1, draftPcm.length, WAV_SAMPLE_RATE);
      buf.copyToChannel(draftPcm as Float32Array<ArrayBuffer>, 0);
      playAudioBuffer = buf;
    }
    // `fftSize 1024` matches the recorder (~23 ms window @ 44.1 kHz, below the RAF interval);
    // `smoothingTimeConstant 0` returns raw frames, smoothed in JS for symmetry with the recorder.
    if (!playAnalyser) {
      playAnalyser = playAudioCtx.createAnalyser();
      playAnalyser.fftSize = 1024;
      playAnalyser.smoothingTimeConstant = 0;
      playAnalyser.connect(playAudioCtx.destination);
    }
    stopActiveSource();

    const source = playAudioCtx.createBufferSource();
    source.buffer = playAudioBuffer;
    source.connect(playAnalyser); // source -> analyser -> destination: bar reads the signal hitting the speakers
    const offsetSec = safeFrom / WAV_SAMPLE_RATE;
    const durationSec = remainingSamples / WAV_SAMPLE_RATE;
    source.start(0, offsetSec, durationSec);
    activeSource = source;
    playbackStartCtxTime = playAudioCtx.currentTime;
    playbackStartOffset = safeFrom;
    playbackSample = safeFrom;
    playing = true;
    source.onended = (): void => {
      if (activeSource === source) {
        activeSource = null;
        playing = false;
        playbackSample = null;
        playbackLevel = 0;
      }
    };
    if (playbackRaf === 0) {
      playbackRaf = requestAnimationFrame(tickPlayback);
    }
  }

  async function playSelection(): Promise<void> {
    if (!draftPcm) return;
    await startPlayback(trimStart);
  }

  function stopPlayback(): void {
    stopActiveSource();
    playing = false;
    playbackSample = null;
    seeking = false;
  }

  function teardownPlayback(): void {
    stopPlayback();
    playAudioBuffer = null;
    if (playAnalyser) {
      try {
        playAnalyser.disconnect();
      } catch {
        /* analyser may already be disconnected */
      }
      playAnalyser = null;
    }
    playLevelBuf = null;
    if (playAudioCtx) {
      playAudioCtx.close().catch(() => undefined);
      playAudioCtx = null;
    }
  }

  function onPlaybackSeek(sample: number): void {
    if (!seeking) {
      seeking = true;
      stopActiveSource(); // stop audio without clearing `playing`: cursor + "stop" mode persist until commit/Stop
    }
    playbackSample = Math.max(trimStart, Math.min(trimEnd, sample));
  }

  async function onPlaybackSeekCommit(sample: number): Promise<void> {
    seeking = false;
    const target = Math.max(trimStart, Math.min(trimEnd - 1, sample));
    if (target >= trimEnd - 1) {
      stopPlayback(); // dragged to/past the end: treat as natural stop
      return;
    }
    await startPlayback(target);
  }

  async function performSlice(): Promise<void> {
    if (!canSlice || !draftPcm) return;
    slicing = true;
    error = null;
    sliceNote = null;
    try {
      // Every full slice uploads, silence included; nothing is pre-filtered.
      const windows = chunkPcmToSlices(draftPcm, trimStart, trimEnd);
      // sha256 of the encoded WAV bytes is the slice's canonical id (daemon filename + cache key).
      const stamped = await Promise.all(
        windows.map(async (samples) => {
          const blob = encodeWavPcm16(samples, WAV_SAMPLE_RATE);
          const buf = await blob.arrayBuffer();
          const id = await sha256Hex(buf);
          return { id, blob };
        })
      );
      // Dedupe byte-identical windows early to skip extra `append` round-trips (IDB would dedupe anyway).
      const seen = new SvelteSet<string>();
      const unique: typeof stamped = [];
      for (const s of stamped) {
        if (seen.has(s.id)) continue;
        seen.add(s.id);
        unique.push(s);
      }
      const created_at = new Date().toISOString();
      for (const { id, blob } of unique) {
        const record = {
          id,
          workspace_id: workspaceId,
          category_name: categoryName,
          blob,
          state: 'local' as const,
          created_at
        };
        await slices.append(record);
        void slices.enqueueUpload(record);
      }
      // Without this note a silent recording looks like it "lost" slices.
      if (unique.length < stamped.length) {
        sliceNote = m.category.input_pane.duplicates_collapsed_suffix(
          stamped.length - unique.length
        );
      }
    } catch (e) {
      error = e instanceof Error ? e.message : m.category.input_pane.error_could_not_slice;
    } finally {
      slicing = false;
    }
  }

  onDestroy(() => {
    recorder.dispose();
    teardownPlayback();
    if (streamAutoStopTimer !== null) {
      clearTimeout(streamAutoStopTimer);
      streamAutoStopTimer = null;
    }
    if (streamRafId !== 0) {
      cancelAnimationFrame(streamRafId);
      streamRafId = 0;
    }
    streamTapDispose?.();
    streamTapDispose = null;
  });

  // Tear down playback on draft removal (Discard), else an active play holds a source over stale PCM.
  $effect(() => {
    if (!draft) stopPlayback();
  });

  async function startRecording(deviceId: string): Promise<void> {
    error = null;
    maxReached = false;
    try {
      op = 'recording';
      await recorder.start({ deviceId: deviceId || undefined });
    } catch {
      op = null; // recorder.error already populated; surfaced inline by the banner
      return;
    }
    void refreshDevices(); // first successful capture grants label visibility
  }

  async function stopRecording(): Promise<void> {
    op = 'finalizing';
    let result: RecorderResult | null;
    try {
      result = await recorder.stop();
    } catch {
      op = null;
      return;
    }
    if (!result) {
      op = null; // tap-too-short (zero samples): leave the prior draft intact
      return;
    }
    try {
      await saveResult(result, 'recorded');
    } catch (e) {
      error = e instanceof Error ? e.message : m.category.input_pane.error_could_not_save_recording;
    } finally {
      op = null;
    }
  }

  function cancelRecording(): void {
    recorder.cancel();
    maxReached = false;
    op = null;
  }

  // Can't reuse `stopRecording`: the recorder has already finalized by the time `onAutoStop` fires.
  async function persistAutoStopRecording(result: RecorderResult | null): Promise<void> {
    try {
      if (result) await saveResult(result, 'recorded');
    } catch (e) {
      error = e instanceof Error ? e.message : m.category.input_pane.error_could_not_save_recording;
    } finally {
      op = null;
    }
  }

  // Stream capture (opus stream -> draft): tap the stream, tick at RAF, and on stop pipe the
  // accumulator through `encodeWavFromChunks` (the mic finalize path).
  function tickStreamDuration(): void {
    if (op !== 'streaming') {
      streamRafId = 0;
      return;
    }
    streamDurationMs = Math.round(performance.now() - streamStartedAtMs);
    streamRafId = requestAnimationFrame(tickStreamDuration);
  }

  function startStream(): void {
    if (op !== null || streams.audioStatus !== 'open') return;
    error = null;
    maxReached = false;
    streamStartedAtMs = performance.now();
    streamDurationMs = 0;
    streamChunks = [];
    streamCapturedSamples = 0;
    op = 'streaming';
    // `op` flips BEFORE attaching the tap so the callback's guard reads 'streaming' from the first
    // packet. Retaining the worker-transferred Float32Array avoids a memcpy.
    streamTapDispose = streams.tap((pcm) => {
      if (op !== 'streaming') return;
      streamChunks.push(pcm);
      streamCapturedSamples += pcm.length;
    });
    if (streamRafId === 0) {
      streamRafId = requestAnimationFrame(tickStreamDuration);
    }
    streamAutoStopTimer = setTimeout(() => {
      streamAutoStopTimer = null;
      maxReached = true;
      void stopStream();
    }, streamMaxDurationMs);
  }

  async function stopStream(): Promise<void> {
    if (op !== 'streaming') return;
    if (streamAutoStopTimer !== null) {
      clearTimeout(streamAutoStopTimer);
      streamAutoStopTimer = null;
    }
    if (streamRafId !== 0) {
      cancelAnimationFrame(streamRafId);
      streamRafId = 0;
    }
    // Detach the tap BEFORE flipping `op` so a same-tick packet can't sneak past the stop boundary.
    streamTapDispose?.();
    streamTapDispose = null;
    op = 'finalizing';
    const chunks = streamChunks;
    const totalSamples = streamCapturedSamples;
    streamChunks = [];
    streamCapturedSamples = 0;
    if (totalSamples <= 0) {
      op = null; // tap-too-short / socket closed mid-capture: leave the prior draft intact
      return;
    }
    try {
      // Widen to `number`: literal types 48000 vs 44100 make the rate-equal check statically `false`.
      const captureRate = streams.sampleRate as number;
      const { blob, outputSamples } = await encodeWavFromChunks(
        chunks,
        totalSamples,
        captureRate,
        WAV_SAMPLE_RATE
      );
      const durationMs = Math.round((outputSamples / WAV_SAMPLE_RATE) * 1000);
      await saveResult(
        { blob, durationMs, sampleRate: WAV_SAMPLE_RATE },
        'imported',
        `live-stream-${new Date().toISOString().replace(/[:.]/g, '-')}.wav`
      );
    } catch (e) {
      error = e instanceof Error ? e.message : m.category.input_pane.error_could_not_capture_stream;
    } finally {
      op = null;
    }
  }

  function cancelStream(): void {
    if (streamAutoStopTimer !== null) {
      clearTimeout(streamAutoStopTimer);
      streamAutoStopTimer = null;
    }
    if (streamRafId !== 0) {
      cancelAnimationFrame(streamRafId);
      streamRafId = 0;
    }
    streamTapDispose?.();
    streamTapDispose = null;
    streamChunks = [];
    streamCapturedSamples = 0;
    streamDurationMs = 0;
    maxReached = false;
    op = null;
  }

  // The `!canStream` guard stays despite the disabled button: a keyboard activation racing a `closed`
  // socket transition bottoms out here, not half-starting a stream.
  async function startCapture(): Promise<void> {
    if (isBusy) return;
    if (selectedSource.kind === 'stream') {
      if (!canStream) return;
      startStream();
    } else {
      await startRecording(selectedSource.deviceId);
    }
  }

  async function stopCapture(): Promise<void> {
    if (op === 'recording') await stopRecording();
    else if (op === 'streaming') await stopStream();
  }

  function cancelCapture(): void {
    if (op === 'recording') cancelRecording();
    else if (op === 'streaming') cancelStream();
  }

  async function saveResult(result: RecorderResult, source: 'recorded'): Promise<void>;
  async function saveResult(
    result: { blob: Blob; durationMs: number; sampleRate: number },
    source: 'imported',
    originalName: string
  ): Promise<void>;
  async function saveResult(
    result: { blob: Blob; durationMs: number; sampleRate: number },
    source: 'recorded' | 'imported',
    originalName?: string
  ): Promise<void> {
    const record: DraftRecord = {
      workspace_id: workspaceId,
      category_name: categoryName,
      blob: result.blob,
      duration_ms: result.durationMs,
      sample_rate: result.sampleRate,
      size_bytes: result.blob.size,
      source,
      created_at: new Date().toISOString(),
      ...(originalName !== undefined ? { original_name: originalName } : {})
    };
    await drafts.save(record);
  }

  let dragging = $state(false);
  let inputEl = $state<HTMLInputElement | undefined>();

  async function importFiles(files: FileList | File[]): Promise<void> {
    if (op !== null) return;
    const list = Array.from(files);
    if (list.length === 0) return;
    const file = list[0];
    if (list.length > 1) {
      error = m.category.input_pane.error_only_one_file;
      return;
    }
    // Reject over-cap before decode (decode inflates the view 4x and OOMs the tab). Clear the picker so
    // a re-pick of the same file refires `change`.
    if (file.size > MAX_IMPORT_BYTES) {
      error = m.category.input_pane.error_file_too_large(
        formatBytes(file.size),
        formatBytes(MAX_IMPORT_BYTES)
      );
      if (inputEl) inputEl.value = '';
      return;
    }
    op = 'importing';
    error = null;
    try {
      const magic = await readWavMagic(file);
      if (!magic.valid) {
        error = magic.reason ?? m.category.input_pane.error_only_wav;
        return;
      }
      const { pcm, sampleRate } = await decodeAudioFile(file);
      const { blob, outputSamples } = await encodeWavFromFloat32(pcm, sampleRate, WAV_SAMPLE_RATE);
      // The daemon rejects sub-1 s clips at train time (TooShort); reject at import
      // so they surface here instead of as training drops.
      if (outputSamples < SLICE_SAMPLES) {
        error = m.category.input_pane.error_clip_too_short(
          (outputSamples / WAV_SAMPLE_RATE).toFixed(1)
        );
        return;
      }
      const durationMs = Math.round((outputSamples / WAV_SAMPLE_RATE) * 1000);
      await saveResult({ blob, durationMs, sampleRate: WAV_SAMPLE_RATE }, 'imported', file.name);
    } catch (e) {
      error = e instanceof Error ? e.message : m.category.input_pane.error_could_not_import;
    } finally {
      op = null;
      if (inputEl) inputEl.value = '';
    }
  }

  function onDrop(e: DragEvent): void {
    e.preventDefault();
    dragging = false;
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) void importFiles(files);
  }
  function onDragOver(e: DragEvent): void {
    e.preventDefault();
    dragging = true;
  }
  function onDragLeave(e: DragEvent): void {
    const next = e.relatedTarget as Node | null;
    if (next && (e.currentTarget as Node).contains(next)) return;
    dragging = false;
  }
  function onPickerChange(e: Event): void {
    const target = e.currentTarget as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      void importFiles(target.files);
    }
  }

  function exportDraft(): void {
    const current = draft;
    if (!current) return;
    const url = URL.createObjectURL(current.blob);
    const a = document.createElement('a');
    a.href = url;
    const stamp = current.created_at.replace(/:/g, '-').replace(/\.\d+Z?$/, '');
    a.download = `${workspaceName}-${categoryName}-${stamp}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  async function discardDraft(): Promise<void> {
    if (!draft) return;
    stopPlayback();
    try {
      await drafts.clear(workspaceId, categoryName);
      maxReached = false;
    } catch (e) {
      error = e instanceof Error ? e.message : m.category.input_pane.error_could_not_discard;
    }
  }

  function dismissError(): void {
    // Dismiss ONLY the recorder-error banner; the local `error` $state has its own dismiss (clearing it
    // here would wipe an unacknowledged import/slice/save failure).
    recorder.reset();
  }

  // Loudness meter (cube-root compressor); reserves space whenever the pane has audio so layout doesn't bounce.
  const isPlayingAudio = $derived(playing);
  const showLevelBar = $derived(recorder.state === 'recording' || isStreaming || draft !== null);
  let streamLevel = $state(0);
  let streamLevelBuf: Float32Array | null = null;
  $effect(() => {
    if (!isStreaming) {
      streamLevel = 0;
      return;
    }
    let cancelled = false;
    let raf = 0;
    const tick = (): void => {
      if (cancelled) return;
      streamLevelBuf ??= new Float32Array(1024);
      streams.snapshot(streamLevelBuf.length, streamLevelBuf);
      let sumSq = 0;
      for (const v of streamLevelBuf) sumSq += v * v;
      const rms = Math.sqrt(sumSq / streamLevelBuf.length);
      streamLevel = streamLevel * 0.5 + rms * 0.5;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (raf !== 0) cancelAnimationFrame(raf);
    };
  });
  const currentLevel = $derived(
    recorder.state === 'recording'
      ? recorder.level
      : isStreaming
        ? streamLevel
        : isPlayingAudio
          ? playbackLevel
          : 0
  );
  const levelPct = $derived(Math.min(100, Math.cbrt(currentLevel) * 130));
  // Inverse clip-path inset: clip the top `(100 - level)%` so the bottom `level%` shows.
  const levelClipInset = $derived(`${Math.max(0, 100 - levelPct)}% 0 0 0`);
  // Out of the template: prettier would wrap this string and break the `style:` attribute.
  const LEVEL_GRADIENT =
    'linear-gradient(to top, var(--color-success-dot) 0%, var(--color-success-dot) 55%, var(--color-warning-dot) 78%, var(--color-danger-dot) 95%)';

  const recorderError = $derived(recorder.error);
  const isRecording = $derived(recorder.state === 'recording');
  const isFinalizing = $derived(op === 'finalizing' || recorder.state === 'finalizing');
  const isImporting = $derived(op === 'importing');
  // 'requesting' (the getUserMedia prompt) folds into the busy gate, else a second activation is swallowed.
  const isRequesting = $derived(recorder.state === 'requesting');
  const isBusy = $derived(
    isRequesting || isRecording || isStreaming || isFinalizing || isImporting
  );
  const canStream = $derived(streams.audioStatus === 'open');

  // Gates the trim-selection `<p>` and the action row's `mt-1.5` to keep gaps balanced.
  const showSelectionStatus = $derived(
    !!draft && !!draftPcm && !isRecording && !isFinalizing && !isImporting
  );

  // Also disabled for a selected-but-closed stream: a keyboard activation would race the start guard.
  const recordDisabled = $derived(isBusy || (selectedSource.kind === 'stream' && !canStream));
  const recordAriaLabel = $derived(
    selectedSource.kind === 'stream'
      ? m.category.input_pane.record_aria_stream
      : m.category.input_pane.record_aria_mic
  );
  // `AudioDecoder` is undefined in an insecure context / old browser (`audioStatus` then sticks at
  // 'closed'), so this flag keeps the not-open tooltip from blaming the daemon.
  const streamUnsupported = $derived(streams.unsupportedReason !== null);
  const recordTitle = $derived<string | undefined>(
    selectedSource.kind === 'stream'
      ? canStream
        ? m.category.input_pane.record_title_stream_open(formatDurationHuman(streamMaxDurationMs))
        : streamUnsupported
          ? m.category.input_pane.record_title_stream_unsupported
          : streams.audioStatus === 'connecting'
            ? m.category.input_pane.record_title_stream_connecting
            : m.category.input_pane.record_title_stream_closed
      : undefined
  );
  // Suffix the socket status only while stream is selected, else `audioStatus` reads its
  // construction-time 'closed' sentinel (a false "· Disconnected") since the worker isn't running.
  const streamOptionLabel = $derived(
    selectedSource.kind !== 'stream' || canStream
      ? m.category.input_pane.source_daemon_stream
      : m.category.input_pane.source_daemon_stream_with_status(socketLabel(streams.audioStatus))
  );

  const displayName = $derived(prettyCategoryName(categoryName));

  // Labels are empty until mic permission is granted once on this origin; fall back to a numbered label.
  function describeDevice(d: MediaDeviceInfo, idx: number): string {
    if (d.label) return d.label;
    const idFrag = d.deviceId.slice(0, 6) || m.category.input_pane.source_mic_default_id;
    return m.category.input_pane.source_mic_fallback(idx + 1, idFrag);
  }

  // Drives the "(remembered) …" phantom <option> when `selectedKey` is a specific mic absent from the
  // enumeration (pre-permission / disconnected). Sentinels have their own <option>; `!== ''` is inert.
  const rememberedMissing = $derived(
    selectedKey !== '' &&
      selectedKey !== STREAM_KEY &&
      selectedKey !== DEFAULT_MIC_KEY &&
      !audioInputs.some((d) => d.deviceId === selectedKey)
  );
  // Saved friendly label, falling back to a short id fragment for legacy bare-id / nulled `initialSaved`.
  const rememberedLabel = $derived(
    initialSaved?.label ??
      m.category.input_pane.source_mic_remembered_fallback(
        selectedKey.slice(0, 6) || m.category.input_pane.source_mic_default_id
      )
  );

  // A user pick wins over recovery: null `initialSaved` so a later reconcile won't re-attach the saved
  // device. (`<select>` fires `change` only on interaction, not programmatic mutations.)
  function onPickInputSource(): void {
    initialSaved = null;
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- `min-h-0` lets every child's `flex-1 min-h-0` shrink. `contain-size` zeros the DPR-sized waveform
     `<canvas>`'s 2:1-aspect OUTWARD contribution, which would else lift every canvas-mounting state
     ~80 px above the canvas-less empty state. -->
<div
  class="flex h-full min-h-0 flex-col gap-1.5 overflow-hidden rounded-md border bg-surface px-3 pt-1.5 pb-3 transition-colors contain-size {dragging
    ? 'border-accent bg-accent-soft/40'
    : 'border-line'}"
  ondrop={onDrop}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  aria-label={m.category.input_pane.pane_aria(displayName)}
>
  <header class="flex min-h-4.75 items-center justify-between gap-1.5">
    <div class="flex translate-y-px items-center gap-1.5">
      <h4 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
        {m.category.input_pane.heading}
      </h4>
      <!-- Counter-shift: Tips' own `-translate-y-px` over-corrects 1 px under the cluster's `translate-y-px`. -->
      <span class="inline-flex translate-y-px">
        <Tips label={m.category.input_pane.tips_label}>
          <ul class="space-y-1.5">
            <li>
              <strong class="font-medium text-fg">{m.category.input_pane.tip_stream_title}</strong>
              {m.category.input_pane.tip_stream_body}
            </li>
            <li>
              <strong class="font-medium text-fg">
                {m.category.input_pane.tip_environment_title}
              </strong>
              {m.category.input_pane.tip_environment_body}
            </li>
            <li>
              <strong class="font-medium text-fg">{m.category.input_pane.tip_meter_title}</strong>
              {m.category.input_pane.tip_meter_body}
            </li>
          </ul>
        </Tips>
      </span>
    </div>
    {#if isRecording}
      <span class="inline-flex items-center gap-1.5 text-[11px] text-fg-secondary">
        <span class="relative inline-flex h-2 w-2">
          <span
            class="absolute inset-0 inline-flex h-full w-full animate-ping rounded-full bg-danger-dot/70"
          ></span>
          <span class="relative inline-flex h-2 w-2 rounded-full bg-danger-dot"></span>
        </span>
        <span class="font-mono tabular-nums">{formatRecordingClock(recorder.durationMs)}</span>
      </span>
    {:else if isStreaming}
      <span class="inline-flex items-center gap-1.5 text-[11px] text-fg-secondary">
        <span class="relative inline-flex h-2 w-2">
          <span
            class="absolute inset-0 inline-flex h-full w-full animate-ping rounded-full bg-accent/70"
          ></span>
          <span class="relative inline-flex h-2 w-2 rounded-full bg-accent"></span>
        </span>
        <span class="font-mono tabular-nums">{formatRecordingClock(streamDurationMs)}</span>
      </span>
    {:else if draft}
      <span class="font-mono text-[11px] tabular-nums text-fg-muted">
        {formatDuration(draft.duration_ms)} · {formatBytes(draft.size_bytes)}
      </span>
    {/if}
  </header>

  <!-- `flex-1 min-h-0` (no floor) lets this slot compress under variable chrome so the pane doesn't
       ratchet taller on an error. -->
  <div class="flex min-h-0 flex-1 gap-2">
    <div class="relative flex-1 overflow-hidden rounded-md bg-canvas">
      <!-- `recorder.state`, not `isFinalizing`: a stream-capture finalize sets `op = 'finalizing'`
           without starting the recorder, so the broader test would mount its idle flat baseline. -->
      {#if isRecording || recorder.state === 'finalizing'}
        <LiveRecorderWaveform {recorder} />
      {:else if isStreaming}
        <EnvelopeWaveform source={streams} />
      {:else if draft && draftPcm}
        <TrimWaveform
          pcm={draftPcm}
          startSamples={trimStart}
          endSamples={trimEnd}
          onChange={onTrimChange}
          onCommit={onTrimCommit}
          {playbackSample}
          onSeek={onPlaybackSeek}
          onSeekCommit={onPlaybackSeekCommit}
        />
      {:else if decodingDraft}
        <div class="flex h-full items-center justify-center text-[11px] text-fg-muted">
          <Spinner class="mr-1.5 h-3 w-3 text-fg-muted" />
          {m.category.input_pane.capture_decoding}
        </div>
      {:else}
        <!-- Empty-state drop zone (drop is wired on the whole pane): advertises it + a click alternative. -->
        <label
          class="flex h-full cursor-pointer flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed border-line-strong px-3 text-center text-[11px] text-fg-muted transition hover:border-line-strong hover:bg-surface-2/40"
          class:border-accent={dragging}
          class:bg-accent-soft={dragging}
          title={m.category.input_pane.drop_zone_title(formatBytes(MAX_IMPORT_BYTES))}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="h-5 w-5 text-fg-subtle"
            aria-hidden="true"
          >
            <path d="M12 4v12" />
            <path d="M6 10l6-6 6 6" />
            <path d="M4 20h16" />
          </svg>
          <span>{m.category.input_pane.drop_zone_idle}</span>
          <span
            class="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary transition group-hover:border-line-strong"
          >
            {m.category.input_pane.drop_zone_browse}
          </span>
          <input
            bind:this={inputEl}
            type="file"
            accept=".wav,audio/wav,audio/wave,audio/x-wav,audio/vnd.wave"
            class="sr-only"
            onchange={onPickerChange}
            disabled={isBusy}
          />
        </label>
      {/if}
    </div>

    <!-- Slot kept across record/idle/playback to avoid the width snap on play start/stop; clip-path
         peels the top of a full-height gradient so the colour-at-height stays stable. -->
    {#if showLevelBar}
      <div
        class="relative w-2 overflow-hidden rounded-full bg-line/60"
        aria-hidden="true"
        aria-label={m.category.input_pane.loudness_aria}
      >
        <div
          class="absolute inset-0 transition-[clip-path] duration-75"
          style:clip-path="inset({levelClipInset})"
          style:background={LEVEL_GRADIENT}
        ></div>
      </div>
    {/if}
  </div>

  <!-- Selection status (draft only): trim range + slice count; a set `sliceNote` overrides the hint. -->
  {#if showSelectionStatus}
    <p
      class="text-[11px] tabular-nums"
      class:text-fg-muted={!!sliceNote || trimRangeSamples >= SLICE_SAMPLES}
      class:text-warning-soft-fg={!sliceNote && trimRangeSamples < SLICE_SAMPLES}
    >
      {#if sliceNote}
        {sliceNote}
      {:else}
        {m.category.input_pane.trim_selection_prefix}
        <span class="font-mono">{(trimRangeMs / 1000).toFixed(1)} s</span>
        ·
        {#if trimRangeSamples < SLICE_SAMPLES}
          {m.category.input_pane.trim_drag_hint}
        {:else}
          {m.category.input_pane.trim_projected_slices(projectedSliceCount)}
          {#if unusedMs >= 10}· <span class="font-mono">{(unusedMs / 1000).toFixed(1)} s</span>
            {m.category.input_pane.trim_unused_label}{/if}
        {/if}
      {/if}
    </p>
  {/if}

  <!-- `mt-1.5` only when the `<p>` is absent: a Button's edge IS its border, so a flush box-gap reads
       ~5.5 px tighter than the header->waveform gap; the `<p>`'s line-box padding balances it itself. -->
  <div class="flex flex-wrap items-center gap-2" class:mt-1.5={!showSelectionStatus}>
    {#if isRecording || isStreaming}
      <!-- One Stop + Discard for both capture paths; aria-label flips with the op. -->
      <Button
        variant="destructive"
        onclick={() => void stopCapture()}
        ariaLabel={isStreaming
          ? m.category.input_pane.capture_stop_aria_stream
          : m.category.input_pane.capture_stop_aria_mic}
      >
        <svg viewBox="0 0 24 24" fill="currentColor" class="h-3 w-3" aria-hidden="true">
          <rect x="6" y="6" width="12" height="12" rx="1.5" />
        </svg>
        {m.category.input_pane.capture_stop_label}
      </Button>
      <Button variant="secondary" onclick={cancelCapture}>
        {m.category.input_pane.capture_discard_label}
      </Button>
    {:else if isFinalizing}
      <Button disabled loading>{m.category.input_pane.capture_encoding}</Button>
    {:else if isImporting}
      <Button disabled loading>{m.category.input_pane.capture_decoding}</Button>
    {:else if draft}
      <!-- No cumulative cap; the `warning` variant (amber) is the sole guard for a large batch. -->
      <Button
        variant={largeBatch ? 'warning' : 'primary'}
        onclick={performSlice}
        disabled={!canSlice}
        loading={slicing}
        ariaLabel={canSlice
          ? m.category.input_pane.slice_aria_enabled(projectedSliceCount)
          : m.category.input_pane.slice_aria_disabled}
        title={canSlice
          ? m.category.input_pane.slice_title_enabled(projectedSliceCount)
          : m.category.input_pane.slice_title_disabled}
      >
        {#if !slicing}
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="h-3 w-3"
            aria-hidden="true"
          >
            <path d="M14.5 17.5l-9-9" />
            <path d="M9 6L6 9l-3-3 3-3z" />
            <path d="M21 15l-3-3 3-3 3 3z" />
            <path d="M14.5 6.5L19 11" />
          </svg>
        {/if}
        {canSlice
          ? m.category.input_pane.slice_label_count(projectedSliceCount)
          : m.category.input_pane.slice_label_bare}
      </Button>

      <Button
        variant="secondary"
        onclick={discardDraft}
        ariaLabel={m.category.input_pane.discard_aria}
        title={m.category.input_pane.discard_title}
      >
        {m.category.input_pane.discard_label}
      </Button>

      <!-- Play/Stop on the trimmed range; cursor + drag-to-seek live inside TrimWaveform. -->
      {#if playing}
        <Button
          variant="secondary"
          onclick={stopPlayback}
          ariaLabel={m.category.input_pane.play_stop_aria}
          title={m.category.input_pane.play_stop_title}
          class="min-h-8.5 px-2"
        >
          <svg viewBox="0 0 24 24" fill="currentColor" class="h-4 w-4" aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="1.5" />
          </svg>
        </Button>
      {:else}
        <Button
          variant="secondary"
          onclick={playSelection}
          disabled={!draftPcm || trimRangeSamples <= 0}
          ariaLabel={m.category.input_pane.play_aria}
          title={m.category.input_pane.play_title}
          class="min-h-8.5 px-2"
        >
          <svg viewBox="0 0 24 24" fill="currentColor" class="h-4 w-4" aria-hidden="true">
            <path d="M8 5v14l11-7z" />
          </svg>
        </Button>
      {/if}

      <Button
        variant="secondary"
        onclick={exportDraft}
        ariaLabel={m.category.input_pane.export_aria}
        title={m.category.input_pane.export_title}
        class="min-h-8.5 px-2"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="h-4 w-4"
          aria-hidden="true"
        >
          <path d="M12 4v12" />
          <path d="M6 14l6 6 6-6" />
          <path d="M4 20h16" />
        </svg>
      </Button>
    {:else}
      <!-- `startCapture` dispatches on the picked source; a stream pick survives a closed socket (Record
           disables, recovery-hint title). Uniform record-dot glyph; aria-label disambiguates. -->
      <Button
        onclick={() => void startCapture()}
        disabled={recordDisabled}
        ariaLabel={recordAriaLabel}
        title={recordTitle}
      >
        <svg viewBox="0 0 24 24" fill="currentColor" class="h-3 w-3" aria-hidden="true">
          <circle cx="12" cy="12" r="6" />
        </svg>
        {m.category.input_pane.record_label}
      </Button>
      <select
        id="input-source-{workspaceId}-{categoryName}"
        bind:value={selectedKey}
        onchange={onPickInputSource}
        class="select-chevron min-w-0 max-w-56 flex-1 truncate rounded-md border border-line bg-surface py-1.5 pl-3 text-sm font-medium text-fg transition hover:border-line-strong hover:bg-surface-2 disabled:cursor-not-allowed disabled:bg-page disabled:text-fg-subtle"
        aria-label={m.category.input_pane.source_aria}
        disabled={isBusy}
      >
        <optgroup label={m.category.input_pane.source_microphone_group}>
          <option value={DEFAULT_MIC_KEY}>{m.category.input_pane.source_system_default_mic}</option>
          {#if rememberedMissing}
            <!-- Phantom for the remembered-but-not-enumerated mic; else <select> shows "System default"
                 while the bound state holds the saved id. -->
            <option value={selectedKey}
              >{m.category.input_pane.source_remembered(rememberedLabel)}</option
            >
          {/if}
          {#each audioInputs as device, idx (device.deviceId || idx)}
            <option value={device.deviceId}>{describeDevice(device, idx)}</option>
          {/each}
        </optgroup>
        <optgroup label={m.category.input_pane.source_live_stream_group}>
          <option value={STREAM_KEY}>{streamOptionLabel}</option>
        </optgroup>
      </select>
    {/if}
  </div>

  {#if maxReached && !isRecording && !isStreaming}
    <!-- `leading-tight` (not `leading-none`) stays above the glyph height so descenders don't clip under `overflow-hidden`. -->
    <span class="text-[11px] leading-tight text-warning-soft-fg"
      >{m.category.input_pane.auto_stopped_at_cap}</span
    >
  {/if}

  <!-- Asymmetric padding (`pl-2.5` vs `pr-1`) compensates for the font's half-leading + ascender-cap delta. -->
  {#if recorderError}
    <div
      class="flex items-center justify-between gap-2 rounded-md border border-danger-line bg-danger-soft py-1 pr-1 pl-2.5 text-xs text-danger-soft-fg"
      role="alert"
    >
      <span class="min-w-0 flex-1">{recorderError}</span>
      <button
        type="button"
        class="shrink-0 rounded-md p-1 text-danger-soft-fg transition hover:bg-danger-soft"
        onclick={dismissError}
        aria-label={m.common.dismiss}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          class="h-3.5 w-3.5"
          aria-hidden="true"
        >
          <path d="M6 6l12 12M6 18L18 6" />
        </svg>
      </button>
    </div>
  {/if}
  {#if error}
    <div
      class="flex items-center justify-between gap-2 rounded-md border border-danger-line bg-danger-soft py-1 pr-1 pl-2.5 text-xs text-danger-soft-fg"
      role="alert"
    >
      <span class="min-w-0 flex-1">{error}</span>
      <button
        type="button"
        class="shrink-0 rounded-md p-1 text-danger-soft-fg transition hover:bg-danger-soft"
        onclick={() => (error = null)}
        aria-label={m.common.dismiss}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          class="h-3.5 w-3.5"
          aria-hidden="true"
        >
          <path d="M6 6l12 12M6 18L18 6" />
        </svg>
      </button>
    </div>
  {/if}
</div>
