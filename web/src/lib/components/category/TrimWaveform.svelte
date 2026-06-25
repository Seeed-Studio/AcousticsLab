<script lang="ts">
  import StaticWaveform from './StaticWaveform.svelte';
  import { SLICE_SAMPLES, WAV_SAMPLE_RATE } from '$lib/audio/wav';
  import { m } from '$lib/i18n';

  // Controlled: parent owns start/endSamples; onChange per drag frame (live), onCommit on pointerup
  // (throttles persistence). Drag uses document-level capture-phase listeners (attached via $effect
  // for the drag's lifetime) to stay glued to the pointer ahead of descendant stopPropagation;
  // setPointerCapture is unreliable (reactive style:left reconciliation can release it) and
  // <svelte:window> misses handle-initiated drags.
  interface Props {
    pcm: Float32Array;
    startSamples: number;
    endSamples: number;
    onChange: (start: number, end: number) => void;
    onCommit: (start: number, end: number) => void;
    minGapSamples?: number;
    color?: string;
    background?: string;
    // null hides the cursor; onSeek per drag frame (scrub), onSeekCommit on drop (restart).
    playbackSample?: number | null;
    onSeek?: (sample: number) => void;
    onSeekCommit?: (sample: number) => void;
  }
  let {
    pcm,
    startSamples,
    endSamples,
    onChange,
    onCommit,
    minGapSamples = SLICE_SAMPLES,
    color,
    background,
    playbackSample = null,
    onSeek,
    onSeekCommit
  }: Props = $props();

  let wrapper = $state<HTMLDivElement | undefined>();
  // null outside a drag; gates the document-listener $effect.
  let dragging = $state<'start' | 'end' | 'playback' | 'window' | null>(null);
  // Wrapper geometry snapshotted at pointerdown (no getBoundingClientRect per move; a rare mid-drag
  // resize then uses a stale rect). Width defaults non-zero so clientXToSample can't divide by zero
  // before the first drag.
  let dragRectLeft = 0;
  let dragRectWidth = 1;
  // Frozen pointer + (start, end) baseline for the window slide: rounding each move's delta off this
  // one baseline keeps width bit-exact (per-move sample->percent->sample rounding crept it).
  let dragAnchorClientX = 0;
  let dragAnchorStart = 0;
  let dragAnchorEnd = 0;

  // `|| 1` guards divide-by-zero on the mount race where the parent flips its decoding draft before
  // pcm arrives (zero-length).
  const totalSamples = $derived(pcm.length || 1);
  const startPct = $derived((startSamples / totalSamples) * 100);
  const endPct = $derived((endSamples / totalSamples) * 100);
  // Clamped to [0, 100] so a rounding overshoot at end-of-clip can't render the grip off-canvas.
  const playbackPct = $derived(
    playbackSample === null
      ? null
      : Math.max(0, Math.min(100, (playbackSample / totalSamples) * 100))
  );

  function clientXToSample(clientX: number, total: number): number {
    if (dragRectWidth <= 0) return 0;
    const pct = (clientX - dragRectLeft) / dragRectWidth;
    const clamped = Math.max(0, Math.min(1, pct));
    return Math.round(clamped * total);
  }

  function startDrag(handle: 'start' | 'end' | 'playback' | 'window', e: PointerEvent): void {
    // preventDefault suppresses OS drag-selection. No stopPropagation needed: capture-phase document
    // listeners fire before any bubbled cancellation.
    e.preventDefault();
    const w = wrapper;
    if (!w) return;
    const rect = w.getBoundingClientRect();
    dragRectLeft = rect.left;
    dragRectWidth = rect.width;
    if (handle === 'window') {
      dragAnchorClientX = e.clientX;
      dragAnchorStart = startSamples;
      dragAnchorEnd = endSamples;
    }
    dragging = handle;
    // Secondary to the document listeners: pointerdown implicitly captures to e.target, so
    // re-capture on currentTarget (the handle) to keep touch gestures rooted there.
    try {
      const el = e.currentTarget as HTMLElement;
      el.setPointerCapture(e.pointerId);
    } catch {
      // Safari throws on released pointer ids; harmless here.
    }
  }

  $effect(() => {
    if (dragging === null) return;

    const onMove = (e: PointerEvent): void => {
      if (dragging === null) return;
      if (dragging === 'window') {
        // Slide both handles by one anchor-relative delta, clamping newStart into [0, total-width]
        // so the window pins to a hit bound without shrinking (newEnd = newStart + frozen width);
        // anchor-relative (not per-move) delta keeps width invariant under rounding.
        const total = pcm.length;
        if (total <= 0 || dragRectWidth <= 0) return;
        const width = dragAnchorEnd - dragAnchorStart;
        const deltaSamples = Math.round(((e.clientX - dragAnchorClientX) * total) / dragRectWidth);
        const maxStart = Math.max(0, total - width);
        const newStart = Math.max(0, Math.min(dragAnchorStart + deltaSamples, maxStart));
        onChange(newStart, newStart + width);
        return;
      }
      const sample = clientXToSample(e.clientX, pcm.length);
      if (dragging === 'start') {
        const newStart = Math.max(0, Math.min(sample, endSamples - minGapSamples));
        onChange(newStart, endSamples);
      } else if (dragging === 'end') {
        const newEnd = Math.min(pcm.length, Math.max(sample, startSamples + minGapSamples));
        onChange(startSamples, newEnd);
      } else if (onSeek) {
        onSeek(Math.max(0, Math.min(pcm.length, sample)));
      }
    };

    const onUp = (): void => {
      if (dragging === null) return;
      const wasDragging = dragging;
      // Clearing dragging only schedules teardown (next effect flush), so listeners stay attached
      // through onCommit while a re-entrant onMove/onUp no-ops; if onCommit recreates this component,
      // component-destroy tears down instead -- no leak either way.
      dragging = null;
      if (wasDragging === 'start' || wasDragging === 'end' || wasDragging === 'window') {
        onCommit(startSamples, endSamples);
      } else if (onSeekCommit && playbackSample !== null) {
        onSeekCommit(playbackSample);
      }
    };

    document.addEventListener('pointermove', onMove, { capture: true });
    document.addEventListener('pointerup', onUp, { capture: true });
    document.addEventListener('pointercancel', onUp, { capture: true });
    return () => {
      document.removeEventListener('pointermove', onMove, { capture: true });
      document.removeEventListener('pointerup', onUp, { capture: true });
      document.removeEventListener('pointercancel', onUp, { capture: true });
    };
  });

  // Keyboard nudge: Arrow = 1/100 of clip (Shift = 1/10), Home/End snap to edge; commits per
  // keystroke (no blur needed).
  function onHandleKey(handle: 'start' | 'end', e: KeyboardEvent): void {
    const step = e.shiftKey
      ? Math.max(1, Math.round(pcm.length / 10))
      : Math.max(1, Math.round(pcm.length / 100));
    let delta = 0;
    if (e.key === 'ArrowLeft') delta = -step;
    else if (e.key === 'ArrowRight') delta = step;
    else if (e.key === 'Home') {
      e.preventDefault();
      if (handle === 'start') {
        onChange(0, endSamples);
        onCommit(0, endSamples);
      } else {
        const newEnd = Math.max(startSamples + minGapSamples, minGapSamples);
        onChange(startSamples, newEnd);
        onCommit(startSamples, newEnd);
      }
      return;
    } else if (e.key === 'End') {
      e.preventDefault();
      if (handle === 'end') {
        onChange(startSamples, pcm.length);
        onCommit(startSamples, pcm.length);
      } else {
        const newStart = Math.min(endSamples - minGapSamples, pcm.length - minGapSamples);
        onChange(Math.max(0, newStart), endSamples);
        onCommit(Math.max(0, newStart), endSamples);
      }
      return;
    } else {
      return;
    }
    e.preventDefault();
    if (handle === 'start') {
      const newStart = Math.max(0, Math.min(startSamples + delta, endSamples - minGapSamples));
      onChange(newStart, endSamples);
      onCommit(newStart, endSamples);
    } else {
      const newEnd = Math.min(
        pcm.length,
        Math.max(endSamples + delta, startSamples + minGapSamples)
      );
      onChange(startSamples, newEnd);
      onCommit(startSamples, newEnd);
    }
  }

  // Keyboard window slide: Arrow = 1/100 of clip (Shift = 1/10), Home/End snap to edge; width
  // preserved via newEnd = newStart + width, never independent per-bound clamps.
  function onWindowKey(e: KeyboardEvent): void {
    const total = pcm.length;
    if (total <= 0) return;
    const width = endSamples - startSamples;
    const maxStart = Math.max(0, total - width);
    if (maxStart === 0) return; // window already fills the clip, nowhere to slide
    const step = e.shiftKey
      ? Math.max(1, Math.round(total / 10))
      : Math.max(1, Math.round(total / 100));
    let newStart: number | null = null;
    if (e.key === 'ArrowLeft') newStart = Math.max(0, startSamples - step);
    else if (e.key === 'ArrowRight') newStart = Math.min(maxStart, startSamples + step);
    else if (e.key === 'Home') newStart = 0;
    else if (e.key === 'End') newStart = maxStart;
    else return;
    e.preventDefault();
    if (newStart === startSamples) return; // pinned at boundary; skip commit
    onChange(newStart, newStart + width);
    onCommit(newStart, newStart + width);
  }

  const startSec = $derived((startSamples / WAV_SAMPLE_RATE).toFixed(2));
  const endSec = $derived((endSamples / WAV_SAMPLE_RATE).toFixed(2));
  // Highest valid window start (total - width) for aria-valuemax; shifts as the handles resize it.
  const windowMax = $derived(Math.max(0, pcm.length - (endSamples - startSamples)));
</script>

<div
  bind:this={wrapper}
  class="relative h-full w-full select-none"
  aria-label={m.category.trim_waveform.handles_aria}
>
  <StaticWaveform {pcm} {color} {background} />

  <!-- Mask over each unselected side; pointer-events-none so it never intercepts handle drags. -->
  <div
    class="pointer-events-none absolute inset-y-0 left-0 bg-canvas-mask"
    style:width="{startPct}%"
  ></div>
  <div
    class="pointer-events-none absolute inset-y-0 bg-canvas-mask"
    style:left="{endPct}%"
    style:right="0"
  ></div>

  <!-- Start handle: 24px hit area (wider than the visible shaft + grip pill) for reliable grab. -->
  <div
    role="slider"
    aria-label={m.category.trim_waveform.handle_start_aria}
    aria-valuemin={0}
    aria-valuemax={pcm.length}
    aria-valuenow={startSamples}
    aria-valuetext={m.category.trim_waveform.value_seconds(startSec)}
    tabindex="0"
    class="group absolute inset-y-0 z-20 flex w-6 -translate-x-1/2 cursor-ew-resize touch-none items-center justify-center"
    class:cursor-grabbing={dragging === 'start'}
    style:left="{startPct}%"
    onpointerdown={(e) => startDrag('start', e)}
    onkeydown={(e) => onHandleKey('start', e)}
  >
    <div
      class="pointer-events-none h-full w-0.75 rounded-full bg-accent shadow-card transition-colors group-hover:bg-accent-hover"
    ></div>
    <div
      class="pointer-events-none absolute top-1/2 left-1/2 h-6 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow ring-1 ring-canvas/80 transition-colors group-hover:bg-accent-hover"
    ></div>
  </div>

  <!-- Selection contour + window-slide hit target. z-auto under the z-20 handles and z-30 cursor so
       edge clicks route to handle/cursor and only the empty middle slides; DOM order
       start->window->end makes tab order match the left-to-right read; touch-none blocks
       horizontal-pan scroll/zoom; ring-inset keeps the focus ring off the handle edges. -->
  <div
    role="slider"
    aria-label={m.category.trim_waveform.selection_aria}
    aria-valuemin={0}
    aria-valuemax={windowMax}
    aria-valuenow={startSamples}
    aria-valuetext={m.category.trim_waveform.value_seconds_range(startSec, endSec)}
    tabindex="0"
    class="absolute inset-y-0 cursor-grab touch-none border-x-2 border-accent/70 focus:outline-none focus-visible:bg-accent/5 focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-inset"
    class:cursor-grabbing={dragging === 'window'}
    style:left="{startPct}%"
    style:width="{Math.max(0, endPct - startPct)}%"
    onpointerdown={(e) => startDrag('window', e)}
    onkeydown={onWindowKey}
  ></div>

  <!-- End handle, mirrored. -->
  <div
    role="slider"
    aria-label={m.category.trim_waveform.handle_end_aria}
    aria-valuemin={0}
    aria-valuemax={pcm.length}
    aria-valuenow={endSamples}
    aria-valuetext={m.category.trim_waveform.value_seconds(endSec)}
    tabindex="0"
    class="group absolute inset-y-0 z-20 flex w-6 -translate-x-1/2 cursor-ew-resize touch-none items-center justify-center"
    class:cursor-grabbing={dragging === 'end'}
    style:left="{endPct}%"
    onpointerdown={(e) => startDrag('end', e)}
    onkeydown={(e) => onHandleKey('end', e)}
  >
    <div
      class="pointer-events-none h-full w-0.75 rounded-full bg-accent shadow-card transition-colors group-hover:bg-accent-hover"
    ></div>
    <div
      class="pointer-events-none absolute top-1/2 left-1/2 h-6 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow ring-1 ring-canvas/80 transition-colors group-hover:bg-accent-hover"
    ></div>
  </div>

  <!-- Optional playback cursor. z-30 above the trim handles so it wins on overlap (e.g. playback
       starting at the trim start). -->
  {#if playbackPct !== null}
    <div
      role="slider"
      aria-label={m.category.trim_waveform.playback_position_aria}
      aria-valuemin={0}
      aria-valuemax={pcm.length}
      aria-valuenow={playbackSample ?? 0}
      tabindex="-1"
      class="group absolute inset-y-0 z-30 flex w-6 -translate-x-1/2 touch-none items-start justify-center"
      class:cursor-ew-resize={!!onSeek}
      class:cursor-grabbing={dragging === 'playback'}
      style:left="{playbackPct}%"
      onpointerdown={onSeek ? (e) => startDrag('playback', e) : undefined}
    >
      <div class="pointer-events-none h-full w-0.5 rounded-full bg-danger-dot/90 shadow-sm"></div>
      <div
        class="pointer-events-none absolute top-0 left-1/2 h-3 w-2.5 -translate-x-1/2 rounded-b-md bg-danger-dot shadow"
      ></div>
    </div>
  {/if}
</div>
