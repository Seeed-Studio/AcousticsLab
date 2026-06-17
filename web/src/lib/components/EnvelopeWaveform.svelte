<script lang="ts">
  import { onMount } from 'svelte';
  import type { PcmSource } from '$lib/audio/pcm-source';
  import {
    nextVisualRenderAt,
    visualDevicePixelRatio
  } from '$lib/components/dashboard/visualRuntime';
  import { hexToRgba } from '$lib/utils/color';
  import { theme } from '$lib/stores/theme.svelte';

  // Min/max envelope waveform; all canvas math consumes the source-owned cursor so any PcmSource works without a math copy.
  interface Props {
    source: PcmSource;
    seconds?: number;
    /** Contour stroke + fill; defaults to --color-accent. */
    color?: string;
    /** Canvas background; defaults to --color-canvas so silence dissolves into chrome. */
    background?: string;
  }
  let { source, seconds = 3, color, background }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state();

  // Theme-flip counter the RAF polls via closure to re-read palette vars. Must be plain `let`, not `$state`: a read-then-write of $state in this effect would subscribe to the signal its own write invalidates -> Svelte 5 effect_update_depth_exceeded freezing the waveform.
  let themeSeq = 0;
  $effect(() => {
    void theme.resolved;
    themeSeq += 1;
  });

  onMount(() => {
    const el = canvas;
    if (!el) return;
    const ctx = el.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) return;

    let hiBuf: Float32Array = new Float32Array(0);
    let loBuf: Float32Array = new Float32Array(0);

    // RAF-coalesced resize: ResizeObserver records dims, the draw frame applies them, so a window-edge drag never flashes blank pixels.
    let pendingW = 1;
    let pendingH = 1;
    let needsResize = true;

    const updatePendingSize = (): void => {
      const dpr = visualDevicePixelRatio();
      const r = el.getBoundingClientRect();
      pendingW = Math.max(1, Math.floor(r.width * dpr));
      pendingH = Math.max(1, Math.floor(r.height * dpr));
      needsResize = true;
    };

    updatePendingSize();
    const ro = new ResizeObserver(updatePendingSize);
    ro.observe(el);
    window.addEventListener('resize', updatePendingSize, { passive: true });

    const readVar = (name: string): string =>
      getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    let activeBackground = background ?? readVar('--color-canvas');
    let activeStroke = color ?? readVar('--color-accent');
    let gridStroke = readVar('--color-line');
    let fillRgba = hexToRgba(activeStroke, 0.15);
    // Lagging mirror of themeSeq; RAF re-resolves palette on divergence. Seeded equal so the first frame skips the just-done mount read.
    let appliedThemeSeq = themeSeq;

    let raf: number | null = null;
    let lastRenderAt = Number.NEGATIVE_INFINITY;

    const draw = (now: DOMHighResTimeStamp): void => {
      const renderAt = nextVisualRenderAt(now, lastRenderAt);
      if (renderAt === null) {
        raf = requestAnimationFrame(draw);
        return;
      }
      lastRenderAt = renderAt;

      // Re-resolve all four palette vars together on flip (no half-mode frame); steady state pays only an int compare.
      if (themeSeq !== appliedThemeSeq) {
        activeBackground = background ?? readVar('--color-canvas');
        activeStroke = color ?? readVar('--color-accent');
        gridStroke = readVar('--color-line');
        fillRgba = hexToRgba(activeStroke, 0.15);
        appliedThemeSeq = themeSeq;
      }

      if (needsResize) {
        if (el.width !== pendingW) el.width = pendingW;
        if (el.height !== pendingH) el.height = pendingH;
        // Length w+2: phantom slots w,w+1 extend the contour past the right edge so the leftward sub-pixel translate leaves no gap (envelopeAt fills [0..w-1]).
        if (hiBuf.length !== pendingW + 2) {
          hiBuf = new Float32Array(pendingW + 2);
          loBuf = new Float32Array(pendingW + 2);
        }
        needsResize = false;
      }
      const w = el.width;
      const h = el.height;
      const mid = h / 2;
      const amp = mid * 0.92;

      ctx.fillStyle = activeBackground;
      ctx.fillRect(0, 0, w, h);

      // Baseline painted un-translated so it stays anchored, not scrolling.
      ctx.strokeStyle = gridStroke;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, mid + 0.5);
      ctx.lineTo(w, mid + 0.5);
      ctx.stroke();

      const sampleRate = source.sampleRate;
      if (sampleRate <= 0) {
        // Idle source: baseline-only paint, skip envelope read so no stale data shows.
        raf = requestAnimationFrame(draw);
        return;
      }
      const endSample = source.renderCursor(now);
      if (endSample <= 0) {
        raf = requestAnimationFrame(draw);
        return;
      }

      // Anti-shimmer: snapping endSample down to a samplesPerBin multiple keeps each bin's contents fixed in absolute time (else raw endSample jitters boundaries ~80 samples/frame at 48kHz/60Hz, flickering dense audio); the remainder rides a sub-pixel translate for continuous scroll, lag bounded by samplesPerBin-1 (~7.5ms, below perception).
      const samplesPerBin = Math.max(1, Math.round((sampleRate * seconds) / w));
      const snappedEnd = Math.floor(endSample / samplesPerBin) * samplesPerBin;
      const subPxOffset = (endSample - snappedEnd) / samplesPerBin;

      source.envelopeAt(snappedEnd, w * samplesPerBin, w, loBuf, hiBuf);

      loBuf[w] = loBuf[w - 1];
      hiBuf[w] = hiBuf[w - 1];
      loBuf[w + 1] = loBuf[w - 1];
      hiBuf[w + 1] = hiBuf[w - 1];

      ctx.save();
      ctx.translate(-subPxOffset, 0);

      ctx.fillStyle = fillRgba;
      ctx.beginPath();
      ctx.moveTo(0, mid - hiBuf[0] * amp);
      for (let x = 1; x <= w + 1; x++) ctx.lineTo(x, mid - hiBuf[x] * amp);
      for (let x = w + 1; x >= 0; x--) ctx.lineTo(x, mid - loBuf[x] * amp);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = activeStroke;
      ctx.lineWidth = 1.25;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(0, mid - hiBuf[0] * amp);
      for (let x = 1; x <= w + 1; x++) ctx.lineTo(x, mid - hiBuf[x] * amp);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, mid - loBuf[0] * amp);
      for (let x = 1; x <= w + 1; x++) ctx.lineTo(x, mid - loBuf[x] * amp);
      ctx.stroke();

      ctx.restore();

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);

    return () => {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
      ro.disconnect();
      window.removeEventListener('resize', updatePendingSize);
    };
  });
</script>

<canvas bind:this={canvas} class="block h-full w-full rounded-md"></canvas>
