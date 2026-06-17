<script lang="ts">
  import { onMount } from 'svelte';
  import { streams } from '$lib/stores/streams.svelte';
  import { fftRadix2, hannWindow } from '$lib/audio/fft';
  import {
    buildSpectrogramLut,
    magnitudeToPaletteIndex,
    SPECTROGRAM_DB_CEILING,
    SPECTROGRAM_DB_FLOOR
  } from '$lib/audio/palette';
  import { theme } from '$lib/stores/theme.svelte';
  import { nextVisualRenderAt, visualDevicePixelRatio } from './visualRuntime';

  interface Props {
    seconds?: number;
    minDb?: number;
    maxDb?: number;
    maxHz?: number;
    fftSize?: number;
    smoothing?: number;
  }
  // dB floor/ceiling default to shared palette constants so all spectrogram surfaces share one dB->colour map; overriding breaks that invariant.
  let {
    seconds = 3,
    minDb = SPECTROGRAM_DB_FLOOR,
    maxDb = SPECTROGRAM_DB_CEILING,
    maxHz = 12_000,
    fftSize = 512,
    smoothing = 0.6
  }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state();

  // Theme-flip signal polled once per RAF frame (LUT is baked into raster pixels + emptyColor, so a mode switch must rebuild both and discard history). MUST be plain `let`, not `$state`: under `$state` the in-effect read-then-write `+= 1` subscribes to the signal it invalidates -> self-feeding effect aborts with effect_update_depth_exceeded, freezing the canvas. Plain `let` leaves theme.resolved as the effect's only dependency.
  let themeRebuildSeq = 0;
  $effect(() => {
    void theme.resolved;
    themeRebuildSeq += 1;
  });

  // JS FFT on the PCM ring buffer, no AudioContext, so the autoplay gesture gate doesn't apply and the spectrogram is alive on cold load. Per-frame catch-up is bounded so slow frames never replay an unbounded column backlog.
  const COLUMN_RATE_HZ = 60;
  const MAX_COLUMNS_PER_FRAME = 4;

  onMount(() => {
    const el = canvas;
    if (!el) return;
    const ctx = el.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'low';

    // fftRadix2 requires power-of-two fftSize; bail rather than let it throw.
    if (!Number.isInteger(Math.log2(fftSize))) return;

    const sampleRate = streams.sampleRate;
    // Real FFT yields N/2 unique bins; cap to maxHz to track the speech band, not the full capture Nyquist range.
    const halfBins = fftSize >> 1;
    const useBins = Math.min(halfBins, Math.ceil((maxHz / (sampleRate / 2)) * halfBins));
    const historyColumns = Math.max(1, Math.ceil(seconds * COLUMN_RATE_HZ));
    const columnStepSamples = Math.max(1, Math.round(sampleRate / COLUMN_RATE_HZ));

    // Working buffers pre-allocated once; never reallocated inside the loop.
    const hann = hannWindow(fftSize);
    const pcmBuf = new Float32Array(fftSize);
    const real = new Float32Array(fftSize);
    const imag = new Float32Array(fftSize);
    // Smooth magnitudes, not dB, to match AnalyserNode's perceptual feel.
    const smoothedMag = new Float32Array(useBins);
    let writeCol = 0;
    let lastColumnEndSample = 0;
    let initializedColumns = false;

    // Hann coherent gain 0.5 => peak magnitude of a unity sinusoid is N/4; scale by 1/(N/2) to land near 0 dB for unity input.
    const magScale = 1 / halfBins;

    let pixelW = 1;
    let pixelH = 1;
    // Shared 256-entry cross-surface colour LUT; Clamped array lets the inner loop drop per-channel min/max guards; rebuilt per theme flip.
    const PALETTE_N = 256;
    let palette = buildSpectrogramLut(PALETTE_N, theme.resolved);
    let emptyColor = `rgb(${palette[0]}, ${palette[1]}, ${palette[2]})`;
    // Seed from the value the mount-time $effect already reached so the first RAF tick treats the LUT as fresh and skips a redundant rebuild.
    let appliedThemeSeq = themeRebuildSeq;

    // Native raster stage: one column = one source pixel, so RAF only scales/scrolls this tiny image via drawImage instead of rebuilding a DPR-sized ImageData each frame. The union type resolves getContext('2d') to the generic overload, hence the casts.
    const hasOffscreenCanvas = typeof OffscreenCanvas === 'function';
    const raster: OffscreenCanvas | HTMLCanvasElement = hasOffscreenCanvas
      ? new OffscreenCanvas(historyColumns, useBins)
      : document.createElement('canvas');
    raster.width = historyColumns;
    raster.height = useBins;
    // Narrow via the typeof flag, not `raster instanceof OffscreenCanvas`: the instanceof RHS is evaluated even on the fallback branch and ReferenceErrors where the identifier is absent (Safari < 16.4).
    const rasterCtx = hasOffscreenCanvas
      ? (raster as OffscreenCanvas).getContext('2d', { alpha: false })
      : (raster as HTMLCanvasElement).getContext('2d', { alpha: false });
    if (!rasterCtx) return;
    rasterCtx.imageSmoothingEnabled = false;
    const columnImage = rasterCtx.createImageData(1, useBins);
    const columnPixels = columnImage.data;

    const clearRaster = (): void => {
      rasterCtx.fillStyle = emptyColor;
      rasterCtx.fillRect(0, 0, historyColumns, useBins);
    };
    clearRaster();

    // ResizeObserver only records dimensions; the canvas reset (which wipes the pixel buffer) is deferred to the next RAF so reset and render share one frame, avoiding a blank flash while dragging the edge.
    let pendingW = 1;
    let pendingH = 1;
    let needsResize = true;

    const applyResizeIfNeeded = (): void => {
      if (!needsResize) return;
      if (el.width !== pendingW) el.width = pendingW;
      if (el.height !== pendingH) el.height = pendingH;
      pixelW = pendingW;
      pixelH = pendingH;
      needsResize = false;
    };

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

    const ingestAt = (endSample: number): void => {
      const pcm = streams.snapshotAt(endSample, fftSize, pcmBuf);
      for (let i = 0; i < fftSize; i++) {
        real[i] = pcm[i] * hann[i];
        imag[i] = 0;
      }
      fftRadix2(real, imag);
      for (let k = 0; k < useBins; k++) {
        const mag = Math.hypot(real[k], imag[k]) * magScale;
        const sm = smoothing * smoothedMag[k] + (1 - smoothing) * mag;
        smoothedMag[k] = sm;
        const p = magnitudeToPaletteIndex(sm, PALETTE_N, minDb, maxDb);
        const src = p * 3;
        const y = useBins - 1 - k;
        const dst = y * 4;
        columnPixels[dst] = palette[src];
        columnPixels[dst + 1] = palette[src + 1];
        columnPixels[dst + 2] = palette[src + 2];
        columnPixels[dst + 3] = 255;
      }
      rasterCtx.putImageData(columnImage, writeCol, 0);
      writeCol = (writeCol + 1) % historyColumns;
    };

    const ingestUntil = (endSample: number): void => {
      if (endSample < fftSize) return;
      if (!initializedColumns) {
        lastColumnEndSample = endSample - columnStepSamples;
        initializedColumns = true;
      }

      // After a throttle or reconnect, restart from a recent edge instead of replaying thousands of stale columns, staying locked to the waveform cursor.
      const backlogColumns = Math.floor((endSample - lastColumnEndSample) / columnStepSamples);
      if (backlogColumns > historyColumns) {
        clearRaster();
        // Discontinuity: drop stale color history and EMA state together.
        smoothedMag.fill(0);
        writeCol = 0;
        lastColumnEndSample = endSample - columnStepSamples;
      } else if (backlogColumns > MAX_COLUMNS_PER_FRAME) {
        lastColumnEndSample = endSample - MAX_COLUMNS_PER_FRAME * columnStepSamples;
      }

      let guard = 0;
      while (
        lastColumnEndSample + columnStepSamples <= endSample &&
        guard < MAX_COLUMNS_PER_FRAME
      ) {
        lastColumnEndSample += columnStepSamples;
        ingestAt(lastColumnEndSample);
        guard++;
      }
    };

    const render = (endSample: number): void => {
      ctx.fillStyle = emptyColor;
      ctx.fillRect(0, 0, pixelW, pixelH);
      if (!initializedColumns) return;

      const columnW = pixelW / historyColumns;
      const phase = Math.max(0, Math.min(1, (endSample - lastColumnEndSample) / columnStepSamples));
      let dx = -phase * columnW;
      const firstCount = historyColumns - writeCol;
      const rasterSource = raster as CanvasImageSource;

      if (firstCount > 0) {
        const dw = firstCount * columnW;
        ctx.drawImage(rasterSource, writeCol, 0, firstCount, useBins, dx, 0, dw, pixelH);
        dx += dw;
      }
      if (writeCol > 0) {
        const dw = writeCol * columnW;
        ctx.drawImage(rasterSource, 0, 0, writeCol, useBins, dx, 0, dw, pixelH);
      }

      // Fill the sub-column gap with the latest column so it scrolls smoothly between FFT hops without flashing a blank strip on the right.
      const gap = phase * columnW;
      if (gap > 0.01) {
        const latestCol = (writeCol + historyColumns - 1) % historyColumns;
        ctx.drawImage(rasterSource, latestCol, 0, 1, useBins, pixelW - gap, 0, gap, pixelH);
      }
    };

    let raf: number | null = null;
    let lastRenderAt = Number.NEGATIVE_INFINITY;
    const tick = (now: DOMHighResTimeStamp): void => {
      const renderAt = nextVisualRenderAt(now, lastRenderAt);
      if (renderAt === null) {
        raf = requestAnimationFrame(tick);
        return;
      }
      lastRenderAt = renderAt;

      // Theme-flip handoff: clear raster + EMA (old columns hold the previous palette and would scroll a visible mode seam, cost: lost history) and reset initializedColumns to snap the ingest cursor to the live edge.
      if (themeRebuildSeq !== appliedThemeSeq) {
        palette = buildSpectrogramLut(PALETTE_N, theme.resolved);
        emptyColor = `rgb(${palette[0]}, ${palette[1]}, ${palette[2]})`;
        clearRaster();
        smoothedMag.fill(0);
        writeCol = 0;
        initializedColumns = false;
        appliedThemeSeq = themeRebuildSeq;
      }

      applyResizeIfNeeded();
      const endSample = streams.renderCursor(now);
      ingestUntil(endSample);
      render(endSample);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      if (raf !== null) cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', updatePendingSize);
    };
  });
</script>

<!-- Palette's first stop is pinned per mode to the --color-canvas token used here, so silence dissolves into the canvas and continues into the waveform panel above as one unified field. -->
<canvas bind:this={canvas} class="block h-full w-full rounded-md bg-canvas"></canvas>
