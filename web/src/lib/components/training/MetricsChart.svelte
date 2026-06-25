<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { EpochMetrics } from '$lib/api/types';
  import { theme } from '$lib/stores/theme.svelte';
  import { visualDevicePixelRatio } from '$lib/components/dashboard/visualRuntime';
  import { locale } from '$lib/stores/locale.svelte';
  import { hexToRgba } from '$lib/utils/color';
  import { m } from '$lib/i18n';

  // render() reads theme/locale only inside a non-reactive RAF, so track both here to repaint an
  // idle chart on mode/language flip.
  $effect(() => {
    void theme.resolved;
    void locale.resolved;
    scheduleRender();
  });

  // Resolved once per render() so a frame is self-consistent across a mid-frame theme flip. An
  // unloaded CSS var reads '' -> raw-string entries no-op fill/strokeStyle (keep prior value),
  // while the hexToRgba-derived ones (tooltip/legend bg, crosshair, val-marker stem) fall back to
  // grey; either way the next RAF corrects the at-worst one off-theme frame.
  interface ChartPalette {
    axisLine: string;
    gridSubtle: string;
    axisLabel: string;
    tooltipBg: string;
    tooltipBorder: string;
    tooltipLabel: string;
    tooltipValue: string;
    legendBg: string;
    legendBorder: string;
    legendLabel: string;
    crosshair: string;
    loss: string;
    val: string;
    train: string;
    valMarkerStem: string;
  }
  function resolvePalette(): ChartPalette {
    const cs = getComputedStyle(document.documentElement);
    const read = (name: string): string => cs.getPropertyValue(name).trim();
    const fg = read('--color-fg');
    const fgSecondary = read('--color-fg-secondary');
    const fgMuted = read('--color-fg-muted');
    const fgSubtle = read('--color-fg-subtle');
    const line = read('--color-line');
    const lineSubtle = read('--color-line-subtle');
    const elevated = read('--color-elevated');
    const loss = read('--color-danger-dot');
    const val = read('--color-success-dot');
    const train = read('--color-accent');
    return {
      axisLine: line,
      gridSubtle: lineSubtle,
      axisLabel: fgSubtle,
      // High-alpha so chart data stays faintly readable through tooltip/legend backings.
      tooltipBg: hexToRgba(elevated, 0.96),
      tooltipBorder: line,
      tooltipLabel: fgMuted,
      tooltipValue: fg,
      legendBg: hexToRgba(elevated, 0.92),
      legendBorder: line,
      legendLabel: fgSecondary,
      // Low alpha so data lines never read as washed out behind the crosshair.
      crosshair: hexToRgba(fgSecondary, 0.35),
      loss,
      val,
      train,
      valMarkerStem: hexToRgba(val, 0.4)
    };
  }

  // Hand-rolled canvas (no chart lib) for a tiny fixed-shape series (X=epoch, loss [0,unbounded],
  // acc [0,1]). No axis tick labels (hover tooltip + JobProgress readout carry the numbers); only
  // acc gridlines at 0.5/1.0, since loss's ceiling is per-run and an unlabelled loss tick would
  // imply an unreadable value.

  interface Props {
    epochs: readonly EpochMetrics[];
    // Set when validation_split === 0 (val_acc all-NaN): drops val line/chip/marker rather than
    // threading NaN through the line-renderer.
    valDisabled?: boolean;
    height?: number;
  }
  let { epochs, valDisabled = false, height = 112 }: Props = $props();

  let canvasEl = $state<HTMLCanvasElement | undefined>();
  let wrapperEl = $state<HTMLDivElement | undefined>();
  let cssW = $state(0);
  // Flagged on any size/prop change (height/epochs/valDisabled effect, onMount, ResizeObserver),
  // acted on inside the RAF, so a burst coalesces into one backing-buffer realloc per paint.
  let needsResize = false;
  let rafId: number | null = null;

  // epochs[] index nearest the pointer (null off-plot); routed through the RAF coalescer so the
  // crosshair repaints at refresh rate, not event rate.
  let hoveredIdx = $state<number | null>(null);

  function scheduleRender(): void {
    if (rafId !== null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      render();
    });
  }

  // Plot gutters keeping data/baseline off the canvas edge; at 0 the line-stroke caps clip against
  // bg-canvas's rounded corners. Legend pill lives inside the plot, so PAD_T reserves no strip.
  const PAD_L = 6;
  const PAD_R = 6;
  const PAD_T = 6;
  const PAD_B = 6;

  function findBestValIdx(): number {
    let bestIdx = -1;
    let bestVal = -Infinity;
    for (let i = 0; i < epochs.length; i++) {
      const v = epochs[i].val_acc;
      if (v === null || !Number.isFinite(v)) continue;
      if (v > bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  // Last paint's x-mapping + epochs snapshot, so the pointer handler maps X back to the nearest
  // epoch without re-deriving the scale.
  let lastXToPx: ((x: number) => number) | null = null;
  let lastEpochs: readonly EpochMetrics[] = [];

  function render(): void {
    const cnv = canvasEl;
    if (!cnv) return;
    const dpr = visualDevicePixelRatio();
    const cssWidth = cssW || cnv.clientWidth;
    if (cssWidth <= 0) return;
    if (needsResize) {
      const targetW = Math.max(1, Math.round(cssWidth * dpr));
      const targetH = Math.max(1, Math.round(height * dpr));
      if (cnv.width !== targetW) cnv.width = targetW;
      if (cnv.height !== targetH) cnv.height = targetH;
      needsResize = false;
    }
    const ctx = cnv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cnv.width / dpr, cnv.height / dpr);

    const palette = resolvePalette();

    const w = cssWidth;
    const h = height;
    const plotW = Math.max(1, w - PAD_L - PAD_R);
    const plotH = Math.max(1, h - PAD_T - PAD_B);

    // Baseline always painted so the chart frame shows before the first epoch lands.
    ctx.strokeStyle = palette.axisLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD_L, PAD_T + plotH + 0.5);
    ctx.lineTo(PAD_L + plotW, PAD_T + plotH + 0.5);
    ctx.stroke();

    if (epochs.length === 0) {
      // Legend painted even on the waiting chart so the colour mapping is visible.
      drawLegendPill(ctx, PAD_L + plotW, PAD_T, palette);
      ctx.fillStyle = palette.axisLabel;
      ctx.font = '500 11px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
      ctx.fillText(m.training.chart.waiting_first_epoch, PAD_L + plotW / 2, PAD_T + plotH / 2);
      lastXToPx = null;
      lastEpochs = [];
      return;
    }

    // X domain 1..maxEpoch, preferring the run's declared total so the axis spans the whole run.
    const last = epochs[epochs.length - 1];
    const xMax = Math.max(1, last.epochs || epochs.length);
    const xMin = 1;
    let maxLoss = 0;
    for (const e of epochs) {
      if (Number.isFinite(e.train_loss) && e.train_loss > maxLoss) maxLoss = e.train_loss;
    }
    if (maxLoss === 0) maxLoss = 1; // all-zero edge case keeps a sane axis
    const lossCeil = maxLoss * 1.05; // headroom so a max-value point doesn't kiss the plot top

    const xToPx = (x: number): number => PAD_L + ((x - xMin) / (xMax - xMin || 1)) * plotW;
    const lossToPx = (l: number): number => PAD_T + plotH - (l / lossCeil) * plotH;
    const accToPx = (a: number): number => PAD_T + plotH - a * plotH;

    lastXToPx = xToPx;
    lastEpochs = epochs;

    // Acc gridlines at 0.5/1.0 keep val_acc's position relative to "half" legible without labels.
    ctx.strokeStyle = palette.gridSubtle;
    ctx.lineWidth = 1;
    for (const a of [0.5, 1.0]) {
      const y = accToPx(a);
      ctx.beginPath();
      ctx.moveTo(PAD_L, y + 0.5);
      ctx.lineTo(PAD_L + plotW, y + 0.5);
      ctx.stroke();
    }

    // Best-val marker drawn before the data lines so they paint over it; skipped at <=1 val point.
    const bestIdx = valDisabled ? -1 : findBestValIdx();
    const bestEntry = bestIdx >= 0 ? epochs[bestIdx] : null;
    const bestVal = bestEntry?.val_acc ?? null;
    if (bestEntry !== null && bestVal !== null && countValPoints() > 1) {
      const x = xToPx(bestEntry.epoch);
      const y = accToPx(Math.max(0, Math.min(1, bestVal)));
      ctx.strokeStyle = palette.valMarkerStem;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, PAD_T);
      ctx.lineTo(x + 0.5, PAD_T + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = palette.val;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Loss line, drawn before the acc lines so they land on top where they overlap.
    ctx.strokeStyle = palette.loss;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let lossStarted = false;
    for (const e of epochs) {
      // Drop non-finite loss explicitly; don't rely on NaN-lineTo being a canvas no-op.
      if (!Number.isFinite(e.train_loss)) continue;
      const x = xToPx(e.epoch);
      const y = lossToPx(Math.max(0, e.train_loss));
      if (!lossStarted) {
        ctx.moveTo(x, y);
        lossStarted = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    if (!valDisabled) {
      ctx.strokeStyle = palette.val;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (const e of epochs) {
        if (e.val_acc === null || !Number.isFinite(e.val_acc)) continue;
        const x = xToPx(e.epoch);
        const y = accToPx(Math.max(0, Math.min(1, e.val_acc)));
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Train-acc line (dashed): a "without holdout" reference next to val_acc.
    ctx.strokeStyle = palette.train;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    let tStarted = false;
    for (const e of epochs) {
      if (!Number.isFinite(e.train_acc)) continue;
      const x = xToPx(e.epoch);
      const y = accToPx(Math.max(0, Math.min(1, e.train_acc)));
      if (!tStarted) {
        ctx.moveTo(x, y);
        tStarted = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Single-point fallback: stroke() with one moveTo and no lineTo draws nothing, so paint a dot
    // per series at n=1.
    if (epochs.length === 1) {
      const e = epochs[0];
      const x = xToPx(e.epoch);
      if (Number.isFinite(e.train_loss)) {
        ctx.fillStyle = palette.loss;
        ctx.beginPath();
        ctx.arc(x, lossToPx(Math.max(0, e.train_loss)), 2, 0, Math.PI * 2);
        ctx.fill();
      }
      if (Number.isFinite(e.train_acc)) {
        // Hollow ring, not a filled dot: train (accent) and val (success) are both greens in dark mode, so
        // where they converge the ring keeps the train marker distinct (mirrors the dashed train line).
        ctx.strokeStyle = palette.train;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(x, accToPx(Math.max(0, Math.min(1, e.train_acc))), 2, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (!valDisabled && e.val_acc !== null && Number.isFinite(e.val_acc)) {
        ctx.fillStyle = palette.val;
        ctx.beginPath();
        ctx.arc(x, accToPx(Math.max(0, Math.min(1, e.val_acc))), 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Legend after the data (backplate occludes crossing lines), before the hover overlay (tooltip
    // sits over the pill near the right edge).
    drawLegendPill(ctx, PAD_L + plotW, PAD_T, palette);

    // Hover overlay drawn last; range guard handles epochs[] truncated since the pointermove fired.
    if (hoveredIdx !== null && hoveredIdx >= 0 && hoveredIdx < epochs.length) {
      const e = epochs[hoveredIdx];
      const x = xToPx(e.epoch);
      ctx.strokeStyle = palette.crosshair;
      ctx.lineWidth = 1;
      ctx.setLineDash([1, 2]);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, PAD_T);
      ctx.lineTo(x + 0.5, PAD_T + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      if (Number.isFinite(e.train_loss)) {
        ctx.fillStyle = palette.loss;
        ctx.beginPath();
        ctx.arc(x, lossToPx(Math.max(0, e.train_loss)), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (Number.isFinite(e.train_acc)) {
        // Hollow ring (see n=1 fallback): keeps the train dot distinct from the val dot where they converge.
        ctx.strokeStyle = palette.train;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(x, accToPx(Math.max(0, Math.min(1, e.train_acc))), 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (!valDisabled && e.val_acc !== null && Number.isFinite(e.val_acc)) {
        ctx.fillStyle = palette.val;
        ctx.beginPath();
        ctx.arc(x, accToPx(Math.max(0, Math.min(1, e.val_acc))), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      // Tooltip: fixed width gives a predictable flip threshold; height tracks line count.
      const lines: { label: string; value: string }[] = [
        { label: m.training.chart.tooltip_epoch, value: `${e.epoch}` },
        { label: m.training.chart.tooltip_loss, value: fmtLoss(e.train_loss) }
      ];
      if (Number.isFinite(e.train_acc)) {
        lines.push({
          label: m.training.chart.tooltip_train,
          value: fmtAcc(e.train_acc)
        });
      }
      if (!valDisabled) {
        lines.push({
          label: m.training.chart.tooltip_val,
          value: fmtAcc(e.val_acc)
        });
      }
      const ttW = 124;
      const lineH = 12;
      const ttH = 6 + lines.length * lineH;
      // Right of the crosshair, flipping left when the box would clip the plot edge.
      let ttX = x + 8;
      if (ttX + ttW > PAD_L + plotW) ttX = x - 8 - ttW;
      const ttY = PAD_T + 2;
      ctx.fillStyle = palette.tooltipBg;
      ctx.strokeStyle = palette.tooltipBorder;
      ctx.lineWidth = 1;
      roundRect(ctx, ttX, ttY, ttW, ttH, 4);
      ctx.fill();
      ctx.stroke();
      // Label left-aligned, value right-aligned, so the value column aligns across rows.
      ctx.textBaseline = 'top';
      for (let i = 0; i < lines.length; i++) {
        const ln = lines[i];
        const y = ttY + 4 + i * lineH;
        ctx.fillStyle = palette.tooltipLabel;
        ctx.textAlign = 'left';
        ctx.font = '500 9px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
        ctx.fillText(ln.label, ttX + 6, y);
        ctx.fillStyle = palette.tooltipValue;
        ctx.textAlign = 'right';
        ctx.font =
          '500 10px ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace';
        ctx.fillText(ln.value, ttX + ttW - 6, y);
      }
    }
  }

  function countValPoints(): number {
    let n = 0;
    for (const e of epochs) {
      if (e.val_acc !== null && Number.isFinite(e.val_acc)) n++;
    }
    return n;
  }

  // Frosted legend pill top-right inside the plot (no vertical row): viable despite both acc lines
  // ending there because the backplate alpha ghosts them and the tooltip flips left at that edge.
  function drawLegendPill(
    ctx: CanvasRenderingContext2D,
    plotRight: number,
    plotTop: number,
    palette: ChartPalette
  ): void {
    interface Item {
      // Same palette source as the data-line stroke, so a swatch can't drift from its series.
      color: string;
      dashed: boolean;
      label: string;
    }
    const items: Item[] = [
      { color: palette.loss, dashed: false, label: m.training.chart.legend_loss },
      { color: palette.train, dashed: true, label: m.training.chart.legend_train }
    ];
    if (!valDisabled) {
      items.push({ color: palette.val, dashed: false, label: m.training.chart.legend_val });
    }

    const SWATCH_W = 6;
    const SWATCH_GAP = 3;
    const ITEM_GAP = 7;
    const INNER_PAD_X = 5;
    const PILL_H = 13;
    const EDGE_INSET = 4;

    ctx.font = '500 9px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.textBaseline = 'middle';

    let contentW = 0;
    for (let i = 0; i < items.length; i++) {
      if (i > 0) contentW += ITEM_GAP;
      contentW += SWATCH_W + SWATCH_GAP + ctx.measureText(items[i].label).width;
    }
    const pillW = contentW + INNER_PAD_X * 2;
    const pillX = plotRight - pillW - EDGE_INSET;
    const pillY = plotTop + EDGE_INSET;

    ctx.fillStyle = palette.legendBg;
    ctx.strokeStyle = palette.legendBorder;
    ctx.lineWidth = 1;
    roundRect(ctx, pillX, pillY, pillW, PILL_H, 3);
    ctx.fill();
    ctx.stroke();

    const yMid = pillY + PILL_H / 2;
    let x = pillX + INNER_PAD_X;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (i > 0) x += ITEM_GAP;
      // Swatch mirrors the data line: dashed rule for the dashed series, filled rect otherwise.
      if (item.dashed) {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(x, yMid);
        ctx.lineTo(x + SWATCH_W, yMid);
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.fillStyle = item.color;
        ctx.fillRect(x, yMid - 1.5, SWATCH_W, 3);
      }
      x += SWATCH_W + SWATCH_GAP;
      ctx.fillStyle = palette.legendLabel;
      ctx.textAlign = 'left';
      ctx.fillText(item.label, x, yMid);
      x += ctx.measureText(item.label).width;
    }
  }

  // Adaptive precision keeps the right-aligned value column <=6 chars across any loss magnitude
  // (a fixed toFixed(3) ran to 8 chars and left the column ragged).
  function fmtLoss(v: number): string {
    if (!Number.isFinite(v)) return m.training.progress.em_dash;
    if (v === 0) return '0';
    const abs = Math.abs(v);
    if (abs < 0.001) return v.toExponential(1);
    if (abs < 0.01) return v.toFixed(4);
    if (abs < 1) return v.toFixed(3);
    if (abs < 10) return v.toFixed(2);
    if (abs < 100) return v.toFixed(1);
    return Math.round(v).toString();
  }

  function fmtAcc(v: number | null): string {
    if (v === null || !Number.isFinite(v)) return m.training.progress.em_dash;
    return `${(v * 100).toFixed(1)}%`;
  }

  function roundRect(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    r: number
  ): void {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  // On the wrapper, not the canvas, so the hit region matches the visible box; padding gutters are
  // excluded below so the crosshair never snaps from within breathing room.
  function onPointerMove(e: PointerEvent): void {
    if (lastXToPx === null || lastEpochs.length === 0) {
      if (hoveredIdx !== null) {
        hoveredIdx = null;
        scheduleRender();
      }
      return;
    }
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const px = e.clientX - rect.left;
    const plotW = Math.max(1, (cssW || rect.width) - PAD_L - PAD_R);
    if (px < PAD_L || px > PAD_L + plotW) {
      if (hoveredIdx !== null) {
        hoveredIdx = null;
        scheduleRender();
      }
      return;
    }
    // Nearest epoch by absolute X distance (linear scan; n is tiny).
    let bestIdx = 0;
    let bestDx = Infinity;
    for (let i = 0; i < lastEpochs.length; i++) {
      const ex = lastXToPx(lastEpochs[i].epoch);
      const dx = Math.abs(ex - px);
      if (dx < bestDx) {
        bestDx = dx;
        bestIdx = i;
      }
    }
    if (bestIdx !== hoveredIdx) {
      hoveredIdx = bestIdx;
      scheduleRender();
    }
  }

  function onPointerLeave(): void {
    if (hoveredIdx !== null) {
      hoveredIdx = null;
      scheduleRender();
    }
  }

  // Repaint on epochs/valDisabled change. Tracking height + flagging resize matters because the
  // ResizeObserver fires only on WIDTH, so a height change would otherwise render into a
  // stale-height (CSS-stretched) buffer until some width resize happened to fire.
  $effect(() => {
    void epochs;
    void valDisabled;
    void height;
    needsResize = true;
    scheduleRender();
  });

  let resizeObs: ResizeObserver | null = null;

  onMount(() => {
    if (!wrapperEl) return;
    cssW = wrapperEl.clientWidth;
    needsResize = true;
    scheduleRender();
    resizeObs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const next = Math.floor(entry.contentRect.width);
        if (next !== cssW) {
          cssW = next;
          needsResize = true;
          scheduleRender();
        }
      }
    });
    resizeObs.observe(wrapperEl);
  });

  onDestroy(() => {
    resizeObs?.disconnect();
    resizeObs = null;
    if (rafId !== null) cancelAnimationFrame(rafId);
    rafId = null;
  });
</script>

<!-- One canvas for every layer so no sibling DOM row eats vertical space. -->
<div
  bind:this={wrapperEl}
  class="relative w-full"
  style="height: {height}px;"
  onpointermove={onPointerMove}
  onpointerleave={onPointerLeave}
  role="img"
  aria-label={m.training.chart.chart_aria}
>
  <!-- bg-canvas is the shared per-mode data-surface floor (matching the dashboard waveform +
       spectrogram canvases) so chrome lands on the same substrate in both modes. -->
  <canvas bind:this={canvasEl} class="block h-full w-full rounded-md bg-canvas"></canvas>
</div>
