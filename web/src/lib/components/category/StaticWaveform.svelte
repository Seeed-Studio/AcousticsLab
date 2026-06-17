<script lang="ts">
  import { onMount } from 'svelte';
  import { visualDevicePixelRatio } from '$lib/components/dashboard/visualRuntime';
  import { envelopeFromRing } from '$lib/audio/ring-buffer';
  import { theme } from '$lib/stores/theme.svelte';

  // Static min/max envelope of a whole draft PCM clip; idempotent draw on mount/resize/pcm-change, no RAF loop. Shares the live waveform's palette so both surfaces read as one primitive.
  interface Props {
    pcm: Float32Array;
    /** Stroke + fill colour; default `--color-accent` (matches live waveform). */
    color?: string;
    /** Background; default `--color-canvas` so silence dissolves into the panel. */
    background?: string;
  }
  let { pcm, color, background }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state();

  onMount(() => {
    const el = canvas;
    if (!el) return;
    const ctx = el.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) return;

    let hiBuf: Float32Array = new Float32Array(0);
    let loBuf: Float32Array = new Float32Array(0);
    let pendingW = 1;
    let pendingH = 1;
    let needsRender = true;

    const updatePendingSize = (): void => {
      const dpr = visualDevicePixelRatio();
      const r = el.getBoundingClientRect();
      pendingW = Math.max(1, Math.floor(r.width * dpr));
      pendingH = Math.max(1, Math.floor(r.height * dpr));
      needsRender = true;
      schedule();
    };

    // void reads register repaint dependencies: pcm reference (new draft into same instance),
    // theme.resolved (flip trigger; CSS-var-derived colours differ light/dark), and color/background
    // overrides (repaint even when pcm/theme hold).
    $effect(() => {
      void pcm;
      void theme.resolved;
      void color;
      void background;
      needsRender = true;
      schedule();
    });

    let rafHandle = 0;
    // Arrow consts, not `function` decls: hoisting would let TS assume they run before `if (!ctx) return`, widening `ctx` back to include null.
    const schedule = (): void => {
      if (rafHandle !== 0) return;
      rafHandle = requestAnimationFrame(() => {
        rafHandle = 0;
        render();
      });
    };

    const render = (): void => {
      if (!needsRender) return;
      if (el.width !== pendingW) el.width = pendingW;
      if (el.height !== pendingH) el.height = pendingH;
      if (hiBuf.length !== pendingW) {
        hiBuf = new Float32Array(pendingW);
        loBuf = new Float32Array(pendingW);
      }
      needsRender = false;

      const w = el.width;
      const h = el.height;
      const mid = h / 2;
      const amp = mid * 0.92;

      // Re-read CSS vars per render (not memoised at mount) so they stay correct across theme flip; render() is not per-frame so the cost is one read per visible state.
      const readVar = (name: string): string =>
        getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      const activeBackground = background ?? readVar('--color-canvas');
      const activeStroke = color ?? readVar('--color-accent');
      const gridStroke = readVar('--color-line');

      ctx.fillStyle = activeBackground;
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = gridStroke;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, mid + 0.5);
      ctx.lineTo(w, mid + 0.5);
      ctx.stroke();

      // `envelopeFromRing` treats `pcm` as a non-wrapping ring (totalWritten = endSample = pcm.length) so static and live share one bin-mapping path.
      if (pcm.length > 0 && w > 0) {
        envelopeFromRing(pcm, pcm.length, pcm.length, pcm.length, w, loBuf, hiBuf);

        // Tint via globalAlpha on the raw stroke colour, not a synthesised rgba(): works for any CSS colour form (shorthand hex, rgb(), oklch(), named) which rgba-string synthesis would mis-render as grey. Restored to 1 before the opaque contour strokes.
        ctx.globalAlpha = 0.15;
        ctx.fillStyle = activeStroke;
        ctx.beginPath();
        ctx.moveTo(0, mid - hiBuf[0] * amp);
        for (let x = 1; x < w; x++) ctx.lineTo(x, mid - hiBuf[x] * amp);
        for (let x = w - 1; x >= 0; x--) ctx.lineTo(x, mid - loBuf[x] * amp);
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.strokeStyle = activeStroke;
        ctx.lineWidth = 1.25;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(0, mid - hiBuf[0] * amp);
        for (let x = 1; x < w; x++) ctx.lineTo(x, mid - hiBuf[x] * amp);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, mid - loBuf[0] * amp);
        for (let x = 1; x < w; x++) ctx.lineTo(x, mid - loBuf[x] * amp);
        ctx.stroke();
      }
    };

    updatePendingSize();
    const ro = new ResizeObserver(updatePendingSize);
    ro.observe(el);
    window.addEventListener('resize', updatePendingSize, { passive: true });

    return () => {
      if (rafHandle !== 0) cancelAnimationFrame(rafHandle);
      ro.disconnect();
      window.removeEventListener('resize', updatePendingSize);
    };
  });
</script>

<canvas bind:this={canvas} class="block h-full w-full rounded-md"></canvas>
