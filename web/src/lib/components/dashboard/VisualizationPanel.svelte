<script lang="ts">
  import { streams } from '$lib/stores/streams.svelte';
  import WaveformCanvas from './WaveformCanvas.svelte';
  import SpectrogramCanvas from './SpectrogramCanvas.svelte';
  import { socketLabel, socketPillClass } from './socketPill';
  import { m } from '$lib/i18n';
</script>

<!-- Height pinned at every breakpoint: below lg the panels stack in an auto-height row where
     the canvases' DPR-sized width/height attrs feed their aspect ratio back into row sizing;
     an explicit height breaks that feedback loop, killing jitter at the lg breakpoint. -->
<section
  class="flex h-(--vis-panel-h) flex-col rounded-xl border border-line bg-surface px-5 pt-3.5 pb-5 shadow-card"
>
  <!-- Meta folds into the header (not a footer) to keep the bottom edge at p-5 and corners symmetric. -->
  <header class="@container/vishdr mb-3 flex items-center justify-between gap-3">
    <div class="flex items-baseline gap-2">
      <h2 class="text-sm font-semibold text-fg">{m.dashboard.visualization_panel.heading}</h2>
      <!-- Segmented so low-precedence facts drop (not wrap) as @container/vishdr narrows: codec
           hides first (@[23rem]), then channels (@[21rem]); rate+window always stay. Each droppable
           segment owns its leading separator so it vanishes cleanly; separators aria-hidden so the
           reading order is just the values. -->
      <span class="text-[11px] text-fg-muted">
        {m.dashboard.visualization_panel.audio_sample_rate}
        <span class="hidden @[21rem]/vishdr:inline">
          <span aria-hidden="true">·</span>
          {m.dashboard.visualization_panel.audio_channels}
        </span>
        <span class="hidden @[23rem]/vishdr:inline">
          <span aria-hidden="true">·</span>
          {m.dashboard.visualization_panel.audio_codec}
        </span>
        <span aria-hidden="true">·</span>
        {m.dashboard.visualization_panel.audio_window}
      </span>
    </div>
    <span
      class="rounded-full px-2 py-0.5 text-[11px] font-medium capitalize tracking-wide transition-colors duration-200 {socketPillClass(
        streams.audioStatus
      )}"
    >
      {socketLabel(streams.audioStatus)}
    </span>
  </header>

  <!-- min-h-0 lets the canvases shrink below intrinsic size so flex actually splits the pinned height evenly. -->
  <div class="flex min-h-0 flex-1 flex-col gap-2">
    <div class="min-h-0 flex-1">
      <WaveformCanvas seconds={3} />
    </div>
    <div class="min-h-0 flex-1">
      <SpectrogramCanvas seconds={3} />
    </div>
  </div>
</section>
