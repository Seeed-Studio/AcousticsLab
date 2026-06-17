<script lang="ts">
  import EnvelopeWaveform from '$lib/components/EnvelopeWaveform.svelte';
  import type { Recorder } from '$lib/audio/recorder.svelte';

  // Renderer reads a smoothed cursor, not the raw sample index, so motion decouples from chunk
  // timing: the worklet posts faster than the 60 Hz RAF and would otherwise step/blink per chunk.
  interface Props {
    recorder: Recorder;
    seconds?: number;
    color?: string;
    background?: string;
  }
  // No color/background default: undefined propagates so EnvelopeWaveform resolves CSS vars.
  let { recorder, seconds = 3, color, background }: Props = $props();
</script>

<EnvelopeWaveform source={recorder} {seconds} {color} {background} />
