<script lang="ts">
  import { streams } from '$lib/stores/streams.svelte';
  import EnvelopeWaveform from '$lib/components/EnvelopeWaveform.svelte';

  // Binds the shared EnvelopeWaveform renderer to the dashboard's `streams` source (the recorder binds the same renderer to a Recorder).
  interface Props {
    seconds?: number;
    color?: string;
    background?: string;
  }
  // color/background stay undefined so EnvelopeWaveform resolves the theme CSS vars, keeping this wrapper theme-agnostic.
  let { seconds = 3, color, background }: Props = $props();
</script>

<EnvelopeWaveform source={streams} {seconds} {color} {background} />
