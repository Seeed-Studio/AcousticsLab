<script lang="ts">
  import PencilIcon from '$lib/components/ui/PencilIcon.svelte';
  import DownloadIcon from '$lib/components/ui/DownloadIcon.svelte';
  import UploadIcon from '$lib/components/ui/UploadIcon.svelte';
  import { m } from '$lib/i18n';

  interface Props {
    onrename: () => void;
    onexport: () => void;
    // Caller locks the import target to the current workspace, skipping the target-pick step.
    onimport: () => void;
  }
  let { onrename, onexport, onimport }: Props = $props();
</script>

<div
  role="toolbar"
  aria-label={m.workspace.tool_island.aria_label}
  class="inline-flex shrink-0 items-center gap-0.5 rounded-full border border-line bg-surface p-1 shadow-card"
>
  <button
    type="button"
    onclick={onrename}
    aria-label={m.workspace.tool_island.rename_aria}
    title={m.workspace.tool_island.rename_title}
    class="rounded-full p-1.5 text-fg-muted transition hover:bg-surface-2 hover:text-fg"
  >
    <PencilIcon />
  </button>
  <button
    type="button"
    onclick={onexport}
    aria-label={m.workspace.tool_island.export_aria}
    title={m.workspace.tool_island.export_title}
    class="rounded-full p-1.5 text-fg-muted transition hover:bg-surface-2 hover:text-fg"
  >
    <DownloadIcon />
  </button>
  <button
    type="button"
    onclick={onimport}
    aria-label={m.workspace.tool_island.import_aria}
    title={m.workspace.tool_island.import_title}
    class="rounded-full p-1.5 text-fg-muted transition hover:bg-surface-2 hover:text-fg"
  >
    <UploadIcon />
  </button>
</div>
