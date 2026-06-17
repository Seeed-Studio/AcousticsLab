<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import TrashIcon from '$lib/components/ui/TrashIcon.svelte';
  import { workspaces } from '$lib/stores/workspaces.svelte';
  import { m } from '$lib/i18n';
  import type { WorkspaceListEntry } from '$lib/api/types';

  interface Props {
    open: boolean;
    // Snapshot taken at open so prose and the confirmed delete survive store-selection changes mid-dialog.
    targets: WorkspaceListEntry[];
    onclose: () => void;
  }
  let { open, targets, onclose }: Props = $props();

  let submitting = $state(false);

  $effect(() => {
    if (open) submitting = false;
  });

  function confirm(): void {
    if (submitting) return;
    submitting = true;
    // Fire-and-forget: store queue serializes through the daemon's single delete slot; close at once so cards drain via the list's `deleting` state, not a spinner here.
    void workspaces.deleteSelected(targets);
    onclose();
  }
</script>

<Modal
  {open}
  title={m.workspace.bulk_delete_dialog.title_count(targets.length)}
  {onclose}
  closeOnBackdrop={!submitting}
>
  <ul class="max-h-60 space-y-1.5 overflow-y-auto">
    {#each targets as t (t.id)}
      <li
        class="rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-xs text-fg wrap-break-word"
      >
        {t.name}
      </li>
    {/each}
  </ul>
  <p class="text-xs text-fg-secondary wrap-break-word">
    {m.workspace.bulk_delete_dialog.body}
  </p>

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button variant="destructive" onclick={confirm} loading={submitting}>
      {#if !submitting}<TrashIcon />{/if}
      {m.workspace.bulk_delete_dialog.submit_count(targets.length)}
    </Button>
  {/snippet}
</Modal>
