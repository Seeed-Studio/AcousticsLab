<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import TrashIcon from '$lib/components/ui/TrashIcon.svelte';
  import { workspaces } from '$lib/stores/workspaces.svelte';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';

  interface Props {
    open: boolean;
    workspaceId: Uuid;
    workspaceName: string;
    onclose: () => void;
    // Fires on confirm before the daemon ack; detail page navigates back to the list, list page ignores it.
    ondeleted?: () => void;
  }
  let { open, workspaceId, workspaceName, onclose, ondeleted }: Props = $props();

  let submitting = $state(false);

  $effect(() => {
    if (open) submitting = false;
  });

  function confirm(): void {
    if (submitting) return;
    submitting = true;
    // Fire-and-forget so the list card transitions through its `deleting` state instead of a spinner; swallow the store's terminal re-throw (no global `unhandledrejection` handler) - its refresh-driven `deleting` revert is the visible failure signal.
    void workspaces.delete(workspaceId).catch(() => undefined);
    ondeleted?.();
    onclose();
  }
</script>

<Modal {open} title={m.workspace.delete_dialog.title} {onclose} closeOnBackdrop={!submitting}>
  <!-- `wrap-break-word` breaks long unbreakable names (user-supplied, no natural break opportunity). -->
  <p
    class="rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-xs text-fg wrap-break-word"
  >
    {workspaceName}
  </p>
  <p class="text-xs text-fg-secondary wrap-break-word">
    {m.workspace.delete_dialog.body}
  </p>

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button variant="destructive" onclick={confirm} loading={submitting}>
      {#if !submitting}<TrashIcon />{/if}
      {m.workspace.delete_dialog.submit}
    </Button>
  {/snippet}
</Modal>
