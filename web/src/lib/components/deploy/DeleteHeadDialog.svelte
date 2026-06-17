<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import TrashIcon from '$lib/components/ui/TrashIcon.svelte';
  import { heads as headsApi } from '$lib/api/endpoints';
  import { errorCopy } from '$lib/utils/error-copy';
  import { formatBytes } from '$lib/utils/format';
  import { m } from '$lib/i18n';
  import type { HeadRecord, Uuid } from '$lib/api/types';

  interface Props {
    open: boolean;
    workspaceId: Uuid;
    // Nullable: the body guards on {#if head} and the submit on disabled={!head}; the parent keeps this snapshot set while `open` is still true so the identity stays painted right up to when Modal's `dialog:not([open])` rule hides the body.
    head: HeadRecord | null;
    onclose: () => void;
    // Fires after a successful DELETE ack so the parent can refresh; the synchronous daemon delete keeps the dialog in `submitting` until the parent closes it.
    ondeleted?: (deletedHeadId: Uuid) => void;
  }
  let { open, workspaceId, head, onclose, ondeleted }: Props = $props();

  let submitting = $state(false);
  let backendError = $state<string | null>(null);

  // Reset on each open so re-opening for a different head shows no stale error.
  $effect(() => {
    if (open) {
      submitting = false;
      backendError = null;
    }
  });

  async function confirm(): Promise<void> {
    if (submitting || !head) return;
    submitting = true;
    backendError = null;
    try {
      const resp = await headsApi.delete(workspaceId, head.head_id);
      ondeleted?.(resp.deleted_head_id);
      onclose();
    } catch (e) {
      backendError = errorCopy(e);
      submitting = false;
    }
  }
</script>

<Modal {open} title={m.deploy.delete_dialog.title} {onclose} closeOnBackdrop={!submitting}>
  <!-- Echoes the clicked row's identity (short id, full UUID on hover via `title`) so the confirmation reads as "yes, this row". -->
  {#if head}
    <div
      class="rounded-md border border-line bg-surface-2 px-3 py-2 text-xs text-fg wrap-break-word"
    >
      <p class="font-mono text-sm font-semibold text-fg" title={head.head_id}>
        {head.head_id.slice(0, 8)}
      </p>
      <p class="mt-0.5 text-[11px] text-fg-muted">
        {m.deploy.delete_dialog.meta_line(
          formatBytes(head.size_bytes),
          head.n_classes,
          head.workspace_revision.id
        )}
      </p>
    </div>
  {/if}
  <p class="text-xs text-fg-secondary wrap-break-word">
    {m.deploy.delete_dialog.body}
  </p>

  {#if backendError}
    <div
      class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
      role="alert"
    >
      {backendError}
    </div>
  {/if}

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button variant="destructive" onclick={confirm} loading={submitting} disabled={!head}>
      {#if !submitting}<TrashIcon />{/if}
      {m.deploy.delete_dialog.submit}
    </Button>
  {/snippet}
</Modal>
