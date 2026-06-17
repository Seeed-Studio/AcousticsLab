<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import TrashIcon from '$lib/components/ui/TrashIcon.svelte';
  import { categories, type CategoryOrigin } from '$lib/stores/categories.svelte';
  import { prettyCategoryName } from './labels';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';

  // Two delete flavours: 'idb' resolves the local-only IDB removal inline (daemon never sees a
  // DELETE); 'server' is fire-and-forget through the global delete queue and closes immediately
  // so the modal never hangs on the async drain, with progress shown by the list row's pill.
  interface Props {
    open: boolean;
    workspaceId: Uuid;
    categoryName: string;
    origin: CategoryOrigin;
    onclose: () => void;
    ondeleted?: () => void;
  }
  let { open, workspaceId, categoryName, origin, onclose, ondeleted }: Props = $props();

  let submitting = $state(false);
  let error = $state<string | null>(null);

  $effect(() => {
    if (open) {
      submitting = false;
      error = null;
    }
  });

  const display = $derived(prettyCategoryName(categoryName));
  const isServerSide = $derived(origin === 'server');

  async function confirm(): Promise<void> {
    if (submitting) return;
    submitting = true;
    if (isServerSide) {
      // Swallow the store's terminal re-throw (no global `unhandledrejection` handler); the dialog
      // is already closing and there's no per-category error surface, so the list row's reverted
      // `deleting` pill is the only failure signal.
      void categories.delete(workspaceId, categoryName).catch(() => undefined);
      ondeleted?.();
      onclose();
      return;
    }
    try {
      await categories.delete(workspaceId, categoryName);
      ondeleted?.();
      onclose();
    } catch (e) {
      error = e instanceof Error ? e.message : m.category.delete_dialog.error_fallback;
    } finally {
      submitting = false;
    }
  }
</script>

<Modal {open} title={m.category.delete_dialog.title} {onclose} closeOnBackdrop={!submitting}>
  <p
    class="rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-xs text-fg wrap-break-word"
  >
    {display}
    <span class="ml-1 text-fg-muted">· {categoryName}</span>
  </p>
  <p class="text-xs text-fg-secondary wrap-break-word">
    {#if isServerSide}
      {m.category.delete_dialog.body_server}
    {:else}
      {m.category.delete_dialog.body_idb}
    {/if}
  </p>

  {#if error}
    <div
      class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
      role="alert"
    >
      {error}
    </div>
  {/if}

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button variant="destructive" onclick={confirm} loading={submitting}>
      {#if !submitting}<TrashIcon />{/if}
      {m.category.delete_dialog.submit}
    </Button>
  {/snippet}
</Modal>
