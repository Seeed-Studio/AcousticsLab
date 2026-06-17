<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import PlusIcon from '$lib/components/ui/PlusIcon.svelte';
  import { inputClass } from '$lib/components/ui/inputClass';
  import { workspaces } from '$lib/stores/workspaces.svelte';
  import { m } from '$lib/i18n';
  import type { WorkspaceMutationResp } from '$lib/api/types';
  import { validateWorkspaceName } from './name-validate';
  import { errorCopy } from '$lib/utils/error-copy';

  interface Props {
    open: boolean;
    onclose: () => void;
    oncreated?: (resp: WorkspaceMutationResp) => void;
  }
  let { open, onclose, oncreated }: Props = $props();

  let name = $state('');
  let submitting = $state(false);
  // Inline because a toast would sit behind the native `<dialog>` top-layer backdrop.
  let backendError = $state<string | null>(null);

  // Null while empty keeps the initial state quiet; otherwise mirrors the daemon's name rules locally to avoid a round-trip.
  const trimmedName = $derived(name.trim());
  const nameError = $derived(trimmedName.length > 0 ? validateWorkspaceName(trimmedName) : null);
  const canSubmit = $derived(!submitting && trimmedName.length > 0 && !nameError);

  // Re-opening must not show stale state from a previous attempt.
  $effect(() => {
    if (open) {
      name = '';
      backendError = null;
      submitting = false;
    }
  });

  let nameInputEl = $state<HTMLInputElement | undefined>();
  // Deferred a tick so the dialog's native auto-focus doesn't clobber it.
  $effect(() => {
    if (!open || !nameInputEl) return;
    const el = nameInputEl;
    queueMicrotask(() => el.focus());
  });

  async function submit(e?: Event): Promise<void> {
    e?.preventDefault();
    // Enter in the input fires form submit while bypassing the button's disabled state.
    if (!canSubmit) return;
    submitting = true;
    backendError = null;
    try {
      const resp = await workspaces.create({ name: trimmedName });
      oncreated?.(resp);
      onclose();
    } catch (e) {
      backendError = errorCopy(e);
    } finally {
      submitting = false;
    }
  }
</script>

<Modal {open} title={m.workspace.create_dialog.title} {onclose} closeOnBackdrop={!submitting}>
  <form onsubmit={submit} class="flex flex-col gap-3">
    <label class="block">
      <span class="mb-1 block text-xs text-fg-secondary"
        >{m.workspace.create_dialog.name_label}</span
      >
      <input
        bind:this={nameInputEl}
        type="text"
        bind:value={name}
        disabled={submitting}
        autocomplete="off"
        spellcheck="false"
        maxlength="128"
        placeholder={m.workspace.create_dialog.name_placeholder}
        aria-invalid={nameError ? true : undefined}
        aria-describedby={nameError ? 'create-workspace-error' : undefined}
        class={inputClass(!!nameError)}
      />
    </label>

    {#if nameError}
      <p id="create-workspace-error" class="-mt-1 text-xs text-danger-soft-fg" role="alert">
        {nameError}
      </p>
    {/if}

    {#if backendError}
      <div
        class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
        role="alert"
      >
        {backendError}
      </div>
    {/if}

    <p class="text-[11px] text-fg-muted">
      {m.workspace.create_dialog.name_help}
    </p>
  </form>

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button onclick={submit} loading={submitting} disabled={!canSubmit}>
      {#if !submitting}<PlusIcon />{/if}
      {m.workspace.create_dialog.submit}
    </Button>
  {/snippet}
</Modal>
