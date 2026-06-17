<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import { inputClass } from '$lib/components/ui/inputClass';
  import { workspaces } from '$lib/stores/workspaces.svelte';
  import { m } from '$lib/i18n';
  import type { Uuid, WorkspaceMutationResp } from '$lib/api/types';
  import { validateWorkspaceName } from './name-validate';
  import { errorCopy } from '$lib/utils/error-copy';

  // Metadata PATCH (name/tags) does not advance workspace_revision, so rename invalidates no
  // workspace-keyed cache; onsaved returns the full resp so a tag-showing caller need not refetch.

  interface Props {
    open: boolean;
    workspaceId: Uuid;
    /// Read once per open transition so mid-open changes don't clobber an in-flight edit.
    currentName: string;
    onclose: () => void;
    /// Caller updates its local name from the response to re-render without another GET.
    onsaved?: (resp: WorkspaceMutationResp) => void;
  }
  let { open, workspaceId, currentName, onclose, onsaved }: Props = $props();

  let name = $state('');
  let submitting = $state(false);
  let backendError = $state<string | null>(null);

  const trimmedName = $derived(name.trim());

  // Empty input shows no error since the disabled submit button is signal enough.
  const nameError = $derived(trimmedName.length > 0 ? validateWorkspaceName(trimmedName) : null);

  // No-op gate: disabling submit when unchanged avoids a redundant PATCH round-trip.
  const isUnchanged = $derived(trimmedName === currentName);

  const canSubmit = $derived(!submitting && trimmedName.length > 0 && !nameError && !isUnchanged);

  // Reset only on the open transition; without this guard the effect re-fires on every reactive
  // read inside and clobbers the in-flight edit.
  let lastOpenSeen = $state(false);
  $effect(() => {
    if (open && !lastOpenSeen) {
      lastOpenSeen = true;
      name = currentName;
      backendError = null;
      submitting = false;
    } else if (!open && lastOpenSeen) {
      lastOpenSeen = false;
    }
  });

  let nameInputEl = $state<HTMLInputElement | undefined>();
  // Autofocus + select-all deferred a tick so it doesn't race the native dialog auto-focus.
  $effect(() => {
    if (!open || !nameInputEl) return;
    const el = nameInputEl;
    queueMicrotask(() => {
      el.focus();
      el.select();
    });
  });

  async function submit(e?: Event): Promise<void> {
    e?.preventDefault();
    // Guard against Enter-in-input bypassing the disabled button.
    if (!canSubmit) return;
    submitting = true;
    backendError = null;
    try {
      const resp = await workspaces.patch(workspaceId, { name: trimmedName });
      onsaved?.(resp);
      onclose();
    } catch (e) {
      backendError = errorCopy(e);
    } finally {
      submitting = false;
    }
  }
</script>

<Modal {open} title={m.workspace.rename_dialog.title} {onclose} closeOnBackdrop={!submitting}>
  <form onsubmit={submit} class="flex flex-col gap-3">
    <label class="block">
      <span class="mb-1 block text-xs text-fg-secondary"
        >{m.workspace.rename_dialog.name_label}</span
      >
      <input
        bind:this={nameInputEl}
        type="text"
        bind:value={name}
        disabled={submitting}
        autocomplete="off"
        spellcheck="false"
        maxlength="128"
        aria-invalid={nameError ? true : undefined}
        aria-describedby={nameError ? 'rename-workspace-error' : undefined}
        class={inputClass(!!nameError)}
      />
    </label>

    {#if nameError}
      <p id="rename-workspace-error" class="-mt-1 text-xs text-danger-soft-fg" role="alert">
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
      {m.workspace.rename_dialog.name_help}
    </p>
  </form>

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button onclick={submit} loading={submitting} disabled={!canSubmit}>
      {m.workspace.rename_dialog.submit}
    </Button>
  {/snippet}
</Modal>
