<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import { inputClass } from '$lib/components/ui/inputClass';
  import { categories } from '$lib/stores/categories.svelte';
  import { validateCategoryName, findCaseInsensitiveDuplicate } from './name-validate';
  import { errorCopy } from '$lib/utils/error-copy';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';

  // Atomic dir rename + IDB re-key; the dir name is the trainer class label, so daemon-side this
  // bumps workspace_revision and marks prior heads stale (inference unaffected).

  interface Props {
    open: boolean;
    workspaceId: Uuid;
    /// Read once per open transition so a mid-open change can't clobber an in-flight edit.
    currentName: string;
    /// Live sibling names (currentName included); duplicate check excludes currentName so a
    /// case-only change or the unchanged name isn't flagged as a self-collision.
    existingNames: string[];
    onclose: () => void;
  }
  let { open, workspaceId, currentName, existingNames, onclose }: Props = $props();

  let name = $state('');
  let submitting = $state(false);
  let backendError = $state<string | null>(null);

  const trimmedName = $derived(name.trim());

  // No-op gate: avoids a redundant round-trip + revision bump on an unchanged name.
  const isUnchanged = $derived(trimmedName === currentName);

  // Mirrors the daemon's AssetPath rules; empty/unchanged stay silent since the disabled button
  // already signals it.
  const localError = $derived.by((): string | null => {
    if (trimmedName.length === 0 || isUnchanged) return null;
    const shape = validateCategoryName(trimmedName);
    if (shape) return shape;
    const others = existingNames.filter((n) => n !== currentName);
    const dup = findCaseInsensitiveDuplicate(trimmedName, others);
    if (dup) {
      return dup === trimmedName
        ? m.category.add_dialog.error_exact_duplicate
        : m.category.add_dialog.error_case_insensitive_duplicate(dup);
    }
    return null;
  });

  const canSubmit = $derived(!submitting && trimmedName.length > 0 && !localError && !isUnchanged);

  // Reset only on the open transition; without the guard the effect re-fires on every reactive
  // read inside (e.g. currentName) and clobbers an in-flight edit.
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
  // Autofocus + select-all on open, deferred a tick so it doesn't race the native dialog focus.
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
      await categories.rename(workspaceId, currentName, trimmedName);
      onclose();
    } catch (e) {
      backendError = errorCopy(e);
    } finally {
      submitting = false;
    }
  }
</script>

<Modal {open} title={m.category.rename_dialog.title} {onclose} closeOnBackdrop={!submitting}>
  <form onsubmit={submit} class="flex flex-col gap-3">
    <label class="block">
      <span class="mb-1 block text-xs text-fg-secondary">{m.category.rename_dialog.name_label}</span
      >
      <input
        bind:this={nameInputEl}
        type="text"
        bind:value={name}
        disabled={submitting}
        autocomplete="off"
        spellcheck="false"
        maxlength="255"
        aria-invalid={localError ? true : undefined}
        aria-describedby={localError ? 'rename-category-error' : undefined}
        class={inputClass(!!localError)}
      />
    </label>

    {#if localError}
      <p id="rename-category-error" class="-mt-1 text-xs text-danger-soft-fg" role="alert">
        {localError}
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
      {m.category.rename_dialog.name_help}
    </p>
  </form>

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button onclick={submit} loading={submitting} disabled={!canSubmit}>
      {m.category.rename_dialog.submit}
    </Button>
  {/snippet}
</Modal>
