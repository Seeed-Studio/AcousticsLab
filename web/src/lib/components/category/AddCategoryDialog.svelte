<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import PlusIcon from '$lib/components/ui/PlusIcon.svelte';
  import { inputClass } from '$lib/components/ui/inputClass';
  import { categories } from '$lib/stores/categories.svelte';
  import { validateCategoryName, findCaseInsensitiveDuplicate } from './name-validate';
  import { errorCopy } from '$lib/utils/error-copy';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';

  interface Props {
    open: boolean;
    workspaceId: Uuid;
    onclose: () => void;
    oncreated?: (name: string) => void;
  }
  let { open, workspaceId, onclose, oncreated }: Props = $props();

  let name = $state('');
  let submitting = $state(false);
  // Inline banner, not a toast: the native <dialog> top-layer hides z-indexed toasts behind its backdrop.
  let backendError = $state<string | null>(null);

  const trimmedName = $derived(name.trim());

  // Mirror the daemon's AssetPath verdict locally without a round-trip; empty input stays unflagged since the disabled submit button is signal enough.
  const localError = $derived.by((): string | null => {
    if (trimmedName.length === 0) return null;
    const shape = validateCategoryName(trimmedName);
    if (shape) return shape;
    // No mandatory-default branch: the sole mandatory name `_background_noise_` is already rejected by the leading-underscore rule.
    const existing = categories.for(workspaceId).entries.map((c) => c.name);
    const dup = findCaseInsensitiveDuplicate(trimmedName, existing);
    if (dup) {
      return dup === trimmedName
        ? m.category.add_dialog.error_exact_duplicate
        : m.category.add_dialog.error_case_insensitive_duplicate(dup);
    }
    return null;
  });

  const canSubmit = $derived(!submitting && trimmedName.length > 0 && !localError);

  // Reset on each open so a re-opened dialog starts clean.
  $effect(() => {
    if (open) {
      name = '';
      backendError = null;
      submitting = false;
    }
  });

  let nameInputEl = $state<HTMLInputElement | undefined>();
  $effect(() => {
    if (!open || !nameInputEl) return;
    const el = nameInputEl;
    // Defer a tick so we don't race the dialog's own auto-focus.
    queueMicrotask(() => el.focus());
  });

  async function submit(e?: Event): Promise<void> {
    e?.preventDefault();
    if (!canSubmit) return;
    submitting = true;
    backendError = null;
    try {
      await categories.create(workspaceId, trimmedName);
      oncreated?.(trimmedName);
      onclose();
    } catch (e) {
      backendError = errorCopy(e);
    } finally {
      submitting = false;
    }
  }
</script>

<Modal {open} title={m.category.add_dialog.title} {onclose} closeOnBackdrop={!submitting}>
  <form onsubmit={submit} class="flex flex-col gap-3">
    <label class="block">
      <span class="mb-1 block text-xs text-fg-secondary">{m.category.add_dialog.name_label}</span>
      <input
        bind:this={nameInputEl}
        type="text"
        bind:value={name}
        disabled={submitting}
        autocomplete="off"
        spellcheck="false"
        maxlength="255"
        placeholder={m.category.add_dialog.name_placeholder}
        aria-invalid={localError ? true : undefined}
        aria-describedby={localError ? 'add-category-error' : undefined}
        class={inputClass(!!localError)}
      />
    </label>

    {#if localError}
      <p id="add-category-error" class="-mt-1 text-xs text-danger-soft-fg" role="alert">
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
      {m.category.add_dialog.name_help_prefix}<code class="font-mono"
        >{m.category.add_dialog.name_help_code_example}</code
      >{m.category.add_dialog.name_help_suffix}
    </p>
  </form>

  {#snippet footer()}
    <Button variant="secondary" onclick={onclose} disabled={submitting}>{m.common.cancel}</Button>
    <Button onclick={submit} loading={submitting} disabled={!canSubmit}>
      {#if !submitting}<PlusIcon />{/if}
      {m.category.add_dialog.submit}
    </Button>
  {/snippet}
</Modal>
