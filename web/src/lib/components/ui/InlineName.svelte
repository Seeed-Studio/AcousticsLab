<script lang="ts">
  import { onMount } from 'svelte';
  import { validateWorkspaceName } from '$lib/components/workspace/name-validate';

  // Zero-layout-shift rename: always-present `border` makes valid<->invalid a 0-px swap (red border = invalid,
  // `disabled` cursor-wait = saving; no text signals). The invalid rose halo MUST stay in the global UNLAYERED
  // `input[aria-invalid='true']:focus-visible` rule: @layer-utilities Tailwind loses to the unlayered focus-ring
  // rule by cascade-layer precedence at any specificity.
  interface Props {
    value: string;
    placeholder?: string;
    ariaLabel?: string;
    // Receives the trimmed/validated/changed draft; may throw to reject (stamps red border, keeps input for amend).
    onsave: (newValue: string) => Promise<void>;
    // Dismissed without committing (Escape, or blur with empty/unchanged/invalid draft).
    oncancel: () => void;
  }
  let { value, placeholder, ariaLabel, onsave, oncancel }: Props = $props();

  // Seed once on mount so a later external `value` change can't clobber an in-progress draft.
  let draft = $state('');
  onMount(() => {
    draft = value;
  });

  let saving = $state(false);
  let backendError = $state(false);
  // Escape sets this so the blur it triggers doesn't re-run commit/discard logic.
  let cancelled = $state(false);

  const trimmed = $derived(draft.trim());
  const unchanged = $derived(trimmed === value);
  // Empty/unchanged drafts are no-op cancels, not errors, so validate only a changed name.
  const localError = $derived(
    trimmed.length > 0 && !unchanged ? validateWorkspaceName(trimmed) : null
  );
  const hasError = $derived(localError !== null || backendError);

  let inputEl = $state<HTMLInputElement | undefined>();
  $effect(() => {
    if (!inputEl) return;
    const el = inputEl;
    queueMicrotask(() => {
      el.focus();
      el.select();
    });
  });

  async function commit(): Promise<void> {
    if (cancelled || saving) return;
    if (trimmed.length === 0 || unchanged) {
      oncancel();
      return;
    }
    if (localError !== null) {
      return; // stay in edit mode; red border already signals it can't commit
    }
    saving = true;
    backendError = false;
    try {
      await onsave(trimmed);
    } catch {
      // Backend reject: red border + refocus so the retry keeps the draft.
      backendError = true;
      queueMicrotask(() => {
        inputEl?.focus();
        inputEl?.select();
      });
    } finally {
      saving = false;
    }
  }

  function cancel(): void {
    cancelled = true;
    draft = value;
    oncancel();
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      cancel();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      void commit();
    }
  }

  function onBlur(): void {
    if (saving || cancelled) return;
    if (trimmed.length === 0 || unchanged || localError !== null) {
      cancel(); // unsavable draft reverts on blur; only a valid changed blur commits
      return;
    }
    void commit();
  }

  // Clear the stale backend-error stamp on any keystroke; `localError` re-lights if the amend is itself invalid.
  function onInput(): void {
    if (backendError) backendError = false;
  }
</script>

<input
  bind:this={inputEl}
  type="text"
  bind:value={draft}
  oninput={onInput}
  onkeydown={onKey}
  onblur={onBlur}
  disabled={saving}
  autocomplete="off"
  spellcheck="false"
  maxlength="128"
  {placeholder}
  aria-label={ariaLabel}
  aria-invalid={hasError ? true : undefined}
  class="block h-7 w-full rounded-md border bg-surface-2 px-2 py-1 text-sm font-semibold text-fg outline-none disabled:cursor-wait disabled:bg-surface-2 {hasError
    ? 'border-danger-dot'
    : 'border-transparent'}"
/>
