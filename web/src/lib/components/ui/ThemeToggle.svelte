<script lang="ts">
  import { theme, type ThemeMode } from '$lib/stores/theme.svelte';
  import { m } from '$lib/i18n';
  import SunIcon from './SunIcon.svelte';
  import MoonIcon from './MoonIcon.svelte';
  import ThemeAutoIcon from './ThemeAutoIcon.svelte';

  // Two presentations off one source of truth (`theme.mode`): sm+ segmented radiogroup with
  // roving tabindex (only the selected segment is tabbable, per WAI-ARIA); <sm popover of
  // menuitemradio options dismissed by outside pointerdown/Escape/focusout (no route-change
  // hook needed -- persistent header, every <sm nav taps outside or moves focus). Icon
  // reflects the operator's CHOICE: 'auto' stays 'auto' regardless of resolved theme.

  const MODES: readonly ThemeMode[] = ['auto', 'light', 'dark'];

  let popoverOpen = $state(false);
  let wrapper = $state<HTMLDivElement | undefined>();

  function selectMode(mode: ThemeMode): void {
    // Param is `mode` not `m` to avoid shadowing the `m.*` i18n proxy.
    theme.setMode(mode);
    popoverOpen = false;
  }
  function togglePopover(): void {
    popoverOpen = !popoverOpen;
  }
  function closePopover(): void {
    popoverOpen = false;
  }

  // WAI-ARIA radiogroup keys; selection wraps at the ends.
  function onSegmentedKeydown(e: KeyboardEvent, idx: number): void {
    let nextIdx: number | null = null;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        nextIdx = (idx + 1) % MODES.length;
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        nextIdx = (idx - 1 + MODES.length) % MODES.length;
        break;
      case 'Home':
        nextIdx = 0;
        break;
      case 'End':
        nextIdx = MODES.length - 1;
        break;
    }
    if (nextIdx === null) return;
    e.preventDefault();
    selectMode(MODES[nextIdx]);
    // Keydown doesn't move focus; focus the now-sole-tabbable segment once selection settles.
    queueMicrotask(() => {
      const buttons = wrapper?.querySelectorAll<HTMLButtonElement>('button[data-segment]');
      buttons?.[nextIdx].focus();
    });
  }

  // Outside-tap / Escape dismissal; listeners attach only while open (zero cost when closed).
  $effect(() => {
    if (!popoverOpen) return;
    const onDown = (e: PointerEvent): void => {
      if (!wrapper?.contains(e.target as Node | null)) closePopover();
    };
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') closePopover();
    };
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey);
    };
  });

  // Close when focus leaves the wrapper; null relatedTarget (non-focusable target / out of
  // document) counts as left.
  function onFocusOut(e: FocusEvent): void {
    const next = e.relatedTarget as Node | null;
    if (!next || !wrapper?.contains(next)) closePopover();
  }

  let currentLabel = $derived(m.theme.options[theme.mode]);
</script>

<div bind:this={wrapper} class="relative" onfocusout={onFocusOut}>
  <div
    role="radiogroup"
    aria-label={m.theme.label}
    class="hidden h-[30px] items-stretch rounded-full border border-line bg-surface p-0.5 shadow-card sm:inline-flex"
  >
    {#each MODES as mode, i (mode)}
      {@const selected = theme.mode === mode}
      {@const label = m.theme.options[mode]}
      <button
        type="button"
        role="radio"
        aria-checked={selected}
        data-segment={mode}
        tabindex={selected ? 0 : -1}
        title={label}
        class="inline-flex w-7 items-center justify-center rounded-full transition duration-200 ease-out {selected
          ? 'bg-surface-2 text-fg'
          : 'text-fg-muted hover:text-fg'}"
        onclick={() => selectMode(mode)}
        onkeydown={(e) => onSegmentedKeydown(e, i)}
      >
        <span class="sr-only">{label}</span>
        {#if mode === 'auto'}
          <ThemeAutoIcon class="h-3.5 w-3.5" />
        {:else if mode === 'light'}
          <SunIcon class="h-3.5 w-3.5" />
        {:else}
          <MoonIcon class="h-3.5 w-3.5" />
        {/if}
      </button>
    {/each}
  </div>

  <button
    type="button"
    class="inline-flex h-[36px] w-[36px] items-center justify-center rounded-full border border-line bg-surface text-fg-muted shadow-card transition hover:border-line-strong hover:text-fg sm:hidden"
    aria-expanded={popoverOpen}
    aria-haspopup="menu"
    aria-label={m.theme.label_with_current(currentLabel)}
    onclick={togglePopover}
  >
    {#if theme.mode === 'auto'}
      <ThemeAutoIcon class="h-4 w-4" />
    {:else if theme.mode === 'light'}
      <SunIcon class="h-4 w-4" />
    {:else}
      <MoonIcon class="h-4 w-4" />
    {/if}
  </button>

  {#if popoverOpen}
    <!-- `sm:hidden` guards the resize edge: if opened at <sm then widened past sm without
         dismissal, the popover would otherwise hover over the segmented control. -->
    <div
      role="menu"
      aria-label={m.theme.label}
      class="absolute right-0 top-full z-30 mt-2 min-w-32 rounded-lg border border-line bg-elevated p-1 shadow-popover sm:hidden"
    >
      {#each MODES as mode (mode)}
        {@const selected = theme.mode === mode}
        <button
          type="button"
          role="menuitemradio"
          aria-checked={selected}
          class="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition {selected
            ? 'bg-surface-2 text-fg'
            : 'text-fg-secondary hover:bg-surface-2 hover:text-fg'}"
          onclick={() => selectMode(mode)}
        >
          <span class="text-fg-muted">
            {#if mode === 'auto'}
              <ThemeAutoIcon class="h-4 w-4" />
            {:else if mode === 'light'}
              <SunIcon class="h-4 w-4" />
            {:else}
              <MoonIcon class="h-4 w-4" />
            {/if}
          </span>
          <span>{m.theme.options[mode]}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>
