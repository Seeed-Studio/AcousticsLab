<script lang="ts">
  import { locale } from '$lib/stores/locale.svelte';
  import { m, SUPPORTED_LOCALES, LOCALE_LABELS, LOCALE_CHIPS, type LocaleCode } from '$lib/i18n';

  // Globe-icon trigger + popover (scales past 2-3 locales without segment-width pressure); hidden
  // while only one locale ships. No "Auto" row: auto-detect is the silent default, and the active
  // (incl. auto-detected) language reads checked off `resolved`.

  let popoverOpen = $state(false);
  let wrapper = $state<HTMLDivElement | undefined>();

  // Lint sees `1 > 1` (dead) until a second locale registers; widening the tuple to silence it
  // loses Record<LocaleCode,_> exhaustiveness.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  const hasMultipleLocales = $derived(SUPPORTED_LOCALES.length > 1);

  // Off `resolved` not `mode`, so the aria-label names the active (incl. auto-detected) language.
  const currentChip = $derived(LOCALE_CHIPS[locale.resolved]);

  // Param is `mode` not `m` to avoid shadowing the imported `m` i18n proxy.
  function selectMode(mode: LocaleCode): void {
    locale.setMode(mode);
    popoverOpen = false;
  }
  function togglePopover(): void {
    popoverOpen = !popoverOpen;
  }
  function closePopover(): void {
    popoverOpen = false;
  }

  // Outside-tap/Escape dismissal, attached only while open so the closed state costs no listeners.
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

  function onFocusOut(e: FocusEvent): void {
    const next = e.relatedTarget as Node | null;
    if (!next || !wrapper?.contains(next)) closePopover();
  }
</script>

{#if hasMultipleLocales}
  <div bind:this={wrapper} class="relative" onfocusout={onFocusOut}>
    <!-- Single globe trigger at every breakpoint (36px <sm / 30px sm+ to align with sibling controls). -->
    <button
      type="button"
      class="inline-flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface text-fg-muted shadow-card transition hover:border-line-strong hover:text-fg sm:h-7.5 sm:w-7.5"
      aria-expanded={popoverOpen}
      aria-haspopup="menu"
      aria-label={m.locale.label_with_current(currentChip)}
      onclick={togglePopover}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="h-4 w-4 sm:h-3.5 sm:w-3.5"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M2 12h20" />
        <path d="M12 2a15 15 0 010 20" />
        <path d="M12 2a15 15 0 000 20" />
      </svg>
    </button>

    {#if popoverOpen}
      <!-- Right-aligned so the popover doesn't overflow the viewport's right edge. -->
      <div
        role="menu"
        aria-label={m.locale.label}
        class="absolute right-0 top-full z-30 mt-2 min-w-32 rounded-lg border border-line bg-elevated p-1 shadow-popover"
      >
        {#each SUPPORTED_LOCALES as code (code)}
          {@const selected = locale.resolved === code}
          <button
            type="button"
            role="menuitemradio"
            aria-checked={selected}
            class="flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-sm font-medium transition {selected
              ? 'bg-surface-2 text-fg'
              : 'text-fg-secondary hover:bg-surface-2 hover:text-fg'}"
            onclick={() => selectMode(code)}
          >
            <span>{LOCALE_LABELS[code]}</span>
            <span class="font-mono text-[10px] text-fg-subtle">{LOCALE_CHIPS[code]}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
{/if}
