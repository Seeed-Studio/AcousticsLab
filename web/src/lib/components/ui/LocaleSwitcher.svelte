<script lang="ts">
  import { locale, type LocaleMode } from '$lib/stores/locale.svelte';
  import { m, SUPPORTED_LOCALES, LOCALE_LABELS, LOCALE_CHIPS, type LocaleCode } from '$lib/i18n';

  // Popover (not a segmented pill) so the picker scales past 2-3 locales without segment-width
  // pressure; auto-hidden while only one locale is registered (a 1-option picker is dead chrome).

  let popoverOpen = $state(false);
  let wrapper = $state<HTMLDivElement | undefined>();

  // Lint sees `1 > 1` (dead) while one locale ships, but the check goes live when a second
  // registers; widening the `as const` tuple to silence it loses Record<LocaleCode,_> exhaustiveness.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  const hasMultipleLocales = $derived(SUPPORTED_LOCALES.length > 1);

  // Off `resolved` not `mode` so in 'auto' the chip names the detected language, not "Auto".
  const currentChip = $derived(LOCALE_CHIPS[locale.resolved]);

  // Param is `mode` not `m` to avoid shadowing the imported `m` i18n proxy.
  function selectMode(mode: LocaleMode): void {
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
    <!-- Two responsive triggers for one popover: sm+ chip, <sm globe icon (toggled by sm: classes). -->
    <button
      type="button"
      class="hidden h-[30px] items-center gap-1 rounded-full border border-line bg-surface px-2.5 text-xs font-medium text-fg-secondary shadow-card transition hover:border-line-strong hover:text-fg sm:inline-flex"
      aria-expanded={popoverOpen}
      aria-haspopup="menu"
      onclick={togglePopover}
    >
      <span>{currentChip}</span>
      <svg
        viewBox="0 0 20 20"
        fill="currentColor"
        class="h-3 w-3 text-fg-muted transition-transform duration-150"
        class:rotate-180={popoverOpen}
        aria-hidden="true"
      >
        <path
          fill-rule="evenodd"
          d="M5.23 7.21a.75.75 0 011.06.02L10 11.06l3.71-3.83a.75.75 0 111.08 1.04l-4.25 4.4a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z"
          clip-rule="evenodd"
        />
      </svg>
    </button>

    <button
      type="button"
      class="inline-flex h-[36px] w-[36px] items-center justify-center rounded-full border border-line bg-surface text-fg-muted shadow-card transition hover:border-line-strong hover:text-fg sm:hidden"
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
        class="h-4 w-4"
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
        <!-- 'auto' (follow-browser) row; aria-checked off `mode` not `resolved` so it reads checked
             only when no language is pinned. Predicate inlined: `{@const}` can't be a direct child here. -->
        <button
          type="button"
          role="menuitemradio"
          aria-checked={locale.mode === 'auto'}
          class="flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-sm font-medium transition {locale.mode ===
          'auto'
            ? 'bg-surface-2 text-fg'
            : 'text-fg-secondary hover:bg-surface-2 hover:text-fg'}"
          onclick={() => selectMode('auto')}
        >
          <span>{m.locale.auto_label}</span>
        </button>
        {#each SUPPORTED_LOCALES as code (code)}
          <!-- Off `mode` not `resolved`, so in 'auto' the detected row isn't announced selected. -->
          {@const selected = locale.mode === code}
          <button
            type="button"
            role="menuitemradio"
            aria-checked={selected}
            class="flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-sm font-medium transition {selected
              ? 'bg-surface-2 text-fg'
              : 'text-fg-secondary hover:bg-surface-2 hover:text-fg'}"
            onclick={() => selectMode(code satisfies LocaleCode)}
          >
            <span>{LOCALE_LABELS[code]}</span>
            <span class="font-mono text-[10px] text-fg-subtle">{LOCALE_CHIPS[code]}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
{/if}
