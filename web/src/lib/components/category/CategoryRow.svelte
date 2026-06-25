<script lang="ts">
  import { slide, fade, scale } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import { categories, type Category } from '$lib/stores/categories.svelte';
  import { slices, type CategorySyncStatus } from '$lib/stores/slices.svelte';
  import { isMandatoryCategory, prettyCategoryName, thresholdFor } from './labels';
  import InputPane from './InputPane.svelte';
  import SlicePane from './SlicePane.svelte';
  import Spinner from '$lib/components/Spinner.svelte';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';

  // Body renders only when `expanded`; header click delegates to the store, which single-expands
  // by collapsing other open rows. `workspaceName` is threaded only for the export filename.
  interface Props {
    workspaceId: Uuid;
    workspaceName: string;
    category: Category;
    expanded: boolean;
    // Kebab passes its own element so the parent (owner of menu/dialogs/gating) anchors the menu
    // under it and excludes it from outside-click dismiss, so re-clicking toggles closed.
    onMenu?: (trigger: HTMLElement) => void;
    menuOpen?: boolean;
  }
  let {
    workspaceId,
    workspaceName,
    category,
    expanded,
    onMenu,
    menuOpen = false
  }: Props = $props();

  const slice = $derived(categories.for(workspaceId));
  const isDeleting = $derived(slice.deleting.has(category.name));
  const display = $derived(prettyCategoryName(category.name));
  // Preserved (mandatory) categories can't be renamed or deleted: kebab is slashed/disabled.
  const isMandatory = $derived(isMandatoryCategory(category.name));

  const sliceCount = $derived(slices.countFor(workspaceId, category.name));
  const threshold = $derived(thresholdFor(category.name));
  const syncStatus = $derived(slices.syncStatusFor(workspaceId, category.name));

  // Row badge keying tone + text off the combined (quota, sync) state.
  type BadgeTone = 'emerald' | 'amber' | 'rose';
  type BadgeIcon = 'check' | null;
  interface Badge {
    tone: BadgeTone;
    icon: BadgeIcon;
    text: string;
    title: string;
  }

  function computeBadge(count: number, N: number, status: CategorySyncStatus): Badge {
    const t = m.category.row;
    const satisfied = count >= N;
    const tally = `${count}/${N}`;
    if (status === 'failed') {
      return {
        tone: 'rose',
        icon: null,
        text: satisfied ? t.badge_failed : t.badge_not_enough_with_state(t.badge_failed),
        title: t.title_failed(tally)
      };
    }
    if (satisfied) {
      if (status === 'synced') {
        return {
          tone: 'emerald',
          icon: 'check',
          text: t.badge_synced,
          title: t.title_synced(tally)
        };
      }
      if (status === 'uploading') {
        return {
          tone: 'amber',
          icon: null,
          text: t.badge_uploading,
          title: t.title_uploading(tally)
        };
      }
      return {
        tone: 'amber',
        icon: null,
        text: t.badge_pending,
        title: t.title_pending(tally)
      };
    }
    if (status === 'empty') {
      return {
        tone: 'amber',
        icon: null,
        text: t.badge_not_enough,
        title: t.title_not_enough_empty(N - count, tally)
      };
    }
    const statusLabel =
      status === 'synced'
        ? t.badge_synced
        : status === 'uploading'
          ? t.badge_uploading
          : t.badge_pending;
    return {
      tone: 'amber',
      icon: null,
      text: t.badge_not_enough_with_state(statusLabel),
      title:
        status === 'synced'
          ? t.title_not_enough_synced(tally, N - count)
          : status === 'uploading'
            ? t.title_not_enough_uploading(tally, N - count)
            : t.title_not_enough_pending(tally, N - count)
    };
  }

  const badge = $derived(computeBadge(sliceCount, threshold, syncStatus));

  // An off-screen mirror measures the label so the wrapper can CSS-interpolate width between
  // labels; else the {#key}-swap `inline-grid` cell sizes to max(old, new) and jolts wider then back.
  let measureEl: HTMLSpanElement | undefined = $state();
  let textWidth: number | null = $state(null);
  $effect(() => {
    void badge.text;
    if (!measureEl) return;
    textWidth = measureEl.getBoundingClientRect().width;
  });

  function onHeaderClick(): void {
    if (isDeleting) return;
    categories.toggleExpand(workspaceId, category.name);
  }

  function onHeaderKey(e: KeyboardEvent): void {
    // Ignore keys from the nested kebab button; only the header itself toggles expansion.
    if (e.target !== e.currentTarget) return;
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    onHeaderClick();
  }

  function onMenuClick(e: MouseEvent): void {
    // Stop propagation so the kebab click doesn't also toggle the row's expansion.
    e.stopPropagation();
    onMenu?.(e.currentTarget as HTMLElement);
  }
</script>

<li
  data-category-name={category.name}
  class="group/row overflow-hidden rounded-lg border bg-surface transition hover:shadow-card {expanded
    ? 'border-line-strong'
    : 'border-line hover:border-line-strong'}"
  class:opacity-60={isDeleting}
>
  <!-- Header and expanded body must share horizontal padding so the chevron aligns with the
       pane border below it. -->
  <div
    role="button"
    tabindex={isDeleting ? -1 : 0}
    aria-expanded={expanded}
    aria-controls="category-body-{category.name}"
    aria-disabled={isDeleting}
    onclick={onHeaderClick}
    onkeydown={onHeaderKey}
    class="flex cursor-pointer items-center gap-2 px-3 py-1.5 transition select-none"
    class:cursor-not-allowed={isDeleting}
    class:pointer-events-none={isDeleting}
  >
    <!-- Disclosure chevron. Path mass sits ~1px above viewBox centre, so collapsed it reads high
         under `items-center`; `translate-y-px` corrects rest, rotate-90 swings the bias horizontal. -->
    <svg
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
      class="h-3.5 w-3.5 shrink-0 text-fg-muted transition-transform duration-200"
      class:translate-y-px={!expanded}
      class:rotate-90={expanded}
    >
      <path
        fill-rule="evenodd"
        d="M7.21 5.23a.75.75 0 011.06.02L12 9l-3.73 3.71a.75.75 0 11-1.06-1.06L9.94 9 7.19 6.29a.75.75 0 01.02-1.06z"
        clip-rule="evenodd"
      />
    </svg>
    <!-- `flex-1 min-w-0` lets the name truncate while the shrink-0 kebab + badge stay anchored
         at the right edge. -->
    <h3 class="min-w-0 flex-1 truncate text-sm font-medium text-fg" title={category.name}>
      {display}
    </h3>
    <!-- Hover-revealed overflow menu: hidden at rest, shown on group-hover/focus-visible to keep
         resting chrome clean without losing the keyboard path. `pointer-coarse:` pins it visible
         on touch devices, which can neither hover nor right-click. Preserved category: disabled, drops aria-haspopup. -->
    {#if !isDeleting}
      <button
        type="button"
        onclick={onMenuClick}
        disabled={isMandatory}
        aria-haspopup={isMandatory ? undefined : 'menu'}
        aria-expanded={isMandatory ? undefined : menuOpen}
        class="pointer-events-none inline-flex shrink-0 items-center justify-center rounded-md p-1 text-fg-subtle opacity-0 transition duration-200 ease-out group-hover/row:pointer-events-auto group-hover/row:opacity-100 hover:bg-surface-2 hover:text-fg-secondary focus-visible:pointer-events-auto focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-fg-subtle pointer-coarse:pointer-events-auto pointer-coarse:opacity-100"
        aria-label={m.category.row.actions_aria(display)}
        title={isMandatory ? m.category.row.actions_title_preserved : m.category.row.actions_title}
      >
        <svg viewBox="0 0 24 24" fill="currentColor" class="h-3.5 w-3.5" aria-hidden="true">
          <circle cx="12" cy="5" r="2" />
          <circle cx="12" cy="12" r="2" />
          <circle cx="12" cy="19" r="2" />
          {#if isMandatory}
            <!-- Prohibition slash; thicker stroke reads over the dots it crosses. -->
            <line
              x1="4"
              y1="4"
              x2="20"
              y2="20"
              stroke="currentColor"
              stroke-width="2.5"
              stroke-linecap="round"
            />
          {/if}
        </svg>
      </button>
    {/if}
    <!-- Sync badge. The icon slot animates width + `mr-1` (on the slot, not parent gap, so the
         gap collapses too) to zero in iconless states; the text wrapper glides on the measured
         `textWidth`; the check scales at paint time so it never perturbs layout. -->
    {#if !isDeleting}
      <span
        class="hidden shrink-0 items-center justify-center overflow-hidden rounded-full px-1.5 py-0.5 text-[10px] font-medium transition-[background-color,color] duration-200 ease-out sm:inline-flex"
        class:bg-success-soft={badge.tone === 'emerald'}
        class:text-success-soft-fg={badge.tone === 'emerald'}
        class:bg-warning-soft={badge.tone === 'amber'}
        class:text-warning-soft-fg={badge.tone === 'amber'}
        class:bg-danger-soft={badge.tone === 'rose'}
        class:text-danger-soft-fg={badge.tone === 'rose'}
        title={badge.title}
      >
        <span
          class="inline-flex h-2.5 shrink-0 items-center justify-center overflow-hidden transition-[width,margin] duration-200 ease-out"
          class:w-2.5={badge.icon === 'check'}
          class:mr-1={badge.icon === 'check'}
          class:w-0={badge.icon !== 'check'}
          aria-hidden="true"
        >
          {#if badge.icon === 'check'}
            <span
              in:scale={{ duration: 240, start: 0.6, easing: cubicOut }}
              out:scale={{ duration: 180, start: 0.6, easing: cubicOut }}
              class="inline-flex"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="3"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="h-2.5 w-2.5"
              >
                <path d="M20 6L9 17l-5-5" />
              </svg>
            </span>
          {/if}
        </span>
        <span
          class="inline-flex items-center justify-center overflow-hidden transition-[width] duration-200 ease-out"
          style:width={textWidth !== null ? `${textWidth}px` : 'auto'}
        >
          <span class="inline-grid grid-cols-1 grid-rows-1 items-center">
            {#key badge.text}
              <span
                in:fade={{ duration: 180, easing: cubicOut }}
                out:fade={{ duration: 180, easing: cubicOut }}
                class="col-start-1 row-start-1 whitespace-nowrap"
              >
                {badge.text}
              </span>
            {/key}
          </span>
        </span>
      </span>
      <!-- Off-screen mirror feeding `textWidth`; its typography must match the visible label. -->
      <span
        bind:this={measureEl}
        aria-hidden="true"
        class="pointer-events-none invisible fixed top-0 left-0 whitespace-nowrap text-[10px] font-medium"
      >
        {badge.text}
      </span>
    {/if}
    {#if isDeleting}
      <span
        class="inline-flex shrink-0 items-center gap-1 rounded-full bg-danger-soft px-1.5 py-0.5 text-[10px] font-medium text-danger-soft-fg capitalize"
      >
        <Spinner class="h-2.5 w-2.5 text-danger-soft-fg" />
        {m.category.row.badge_deleting}
      </span>
    {/if}
  </div>

  {#if expanded}
    <!-- Slide animates the body, not the row's border, which would jitter against the parent gap. -->
    <div
      id="category-body-{category.name}"
      transition:slide={{ duration: 200, easing: cubicOut }}
      class="border-t border-line-subtle bg-surface-2 px-3 py-3"
    >
      <!-- Two-pane layout (Input | Slices at md+, stacked below). `min-h-80` + items-stretch welds
           both to one baseline so a new batch never grows the row (SlicePane scrolls internally).
           Panes use `contain: size` to zero intrinsic height (else the waveform's 2:1 aspect lifts
           the track each state change); since that makes stacked rows 0-content, each gets its own
           `minmax(16rem,1fr)` floor (256px). `md:grid-rows-1` collapses to one row at md+. -->
      <div
        class="grid min-h-80 grid-cols-1 grid-rows-[minmax(16rem,1fr)_minmax(16rem,1fr)] gap-3 md:grid-cols-2 md:grid-rows-1"
      >
        <InputPane {workspaceId} {workspaceName} categoryName={category.name} />
        <SlicePane {workspaceId} categoryName={category.name} />
      </div>
    </div>
  {/if}
</li>
