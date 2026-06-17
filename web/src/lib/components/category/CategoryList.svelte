<script lang="ts">
  import { untrack } from 'svelte';
  import CategoryRow from './CategoryRow.svelte';
  import AddCategoryDialog from './AddCategoryDialog.svelte';
  import DeleteCategoryDialog from './DeleteCategoryDialog.svelte';
  import RenameCategoryDialog from './RenameCategoryDialog.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import LoadingRow from '$lib/components/LoadingRow.svelte';
  import ContextMenu, { type MenuSection } from '$lib/components/ui/ContextMenu.svelte';
  import { categories, type Category } from '$lib/stores/categories.svelte';
  import { slices } from '$lib/stores/slices.svelte';
  import { isMandatoryCategory } from './labels';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';

  interface Props {
    workspaceId: Uuid;
    // Compared against the persisted workspace_sync record on bulk mount; equality skips every per-category GET.
    workspaceRevision: number;
    // Export-filename label; defaults so the list renders before the detail fetch resolves.
    workspaceName?: string;
  }
  let { workspaceId, workspaceRevision, workspaceName = 'workspace' }: Props = $props();

  // Tracked isStale read is the dependency (mount, workspaceId change, poller flip); untrack
  // wraps refresh so its writes to the same slice it reads don't re-queue this effect forever.
  $effect(() => {
    const id = workspaceId;
    void categories.isStale(id);
    untrack(() => {
      void categories.refresh(id);
    });
  });

  const slice = $derived(categories.for(workspaceId));

  // One IDB walk partitions every workspace slice into per-category badges and auto-resumes
  // cross-reload pending uploads; the store's workspacesLoaded set makes re-fires idempotent,
  // and untrack avoids the same store-writes-re-queue-effect loop as above.
  $effect(() => {
    const id = workspaceId;
    const cats = slice.entries;
    const rev = workspaceRevision;
    if (cats.length === 0) return;
    const names = cats.map((c) => c.name);
    untrack(() => {
      void slices.refreshForWorkspace(id, names, rev);
    });
  });

  let addOpen = $state(false);
  let deleteTarget = $state<Category | null>(null);
  let renameTarget = $state<Category | null>(null);

  let menuOpen = $state(false);
  let menuX = $state(0);
  let menuY = $state(0);
  // Set for kebab-button opens; null for cursor (right-click) opens.
  let menuAnchor = $state<DOMRect | null>(null);
  // The kebab that opened the menu; ContextMenu excludes it from outside-click dismiss so a
  // re-click can toggle the menu closed (see onRowMenu).
  let menuTrigger = $state<HTMLElement | null>(null);
  // Drives the kebab's aria-expanded; null when closed or anchored to the empty-list Add area.
  let activeMenuCat = $state<string | null>(null);
  let menuSections = $state<MenuSection[]>([]);

  function onListContextMenu(e: MouseEvent): void {
    const target = e.target as HTMLElement | null;
    if (!target) return;
    if (target.closest('input, textarea')) return;
    const rowEl = target.closest<HTMLElement>('[data-category-name]');
    const name = rowEl?.dataset.categoryName ?? null;
    const cat = name ? (slice.entries.find((c) => c.name === name) ?? null) : null;
    const sections = buildMenu(cat);
    if (sections.length === 0) return;
    e.preventDefault();
    // Inner handler wins so the page-root workspace menu doesn't also open at this cursor.
    e.stopPropagation();
    menuAnchor = null;
    menuTrigger = null;
    activeMenuCat = cat?.name ?? null;
    menuX = e.clientX;
    menuY = e.clientY;
    menuSections = sections;
    menuOpen = true;
  }

  // Re-clicking the kebab while its own menu is open toggles closed (the trigger is excluded
  // from outside-dismiss, so this click must decide; otherwise dismiss would race the reopen).
  function onRowMenu(cat: Category, trigger: HTMLElement): void {
    if (menuOpen && activeMenuCat === cat.name) {
      closeMenu();
      return;
    }
    const sections = buildMenu(cat);
    if (sections.length === 0) return;
    menuAnchor = trigger.getBoundingClientRect();
    menuTrigger = trigger;
    activeMenuCat = cat.name;
    menuSections = sections;
    menuOpen = true;
  }

  function closeMenu(): void {
    menuOpen = false;
    menuAnchor = null;
    menuTrigger = null;
    activeMenuCat = null;
  }

  function buildMenu(cat: Category | null): MenuSection[] {
    const t = m.category.list;
    if (cat) {
      const mandatory = isMandatoryCategory(cat.name);
      const deleting = slice.deleting.has(cat.name);
      // Block rename during an in-flight upload/delete: both bake the old category name into the
      // daemon path, so renaming mid-flight would re-create or orphan the old directory. Mirrors
      // the store-level gate in categories.rename.
      const status = slices.syncStatusFor(workspaceId, cat.name);
      const busy =
        status === 'uploading' ||
        status === 'pending' ||
        status === 'failed' ||
        slices.hasInflightDeletes(workspaceId, cat.name);
      return [
        {
          items: [
            {
              label: t.menu_rename,
              disabled: mandatory || deleting || busy,
              hint: mandatory ? t.menu_hint_preserved : busy ? t.menu_rename_hint_busy : undefined,
              onclick: () => (renameTarget = cat)
            },
            {
              label: t.menu_delete,
              variant: 'destructive',
              disabled: mandatory || deleting,
              hint: mandatory ? t.menu_hint_preserved : undefined,
              onclick: () => (deleteTarget = cat)
            }
          ]
        }
      ];
    }
    return [
      {
        items: [
          {
            label: t.menu_add,
            onclick: () => (addOpen = true)
          }
        ]
      }
    ];
  }
</script>

<section class="rounded-xl border border-line bg-surface px-5 pt-3.5 pb-5 shadow-card">
  <!-- items-center keeps the Add button centred against the title+description block however the
       description wraps on narrow viewports. -->
  <header class="mb-3 flex items-center justify-between gap-3">
    <div class="min-w-0">
      <h2 class="text-sm font-semibold text-fg">{m.category.list.heading}</h2>
      <p class="mt-0.5 text-xs text-fg-muted">
        {m.category.list.description}
      </p>
    </div>
    <Button onclick={() => (addOpen = true)} ariaLabel={m.category.list.add_button_aria}>
      {m.category.list.add_button}
    </Button>
  </header>

  {#if !slice.loaded}
    <LoadingRow size="section" label={m.category.list.loading} />
  {:else if slice.error && slice.entries.length === 0}
    <div
      class="rounded-md border border-warning-line bg-warning-soft px-3 py-2 text-xs text-warning-soft-fg"
      role="alert"
    >
      {m.category.list.load_error(slice.error)}
    </div>
  {:else}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div oncontextmenu={onListContextMenu}>
      <ul class="flex flex-col gap-2">
        {#each slice.entries as cat (cat.name)}
          <CategoryRow
            {workspaceId}
            {workspaceName}
            category={cat}
            expanded={slice.expandedName === cat.name}
            onMenu={(el: HTMLElement) => onRowMenu(cat, el)}
            menuOpen={menuOpen && activeMenuCat === cat.name}
          />
        {/each}
      </ul>
    </div>
  {/if}
</section>

<AddCategoryDialog open={addOpen} {workspaceId} onclose={() => (addOpen = false)} />

{#if deleteTarget}
  <DeleteCategoryDialog
    open={true}
    {workspaceId}
    categoryName={deleteTarget.name}
    origin={deleteTarget.origin}
    onclose={() => (deleteTarget = null)}
  />
{/if}

{#if renameTarget}
  <RenameCategoryDialog
    open={true}
    {workspaceId}
    currentName={renameTarget.name}
    existingNames={slice.entries.map((c) => c.name)}
    onclose={() => (renameTarget = null)}
  />
{/if}

<ContextMenu
  open={menuOpen}
  x={menuX}
  y={menuY}
  anchorRect={menuAnchor}
  triggerEl={menuTrigger}
  sections={menuSections}
  onclose={closeMenu}
/>
