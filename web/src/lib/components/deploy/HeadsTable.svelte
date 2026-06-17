<script lang="ts">
  import { onDestroy } from 'svelte';
  import { fade } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import ContextMenu, { type MenuSection } from '$lib/components/ui/ContextMenu.svelte';
  import StatusBadge from '$lib/components/ui/StatusBadge.svelte';
  import Spinner from '$lib/components/Spinner.svelte';
  import { config as configStore } from '$lib/stores/config.svelte';
  import { errorCopy } from '$lib/utils/error-copy';
  import { exportHead } from '$lib/api/heads-export';
  import HeadRow from './HeadRow.svelte';
  import DeleteHeadDialog from './DeleteHeadDialog.svelte';
  import { m } from '$lib/i18n';
  import type { ActiveResp, HeadRecord, Uuid } from '$lib/api/types';

  // `onchanged` runs post-swap and must not throw: a refresh error would misread as a deploy failure.

  interface Props {
    workspaceId: Uuid;
    // Seeds the alpkg export filename slug and manifest `source.workspace_name`; missing name falls
    // back to the literal slug `workspace`, yielding `workspace-head-<id8>.alpkg`.
    workspaceName: string;
    heads: readonly HeadRecord[];
    liveRevision: number;
    onchanged: () => Promise<void> | void;
  }
  let { workspaceId, workspaceName, heads, liveRevision, onchanged }: Props = $props();

  // Mirror of the daemon's `MAX_HEADS_PER_WORKSPACE` rotation cap; no API exposes it.
  const HEAD_HISTORY_CAP = 3;

  const active = $derived(configStore.active);
  // Null (no row highlighted) when the active head belongs to another workspace.
  const deployedHeadId = $derived<Uuid | null>(
    active?.origin === 'head' && active.source_workspace_id === workspaceId
      ? active.source_head_id
      : null
  );
  const defaultDeployed = $derived(active?.origin === 'default');

  // Newest-first; strict-weak comparator keeps order stable when two heads share `created_at`.
  const ordered = $derived(
    heads
      .slice()
      .sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0))
  );

  // Newest head at the live workspace revision (drives the "Latest" pill).
  const latestHeadId = $derived.by<Uuid | null>(() => {
    for (const h of ordered) {
      if (h.workspace_revision.id === liveRevision) return h.head_id;
    }
    return null;
  });

  let busyHeadId = $state<Uuid | null>(null);
  let deployingDefault = $state(false);
  let exportingHeadId = $state<Uuid | null>(null);

  let deleteOpen = $state(false);
  let deleteHead = $state<HeadRecord | null>(null);

  // Cleared at the start of the next action so a retry isn't shadowed by a stale banner.
  type ActionError =
    | { kind: 'deploy-head'; headId: Uuid; message: string }
    | { kind: 'deploy-default'; message: string }
    | { kind: 'export-head'; headId: Uuid; message: string };
  let actionError = $state<ActionError | null>(null);

  // Pre-deploy active record for one-step revert (daemon active is single-slot); clears on workspace
  // swap (parent keys by id).
  let previousActive = $state<ActiveResp | null>(null);

  // Revert target to stash for a pending deploy, or null when it won't change runtime state (redundant
  // deploy on the active target, or default-while-default). Returns rather than assigns so the caller
  // commits only AFTER the deploy lands: eager assignment briefly makes `previousActive ===
  // configStore.active`, which `showRevert` reads as "nothing to roll back to" (flicker off).
  function priorRevertTarget(
    intent: { kind: 'head'; headId: Uuid } | { kind: 'default' }
  ): ActiveResp | null {
    const cur = configStore.active;
    if (cur === null) return null;
    if (intent.kind === 'head') {
      if (cur.origin === 'head' && cur.source_head_id === intent.headId) return null;
    } else if (cur.origin === 'default') {
      return null;
    }
    return cur;
  }

  async function deployHead(headId: Uuid): Promise<void> {
    if (interactionBlocked) return;
    busyHeadId = headId;
    actionError = null;
    const prior = priorRevertTarget({ kind: 'head', headId });
    try {
      await configStore.activateHead(workspaceId, headId);
      if (prior !== null) previousActive = prior;
      await onchanged();
    } catch (e) {
      actionError = { kind: 'deploy-head', headId, message: errorCopy(e) };
    } finally {
      if (busyHeadId === headId) busyHeadId = null;
    }
  }

  async function deployDefault(): Promise<void> {
    if (interactionBlocked) return;
    deployingDefault = true;
    actionError = null;
    const prior = priorRevertTarget({ kind: 'default' });
    try {
      await configStore.activateDefault();
      if (prior !== null) previousActive = prior;
      await onchanged();
    } catch (e) {
      actionError = { kind: 'deploy-default', message: errorCopy(e) };
    } finally {
      deployingDefault = false;
    }
  }

  // Dispatch against the prior record's own workspace id (not this `workspaceId`) so a
  // cross-workspace prior reverts correctly.
  async function revert(): Promise<void> {
    const prev = previousActive;
    if (prev === null) return;
    if (interactionBlocked) return;
    if (prev.origin === 'head') {
      const headId = prev.source_head_id;
      const wsId = prev.source_workspace_id;
      busyHeadId = headId;
      actionError = null;
      // Next revert target, committed only after success.
      const rolledBackFrom = configStore.active;
      try {
        await configStore.activateHead(wsId, headId);
        previousActive = rolledBackFrom;
        await onchanged();
      } catch (e) {
        actionError = { kind: 'deploy-head', headId, message: errorCopy(e) };
      } finally {
        if (busyHeadId === headId) busyHeadId = null;
      }
    } else {
      deployingDefault = true;
      actionError = null;
      const rolledBackFrom = configStore.active;
      try {
        await configStore.activateDefault();
        previousActive = rolledBackFrom;
        await onchanged();
      } catch (e) {
        actionError = { kind: 'deploy-default', message: errorCopy(e) };
      } finally {
        deployingDefault = false;
      }
    }
  }

  let exportAbort: AbortController | null = null;

  // Export is daemon-read-only, so the `interactionBlocked` gate is purely a client guardrail
  // against two concurrent SaveAs dialogs.
  async function exportHeadAction(head: HeadRecord): Promise<void> {
    if (interactionBlocked) return;
    exportingHeadId = head.head_id;
    actionError = null;
    const ctrl = new AbortController();
    exportAbort = ctrl;
    try {
      await exportHead({ workspaceId, workspaceName, head }, { signal: ctrl.signal });
    } catch (e) {
      // Our own teardown abort is not an operator-facing error.
      if (!ctrl.signal.aborted) {
        actionError = { kind: 'export-head', headId: head.head_id, message: errorCopy(e) };
      }
    } finally {
      if (exportAbort === ctrl) exportAbort = null;
      if (exportingHeadId === head.head_id) exportingHeadId = null;
    }
  }

  // Abort in-flight export on unmount so an orphaned download doesn't fire for the workspace we left.
  onDestroy(() => exportAbort?.abort());

  const revertLabel = $derived<string | null>(
    previousActive === null
      ? null
      : previousActive.origin === 'default'
        ? m.deploy.heads_table.revert_to_default
        : m.deploy.heads_table.revert_to_id(previousActive.source_head_id.slice(0, 8))
  );

  // Hidden when the prior IS the currently active record (nothing to roll back to).
  const showRevert = $derived.by(() => {
    if (previousActive === null) return false;
    const cur = configStore.active;
    if (cur === null) return true;
    if (previousActive.origin === 'default') return cur.origin !== 'default';
    return !(
      cur.origin === 'head' &&
      cur.source_head_id === previousActive.source_head_id &&
      cur.source_workspace_id === previousActive.source_workspace_id
    );
  });

  function dismissActionError(): void {
    actionError = null;
  }

  function requestDelete(head: HeadRecord): void {
    if (interactionBlocked) return;
    deleteHead = head;
    deleteOpen = true;
  }

  function onDeleteClose(): void {
    deleteOpen = false;
  }

  async function onDeleted(deletedId: Uuid): Promise<void> {
    // Clear the revert target if it pointed at the deleted head so the affordance can't dangle.
    if (
      previousActive !== null &&
      previousActive.origin === 'head' &&
      previousActive.source_head_id === deletedId
    ) {
      previousActive = null;
    }
    await onchanged();
  }

  // Any in-flight deploy/default/export blocks every sibling action; handlers clear their slot in
  // `finally` so a throw can't strand it.
  const interactionBlocked = $derived(
    busyHeadId !== null || deployingDefault || exportingHeadId !== null
  );

  const canDeployDefault = $derived(!defaultDeployed && !interactionBlocked);

  // Lags `defaultDeployed` by the spinner's out:fade so spinner and Active badge never overlap on a
  // mid-deploy flip; initial value mirrors `defaultDeployed` so an already-default list paints the
  // badge at once, and revert (true -> false) is immediate.
  const DEFAULT_SPINNER_OUT_DURATION_MS = 140;
  // svelte-ignore state_referenced_locally
  let defaultActiveMounted = $state(defaultDeployed);
  let defaultActiveMountTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    const target = defaultDeployed;
    if (target === defaultActiveMounted) return;
    if (target) {
      defaultActiveMountTimer = setTimeout(() => {
        defaultActiveMounted = true;
        defaultActiveMountTimer = null;
      }, DEFAULT_SPINNER_OUT_DURATION_MS);
    } else {
      defaultActiveMounted = false;
    }
    return (): void => {
      if (defaultActiveMountTimer !== null) {
        clearTimeout(defaultActiveMountTimer);
        defaultActiveMountTimer = null;
      }
    };
  });

  // One ContextMenu; the scroller delegates `oncontextmenu` and walks `data-head-id` to the row. A
  // right-click outside any head row leaves `preventDefault` un-called, so the workspace page's own
  // context menu takes over there.
  let menuOpen = $state(false);
  let menuX = $state(0);
  let menuY = $state(0);
  let menuSections = $state<MenuSection[]>([]);

  function onListContextMenu(e: MouseEvent): void {
    const target = e.target as HTMLElement | null;
    if (!target) return;
    const rowEl = target.closest<HTMLElement>('[data-head-id]');
    const headId = rowEl?.dataset.headId ?? null;
    const head = headId ? (heads.find((h) => h.head_id === headId) ?? null) : null;
    if (head === null) return;
    const sections = buildMenu(head);
    if (sections.length === 0) return;
    e.preventDefault();
    // Stop the workspace page's root `oncontextmenu` from also opening here.
    e.stopPropagation();
    menuX = e.clientX;
    menuY = e.clientY;
    menuSections = sections;
    menuOpen = true;
  }

  function buildMenu(head: HeadRecord): MenuSection[] {
    const isThisDeployed = deployedHeadId === head.head_id;
    const isThisExporting = exportingHeadId === head.head_id;
    // Deploy disabled on the active head (daemon no-op) or any sibling mutation; Export by ANOTHER
    // mutation and also while THIS row exports (which additionally swaps the label to "exporting");
    // Delete on the active head (daemon 409s) or any mutation.
    const deployDisabled = isThisDeployed || interactionBlocked;
    const deployHint = isThisDeployed ? m.deploy.heads_table.menu_hint_active : undefined;
    const exportDisabled = interactionBlocked && !isThisExporting;
    const deleteDisabled = isThisDeployed || interactionBlocked;
    const deleteHint = isThisDeployed ? m.deploy.heads_table.menu_hint_deployed : undefined;
    const t = m.deploy.heads_table;
    return [
      {
        items: [
          {
            label: t.menu_deploy,
            disabled: deployDisabled,
            hint: deployHint,
            onclick: () => void deployHead(head.head_id)
          },
          {
            label: isThisExporting ? t.menu_exporting : t.menu_export,
            disabled: exportDisabled || isThisExporting,
            onclick: () => void exportHeadAction(head)
          },
          {
            label: t.menu_delete,
            variant: 'destructive',
            disabled: deleteDisabled,
            hint: deleteHint,
            onclick: () => {
              requestDelete(head);
            }
          }
        ]
      }
    ];
  }
</script>

<!-- Fills the parent's h-80 (320 px) budget so the inner scroller absorbs a long list without pushing
     action chrome below the fold; `overflow-hidden` clips the scroller's rounded inner edge. -->
<section
  class="flex h-full min-h-0 flex-col overflow-hidden rounded-md border border-line bg-surface px-3 pt-1.5 pb-3"
>
  <!-- `min-h-5.25` (21 px) so the optional revert button mounting doesn't reflow the header; `mb-1`
       offsets the +2 px to keep header+margin at 25 px, preserving the scroller's 277 px capacity
       (cap=3 + default fallback ~274 px must fit). -->
  <header class="@container/heads mb-1 flex min-h-5.25 items-center justify-between gap-1.5">
    <!-- Intentionally NO `translate-y-px`: items-center on this 21 px header already lands the heading
         cap 1 px lower than sibling panes' 19 px+translate headers; adding it would compound. -->
    <div class="flex items-baseline gap-1.5">
      <h4 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
        {m.deploy.heads_table.heading}
      </h4>
      <!-- Suffix hides below @[18rem]/heads only while the revert button is present (they'd compete on
           a narrow card). `&& revertLabel` isn't a visibility term but narrows `revertLabel` non-null
           for the button label below - keep it. -->
      <span class="text-[10px] text-fg-subtle tabular-nums">
        {m.deploy.heads_table.count_label(heads.length)}<span
          class={showRevert && revertLabel ? 'hidden @[18rem]/heads:inline' : 'inline'}
          >{m.deploy.heads_table.count_retained(HEAD_HISTORY_CAP)}</span
        >
      </span>
    </div>
    {#if showRevert && revertLabel}
      <!-- `-translate-y-px` centres the 2 px-top-heavy button via transform (not margin) so the layout
           pass keeps it at items-center and the scroller budget reads from flow. -->
      <button
        type="button"
        onclick={revert}
        disabled={interactionBlocked}
        class="inline-flex shrink-0 -translate-y-px items-center rounded-md border border-line bg-surface px-1.5 py-0.5 text-[10px] leading-tight font-medium text-fg-secondary transition duration-200 ease-out hover:border-line-strong hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-60"
        title={m.deploy.heads_table.revert_title}
      >
        {revertLabel}
      </button>
    {/if}
  </header>

  <!-- `pr-1`/`-mr-1` keep the scrollbar off row borders while the right inset still matches `px-3`. No
       empty-state notice: at zero heads the always-present fallback row is the only deploy target.
       `oncontextmenu` delegated here (not per-row) so right-clicks outside a head row fall through to
       the workspace page's menu. -->
  <div
    class="-mr-1 min-h-0 flex-1 overflow-y-auto pr-1"
    oncontextmenu={onListContextMenu}
    role="presentation"
  >
    <ul class="flex flex-col gap-2">
      {#each ordered as head (head.head_id)}
        <HeadRow
          {head}
          {workspaceId}
          isLatest={head.head_id === latestHeadId}
          isDeployed={deployedHeadId === head.head_id}
          busy={interactionBlocked &&
            busyHeadId !== head.head_id &&
            exportingHeadId !== head.head_id}
          isExporting={exportingHeadId === head.head_id}
          isDeploying={busyHeadId === head.head_id}
          ondeploy={deployHead}
          onexport={exportHeadAction}
        />
      {/each}

      <!-- Daemon-default fallback row, always present as the escape hatch when every head is unfit; the
           whole row is the click target. Interactive role sits on the inner `<div>`, not the `<li>`:
           Svelte's a11y lint refuses it on a list element. -->
      <li>
        <div
          role="button"
          tabindex={canDeployDefault ? 0 : -1}
          aria-disabled={!canDeployDefault}
          aria-label={defaultDeployed
            ? m.deploy.heads_table.default_aria_active
            : m.deploy.heads_table.default_aria_deploy}
          title={defaultDeployed
            ? m.deploy.heads_table.default_title_active
            : deployingDefault
              ? m.deploy.heads_table.default_title_deploying
              : interactionBlocked
                ? m.deploy.heads_table.default_title_busy
                : m.deploy.heads_table.default_title_idle}
          onclick={() => void deployDefault()}
          onkeydown={(e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            if (!canDeployDefault) return;
            e.preventDefault();
            void deployDefault();
          }}
          class="flex flex-wrap items-center justify-between gap-3 rounded-md border border-dashed px-3 py-2.5 transition-colors focus-visible:ring-2 focus-visible:ring-accent-line focus-visible:outline-none"
          class:border-accent={defaultDeployed}
          class:bg-accent-soft={defaultDeployed}
          class:border-line-strong={!defaultDeployed}
          class:bg-surface-2={!defaultDeployed}
          class:hover:border-accent={canDeployDefault}
          class:hover:bg-accent-soft={canDeployDefault}
          class:cursor-pointer={canDeployDefault}
          class:cursor-wait={deployingDefault}
          class:cursor-default={defaultDeployed && !deployingDefault}
          class:cursor-not-allowed={interactionBlocked && !defaultDeployed && !deployingDefault}
        >
          <div class="min-w-0 flex-1">
            <div class="flex flex-wrap items-center gap-x-2 gap-y-1">
              <p class="text-sm font-semibold text-fg">
                {m.deploy.heads_table.default_row_headline}
              </p>
              {#if deployingDefault && !defaultDeployed}
                <!-- Spinner in the slot the Active badge takes once the deploy lands; the
                     `defaultActiveMounted` lag keeps the two from overlapping. -->
                <span
                  in:fade={{ duration: 180, easing: cubicOut }}
                  out:fade={{ duration: 140, easing: cubicOut }}
                  aria-hidden="true"
                  class="pointer-events-none inline-flex shrink-0 items-center justify-center"
                >
                  <Spinner class="h-3.5 w-3.5 text-accent" />
                </span>
              {/if}
              {#if defaultActiveMounted}
                <StatusBadge
                  size="xs"
                  label={m.deploy.head_row.pill_active}
                  tone="accent"
                  title={m.deploy.heads_table.default_active_title}
                />
              {/if}
            </div>
            <!-- `truncate`: wrapping on a narrow card would grow this always-present row past the
                 scroller's height budget, and the text is non-essential. -->
            <p class="mt-1 truncate text-[11px] text-fg-muted">
              {m.deploy.heads_table.default_row_description}
            </p>
          </div>
        </div>
      </li>
    </ul>
  </div>

  <!-- Pinned below the scroller so a failure doesn't shove still-correct heads down. Layout branches
       on whether the message has text: multi-line (real failure) uses `items-start` + corner-pinned
       dismiss; single-line (blank message) uses `items-center` + asymmetric `pl-2.5`. -->
  {#if actionError}
    <!-- Snapshot into a local so the kind-discriminated branches narrow once; re-reading the `$state`
         proxy per access would widen back to the union and lose `.headId`. -->
    {@const err = actionError}
    {@const hasMessage = err.message.trim().length > 0}
    <div
      in:fade={{ duration: 200, easing: cubicOut }}
      out:fade={{ duration: 160, easing: cubicOut }}
      class="mt-1.5 flex justify-between gap-2 rounded-md border border-danger-line bg-danger-soft text-xs text-danger-soft-fg"
      class:items-start={hasMessage}
      class:items-center={!hasMessage}
      class:px-3={hasMessage}
      class:py-2={hasMessage}
      class:py-1={!hasMessage}
      class:pr-1={!hasMessage}
      class:pl-2.5={!hasMessage}
      role="alert"
    >
      <div class="min-w-0">
        <p class="font-medium">
          {#if err.kind === 'deploy-head'}
            {m.deploy.heads_table.error_deploy_head}
            <span class="font-mono text-[10px]" title={err.headId}>
              {err.headId.slice(0, 8)}
            </span>
          {:else if err.kind === 'export-head'}
            {m.deploy.heads_table.error_export_head}
            <span class="font-mono text-[10px]" title={err.headId}>
              {err.headId.slice(0, 8)}
            </span>
          {:else}
            {m.deploy.heads_table.error_deploy_default}
          {/if}
        </p>
        {#if hasMessage}
          <p class="mt-0.5 wrap-break-word">{err.message}</p>
        {/if}
      </div>
      <button
        type="button"
        onclick={dismissActionError}
        aria-label={m.common.dismiss}
        class="shrink-0 rounded-md p-1 text-danger-soft-fg transition hover:bg-danger-soft"
        class:-mt-1={hasMessage}
        class:-mr-2={hasMessage}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          class="h-3.5 w-3.5"
          aria-hidden="true"
        >
          <path d="M6 6l12 12M6 18L18 6" />
        </svg>
      </button>
    </div>
  {/if}
</section>

<DeleteHeadDialog
  open={deleteOpen}
  {workspaceId}
  head={deleteHead}
  onclose={onDeleteClose}
  ondeleted={onDeleted}
/>

<!-- Rendered at body end so its `position: fixed` chrome (own `z-50`) paints above the card and tab
     strip without extra stacking discipline. -->
<ContextMenu
  open={menuOpen}
  x={menuX}
  y={menuY}
  sections={menuSections}
  onclose={() => (menuOpen = false)}
/>
