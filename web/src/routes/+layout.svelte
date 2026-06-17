<script lang="ts">
  import '../app.css';
  import { onDestroy, type Snippet } from 'svelte';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import { health } from '$lib/stores/health.svelte';
  import { config } from '$lib/stores/config.svelte';
  import { theme } from '$lib/stores/theme.svelte';
  import { locale } from '$lib/stores/locale.svelte';
  import { m } from '$lib/i18n';
  import HealthBadge from '$lib/components/HealthBadge.svelte';
  import ThemeToggle from '$lib/components/ui/ThemeToggle.svelte';
  import LocaleSwitcher from '$lib/components/ui/LocaleSwitcher.svelte';

  interface Props {
    children?: Snippet;
  }
  let { children }: Props = $props();

  // `$derived` (not const) so labels reconstitute on locale switch; a const would capture stale `m.*` at script-init.
  const TABS = $derived([
    { href: resolve('/'), label: m.nav.dashboard },
    { href: resolve('/workspaces'), label: m.nav.workspaces }
  ]);

  function isActive(href: string): boolean {
    const root = resolve('/');
    if (href === root) return page.url.pathname === root;
    return page.url.pathname === href || page.url.pathname.startsWith(href + '/');
  }

  let currentTabLabel = $derived(TABS.find((t) => isActive(t.href))?.label ?? m.nav.menu_fallback);

  // Below sm the tab row collapses to a "current tab + chevron" drop-down, dismissed on tap-outside, Escape, focusout, or route change.
  let mobileMenuOpen = $state(false);
  let mobileMenuWrapper = $state<HTMLDivElement | undefined>();

  function closeMobileMenu(): void {
    mobileMenuOpen = false;
  }
  function toggleMobileMenu(): void {
    mobileMenuOpen = !mobileMenuOpen;
  }
  function onMobileMenuFocusOut(e: FocusEvent): void {
    const next = e.relatedTarget as Node | null;
    if (!next || !mobileMenuWrapper?.contains(next)) closeMobileMenu();
  }

  // Plain `let`, not $state, so read/write inside the effect doesn't retrigger it: page.url.pathname is the only dependency.
  let lastPath: string | null = null;
  $effect(() => {
    const p = page.url.pathname;
    if (lastPath !== null && lastPath !== p) mobileMenuOpen = false;
    lastPath = p;
  });

  $effect(() => {
    if (!mobileMenuOpen) return;
    const onDown = (e: PointerEvent): void => {
      const root = mobileMenuWrapper;
      if (!root) return;
      if (!root.contains(e.target as Node | null)) closeMobileMenu();
    };
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') closeMobileMenu();
    };
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey);
    };
  });

  // Auto-reconnect throttle ref. Plain `let`, not `$state`: a reactive ref would re-enter the effect on every schedule, defeating the throttle.
  let pendingRefreshTimer: ReturnType<typeof setTimeout> | null = null;
  const RETRY_THROTTLE_MS = 2_000;

  // Bootstrap at script level, not onMount: child onMount fires bottom-up, so a parent onMount wouldn't have these
  // stores ready when child canvases/pollers mount (ssr=false keeps it browser-only). Streams are deliberately NOT started here: the
  // opus + inference WebSockets are refcount-gated via `streams.acquire()`, so routes that never read the live stream
  // avoid the worker's ~50 Hz Opus decode loop and ~4 KB/s daemon bandwidth (32 kbps encoder).
  health.start();
  theme.start();
  locale.start();
  void config.refresh();
  onDestroy(() => {
    health.stop();
    theme.stop();
    locale.stop();
    if (pendingRefreshTimer !== null) {
      clearTimeout(pendingRefreshTimer);
      pendingRefreshTimer = null;
    }
  });

  // Auto-reconnect REST config when the daemon returns (WS streams reconnect via the worker's own backoff),
  // piggybacking on health's 2 s poll; the `config.error` guard keeps it inert on the steady-state path
  // (`unhealthy` is a subsystem fault, still reachable, so it too retries). Throttled because a failed
  // `config.refresh` - or any action through `config.guard`, which also sets `config.error` - re-triggers this
  // effect synchronously; without the throttle a fast-failing daemon spins at network RTT and pins the browser
  // thread. The schedule is idempotent (re-fire while a timer is pending no-ops, preserving the original wait
  // window) and cancelled when the error clears or the daemon goes unreachable.
  $effect(() => {
    const level = health.level;
    const reachable = level === 'ok' || level === 'degraded' || level === 'unhealthy';

    if (!reachable || config.error === null) {
      // Recovered or unreachable: cancel any pending retry, else a late firing runs a redundant refresh or re-enters the loop.
      if (pendingRefreshTimer !== null) {
        clearTimeout(pendingRefreshTimer);
        pendingRefreshTimer = null;
      }
      return;
    }

    if (pendingRefreshTimer !== null) return;

    pendingRefreshTimer = setTimeout(() => {
      pendingRefreshTimer = null;
      void config.refresh();
    }, RETRY_THROTTLE_MS);
  });
</script>

<div class="flex min-h-screen flex-col">
  <!-- Contextual "current tab" dropdown below sm (not a generic hamburger) so the operator always sees where they are. -->
  <header class="border-b border-line bg-surface">
    <div class="mx-auto flex h-14 max-w-7xl items-center justify-between gap-2 px-4 sm:gap-4">
      <div class="flex min-w-0 items-center gap-3 sm:gap-6">
        <a
          href={resolve('/')}
          class="flex shrink-0 items-center gap-2 text-fg"
          aria-label={m.nav.home_aria}
        >
          <span class="inline-block h-2.5 w-2.5 rounded-full bg-accent"></span>
          <span class="hidden text-base font-semibold tracking-tight sm:inline">{m.app.name}</span>
        </a>

        <div
          bind:this={mobileMenuWrapper}
          class="relative sm:hidden"
          role="group"
          aria-label={m.nav.primary_nav_aria}
          onfocusout={onMobileMenuFocusOut}
        >
          <!-- Trigger geometry (text-sm, px-3 py-1.5) kept in lockstep with the other header controls so the mobile bar aligns. -->
          <button
            type="button"
            class="inline-flex items-center gap-1.5 rounded-md border border-line bg-surface px-3 py-1.5 text-sm font-medium text-fg transition hover:border-line-strong"
            aria-expanded={mobileMenuOpen}
            aria-haspopup="menu"
            onclick={toggleMobileMenu}
          >
            <span>{currentTabLabel}</span>
            <svg
              viewBox="0 0 20 20"
              fill="currentColor"
              class="h-3.5 w-3.5 text-fg-muted transition-transform duration-150"
              class:rotate-180={mobileMenuOpen}
              aria-hidden="true"
            >
              <path
                fill-rule="evenodd"
                d="M5.23 7.21a.75.75 0 011.06.02L10 11.06l3.71-3.83a.75.75 0 111.08 1.04l-4.25 4.4a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z"
                clip-rule="evenodd"
              />
            </svg>
          </button>

          {#if mobileMenuOpen}
            <div
              role="menu"
              aria-label={m.nav.primary_nav_aria}
              class="absolute top-full left-0 z-30 mt-2 min-w-44 rounded-lg border border-line bg-elevated p-1 shadow-popover"
            >
              {#each TABS as tab (tab.href)}
                <a
                  href={tab.href}
                  role="menuitem"
                  onclick={closeMobileMenu}
                  class="block rounded-md px-4 py-2 text-sm font-medium transition"
                  class:bg-surface-2={isActive(tab.href)}
                  class:text-fg={isActive(tab.href)}
                  class:text-fg-secondary={!isActive(tab.href)}
                  class:hover:bg-surface-2={!isActive(tab.href)}
                  class:hover:text-fg={!isActive(tab.href)}>{tab.label}</a
                >
              {/each}
            </div>
          {/if}
        </div>

        <nav class="hidden items-center gap-1 sm:flex">
          {#each TABS as tab (tab.href)}
            <a
              href={tab.href}
              class="rounded-md px-3 py-1.5 text-sm font-medium transition"
              class:bg-surface-2={isActive(tab.href)}
              class:text-fg={isActive(tab.href)}
              class:text-fg-muted={!isActive(tab.href)}
              class:hover:text-fg={!isActive(tab.href)}>{tab.label}</a
            >
          {/each}
        </nav>
      </div>

      <div class="flex shrink-0 items-center gap-3">
        <LocaleSwitcher />
        <ThemeToggle />
        <HealthBadge />
      </div>
    </div>
  </header>

  <main class="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
    {@render children?.()}
  </main>
</div>
