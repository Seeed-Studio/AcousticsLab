// An inline dep-free IIFE in app.html applies these same rules before paint (anti-FOUC); this
// store re-applies on first effect (no-op cold) and owns every change once Svelte is alive.

export type ThemeMode = 'auto' | 'light' | 'dark';
export type ResolvedTheme = 'light' | 'dark';

const STORAGE_KEY = 'acousticslab-theme';

// Anything but strict 'light'/'dark' (unset, stale, or Safari-private throw) -> 'auto' (OS-driven).
function readInitialMode(): ThemeMode {
  if (typeof localStorage === 'undefined') return 'auto';
  try {
    const m = localStorage.getItem(STORAGE_KEY);
    if (m === 'light' || m === 'dark') return m;
    return 'auto';
  } catch {
    return 'auto';
  }
}

// Seed snapshot; later changes flow through the matchMedia listener wired in start().
function readInitialSystemPrefersDark(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  } catch {
    return false;
  }
}

class ThemeStore {
  mode = $state<ThemeMode>(readInitialMode());

  systemPrefersDark = $state(readInitialSystemPrefersDark());

  // Getter, not $derived: reading the $state fields here makes Svelte 5 re-run callers reactively.
  get resolved(): ResolvedTheme {
    if (this.mode === 'dark') return 'dark';
    if (this.mode === 'light') return 'light';
    return this.systemPrefersDark ? 'dark' : 'light';
  }

  private mql: MediaQueryList | null = null;
  private disposeEffects: (() => void) | null = null;

  // Arrow-property, not a method, so add/removeEventListener share one stable bound identity.
  private onSystemChange = (e: MediaQueryListEvent): void => {
    this.systemPrefersDark = e.matches;
  };

  // `storage` fires only in OTHER same-origin tabs (cross-tab sync); `e.key === null` is a clear() -> 'auto'.
  private onStorage = (e: StorageEvent): void => {
    if (e.key !== null && e.key !== STORAGE_KEY) return;
    if (e.key === null) {
      this.mode = 'auto';
      return;
    }
    const v = e.newValue;
    if (v === 'light' || v === 'dark') {
      this.mode = v;
    } else if (v === null) {
      this.mode = 'auto';
    }
  };

  start(): void {
    if (this.disposeEffects !== null) return;
    if (typeof document === 'undefined') return;

    // Safari < 14 threw on MediaQueryList.addEventListener; catch leaves the seed value pinned.
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      try {
        this.mql = window.matchMedia('(prefers-color-scheme: dark)');
        this.mql.addEventListener('change', this.onSystemChange);
      } catch {
        this.mql = null;
      }
    }

    if (typeof window !== 'undefined') {
      window.addEventListener('storage', this.onStorage);
    }

    // $effect.root: manual effect scope for this module-scope singleton (no component); disposed in stop().
    this.disposeEffects = $effect.root(() => {
      // colorScheme paints native controls to match (else a dark page gets light macOS scrollbars); html ONLY so body inherits.
      $effect(() => {
        const r = this.resolved;
        const html = document.documentElement;
        html.classList.toggle('dark', r === 'dark');
        html.style.colorScheme = r;
      });

      // 'auto' removes the key (not stored literally) so it reads back as "no preference".
      $effect(() => {
        const m = this.mode;
        try {
          if (m === 'auto') localStorage.removeItem(STORAGE_KEY);
          else localStorage.setItem(STORAGE_KEY, m);
        } catch {
          // private-browsing / quota: ignore
        }
      });
    });
  }

  stop(): void {
    if (this.mql !== null) {
      this.mql.removeEventListener('change', this.onSystemChange);
      this.mql = null;
    }
    if (typeof window !== 'undefined') {
      window.removeEventListener('storage', this.onStorage);
    }
    if (this.disposeEffects !== null) {
      this.disposeEffects();
      this.disposeEffects = null;
    }
  }

  setMode(m: ThemeMode): void {
    this.mode = m;
  }
}

export const theme = new ThemeStore();
