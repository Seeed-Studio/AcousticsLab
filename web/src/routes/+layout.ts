import { locale } from '$lib/stores/locale.svelte';
import { ensureCatalog } from '$lib/i18n';

export const prerender = true;
export const ssr = false;
export const trailingSlash = 'never';

// Block initial paint on the resolved catalog so a non-default locale doesn't flash en (the switcher
// preloads on switch; cross-tab sync still flashes briefly until its chunk lands). Prerender resolves
// to the bundled en — an instant no-op.
export async function load(): Promise<void> {
  await ensureCatalog(locale.resolved);
}
