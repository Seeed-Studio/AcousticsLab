// Dynamic [id] can't be enumerated at build time, so override the layout's prerender; ssr is restated (already off in the layout) to keep the contract local.
export const prerender = false;
export const ssr = false;
