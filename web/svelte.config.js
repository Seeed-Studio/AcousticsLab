import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { loadEnv } from 'vite';
import { normalizeBasePath } from './base-path.js';

// Reverse-proxy mount prefix (e.g. `/extension/acousticslab`), read at
// build/dev time from `VITE_BASE_PATH` -- an actual env var OR a
// `.env*` file (`loadEnv('', …)`-style: prefix `''` pulls every key,
// including non-`VITE_`-exposed ones, from both sources).  Empty (the
// default) keeps a root mount, byte-identical to before.
//
// The SAME var is read by `src/lib/api/base.ts` to prefix same-origin
// API/SSE/WebSocket calls, so static assets, internal links, and
// backend calls all sit under one shared prefix behind the gateway.
const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';
const base = normalizeBasePath(loadEnv(mode, process.cwd(), '').VITE_BASE_PATH);

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html',
      precompress: false,
      strict: false
    }),
    paths: { base },
    alias: {
      $lib: 'src/lib',
      $proto: 'src/lib/proto'
    }
  }
};

export default config;
