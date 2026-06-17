import type { SocketState } from '$lib/stream/client';
import { m } from '$lib/i18n';

// Centralised socket-status pill vocabulary so every pill shares one lifecycle/colour set, preventing drift when one is restyled.

// Read the catalog on every call (never cache a Record snapshot): the proxy `get` re-reads
// `locale.resolved` (a getter over the `mode`/`detected` $state), which is what re-renders the label on locale switch.
export function socketLabel(state: SocketState): string {
  return m.streams.socket_status[state];
}

export function socketPillClass(state: SocketState): string {
  switch (state) {
    case 'open':
      return 'bg-success-soft text-success-soft-fg';
    case 'connecting':
      return 'bg-warning-soft text-warning-soft-fg';
    default:
      return 'bg-danger-soft text-danger-soft-fg';
  }
}
