import { active as activeApi, inference as inferenceApi, mic as micApi } from '$lib/api/endpoints';
import type { ActiveResp, InferenceCfg, MicPolicy, MicState, Uuid } from '$lib/api/types';

class ConfigStore {
  mic = $state<MicState | null>(null);
  inference = $state<InferenceCfg | null>(null);
  active = $state<ActiveResp | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const [m, i, a] = await Promise.all([micApi.get(), inferenceApi.get(), activeApi.get()]);
      this.mic = m;
      this.inference = i;
      this.active = a;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  // Post-delete truth patch: the daemon keeps orphaned runtimes serving and GET re-derives
  // `source_workspace_alive` from is_dir(), so after a terminal delete this flip equals the next GET.
  markWorkspaceDetached(workspaceId: Uuid): void {
    const a = this.active;
    if (
      a?.origin === 'head' &&
      a.source_workspace_id === workspaceId &&
      a.source_workspace_alive !== false
    ) {
      this.active = { ...a, source_workspace_alive: false };
    }
  }

  // Active-only revalidation: full refresh() re-assigns mic/inference, whose sync-effects in
  // ConfigurationControls clobber unapplied slider edits. The `before` guard (both paths) drops
  // results that lost the race to a newer write; failure hands recovery to the layout's throttled
  // auto-reconnect, and success leaves `error` alone (mic/inference may still be stale).
  async refreshActive(): Promise<void> {
    const before = this.active;
    try {
      const a = await activeApi.get();
      if (this.active === before) this.active = a;
    } catch (e) {
      if (this.active === before) this.error = e instanceof Error ? e.message : String(e);
    }
  }

  async setMicPolicy(policy: MicPolicy): Promise<void> {
    await this.guard(async () => {
      this.mic = await micApi.set(policy);
    });
  }

  async setInferenceCfg(cfg: Partial<InferenceCfg>): Promise<void> {
    await this.guard(async () => {
      this.inference = await inferenceApi.set(cfg);
    });
  }

  async activateDefault(): Promise<void> {
    await this.guard(async () => {
      this.active = await activeApi.setDefault();
    });
  }

  async activateHead(workspace_id: string, head_id: string): Promise<void> {
    await this.guard(async () => {
      this.active = await activeApi.setHead(workspace_id, head_id);
    });
  }

  // Re-throws after setting `error` so callers can distinguish failure and roll back optimistic UI.
  private async guard(fn: () => Promise<void>): Promise<void> {
    this.loading = true;
    try {
      await fn();
      this.error = null;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }
}

export const config = new ConfigStore();
