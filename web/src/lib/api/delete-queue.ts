// One global queue because the daemon serializes the entire delete family (max_delete_jobs = 1):
// parallel DELETEs would 409 even across unrelated resources. Each link fires after the prior
// job's terminal SSE event lands.

let chain: Promise<unknown> = Promise.resolve();

// Returned promise mirrors task's outcome, but the chain itself never rejects (per-link catch)
// so one failed delete can't stall the queue. task owns the full DELETE-ack + SSE-terminal lifecycle.
export function enqueueDelete<T>(task: () => Promise<T>): Promise<T> {
  const work = chain.then(() => task());
  chain = work.catch(() => undefined);
  return work;
}
