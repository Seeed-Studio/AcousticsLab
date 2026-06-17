// Classifies operator-dropped files into a TFJS bundle (one model.json manifest, its declared
// shards, and a labels source). Pure apart from reading the dropped `File`s, so re-runnable on a
// stale drop without side effects.

import type { LabelsFormat } from '$lib/api/types';
import { m } from '$lib/i18n';

export interface ClassifiedTfjsBundle {
  modelJson: File | null;
  shards: File[];
  labels: File | null;
  labelsFormat: LabelsFormat | null;
  /// Files matching no role; surfaced so the dialog reports them rather than dropping silently.
  unknown: File[];
  /// Present once `modelJson` parsed; `nClasses` is always null (class count left to the worker).
  parsed: {
    declaredShards: string[];
    nClasses: number | null;
  } | null;
  diagnostics: TfjsDiagnostic[];
  /// True iff every required slot is populated, every declared shard present, and no blocker
  /// fired; false disables the dialog's Import button.
  ready: boolean;
}

export type TfjsDiagnosticSeverity = 'blocker' | 'warning';

export interface TfjsDiagnostic {
  severity: TfjsDiagnosticSeverity;
  message: string;
}

/// Classify a `File[]` drop. Reads `model.json`'s bytes only when exactly one is present; all
/// other files are inspected by name/extension alone.
export async function classifyTfjsBundle(files: File[]): Promise<ClassifiedTfjsBundle> {
  const result: ClassifiedTfjsBundle = {
    modelJson: null,
    shards: [],
    labels: null,
    labelsFormat: null,
    unknown: [],
    parsed: null,
    diagnostics: [],
    ready: false
  };

  if (files.length === 0) {
    result.diagnostics.push({
      severity: 'blocker',
      message: m.workspace.import_dialog.pick_file.tfjs_diag_empty_drop
    });
    return result;
  }

  // Lowercase the basename so case-typo'd filenames (`Model.json`) still classify.
  const modelJsons: File[] = [];
  const labelsLines: File[] = [];
  const labelsMetadata: File[] = [];
  const possibleShards: File[] = [];

  for (const f of files) {
    const lower = f.name.toLowerCase();
    if (lower === 'model.json') {
      modelJsons.push(f);
    } else if (lower === 'labels.txt') {
      labelsLines.push(f);
    } else if (lower === 'metadata.json' || lower.endsWith('.metadata.json')) {
      labelsMetadata.push(f);
    } else if (lower.endsWith('.bin') || /-shard\d+of\d+$/i.test(lower)) {
      // Over-inclusive candidate; the manifest's declared list prunes anything it doesn't name.
      possibleShards.push(f);
    } else {
      result.unknown.push(f);
    }
  }

  if (modelJsons.length === 0) {
    result.diagnostics.push({
      severity: 'blocker',
      message: m.workspace.import_dialog.pick_file.tfjs_diag_no_model_json
    });
  } else if (modelJsons.length > 1) {
    result.diagnostics.push({
      severity: 'blocker',
      message: m.workspace.import_dialog.pick_file.tfjs_diag_ambiguous_model_json(modelJsons.length)
    });
  } else {
    result.modelJson = modelJsons[0];
  }

  // Accept exactly one labels source (both is a blocker: the worker reads one, intent ambiguous).
  // Diagnostics quote names with double quotes; the plain-text panel renders backticks literally.
  if (labelsLines.length > 1) {
    result.diagnostics.push({
      severity: 'blocker',
      message: m.workspace.import_dialog.pick_file.tfjs_diag_multiple_labels_txt
    });
  } else if (labelsMetadata.length > 1) {
    result.diagnostics.push({
      severity: 'blocker',
      message: m.workspace.import_dialog.pick_file.tfjs_diag_multiple_metadata_json
    });
  } else if (labelsLines.length === 1 && labelsMetadata.length === 1) {
    result.diagnostics.push({
      severity: 'blocker',
      message: m.workspace.import_dialog.pick_file.tfjs_diag_both_labels
    });
  } else if (labelsLines.length === 1) {
    result.labels = labelsLines[0];
    result.labelsFormat = 'lines';
  } else if (labelsMetadata.length === 1) {
    result.labels = labelsMetadata[0];
    result.labelsFormat = 'tfjs_metadata';
  } else {
    result.diagnostics.push({
      severity: 'blocker',
      message: m.workspace.import_dialog.pick_file.tfjs_diag_no_labels
    });
  }

  // Parse failures are blockers: catch a malformed manifest here to skip the worker round-trip.
  if (result.modelJson) {
    try {
      const parsed = await parseModelJsonShards(result.modelJson);
      result.parsed = parsed;
      const declared = new Set(parsed.declaredShards);
      // Same-basename staged files differ in bytes (drop is deduped by name+size+lastModified),
      // so the name -> File Map below would pick wrong bytes and trip a downstream CRC/size
      // mismatch; block so the operator re-drops. Only declared-name collisions matter.
      const shardNameCounts = new Map<string, number>();
      for (const f of possibleShards) {
        shardNameCounts.set(f.name, (shardNameCounts.get(f.name) ?? 0) + 1);
      }
      const collidingShardNames: string[] = [];
      for (const [name, count] of shardNameCounts) {
        if (count > 1 && declared.has(name)) collidingShardNames.push(name);
      }
      if (collidingShardNames.length > 0) {
        const quoted = collidingShardNames
          .slice(0, 3)
          .map((n) => `"${n}"`)
          .join(', ');
        result.diagnostics.push({
          severity: 'blocker',
          message:
            collidingShardNames.length === 1
              ? m.workspace.import_dialog.pick_file.tfjs_diag_shard_collision_one(quoted)
              : m.workspace.import_dialog.pick_file.tfjs_diag_shard_collision_many(
                  quoted,
                  collidingShardNames.length > 3
                )
        });
      }
      const provided = new Map(possibleShards.map((f) => [f.name, f]));
      const matched: File[] = [];
      const missing: string[] = [];
      for (const name of parsed.declaredShards) {
        const f = provided.get(name);
        if (f) matched.push(f);
        else missing.push(name);
      }
      result.shards = matched;
      // Unreferenced candidate shards go to `unknown` so they're reported, not silently shipped.
      for (const f of possibleShards) {
        if (!declared.has(f.name)) result.unknown.push(f);
      }
      if (missing.length > 0) {
        const quoted = missing
          .slice(0, 3)
          .map((n) => `"${n}"`)
          .join(', ');
        result.diagnostics.push({
          severity: 'blocker',
          message:
            missing.length === 1
              ? m.workspace.import_dialog.pick_file.tfjs_diag_missing_shard_one(quoted)
              : m.workspace.import_dialog.pick_file.tfjs_diag_missing_shards_many(
                  missing.length,
                  quoted,
                  missing.length > 3
                )
        });
      }
    } catch (e) {
      result.diagnostics.push({
        severity: 'blocker',
        message:
          e instanceof Error
            ? e.message
            : m.workspace.import_dialog.pick_file.error_could_not_read_model_json
      });
    }
  }

  const blocked = result.diagnostics.some((d) => d.severity === 'blocker');
  result.ready =
    !blocked &&
    result.modelJson !== null &&
    result.labels !== null &&
    result.labelsFormat !== null &&
    result.parsed !== null;

  return result;
}

/// Read class labels for an operator preview. `'lines'` trims and drops empties so trailing
/// newlines don't surface as phantom classes; `'tfjs_metadata'` reads `wordLabels`
/// (Speech-Commands) or `words` (Teachable Machine), preferring `wordLabels`. Returns `null` on
/// any parse failure; this is preview-only (the worker is authoritative), so null just skips it.
export async function parseTfjsLabels(file: File, format: LabelsFormat): Promise<string[] | null> {
  try {
    const text = await file.text();
    if (format === 'lines') {
      const out: string[] = [];
      for (const raw of text.split(/\r?\n/)) {
        const trimmed = raw.trim();
        if (trimmed.length > 0) out.push(trimmed);
      }
      return out.length > 0 ? out : null;
    }
    const parsed = JSON.parse(text) as unknown;
    if (parsed === null || typeof parsed !== 'object') return null;
    const obj = parsed as Record<string, unknown>;
    const candidate = obj.wordLabels ?? obj.words;
    if (!Array.isArray(candidate)) return null;
    const out: string[] = [];
    for (const v of candidate as unknown[]) {
      if (typeof v === 'string' && v.length > 0) out.push(v);
    }
    return out.length > 0 ? out : null;
  } catch {
    return null;
  }
}

interface ParsedModelJson {
  declaredShards: string[];
  nClasses: number | null;
}

/// Flatten `weightsManifest[].paths` into an order-preserved shard list. Throws (blocker message)
/// on a non-object body, missing `weightsManifest` array, or zero shards. `nClasses` is always
/// null: its `modelTopology` path varies across Teachable Machine vs tfjs-models, left to the worker.
async function parseModelJsonShards(modelJson: File): Promise<ParsedModelJson> {
  let parsed: unknown;
  try {
    const text = await modelJson.text();
    parsed = JSON.parse(text);
  } catch (e) {
    throw new Error(m.workspace.import_dialog.pick_file.tfjs_diag_model_json_not_json, {
      cause: e
    });
  }
  if (parsed === null || typeof parsed !== 'object') {
    throw new Error(m.workspace.import_dialog.pick_file.tfjs_diag_model_json_not_object);
  }
  const obj = parsed as Record<string, unknown>;
  const wm = obj.weightsManifest;
  if (!Array.isArray(wm)) {
    throw new Error(m.workspace.import_dialog.pick_file.tfjs_diag_model_json_no_manifest);
  }
  const declaredShards: string[] = [];
  for (const groupRaw of wm as unknown[]) {
    if (groupRaw === null || typeof groupRaw !== 'object') continue;
    const paths = (groupRaw as Record<string, unknown>).paths;
    if (!Array.isArray(paths)) continue;
    for (const p of paths as unknown[]) {
      if (typeof p === 'string' && p.length > 0) declaredShards.push(p);
    }
  }
  if (declaredShards.length === 0) {
    throw new Error(m.workspace.import_dialog.pick_file.tfjs_diag_model_json_no_shards);
  }
  return { declaredShards, nClasses: null };
}
