// Convert producers for `POST /workspaces/{id}/convert`: bodies are internally tagged on
// `converter_type` for daemon per-variant dispatch; helpers hide the discriminator.

import { api } from './http';
import type {
  AlpkgConvertParams,
  ConvertStartResp,
  LabelsFormat,
  TfjsConvertParams,
  Uuid
} from './types';

export const converter = {
  /// `.alpkg` head import. `manifestPath` is converter-rooted (no leading slash) `<head_id>.json`; daemon derives sibling `<parent>/<head_id>.mpk`, so upload both under one dir first. Head id is lifted from the manifest, not allocated. 409 `conflict` = convert semaphore busy or a concurrent WorkspaceDelete holds the slot.
  startAlpkg: (workspaceId: Uuid, params: { manifestPath: string }) => {
    const body: AlpkgConvertParams = {
      converter_type: 'alpkg',
      manifest_path: params.manifestPath
    };
    return api.post<ConvertStartResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/convert`,
      body
    );
  },

  /// TFJS bundle conversion (operator uploads `model.json` + its `weightsManifest[].paths` shards + labels). Body omits the shard list (daemon derives it from the manifest) to satisfy `deny_unknown_fields`. Head id is daemon-allocated, not lifted.
  startTfjs: (
    workspaceId: Uuid,
    params: {
      modelJsonPath: string;
      labelsPath: string;
      labelsFormat: LabelsFormat;
    }
  ) => {
    const body: TfjsConvertParams = {
      converter_type: 'tfjs',
      model_json_path: params.modelJsonPath,
      labels_path: params.labelsPath,
      labels_format: params.labelsFormat
    };
    return api.post<ConvertStartResp>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/convert`,
      body
    );
  }
};
