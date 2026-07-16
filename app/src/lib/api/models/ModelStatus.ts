/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Response model for model status.
 */
export type ModelStatus = {
  model_name: string;
  display_name: string;
  model_size?: string;
  hf_repo_id?: string | null;
  downloaded: boolean;
  downloading?: boolean; // True if download is in progress
  size_mb?: number | null;
  loaded?: boolean;
  engine?: string;
  modality?: string;
  runtime_group?: string;
  capabilities?: string[];
  languages?: string[];
  description?: string;
  precision_options?: string[];
  default_precision?: string | null;
  recommended?: boolean;
  min_vram_gb?: number | null;
  audio_sample_rate?: number | null;
  audio_channels?: number | null;
  audio_format?: string | null;
  shares_cache_with?: string[];
};
