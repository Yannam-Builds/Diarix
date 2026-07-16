/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ModelStatus = {
  description: `Response model for model status.`,
  properties: {
    model_name: {
      type: 'string',
      isRequired: true,
    },
    display_name: {
      type: 'string',
      isRequired: true,
    },
    model_size: { type: 'string' },
    hf_repo_id: { type: 'any-of', contains: [{ type: 'string' }, { type: 'null' }] },
    downloaded: {
      type: 'boolean',
      isRequired: true,
    },
    size_mb: {
      type: 'any-of',
      contains: [
        {
          type: 'number',
        },
        {
          type: 'null',
        },
      ],
    },
    loaded: {
      type: 'boolean',
    },
    engine: { type: 'string' },
    modality: { type: 'string' },
    runtime_group: { type: 'string' },
    capabilities: { type: 'array', contains: { type: 'string' } },
    languages: { type: 'array', contains: { type: 'string' } },
    description: { type: 'string' },
    precision_options: { type: 'array', contains: { type: 'string' } },
    default_precision: { type: 'any-of', contains: [{ type: 'string' }, { type: 'null' }] },
    recommended: { type: 'boolean' },
    min_vram_gb: { type: 'any-of', contains: [{ type: 'number' }, { type: 'null' }] },
    audio_sample_rate: { type: 'any-of', contains: [{ type: 'number' }, { type: 'null' }] },
    audio_channels: { type: 'any-of', contains: [{ type: 'number' }, { type: 'null' }] },
    audio_format: { type: 'any-of', contains: [{ type: 'string' }, { type: 'null' }] },
    shares_cache_with: { type: 'array', contains: { type: 'string' } },
  },
} as const;
