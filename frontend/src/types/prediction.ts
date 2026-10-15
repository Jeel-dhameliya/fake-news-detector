/**
 * Type definitions for TruthLens prediction and health API contracts.
 */

export type PredictedLabel = 'FAKE' | 'REAL' | string;

export interface PredictionRequest {
  text: string;
}

export interface PredictionResponse {
  predicted_label: PredictedLabel;
  confidence: number;
  model_confidence: number;
  class_probabilities: {
    FAKE?: number;
    REAL?: number;
    [label: string]: number | undefined;
  };
  model_id: string;
  device: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  api_liveness?: string;
  model_readiness?: 'ready' | 'unloaded' | string;
  model_loaded?: boolean;
  model_id?: string;
  device?: string | null;
}

export interface ApiError {
  message: string;
  status?: number;
  detail?: string | Array<{ msg: string; loc?: string[] }>;
}
