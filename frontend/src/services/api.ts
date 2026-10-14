/**
 * API service for interacting with the TruthLens FastAPI backend.
 */

import { HealthResponse, PredictionRequest, PredictionResponse } from '../types/prediction';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export class ApiRequestError extends Error {
  status?: number;
  detail?: string;

  constructor(message: string, status?: number, detail?: string) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Sends text to the classifier endpoint for prediction.
 */
export async function predictNews(text: string): Promise<PredictionResponse> {
  const payload: PredictionRequest = { text };

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    throw new ApiRequestError(
      `Network connection error: Unable to reach TruthLens backend (${errorMsg}). Please verify the backend is running at http://127.0.0.1:8000.`,
      0
    );
  }

  if (!response.ok) {
    let errorDetail = '';
    try {
      const errorJson = await response.json();
      if (typeof errorJson.detail === 'string') {
        errorDetail = errorJson.detail;
      } else if (Array.isArray(errorJson.detail)) {
        // FastAPI validation errors (e.g. 422)
        errorDetail = errorJson.detail
          .map((item: { msg?: string; message?: string }) => item.msg || item.message || JSON.stringify(item))
          .join(', ');
      } else if (errorJson.message) {
        errorDetail = errorJson.message;
      }
    } catch {
      errorDetail = response.statusText || 'Unknown server response';
    }

    if (response.status === 422) {
      throw new ApiRequestError(
        errorDetail || 'Validation failed: The provided text is empty, whitespace-only, or invalid.',
        422,
        errorDetail
      );
    } else if (response.status === 503) {
      throw new ApiRequestError(
        errorDetail || 'Classification model service is temporarily unavailable. The model may be initializing.',
        503,
        errorDetail
      );
    } else if (response.status >= 500) {
      throw new ApiRequestError(
        errorDetail || 'A server error occurred during model inference computation.',
        response.status,
        errorDetail
      );
    } else {
      throw new ApiRequestError(
        errorDetail || `Request failed with status ${response.status}`,
        response.status,
        errorDetail
      );
    }
  }

  const data: PredictionResponse = await response.json();
  return data;
}

/**
 * Checks backend health readiness.
 */
export async function checkBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new ApiRequestError('Backend health check returned non-200 status', response.status);
  }
  return response.json();
}
