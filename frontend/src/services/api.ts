/**
 * API service for interacting with the TruthLens FastAPI backend.
 */

import { HealthResponse, PredictionRequest, PredictionResponse } from '../types/prediction';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export const PREDICTION_TIMEOUT_MS = 15000;
export const HEALTH_TIMEOUT_MS = 15000;

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
 * Sends text to the classifier endpoint for prediction with a 15-second timeout.
 */
export async function predictNews(
  text: string,
  timeoutMs: number = PREDICTION_TIMEOUT_MS,
  fetchFn: typeof fetch = fetch
): Promise<PredictionResponse> {
  const payload: PredictionRequest = { text };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetchFn(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err: unknown) {
    const isAbort = (err as { name?: string })?.name === 'AbortError' || controller.signal.aborted;
    if (isAbort) {
      throw new ApiRequestError(
        `Request timed out: The analysis took longer than ${Math.round(timeoutMs / 1000)} seconds to complete. The server may be busy or initializing; please retry.`,
        408,
        'TIMEOUT'
      );
    }
    const errorMsg = err instanceof Error ? err.message : String(err);
    throw new ApiRequestError(
      `Network connection error: Unable to reach TruthLens backend (${errorMsg}). Please verify the backend is running at http://127.0.0.1:8000.`,
      0
    );
  } finally {
    clearTimeout(timer);
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
 * Checks backend health readiness with a 5-second timeout.
 */
export async function checkBackendHealth(
  timeoutMs: number = HEALTH_TIMEOUT_MS,
  fetchFn: typeof fetch = fetch
): Promise<HealthResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchFn(`${API_BASE_URL}/health`, {
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new ApiRequestError('Backend health check returned non-200 status', response.status);
    }
    const data: HealthResponse = await response.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiRequestError) {
      throw err;
    }
    const isAbort = (err as { name?: string })?.name === 'AbortError' || controller.signal.aborted;
    if (isAbort) {
      throw new ApiRequestError(
        'Backend health check timed out. The server may be busy or starting up.',
        408,
        'TIMEOUT'
      );
    }
    const errorMsg = err instanceof Error ? err.message : String(err);
    throw new ApiRequestError(
      `Unable to reach backend health endpoint (${errorMsg}).`,
      0
    );
  } finally {
    clearTimeout(timer);
  }
}
