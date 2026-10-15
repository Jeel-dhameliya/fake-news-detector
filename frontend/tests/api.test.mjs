import test from 'node:test';
import assert from 'node:assert/strict';

export class ApiRequestError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
    this.detail = detail;
  }
}

// Logic mirroring predictNews in frontend/src/services/api.ts
async function predictNews(text, timeoutMs = 15000, fetchFn = fetch) {
  const payload = { text };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response;
  try {
    response = await fetchFn('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err) {
    const isAbort = err?.name === 'AbortError' || controller.signal.aborted;
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
        errorDetail = errorJson.detail
          .map((item) => item.msg || item.message || JSON.stringify(item))
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

  return response.json();
}

// Logic mirroring checkBackendHealth in frontend/src/services/api.ts
async function checkBackendHealth(timeoutMs = 15000, fetchFn = fetch) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchFn('/api/health', {
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new ApiRequestError('Backend health check returned non-200 status', response.status);
    }
    return await response.json();
  } catch (err) {
    if (err instanceof ApiRequestError) {
      throw err;
    }
    const isAbort = err?.name === 'AbortError' || controller.signal.aborted;
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

// Logic mirroring Navbar health status resolution
function parseHealthForNavbar(data) {
  if (data.status === 'healthy' || data.api_liveness === 'alive') {
    return {
      backendStatus: 'online',
      modelState: data.model_readiness === 'ready' ? 'ready' : (data.model_readiness === 'unloaded' ? 'unloaded' : 'unknown'),
      version: data.version || null,
    };
  }
  return {
    backendStatus: 'offline',
    modelState: 'unknown',
    version: null,
  };
}

// Logic mirroring ResultsSection probability extraction & safe percentage formatting
function formatClassProbabilities(result) {
  const isReal = result.predicted_label.toUpperCase() === 'REAL';
  const confidence = typeof result.confidence === 'number' && Number.isFinite(result.confidence) ? result.confidence : 0;
  const fakeProb = result.class_probabilities?.FAKE ?? (isReal ? 1 - confidence : confidence);
  const realProb = result.class_probabilities?.REAL ?? (isReal ? confidence : 1 - confidence);

  const fakePercent = (Math.max(0, Math.min(1, fakeProb)) * 100).toFixed(1);
  const realPercent = (Math.max(0, Math.min(1, realProb)) * 100).toFixed(1);

  return { fakePercent, realPercent };
}

// ---------------------- PREDICTION TESTS ----------------------

test('predictNews successfully parses valid model prediction contract', async () => {
  const mockFetch = async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      predicted_label: 'REAL',
      confidence: 0.982,
      model_confidence: 0.982,
      class_probabilities: { FAKE: 0.018, REAL: 0.982 },
      model_id: 'dima806/fake-news-classifier',
      device: 'mps',
    }),
  });

  const result = await predictNews('Valid factual news statement', 15000, mockFetch);
  assert.equal(result.predicted_label, 'REAL');
  assert.equal(result.confidence, 0.982);
  assert.equal(result.model_confidence, 0.982);
  assert.equal(result.class_probabilities.REAL, 0.982);
  assert.equal(result.class_probabilities.FAKE, 0.018);
  assert.equal(result.model_id, 'dima806/fake-news-classifier');
  assert.equal(result.device, 'mps');
});

test('predictNews handles HTTP 422 validation errors with parsed message', async () => {
  const mockFetch = async () => ({
    ok: false,
    status: 422,
    json: async () => ({
      detail: [{ loc: ['body', 'text'], msg: 'Value error, Input text cannot be empty or whitespace-only.' }],
    }),
  });

  await assert.rejects(
    async () => {
      await predictNews('   ', 15000, mockFetch);
    },
    (err) => {
      assert.equal(err.status, 422);
      assert.match(err.message, /Input text cannot be empty or whitespace-only/);
      return true;
    }
  );
});

test('predictNews handles HTTP 503 model service unavailable', async () => {
  const mockFetch = async () => ({
    ok: false,
    status: 503,
    json: async () => ({
      detail: 'Classification model service is temporarily unavailable.',
    }),
  });

  await assert.rejects(
    async () => {
      await predictNews('Sample article', 15000, mockFetch);
    },
    (err) => {
      assert.equal(err.status, 503);
      assert.match(err.message, /temporarily unavailable/);
      return true;
    }
  );
});

test('predictNews handles HTTP 500 server error', async () => {
  const mockFetch = async () => ({
    ok: false,
    status: 500,
    statusText: 'Internal Server Error',
    json: async () => {
      throw new Error('Not JSON');
    },
  });

  await assert.rejects(
    async () => {
      await predictNews('Sample article', 15000, mockFetch);
    },
    (err) => {
      assert.equal(err.status, 500);
      assert.match(err.message, /server error/i);
      return true;
    }
  );
});

test('predictNews handles network connection failure (status 0) without stack leakage', async () => {
  const mockFetch = async () => {
    throw new Error('connect ECONNREFUSED 127.0.0.1:8000');
  };

  await assert.rejects(
    async () => {
      await predictNews('Sample article', 15000, mockFetch);
    },
    (err) => {
      assert.equal(err.status, 0);
      assert.match(err.message, /Network connection error/);
      assert.match(err.message, /127\.0\.0\.1:8000/);
      return true;
    }
  );
});

test('predictNews handles request timeout deterministically (status 408, TIMEOUT detail)', async () => {
  // Deterministic mock fetch: respects AbortSignal and throws AbortError immediately when aborted
  const mockFetch = async (_url, options) => {
    return new Promise((_, reject) => {
      if (options?.signal) {
        options.signal.addEventListener('abort', () => {
          const abortErr = new Error('The operation was aborted');
          abortErr.name = 'AbortError';
          reject(abortErr);
        });
      }
    });
  };

  // Run with an immediate 1ms timeout — deterministic, no 15-second wait
  await assert.rejects(
    async () => {
      await predictNews('Long inference article', 1, mockFetch);
    },
    (err) => {
      assert.equal(err.status, 408);
      assert.equal(err.detail, 'TIMEOUT');
      assert.match(err.message, /Request timed out/i);
      assert.match(err.message, /please retry/i);
      return true;
    }
  );
});

// ---------------------- HEALTH CHECK TESTS ----------------------

test('checkBackendHealth successfully parses health payload with liveness and readiness', async () => {
  const mockFetch = async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      status: 'healthy',
      api_liveness: 'alive',
      model_readiness: 'unloaded',
      model_loaded: false,
      model_id: 'dima806/fake-news-classifier',
      device: 'cpu',
      version: '1.0.0',
    }),
  });

  const data = await checkBackendHealth(15000, mockFetch);
  assert.equal(data.status, 'healthy');
  assert.equal(data.api_liveness, 'alive');
  assert.equal(data.model_readiness, 'unloaded');
  assert.equal(data.model_loaded, false);
  assert.equal(data.model_id, 'dima806/fake-news-classifier');

  const navbarState = parseHealthForNavbar(data);
  // An unloaded model must NOT be marked as an offline backend!
  assert.equal(navbarState.backendStatus, 'online');
  assert.equal(navbarState.modelState, 'unloaded');
  assert.equal(navbarState.version, '1.0.0');
});

test('checkBackendHealth handles ready model payload properly', async () => {
  const mockFetch = async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      status: 'healthy',
      api_liveness: 'alive',
      model_readiness: 'ready',
      model_loaded: true,
      model_id: 'dima806/fake-news-classifier',
      device: 'mps',
      version: '1.0.0',
    }),
  });

  const data = await checkBackendHealth(15000, mockFetch);
  const navbarState = parseHealthForNavbar(data);
  assert.equal(navbarState.backendStatus, 'online');
  assert.equal(navbarState.modelState, 'ready');
});

test('checkBackendHealth handles timeout deterministically (status 408)', async () => {
  const mockFetch = async (_url, options) => {
    return new Promise((_, reject) => {
      if (options?.signal) {
        options.signal.addEventListener('abort', () => {
          const abortErr = new Error('The operation was aborted');
          abortErr.name = 'AbortError';
          reject(abortErr);
        });
      }
    });
  };

  await assert.rejects(
    async () => {
      await checkBackendHealth(1, mockFetch);
    },
    (err) => {
      assert.equal(err.status, 408);
      assert.equal(err.detail, 'TIMEOUT');
      assert.match(err.message, /timed out/i);
      return true;
    }
  );
});

test('checkBackendHealth handles network failure (status 0)', async () => {
  const mockFetch = async () => {
    throw new Error('fetch failed');
  };

  await assert.rejects(
    async () => {
      await checkBackendHealth(15000, mockFetch);
    },
    (err) => {
      assert.equal(err.status, 0);
      assert.match(err.message, /Unable to reach backend health endpoint/);
      return true;
    }
  );
});

// ---------------------- PROBABILITY & DEMO TESTS ----------------------

test('formatClassProbabilities safely handles missing or non-finite probability fields', () => {
  const resultMissingProbs = {
    predicted_label: 'REAL',
    confidence: 0.95,
  };
  const { fakePercent, realPercent } = formatClassProbabilities(resultMissingProbs);
  assert.equal(realPercent, '95.0');
  assert.equal(fakePercent, '5.0');

  const resultUnclamped = {
    predicted_label: 'FAKE',
    confidence: 1.2,
    class_probabilities: { FAKE: 1.2, REAL: -0.2 },
  };
  const clamped = formatClassProbabilities(resultUnclamped);
  assert.equal(clamped.fakePercent, '100.0');
  assert.equal(clamped.realPercent, '0.0');
});

test('demo samples adhere to backend validation constraints', () => {
  const DEMO_SAMPLES = [
    'NASA confirmed today that the James Webb Space Telescope detected water vapor in the atmosphere of a rocky exoplanet orbiting within its host star habitable zone.',
    'SHOCKING PROOF: Government secretly installing microchips in tap water to control thoughts of citizens across nationwide network! Whistleblowers reveal explosive hidden documents!',
    'The government announced a new public health policy today regarding regional immunization clinics and preventative care programs.',
  ];

  for (const sample of DEMO_SAMPLES) {
    assert.ok(sample.trim().length > 0, 'Demo sample must not be empty or whitespace-only');
    assert.ok(sample.length <= 50000, 'Demo sample must not exceed MAX_INPUT_LENGTH');
  }
});
