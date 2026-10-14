import test from 'node:test';
import assert from 'node:assert/strict';

// Helper mimicking predictNews API service logic
async function mockPredictNews(mockFetch, text) {
  const payload = { text };
  let response;
  try {
    response = await mockFetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    throw new Error(`Network connection error: Unable to reach TruthLens backend (${err.message}).`);
  }

  if (!response.ok) {
    let errorDetail = '';
    try {
      const errorJson = await response.json();
      if (typeof errorJson.detail === 'string') {
        errorDetail = errorJson.detail;
      } else if (Array.isArray(errorJson.detail)) {
        errorDetail = errorJson.detail.map((item) => item.msg || JSON.stringify(item)).join(', ');
      }
    } catch {
      errorDetail = response.statusText || 'Unknown server response';
    }

    const err = new Error(errorDetail || `Request failed with status ${response.status}`);
    err.status = response.status;
    err.detail = errorDetail;
    throw err;
  }

  return response.json();
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

  const result = await mockPredictNews(mockFetch, 'Valid factual news statement');
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
      await mockPredictNews(mockFetch, '   ');
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
      await mockPredictNews(mockFetch, 'Sample article');
    },
    (err) => {
      assert.equal(err.status, 503);
      assert.match(err.message, /temporarily unavailable/);
      return true;
    }
  );
});

test('predictNews handles network connection failure', async () => {
  const mockFetch = async () => {
    throw new Error('fetch failed: ECONNREFUSED');
  };

  await assert.rejects(
    async () => {
      await mockPredictNews(mockFetch, 'Sample article');
    },
    (err) => {
      assert.match(err.message, /Network connection error/);
      return true;
    }
  );
});

test('formatClassProbabilities safely handles missing or non-finite probability fields', () => {
  // Case: class_probabilities is missing
  const resultMissingProbs = {
    predicted_label: 'REAL',
    confidence: 0.95,
  };
  const { fakePercent, realPercent } = formatClassProbabilities(resultMissingProbs);
  assert.equal(realPercent, '95.0');
  assert.equal(fakePercent, '5.0');

  // Case: probabilities out of [0, 1] range are clamped
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
