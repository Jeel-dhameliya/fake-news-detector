import React from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Cpu,
  Zap,
  AlertTriangle,
  BarChart2,
} from 'lucide-react';

import { PredictionResponse } from '../types/prediction';

interface ResultsSectionProps {
  result: PredictionResponse;
}

export const ResultsSection: React.FC<ResultsSectionProps> = ({ result }) => {
  const isReal = result.predicted_label.toUpperCase() === 'REAL';
  const confidencePercent = (result.confidence * 100).toFixed(1);

  // Safely extract probabilities
  const fakeProb = result.class_probabilities?.FAKE ?? (isReal ? 1 - result.confidence : result.confidence);
  const realProb = result.class_probabilities?.REAL ?? (isReal ? result.confidence : 1 - result.confidence);

  const fakePercent = (Math.max(0, Math.min(1, fakeProb)) * 100).toFixed(1);
  const realPercent = (Math.max(0, Math.min(1, realProb)) * 100).toFixed(1);

  return (
    <div
      aria-live="polite"
      className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300"
    >
      {/* Primary Classification Verdict Card */}
      <div
        className={`rounded-xl border p-6 sm:p-8 shadow-verdict transition-all ${
          isReal
            ? 'bg-white border-emerald-600/70'
            : 'bg-white border-rose-600/70'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-stone-200">
          <div className="flex items-center gap-4">
            <div
              className={`w-14 h-14 rounded-xl flex items-center justify-center shrink-0 shadow-classic ${
                isReal
                  ? 'bg-emerald-50 border border-emerald-200 text-emerald-800'
                  : 'bg-rose-50 border border-rose-200 text-rose-800'
              }`}
            >
              {isReal ? (
                <ShieldCheck className="w-8 h-8" />
              ) : (
                <ShieldAlert className="w-8 h-8" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-serif uppercase tracking-widest text-stone-500 font-semibold">
                  Official Verdict &bull;
                </span>
                <span className="text-xs font-serif italic text-stone-500">
                  Binary Classifier Projection
                </span>
              </div>
              <div className="flex items-center gap-3">
                <h3
                  className={`text-3xl sm:text-4xl font-serif font-black tracking-tight ${
                    isReal ? 'text-emerald-900' : 'text-rose-950'
                  }`}
                >
                  {result.predicted_label}
                </h3>
                <span
                  className={`px-3.5 py-1 rounded-full text-xs font-serif font-bold italic border ${
                    isReal
                      ? 'bg-emerald-50 border-emerald-300 text-emerald-900'
                      : 'bg-rose-50 border-rose-300 text-rose-900'
                  }`}
                >
                  {confidencePercent}% Model Output Score
                </span>
              </div>
            </div>
          </div>

          {/* Model & Device Metadata */}
          <div className="flex sm:flex-col items-end gap-1.5 text-xs text-stone-600 font-serif">
            <div className="inline-flex items-center gap-1.5 font-mono text-[11px] bg-stone-50 px-2 py-0.5 rounded border border-stone-200">
              <Cpu className="w-3.5 h-3.5 text-amber-700" />
              <span>{result.model_id}</span>
            </div>
            <div className="inline-flex items-center gap-1.5 font-mono text-[11px] bg-stone-50 px-2 py-0.5 rounded border border-stone-200">
              <Zap className="w-3.5 h-3.5 text-amber-700" />
              <span>Device: <strong className="text-stone-900">{result.device}</strong></span>
            </div>
          </div>
        </div>

        {/* Probability Distribution */}
        <div className="pt-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-3.5">
            <span className="text-sm font-serif font-bold text-stone-900 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-amber-700" />
              Normalized Logit Probability Distribution
            </span>
            <span className="text-xs font-serif italic text-stone-500">
              Internal softmax preference (not a verified calibrated probability of real-world factuality)
            </span>
          </div>

          {/* Dual Bar Visualizer */}
          <div className="space-y-3.5">
            {/* REAL Class */}
            <div>
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-emerald-900 font-serif font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-600"></span>
                  Class Label: REAL
                </span>
                <span className="font-mono text-stone-900 font-bold">{realPercent}%</span>
              </div>
              <div className="w-full bg-stone-100 rounded-full h-2.5 overflow-hidden border border-stone-200 shadow-inner">
                <div
                  className="bg-gradient-to-r from-emerald-600 to-teal-600 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${realPercent}%` }}
                ></div>
              </div>
            </div>

            {/* FAKE Class */}
            <div>
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-rose-950 font-serif font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-600"></span>
                  Class Label: FAKE
                </span>
                <span className="font-mono text-stone-900 font-bold">{fakePercent}%</span>
              </div>
              <div className="w-full bg-stone-100 rounded-full h-2.5 overflow-hidden border border-stone-200 shadow-inner">
                <div
                  className="bg-gradient-to-r from-rose-600 to-amber-700 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${fakePercent}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mandatory Responsible Presentation & Benchmark Limitations Notice */}
      <aside
        aria-label="Model Limitations and Benchmark Notice"
        className="rounded-xl border border-amber-300 bg-[#fefcf6] p-6 sm:p-7 shadow-elevated"
      >
        <div className="flex items-start gap-3.5">
          <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="space-y-3 text-xs sm:text-sm leading-relaxed text-stone-800 font-serif">
            <h4 className="font-bold text-amber-950 text-base">
              Academic Benchmark Performance &amp; Epistemological Limits
            </h4>
            <p className="text-stone-700">
              TruthLens computes syntactic and semantic text classification using fine-tuned transformer weights. TruthLens does{' '}
              <strong className="text-stone-950">not independently verify claims against authoritative facts or live external evidence</strong>.
            </p>
            <ul className="list-disc pl-5 space-y-2 text-stone-700">
              <li>
                <strong className="text-stone-900">Model Output Score vs. Ground Truth:</strong> The displayed softmax score reflects the model's internal statistical preference over binary labels under its training distribution. It is{' '}
                <strong className="text-stone-950">not a calibrated probability of real-world veracity</strong>.
              </li>
              <li>
                <strong className="text-stone-900">Empirical Benchmark Limitation:</strong> On the evaluated LIAR benchmark test split (1,016 short political claims evaluated under the project's documented binary-label mapping with half-true excluded), the current model (<code className="text-amber-900 font-mono text-xs bg-amber-100 px-1.5 py-0.5 rounded border border-amber-200">dima806/fake-news-classifier</code>) achieved{' '}
                <strong className="text-stone-950">49.21% accuracy</strong>, performing below the <strong className="text-stone-950">54.72% majority-class baseline</strong>.
              </li>
              <li>
                <strong className="text-stone-900">Scope of Findings:</strong> These metrics reflect performance under this specific protocol, not universal accuracy across arbitrary text domains.
              </li>
              <li>
                <strong className="text-stone-900">Independent Verification:</strong> Statistical language models cannot substitute for primary journalistic sourcing and credentialed fact-checking archives.
              </li>
            </ul>
          </div>
        </div>
      </aside>
    </div>
  );
};
