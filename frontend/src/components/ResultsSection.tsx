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
        className={`rounded-2xl border p-6 sm:p-8 backdrop-blur-sm shadow-xl transition-all ${
          isReal
            ? 'bg-slate-900/90 border-emerald-500/30 shadow-emerald-950/20'
            : 'bg-slate-900/90 border-rose-500/30 shadow-rose-950/20'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800/80">
          <div className="flex items-center gap-4">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 shadow-lg ${
                isReal
                  ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
                  : 'bg-rose-500/10 border border-rose-500/30 text-rose-400'
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
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Model Prediction
                </span>
                <span className="text-[11px] text-slate-500 font-normal">
                  (Binary Classifier Verdict)
                </span>
              </div>
              <div className="flex items-center gap-3">
                <h3
                  className={`text-2xl sm:text-3xl font-extrabold tracking-tight ${
                    isReal ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {result.predicted_label}
                </h3>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold border ${
                    isReal
                      ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                      : 'bg-rose-950/60 border-rose-800 text-rose-300'
                  }`}
                >
                  {confidencePercent}% Confidence
                </span>
              </div>
            </div>
          </div>

          {/* Model & Device Metadata */}
          <div className="flex sm:flex-col items-end gap-1.5 text-xs text-slate-400">
            <div className="inline-flex items-center gap-1.5 font-mono text-[11px]">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>{result.model_id}</span>
            </div>
            <div className="inline-flex items-center gap-1.5 font-mono text-[11px]">
              <Zap className="w-3.5 h-3.5 text-indigo-400" />
              <span>Device: <strong className="text-slate-300">{result.device}</strong></span>
            </div>
          </div>
        </div>

        {/* Probability Distribution */}
        <div className="pt-6">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              Class Probability Distribution
            </span>
            <span className="text-xs text-slate-400">
              Softmax output over fine-tuned classes
            </span>
          </div>

          {/* Dual Bar Visualizer */}
          <div className="space-y-3">
            {/* REAL Class */}
            <div>
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-emerald-300 font-medium flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                  Class: REAL
                </span>
                <span className="font-mono text-slate-200 font-semibold">{realPercent}%</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                <div
                  className="bg-gradient-to-r from-emerald-500 to-teal-400 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${realPercent}%` }}
                ></div>
              </div>
            </div>

            {/* FAKE Class */}
            <div>
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-rose-300 font-medium flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-400"></span>
                  Class: FAKE
                </span>
                <span className="font-mono text-slate-200 font-semibold">{fakePercent}%</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                <div
                  className="bg-gradient-to-r from-rose-500 to-amber-500 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${fakePercent}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mandatory Responsible Presentation & Interpretation Notice */}
      <div className="rounded-2xl border border-amber-500/30 bg-amber-950/20 p-5 sm:p-6 backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-2 text-xs leading-relaxed text-amber-200/90">
            <h4 className="font-semibold text-amber-300 text-sm">
              Critical Interpretation Notice & Model Limitations
            </h4>
            <p>
              This verdict is an <strong>automated statistical machine-learning prediction</strong> derived from
              a BERT transformer fine-tuned on historical benchmark news datasets. It is{' '}
              <strong>not proof of factual truth or falsity</strong>.
            </p>
            <ul className="list-disc pl-4 space-y-1 text-amber-300/80">
              <li>
                <strong>Model Confidence vs. Factuality:</strong> The confidence score reflects how closely the sentence
                matches statistical patterns from the model's training data. High confidence does <em>not</em> guarantee factual accuracy.
              </li>
              <li>
                <strong>No Verification of Underlying Claims:</strong> A <code className="text-emerald-300 font-mono">REAL</code> prediction
                does not independently verify a statement, and a <code className="text-rose-300 font-mono">FAKE</code> prediction
                does not prove malicious misinformation.
              </li>
              <li>
                <strong>Known Domain Sensitivities:</strong> Sentence length, grammatical style, and phrasing can influence
                predictions independently of real-world veracity. Always verify news claims using primary sources and professional fact-checking organizations.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
