import React from 'react';
import { ShieldCheck, BarChart3, HelpCircle, AlertCircle } from 'lucide-react';

export const EmptyState: React.FC = () => {
  return (
    <div className="bg-slate-900/40 rounded-2xl border border-slate-800/80 p-8 sm:p-10 text-center backdrop-blur-sm">
      <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700/80 mx-auto flex items-center justify-center mb-5 text-cyan-400">
        <ShieldCheck className="w-8 h-8 text-cyan-400" />
      </div>

      <h3 className="text-xl font-semibold text-white mb-2">
        Awaiting Statement Analysis
      </h3>

      <p className="text-sm text-slate-400 max-w-md mx-auto mb-8 leading-relaxed">
        Enter a news article excerpt, headline, or claim above, or select one of the demonstration inputs to evaluate the text using the BERT classification pipeline.
      </p>

      {/* Overview of Current Features */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-left max-w-2xl mx-auto">
        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/70">
          <div className="flex items-center gap-2 mb-2 text-cyan-400">
            <BarChart3 className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Classification
            </span>
          </div>
          <p className="text-xs text-slate-400 leading-normal">
            Binary <span className="text-slate-300">REAL</span> vs.{' '}
            <span className="text-slate-300">FAKE</span> prediction computed via fine-tuned Transformer logits.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/70">
          <div className="flex items-center gap-2 mb-2 text-indigo-400">
            <HelpCircle className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Confidence
            </span>
          </div>
          <p className="text-xs text-slate-400 leading-normal">
            Normalized softmax probability distribution reflecting model certainty relative to its training corpus.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/70">
          <div className="flex items-center gap-2 mb-2 text-sky-400">
            <AlertCircle className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Responsible AI
            </span>
          </div>
          <p className="text-xs text-slate-400 leading-normal">
            Clear interpretation caveats highlighting statistical limits and dataset sensitivities.
          </p>
        </div>
      </div>
    </div>
  );
};
