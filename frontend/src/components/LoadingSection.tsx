import React from 'react';
import { Loader2, Cpu } from 'lucide-react';

export const LoadingSection: React.FC = () => {
  return (
    <div
      role="status"
      aria-live="polite"
      className="bg-slate-900/60 rounded-2xl border border-cyan-500/20 p-8 sm:p-10 text-center backdrop-blur-sm shadow-xl"
    >
      <div className="w-16 h-16 rounded-2xl bg-cyan-950/60 border border-cyan-800/80 mx-auto flex items-center justify-center mb-5 text-cyan-400">
        <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
      </div>

      <h3 className="text-xl font-semibold text-white mb-2">
        Evaluating Statement
      </h3>

      <p className="text-sm text-slate-300 max-w-md mx-auto mb-6 leading-relaxed">
        Tokenizing text and performing PyTorch forward pass on Hugging Face BERT classifier...
      </p>

      {/* Progress Skeleton Bar */}
      <div className="max-w-md mx-auto space-y-2">
        <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
          <div className="bg-gradient-to-r from-cyan-500 via-sky-400 to-indigo-500 h-2 rounded-full w-2/3 animate-pulse"></div>
        </div>
        <div className="flex items-center justify-center gap-2 text-xs text-slate-400 font-mono">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>Endpoint: POST /api/predict</span>
        </div>
      </div>
    </div>
  );
};
