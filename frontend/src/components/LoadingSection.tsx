import React from 'react';
import { Loader2, Cpu } from 'lucide-react';

export const LoadingSection: React.FC = () => {
  return (
    <div
      role="status"
      aria-live="polite"
      className="bg-white rounded-xl border border-stone-200 p-8 sm:p-12 text-center shadow-classic"
    >
      <div className="w-16 h-16 rounded-xl bg-amber-50 border border-amber-200 mx-auto flex items-center justify-center mb-5 text-amber-800 shadow-classic">
        <Loader2 className="w-8 h-8 animate-spin text-amber-800" />
      </div>

      <h3 className="text-2xl font-serif font-bold text-stone-900 mb-2">
        Conducting Linguistic Analysis
      </h3>

      <p className="text-sm sm:text-base font-serif italic text-stone-600 max-w-md mx-auto mb-6 leading-relaxed">
        Tokenizing statement structure and executing forward pass across fine-tuned BERT transformer layers...
      </p>

      {/* Progress Skeleton Bar */}
      <div className="max-w-md mx-auto space-y-2.5">
        <div className="w-full bg-stone-100 rounded-full h-2 overflow-hidden border border-stone-200">
          <div className="bg-gradient-to-r from-amber-600 via-amber-700 to-stone-900 h-2 rounded-full w-2/3 animate-pulse"></div>
        </div>
        <div className="flex items-center justify-center gap-2 text-xs text-stone-500 font-mono">
          <Cpu className="w-3.5 h-3.5 text-amber-800" />
          <span>Inference Endpoint: POST /api/predict</span>
        </div>
      </div>
    </div>
  );
};
