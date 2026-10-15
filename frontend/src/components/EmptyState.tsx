import React from 'react';
import { ShieldCheck, BarChart3, HelpCircle, AlertCircle } from 'lucide-react';

export const EmptyState: React.FC = () => {
  return (
    <div className="bg-white rounded-xl border border-stone-200/90 p-8 sm:p-12 text-center shadow-classic">
      <div className="w-16 h-16 rounded-xl bg-amber-50 border border-amber-200/80 mx-auto flex items-center justify-center mb-5 text-amber-800 shadow-classic">
        <ShieldCheck className="w-8 h-8 text-amber-800" />
      </div>

      <h3 className="text-2xl font-serif font-bold text-stone-950 mb-2">
        Awaiting Claim Submission
      </h3>

      <p className="text-base font-serif italic text-stone-600 max-w-md mx-auto mb-8 leading-relaxed">
        Submit a news passage, headline, or dispatch above to initiate transformer-based linguistic credibility evaluation.
      </p>

      {/* Overview of Current Features */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-left max-w-2xl mx-auto">
        <div className="p-5 rounded-lg bg-stone-50/80 border border-stone-200 shadow-classic">
          <div className="flex items-center gap-2 mb-2 text-amber-800">
            <BarChart3 className="w-4 h-4" />
            <span className="text-xs font-serif uppercase tracking-widest font-bold text-stone-900">
              Classification
            </span>
          </div>
          <p className="text-xs font-serif text-stone-600 leading-relaxed">
            Binary <span className="font-bold text-stone-900">REAL</span> vs.{' '}
            <span className="font-bold text-stone-900">FAKE</span> projection derived from fine-tuned BERT transformer logits.
          </p>
        </div>

        <div className="p-5 rounded-lg bg-stone-50/80 border border-stone-200 shadow-classic">
          <div className="flex items-center gap-2 mb-2 text-amber-800">
            <HelpCircle className="w-4 h-4" />
            <span className="text-xs font-serif uppercase tracking-widest font-bold text-stone-900">
              Confidence
            </span>
          </div>
          <p className="text-xs font-serif text-stone-600 leading-relaxed">
            Normalized softmax probability distribution reflecting model preference relative to its fine-tuning distribution.
          </p>
        </div>

        <div className="p-5 rounded-lg bg-stone-50/80 border border-stone-200 shadow-classic">
          <div className="flex items-center gap-2 mb-2 text-amber-800">
            <AlertCircle className="w-4 h-4" />
            <span className="text-xs font-serif uppercase tracking-widest font-bold text-stone-900">
              Epistemics
            </span>
          </div>
          <p className="text-xs font-serif text-stone-600 leading-relaxed">
            Rigorous contextual disclosures distinguishing syntax patterns from factual ground truth verification.
          </p>
        </div>
      </div>
    </div>
  );
};
