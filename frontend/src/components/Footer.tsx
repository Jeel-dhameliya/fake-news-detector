import React from 'react';
import { Info, AlertTriangle } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-stone-200/90 bg-[#f7f6f0] mt-16 py-10 text-xs text-stone-600">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-7">
        {/* Capabilities and Limitations Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pb-7 border-b border-stone-200">
          <div className="space-y-2">
            <h4 className="font-serif font-bold text-stone-900 flex items-center gap-2 text-xs uppercase tracking-widest">
              <Info className="w-4 h-4 text-amber-700" />
              Verified Engine Capabilities
            </h4>
            <p className="font-serif text-stone-600 leading-relaxed text-xs sm:text-[13px]">
              TruthLens executes sequence classification using fine-tuned BERT transformer weights (<code className="text-amber-900 font-mono text-[11px] bg-white px-1.5 py-0.5 rounded border border-stone-300">dima806/fake-news-classifier</code>)
              served via a local asynchronous FastAPI application. It generates contextual embeddings, computes classification logits, and outputs
              normalized probability distributions for submitted news text.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-serif font-bold text-stone-900 flex items-center gap-2 text-xs uppercase tracking-widest">
              <AlertTriangle className="w-4 h-4 text-amber-700" />
              Architectural Limitations
            </h4>
            <p className="font-serif text-stone-600 leading-relaxed text-xs sm:text-[13px]">
              TruthLens does <strong className="text-stone-900">not</strong> execute live web queries, cross-reference external fact-check databases, verify cited publisher URLs,
              or calibrate probabilities against factual ground truth. Softmax outputs measure syntactic text alignment with training labels and cannot serve as the sole criterion for editorial truth.
            </p>
          </div>
        </div>

        {/* Bottom Metadata */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-stone-500 font-serif italic text-xs">
          <span>
            TruthLens &mdash; Academic Capstone Research Prototype &bull; Class of 2026
          </span>
          <div className="flex items-center space-x-3 font-sans not-italic text-[11px] text-stone-500">
            <span>FastAPI &bull; PyTorch MPS &bull; React 18 &bull; Vite &bull; Tailwind CSS</span>
            <span>&bull;</span>
            <span className="text-amber-800 font-semibold">Research Monograph</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
