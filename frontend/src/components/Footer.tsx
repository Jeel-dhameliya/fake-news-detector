import React from 'react';
import { Info, AlertTriangle } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950/80 mt-12 py-8 text-xs text-slate-400">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        {/* Capabilities and Limitations Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-6 border-b border-slate-800/60">
          <div className="space-y-2">
            <h4 className="font-semibold text-slate-200 flex items-center gap-1.5 text-xs uppercase tracking-wider">
              <Info className="w-3.5 h-3.5 text-cyan-400" />
              Current Capabilities
            </h4>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              TruthLens performs text classification using a fine-tuned BERT transformer (<code>dima806/fake-news-classifier</code>)
              deployed through a local FastAPI backend. It extracts token-level representations, computes logits, and outputs
              class probabilities for submitted text.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold text-slate-200 flex items-center gap-1.5 text-xs uppercase tracking-wider">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              System Limitations
            </h4>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              TruthLens does <strong>not</strong> perform real-time web crawling, check live news databases, verify source URLs,
              or rank domain credibility. Outputs reflect historical machine-learning patterns and should not be used as the sole
              basis for evaluating journalistic credibility.
            </p>
          </div>
        </div>

        {/* Bottom Metadata */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-slate-500 text-[11px]">
          <span>
            TruthLens — AI Misinformation Detection Capstone Project
          </span>
          <div className="flex items-center space-x-3">
            <span>FastAPI &bull; PyTorch &bull; React &bull; Vite &bull; Tailwind CSS</span>
            <span>&bull;</span>
            <span className="text-cyan-400/90 font-medium">Milestone 3 Complete</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
