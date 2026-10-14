import React from 'react';
import { Cpu, Eye, ShieldAlert, Sparkles } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <div className="text-center max-w-3xl mx-auto pt-8 pb-6 px-4">
      <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-950/60 border border-indigo-800/80 text-indigo-300 text-xs font-medium mb-4">
        <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
        <span>Transparent, Explainable Misinformation Detection</span>
      </div>

      <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-white mb-4">
        TruthLens <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-400">— AI Fake News Detector</span>
      </h1>

      <p className="text-base sm:text-lg text-slate-300 font-normal leading-relaxed mb-6">
        Evaluate news statements and articles through fine-tuned Transformer models, token-level interpretability (XAI), and domain reputation signals.
      </p>

      {/* Feature Pills */}
      <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-xs">
        <span className="inline-flex items-center px-3 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300">
          <Cpu className="w-3.5 h-3.5 mr-1.5 text-cyan-400" />
          Hugging Face BERT Classifier
        </span>
        <span className="inline-flex items-center px-3 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300">
          <Eye className="w-3.5 h-3.5 mr-1.5 text-indigo-400" />
          Explainable AI (LIME Token Heatmap)
        </span>
        <span className="inline-flex items-center px-3 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300">
          <ShieldAlert className="w-3.5 h-3.5 mr-1.5 text-sky-400" />
          Verifiable Domain Trust
        </span>
      </div>
    </div>
  );
};
