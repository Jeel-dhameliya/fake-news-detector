import React from 'react';
import { Cpu, Zap, Shield, Sparkles } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <div className="text-center max-w-3xl mx-auto pt-4 pb-6 px-4">
      <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/80 text-cyan-300 text-xs font-medium mb-4">
        <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
        <span>Fine-Tuned Transformer Classification Pipeline</span>
      </div>

      <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-white mb-4">
        TruthLens <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-400">— AI Misinformation Detector</span>
      </h1>

      <p className="text-sm sm:text-base text-slate-300 font-normal leading-relaxed mb-6">
        Evaluate news headlines, claims, and article excerpts using a fine-tuned BERT text-classification model running on an Apple Silicon / CPU FastAPI backend.
      </p>

      {/* Verified Feature Capabilities (No Unimplemented Claims) */}
      <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-xs">
        <span className="inline-flex items-center px-3 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300">
          <Cpu className="w-3.5 h-3.5 mr-1.5 text-cyan-400" />
          Hugging Face BERT (<code className="text-cyan-300 font-mono text-[11px] ml-1">dima806</code>)
        </span>
        <span className="inline-flex items-center px-3 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300">
          <Zap className="w-3.5 h-3.5 mr-1.5 text-indigo-400" />
          Hardware Acceleration (Apple Silicon MPS / CPU)
        </span>
        <span className="inline-flex items-center px-3 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300">
          <Shield className="w-3.5 h-3.5 mr-1.5 text-sky-400" />
          Probability Distribution Output
        </span>
      </div>
    </div>
  );
};
