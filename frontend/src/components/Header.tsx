import React from 'react';
import { Cpu, Zap, Shield, Sparkles } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <div className="text-center max-w-3xl mx-auto pt-2 pb-6 px-4">
      {/* Editorial Badge */}
      <div className="inline-flex items-center space-x-2 px-3.5 py-1 rounded-full bg-amber-50 border border-amber-200/90 text-amber-900 text-xs font-serif italic mb-5 shadow-classic">
        <Sparkles className="w-3.5 h-3.5 text-amber-700" />
        <span>Natural Language Processing &bull; Algorithmic Credibility Analysis</span>
      </div>

      {/* Classic Masthead Heading */}
      <h1 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-bold tracking-tight text-stone-950 mb-3 leading-[1.15]">
        TruthLens <span className="font-serif italic font-normal text-stone-700 block sm:inline">&mdash; AI Misinformation Detector</span>
      </h1>

      <p className="text-base sm:text-lg font-serif italic text-stone-600 max-w-2xl mx-auto mb-4 leading-relaxed">
        Distinguishing syntactic classification from verified truth across news reporting and public statements.
      </p>

      <p className="text-xs sm:text-sm text-stone-500 font-sans max-w-xl mx-auto leading-relaxed mb-7">
        Automated sequence inference powered by fine-tuned transformer weights on an accelerated FastAPI server.
      </p>

      {/* Verified Feature Capabilities (Classic Editorial Badges) */}
      <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-xs">
        <span className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-white border border-stone-200 text-stone-800 shadow-classic">
          <Cpu className="w-3.5 h-3.5 mr-1.5 text-amber-700" />
          Hugging Face BERT (<code className="text-amber-800 font-mono text-[11px] ml-1 bg-amber-50 px-1 py-0.5 rounded border border-amber-200">dima806</code>)
        </span>
        <span className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-white border border-stone-200 text-stone-800 shadow-classic">
          <Zap className="w-3.5 h-3.5 mr-1.5 text-amber-700" />
          Hardware Acceleration (Apple Silicon MPS / CPU)
        </span>
        <span className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-white border border-stone-200 text-stone-800 shadow-classic">
          <Shield className="w-3.5 h-3.5 mr-1.5 text-amber-700" />
          Softmax Score Distribution
        </span>
      </div>
    </div>
  );
};
