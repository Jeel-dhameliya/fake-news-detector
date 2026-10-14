import React from 'react';
import { ShieldCheck, BarChart3, HelpCircle, Globe2 } from 'lucide-react';

export const EmptyState: React.FC = () => {
  return (
    <div className="bg-slate-900/40 rounded-2xl border border-slate-800/80 p-8 sm:p-10 text-center backdrop-blur-sm">
      <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700/80 mx-auto flex items-center justify-center mb-5 text-cyan-400">
        <ShieldCheck className="w-8 h-8 text-cyan-400" />
      </div>

      <h3 className="text-xl font-semibold text-white mb-2">
        Awaiting Analysis
      </h3>

      <p className="text-sm text-slate-400 max-w-md mx-auto mb-8 leading-relaxed">
        Once text is submitted for classification, detailed transparency metrics and verification results will be rendered here.
      </p>

      {/* Overview of Planned Modules */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-left max-w-2xl mx-auto">
        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/70">
          <div className="flex items-center gap-2 mb-2 text-cyan-400">
            <BarChart3 className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">Classification</span>
          </div>
          <p className="text-xs text-slate-400 leading-normal">
            Binary Fake vs. Real verdict with calibrated probability distribution.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/70">
          <div className="flex items-center gap-2 mb-2 text-indigo-400">
            <HelpCircle className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">XAI Heatmap</span>
          </div>
          <p className="text-xs text-slate-400 leading-normal">
            LIME token-importance weights showing words influencing the prediction.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/70">
          <div className="flex items-center gap-2 mb-2 text-sky-400">
            <Globe2 className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">Domain Safety</span>
          </div>
          <p className="text-xs text-slate-400 leading-normal">
            Audited publisher reputation & transparent domain categorization signals.
          </p>
        </div>
      </div>
    </div>
  );
};
