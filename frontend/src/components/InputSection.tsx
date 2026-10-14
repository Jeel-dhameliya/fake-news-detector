import React, { useState } from 'react';
import { Search, Link as LinkIcon, AlertCircle, FileText, RotateCcw } from 'lucide-react';

export const InputSection: React.FC = () => {
  const [newsText, setNewsText] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');

  const handleClear = () => {
    setNewsText('');
    setSourceUrl('');
  };

  const loadSample = (sampleText: string, sampleUrl: string = '') => {
    setNewsText(sampleText);
    setSourceUrl(sampleUrl);
  };

  return (
    <section aria-labelledby="input-heading" className="bg-slate-900/80 rounded-2xl border border-slate-800 shadow-xl p-5 sm:p-7 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <h2 id="input-heading" className="text-lg font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-cyan-400" />
          News Article or Claim
        </h2>
        
        {newsText && (
          <button
            onClick={handleClear}
            className="text-xs text-slate-400 hover:text-slate-200 inline-flex items-center gap-1 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      {/* Main Textarea */}
      <div className="relative mb-4">
        <label htmlFor="news-content-input" className="sr-only">
          Paste news article text or headline
        </label>
        <textarea
          id="news-content-input"
          value={newsText}
          onChange={(e) => setNewsText(e.target.value)}
          placeholder="Paste full news article text, headline, or excerpt here for misinformation and credibility assessment..."
          rows={6}
          className="w-full bg-slate-950/70 border border-slate-800 rounded-xl p-4 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/80 transition-all text-sm sm:text-base resize-y leading-relaxed"
        />
        <div className="flex justify-between items-center mt-1 px-1">
          <span className="text-xs text-slate-500">
            Min. 50 characters recommended for high-confidence model classification
          </span>
          <span className="text-xs font-mono text-slate-400">
            {newsText.length} characters
          </span>
        </div>
      </div>

      {/* Optional Source URL Input */}
      <div className="mb-6">
        <label htmlFor="source-url-input" className="block text-xs font-medium text-slate-400 mb-1.5">
          Source URL or Publisher Domain <span className="text-slate-500 font-normal">(Optional domain reputation check)</span>
        </label>
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
            <LinkIcon className="w-4 h-4" />
          </div>
          <input
            id="source-url-input"
            type="url"
            value={sourceUrl}
            onChange={(e) => setSourceUrl(e.target.value)}
            placeholder="e.g. https://www.reuters.com/... or https://suspicious-outlet.biz/..."
            className="w-full bg-slate-950/70 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/80 transition-all"
          />
        </div>
      </div>

      {/* Sample Presets for Quick Testing */}
      <div className="mb-6 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-slate-400 font-medium">Quick sample presets:</span>
        <button
          type="button"
          onClick={() => loadSample(
            "NASA confirmed today that the James Webb Space Telescope detected water vapor in the atmosphere of a rocky exoplanet orbiting within its star's habitable zone.",
            "https://www.nasa.gov"
          )}
          className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700/80 text-cyan-300 border border-slate-700 transition-colors"
        >
          Scientific Discovery (Real)
        </button>
        <button
          type="button"
          onClick={() => loadSample(
            "SHOCKING PROOF: Government secretly installing microchips in tap water to control thoughts of citizens across nationwide network! You won't believe what whistleblowers revealed!",
            "http://thetruthrevealed-breaking.xyz"
          )}
          className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700/80 text-amber-300 border border-slate-700 transition-colors"
        >
          Sensationalist Conspiracy (Fake)
        </button>
      </div>

      {/* Action Button & Milestone Notice */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800/80">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <AlertCircle className="w-4 h-4 text-cyan-400 shrink-0" />
          <span>
            Model inference pipeline will be activated in <strong>Milestone 2</strong>.
          </span>
        </div>

        <button
          type="button"
          disabled
          aria-disabled="true"
          className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-slate-800 text-slate-400 font-medium text-sm border border-slate-700/80 cursor-not-allowed shadow-inner transition-colors"
        >
          <Search className="w-4 h-4 text-slate-500" />
          Analyze Credibility (Milestone 2)
        </button>
      </div>
    </section>
  );
};
