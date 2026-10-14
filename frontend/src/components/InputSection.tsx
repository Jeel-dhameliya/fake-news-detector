import React from 'react';
import { FileText, RotateCcw, Search, Loader2, Sparkles, HelpCircle } from 'lucide-react';

interface InputSectionProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  onClear: () => void;
  isLoading: boolean;
  onSelectSample: (text: string) => void;
}

const MAX_CHAR_LIMIT = 50000;

interface DemoSample {
  id: string;
  category: string;
  tag: string;
  text: string;
}

const DEMO_SAMPLES: DemoSample[] = [
  {
    id: 'science-sample',
    category: 'Scientific Excerpt',
    tag: 'Demo Sample 1',
    text: 'NASA confirmed today that the James Webb Space Telescope detected water vapor in the atmosphere of a rocky exoplanet orbiting within its host star habitable zone.',
  },
  {
    id: 'conspiracy-sample',
    category: 'Conspiracy Headline',
    tag: 'Demo Sample 2',
    text: 'SHOCKING PROOF: Government secretly installing microchips in tap water to control thoughts of citizens across nationwide network! Whistleblowers reveal explosive hidden documents!',
  },
  {
    id: 'policy-sample',
    category: 'Public Policy',
    tag: 'Demo Sample 3',
    text: 'The government announced a new public health policy today regarding regional immunization clinics and preventative care programs.',
  },
];

export const InputSection: React.FC<InputSectionProps> = ({
  value,
  onChange,
  onAnalyze,
  onClear,
  isLoading,
  onSelectSample,
}) => {
  const isWhitespaceOnly = !value.trim();
  const isOverLimit = value.length > MAX_CHAR_LIMIT;
  const isAnalyzeDisabled = isLoading || isWhitespaceOnly || isOverLimit;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Allow Ctrl+Enter or Cmd+Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !isAnalyzeDisabled) {
      e.preventDefault();
      onAnalyze();
    }
  };

  return (
    <section
      aria-labelledby="input-panel-heading"
      className="bg-slate-900/80 rounded-2xl border border-slate-800 shadow-xl p-5 sm:p-7 backdrop-blur-sm transition-all"
    >
      {/* Panel Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 id="input-panel-heading" className="text-lg font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-cyan-400" />
          News Article or Statement
        </h2>

        {value && (
          <button
            type="button"
            onClick={onClear}
            disabled={isLoading}
            className="text-xs text-slate-400 hover:text-slate-200 inline-flex items-center gap-1.5 transition-colors px-2 py-1 rounded-md hover:bg-slate-800 disabled:opacity-50"
            title="Clear input and results"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      {/* Main Input Textarea */}
      <div className="relative mb-3">
        <label htmlFor="news-content-input" className="sr-only">
          Paste news article text, headline, or claim to analyze
        </label>
        <textarea
          id="news-content-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          placeholder="Paste news headline, article excerpt, or claim statement here for BERT transformer classification..."
          rows={6}
          aria-describedby="char-count input-hint"
          className="w-full bg-slate-950/70 border border-slate-800 rounded-xl p-4 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/80 transition-all text-sm sm:text-base resize-y leading-relaxed disabled:opacity-60 disabled:cursor-not-allowed"
        />

        {/* Live Counters & Safe Length Feedback */}
        <div id="char-count" className="flex justify-between items-center mt-1 px-1">
          <span id="input-hint" className="text-xs text-slate-400">
            Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-[11px] text-slate-300 font-mono">Cmd/Ctrl + Enter</kbd> to analyze
          </span>
          <span
            className={`text-xs font-mono transition-colors ${
              isOverLimit ? 'text-rose-400 font-semibold' : 'text-slate-400'
            }`}
          >
            {value.length.toLocaleString()} / {MAX_CHAR_LIMIT.toLocaleString()} chars
          </span>
        </div>
      </div>

      {/* Demonstration Sample Presets */}
      <div className="mb-6 pt-2">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            Demonstration inputs (for UI testing):
          </span>
          <span className="text-[11px] text-slate-500 flex items-center gap-1">
            <HelpCircle className="w-3 h-3" />
            Not verified ground truth
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {DEMO_SAMPLES.map((sample) => (
            <button
              key={sample.id}
              type="button"
              disabled={isLoading}
              onClick={() => onSelectSample(sample.text)}
              className="text-left px-3 py-1.5 rounded-lg bg-slate-950/70 hover:bg-slate-800 border border-slate-800/80 hover:border-slate-700 text-xs text-slate-300 transition-all focus:outline-none focus:ring-1 focus:ring-cyan-500/60 disabled:opacity-50"
            >
              <span className="text-cyan-400 font-medium mr-1.5">[{sample.tag}]</span>
              {sample.category}
            </button>
          ))}
        </div>
      </div>

      {/* Action Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800/80">
        <div className="text-xs text-slate-400">
          Endpoint target: <code className="text-cyan-300 font-mono text-[11px]">POST /api/predict</code>
        </div>

        <button
          type="button"
          onClick={onAnalyze}
          disabled={isAnalyzeDisabled}
          aria-busy={isLoading}
          className={`w-full sm:w-auto inline-flex items-center justify-center gap-2 px-7 py-3 rounded-xl font-medium text-sm transition-all shadow-lg ${
            isAnalyzeDisabled
              ? 'bg-slate-800 text-slate-500 border border-slate-700/60 cursor-not-allowed shadow-none'
              : 'bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white shadow-cyan-500/20 active:scale-[0.98]'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Analyzing Statement...</span>
            </>
          ) : (
            <>
              <Search className="w-4 h-4" />
              <span>Analyze Statement</span>
            </>
          )}
        </button>
      </div>
    </section>
  );
};
