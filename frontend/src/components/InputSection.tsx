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
      className="bg-white rounded-xl border border-stone-200/90 shadow-elevated p-6 sm:p-8 transition-all"
    >
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-4 mb-4 border-b border-stone-100">
        <div>
          <h2 id="input-panel-heading" className="text-xl sm:text-2xl font-serif font-bold text-stone-900 flex items-center gap-2.5">
            <FileText className="w-5 h-5 text-amber-700" />
            Claim Examination Manuscript
          </h2>
          <p className="text-xs font-serif italic text-stone-500 mt-0.5">
            Submit a news passage, headline, or political assertion for transformer evaluation
          </p>
        </div>

        {value && (
          <button
            type="button"
            onClick={onClear}
            disabled={isLoading}
            className="text-xs font-serif text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5 transition-colors px-2.5 py-1 rounded-md hover:bg-stone-100 border border-stone-200/60 disabled:opacity-50"
            title="Clear input and results"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Form
          </button>
        )}
      </div>

      {/* Main Input Textarea */}
      <div className="relative mb-4">
        <label htmlFor="news-content-input" className="sr-only">
          Paste news article text, headline, or claim to analyze
        </label>
        <textarea
          id="news-content-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          placeholder="Paste or type a journalistic excerpt, report, or headline to evaluate with the classification engine..."
          rows={6}
          aria-describedby="char-count input-hint"
          className="w-full bg-[#fdfcf7] border border-stone-300 rounded-lg p-4 sm:p-5 text-stone-900 placeholder:text-stone-400 placeholder:font-serif placeholder:italic focus:bg-white focus:outline-none focus:ring-2 focus:ring-amber-600/20 focus:border-amber-700 transition-all font-serif text-base sm:text-lg resize-y leading-relaxed disabled:opacity-60 disabled:cursor-not-allowed shadow-inner"
        />

        {/* Live Counters & Safe Length Feedback */}
        <div id="char-count" className="flex justify-between items-center mt-2 px-1">
          <span id="input-hint" className="text-xs font-serif italic text-stone-500">
            Press <kbd className="px-1.5 py-0.5 rounded bg-stone-100 border border-stone-300 text-[11px] text-stone-700 font-mono">Cmd/Ctrl + Enter</kbd> to analyze
          </span>
          <span
            className={`text-xs font-mono transition-colors ${
              isOverLimit ? 'text-rose-700 font-semibold' : 'text-stone-500'
            }`}
          >
            {value.length.toLocaleString()} / {MAX_CHAR_LIMIT.toLocaleString()} chars
          </span>
        </div>
      </div>

      {/* Demonstration Sample Presets */}
      <div className="mb-6 pt-3 pb-1 border-t border-stone-100">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs font-serif font-semibold text-stone-700 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-700" />
            Archival test dispatches (for demonstration):
          </span>
          <span className="text-[11px] font-serif italic text-stone-400 flex items-center gap-1">
            <HelpCircle className="w-3 h-3" />
            Unverified demonstration data
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {DEMO_SAMPLES.map((sample) => (
            <button
              key={sample.id}
              type="button"
              disabled={isLoading}
              onClick={() => onSelectSample(sample.text)}
              className="text-left px-3.5 py-2 rounded-lg bg-stone-50 hover:bg-amber-50/70 border border-stone-200/90 hover:border-amber-300 text-xs text-stone-800 transition-all focus:outline-none focus:ring-1 focus:ring-amber-600/60 disabled:opacity-50 shadow-classic"
            >
              <span className="font-serif font-bold text-amber-800 mr-1.5">[{sample.tag}]</span>
              <span className="font-serif">{sample.category}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Action Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-stone-200/80">
        <div className="text-xs font-serif text-stone-500">
          Inference Target: <code className="text-stone-800 bg-stone-100 px-2 py-0.5 rounded border border-stone-200 font-mono text-[11px]">POST /api/predict</code>
        </div>

        <button
          type="button"
          onClick={onAnalyze}
          disabled={isAnalyzeDisabled}
          aria-busy={isLoading}
          className={`w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-3 rounded-lg font-serif font-bold text-sm tracking-wide transition-all shadow-md ${
            isAnalyzeDisabled
              ? 'bg-stone-100 text-stone-400 border border-stone-200 cursor-not-allowed shadow-none'
              : 'bg-stone-900 hover:bg-stone-800 text-amber-100 hover:text-white border border-stone-800 shadow-stone-900/10 active:scale-[0.99]'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-amber-300" />
              <span>Executing Classification...</span>
            </>
          ) : (
            <>
              <Search className="w-4 h-4 text-amber-400" />
              <span>Evaluate Statement</span>
            </>
          )}
        </button>
      </div>
    </section>
  );
};
