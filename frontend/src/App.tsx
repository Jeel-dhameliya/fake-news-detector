import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Header } from './components/Header';
import { InputSection } from './components/InputSection';
import { ResultsSection } from './components/ResultsSection';
import { LoadingSection } from './components/LoadingSection';
import { ErrorSection } from './components/ErrorSection';
import { EmptyState } from './components/EmptyState';
import { Footer } from './components/Footer';
import { predictNews } from './services/api';
import { PredictionResponse } from './types/prediction';

export const App: React.FC = () => {
  const [newsText, setNewsText] = useState<string>('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<Error | string | null>(null);

  /**
   * Handle text changes from the user and clear stale results.
   */
  const handleTextChange = (newText: string) => {
    setNewsText(newText);
    // Automatically clear stale outputs when user modifies input
    if (result !== null || error !== null || status !== 'idle') {
      setResult(null);
      setError(null);
      setStatus('idle');
    }
  };

  /**
   * Clears the input and all prediction results.
   */
  const handleClear = () => {
    setNewsText('');
    setResult(null);
    setError(null);
    setStatus('idle');
  };

  /**
   * Loads a demonstration sample and clears stale results.
   */
  const handleSelectSample = (sampleText: string) => {
    setNewsText(sampleText);
    setResult(null);
    setError(null);
    setStatus('idle');
  };

  /**
   * Performs prediction against the FastAPI backend.
   */
  const handleAnalyze = async () => {
    // Guard against duplicate concurrent submissions
    if (status === 'loading') {
      return;
    }

    const trimmed = newsText.trim();
    if (!trimmed) {
      setError(new Error('Please enter text to analyze.'));
      setStatus('error');
      return;
    }

    setStatus('loading');
    setError(null);

    try {
      const response = await predictNews(trimmed);
      setResult(response);
      setStatus('success');
    } catch (err) {
      setError(err instanceof Error ? err : String(err));
      setStatus('error');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-navy-900 text-slate-100 selection:bg-cyan-500 selection:text-navy-900">
      <Navbar />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-10 space-y-8">
        <Header />

        {/* Input Panel */}
        <InputSection
          value={newsText}
          onChange={handleTextChange}
          onAnalyze={handleAnalyze}
          onClear={handleClear}
          isLoading={status === 'loading'}
          onSelectSample={handleSelectSample}
        />

        {/* Output Section */}
        <section aria-label="Prediction Results" className="pt-2">
          {status === 'loading' && <LoadingSection />}

          {status === 'error' && error && (
            <ErrorSection error={error} onRetry={handleAnalyze} />
          )}

          {status === 'success' && result && (
            <ResultsSection result={result} />
          )}

          {status === 'idle' && !result && !error && <EmptyState />}
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default App;
