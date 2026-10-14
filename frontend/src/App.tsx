import React from 'react';
import { Navbar } from './components/Navbar';
import { Header } from './components/Header';
import { InputSection } from './components/InputSection';
import { EmptyState } from './components/EmptyState';

export const App: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-navy-900 text-slate-100 selection:bg-cyan-500 selection:text-navy-900">
      <Navbar />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <Header />
        
        <InputSection />

        <section aria-label="Analysis Output">
          <EmptyState />
        </section>
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-950/70 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <span>
            TruthLens — College Capstone Project on AI Misinformation Detection
          </span>
          <div className="flex items-center space-x-4">
            <span>FastAPI &bull; React &bull; Transformers</span>
            <span>&bull;</span>
            <span className="text-slate-400">Milestone 1 Complete</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
