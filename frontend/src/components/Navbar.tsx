import React, { useEffect, useState } from 'react';
import { ShieldCheck, Activity, Github } from 'lucide-react';

export const Navbar: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    // Check backend health
    fetch('/api/health')
      .then((res) => {
        if (res.ok) return res.json();
        throw new Error('Not healthy');
      })
      .then(() => setBackendStatus('online'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  return (
    <header className="border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white">TruthLens</span>
              <span className="text-[11px] font-semibold uppercase px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300">
                Milestone 1
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">AI Fake News & Misinformation Detector</p>
          </div>
        </div>

        {/* Right Status & Links */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <Activity className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">Backend API:</span>
            {backendStatus === 'online' ? (
              <span className="inline-flex items-center text-emerald-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
                Connected
              </span>
            ) : backendStatus === 'offline' ? (
              <span className="inline-flex items-center text-amber-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-amber-400 mr-1.5"></span>
                Offline (Start API)
              </span>
            ) : (
              <span className="text-slate-500">Connecting...</span>
            )}
          </div>

          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="p-2 text-slate-400 hover:text-slate-200 transition-colors rounded-lg hover:bg-slate-800/60"
            title="GitHub Repository"
            aria-label="GitHub Repository"
          >
            <Github className="w-5 h-5" />
          </a>
        </div>
      </div>
    </header>
  );
};
