import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ShieldCheck, Activity, Github, Cpu } from 'lucide-react';
import { checkBackendHealth } from '../services/api';

export const Navbar: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [serviceVersion, setServiceVersion] = useState<string | null>(null);
  const [modelState, setModelState] = useState<'ready' | 'unloaded' | 'unknown'>('unknown');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const isCheckingRef = useRef(false);

  const pollHealth = useCallback(async () => {
    if (isCheckingRef.current) return;
    isCheckingRef.current = true;
    setIsRefreshing(true);

    try {
      const data = await checkBackendHealth();
      // api_liveness confirms backend process availability
      if (data.status === 'healthy' || data.api_liveness === 'alive') {
        setBackendStatus('online');
        setServiceVersion(data.version || null);
        if (data.model_readiness === 'ready') {
          setModelState('ready');
        } else if (data.model_readiness === 'unloaded') {
          setModelState('unloaded');
        } else {
          setModelState('unknown');
        }
      } else {
        setBackendStatus('offline');
        setModelState('unknown');
      }
    } catch {
      setBackendStatus('offline');
      setModelState('unknown');
    } finally {
      isCheckingRef.current = false;
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    // Initial health check
    pollHealth();

    // 30-second polling interval for automatic connectivity recovery
    const timer = setInterval(() => {
      pollHealth();
    }, 30000);

    return () => clearInterval(timer);
  }, [pollHealth]);

  return (
    <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white">TruthLens</span>
              <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300">
                Research
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">AI Misinformation & Fake News Classifier</p>
          </div>
        </div>

        {/* Right Status Indicators & Links */}
        <div className="flex items-center space-x-3 sm:space-x-4">
          {/* Active Model Pill */}
          <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-mono text-[11px] text-slate-300">
              BERT ({modelState === 'ready' ? 'Loaded' : 'On-Demand'})
            </span>
          </div>

          {/* Backend Health Status with Accessible Recheck Trigger */}
          <button
            type="button"
            onClick={() => pollHealth()}
            disabled={isRefreshing}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs hover:border-slate-700 transition-colors focus:outline-none focus:ring-1 focus:ring-cyan-500/50 cursor-pointer disabled:opacity-70"
            title="Backend health check (polls every 30s; click to recheck now)"
            aria-label="Backend health status. Click to recheck connectivity."
          >
            <Activity className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-cyan-400' : 'text-slate-400'}`} />
            <span className="text-slate-400 hidden sm:inline">Backend:</span>
            {backendStatus === 'online' ? (
              <span className="inline-flex items-center text-emerald-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
                Online {serviceVersion ? `(v${serviceVersion})` : ''}
              </span>
            ) : backendStatus === 'offline' ? (
              <span className="inline-flex items-center text-amber-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-amber-400 mr-1.5"></span>
                Offline (Click to retry)
              </span>
            ) : (
              <span className="text-slate-500">Checking...</span>
            )}
          </button>

          {/* Repository Link */}
          <a
            href="https://github.com/Jeel-dhameliya/fake-news-detector"
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
