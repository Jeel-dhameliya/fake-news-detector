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
    <header className="border-b border-stone-200/90 bg-[#faf9f5]/90 backdrop-blur-md sticky top-0 z-50 shadow-classic">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-18 py-3.5 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-stone-900 border border-amber-600/30 flex items-center justify-center shadow-sm text-amber-400">
            <ShieldCheck className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2.5">
              <span className="text-2xl font-serif font-bold tracking-tight text-stone-950">
                TruthLens
              </span>
              <span className="text-[10px] font-sans font-semibold uppercase tracking-wider px-2 py-0.5 rounded border border-stone-300 bg-stone-100/80 text-stone-700">
                Research Edition
              </span>
            </div>
            <p className="text-xs font-serif italic text-stone-500 hidden sm:block">
              Automated Misinformation & Text Credibility Analysis
            </p>
          </div>
        </div>

        {/* Right Status Indicators & Links */}
        <div className="flex items-center space-x-3 sm:space-x-4">
          {/* Active Model Pill */}
          <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-white border border-stone-200 shadow-classic text-xs text-stone-700">
            <Cpu className="w-3.5 h-3.5 text-amber-600" />
            <span className="font-mono text-[11px] text-stone-800">
              BERT ({modelState === 'ready' ? 'Loaded' : 'On-Demand'})
            </span>
          </div>

          {/* Backend Health Status with Accessible Recheck Trigger */}
          <button
            type="button"
            onClick={() => pollHealth()}
            disabled={isRefreshing}
            className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-white border border-stone-200 shadow-classic text-xs hover:bg-stone-50 hover:border-stone-300 transition-all focus:outline-none focus:ring-1 focus:ring-amber-500/50 cursor-pointer disabled:opacity-70"
            title="Backend health check (polls every 30s; click to recheck now)"
            aria-label="Backend health status. Click to recheck connectivity."
          >
            <Activity className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-amber-600' : 'text-stone-400'}`} />
            <span className="text-stone-500 hidden sm:inline font-serif italic">Engine:</span>
            {backendStatus === 'online' ? (
              <span className="inline-flex items-center text-emerald-800 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-600 mr-1.5 animate-pulse"></span>
                Active {serviceVersion ? `(v${serviceVersion})` : ''}
              </span>
            ) : backendStatus === 'offline' ? (
              <span className="inline-flex items-center text-rose-800 font-medium">
                <span className="w-2 h-2 rounded-full bg-rose-600 mr-1.5"></span>
                Disconnected (Retry)
              </span>
            ) : (
              <span className="text-stone-400">Verifying...</span>
            )}
          </button>

          {/* Repository Link */}
          <a
            href="https://github.com/Jeel-dhameliya/fake-news-detector"
            target="_blank"
            rel="noreferrer"
            className="p-2 text-stone-500 hover:text-stone-900 transition-colors rounded-lg hover:bg-stone-100 border border-transparent hover:border-stone-200"
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
