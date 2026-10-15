import React from 'react';
import { AlertCircle, RotateCcw, WifiOff, ServerCrash, Ban, Clock } from 'lucide-react';
import { ApiRequestError } from '../services/api';

interface ErrorSectionProps {
  error: Error | string;
  onRetry: () => void;
}

export const ErrorSection: React.FC<ErrorSectionProps> = ({ error, onRetry }) => {
  const isApiError = error instanceof ApiRequestError;
  const status = isApiError ? error.status : undefined;
  const detail = isApiError ? error.detail : undefined;
  const message = typeof error === 'string' ? error : error.message;

  // Select appropriate icon based on status
  let IconComponent = AlertCircle;
  let errorTitle = 'Classification Error';

  if (status === 422) {
    IconComponent = Ban;
    errorTitle = 'Input Validation Error (HTTP 422)';
  } else if (status === 503) {
    IconComponent = ServerCrash;
    errorTitle = 'Model Service Unavailable (HTTP 503)';
  } else if (status === 408 || detail === 'TIMEOUT' || message.includes('timed out')) {
    IconComponent = Clock;
    errorTitle = 'Request Timeout (15-Second Limit)';
  } else if (status === 0 || message.includes('Network connection error')) {
    IconComponent = WifiOff;
    errorTitle = 'Backend Connection Error';
  } else if (status && status >= 500) {
    IconComponent = ServerCrash;
    errorTitle = `Server Error (HTTP ${status})`;
  }

  return (
    <div
      role="alert"
      className="bg-rose-950/20 rounded-2xl border border-rose-500/30 p-6 sm:p-8 backdrop-blur-sm shadow-xl"
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center shrink-0 text-rose-400 mt-0.5 sm:mt-0">
            <IconComponent className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-rose-300 mb-1">
              {errorTitle}
            </h3>
            <p className="text-xs sm:text-sm text-rose-200/90 leading-relaxed max-w-xl">
              {message}
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onRetry}
          className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-rose-300 hover:text-rose-200 border border-rose-800/60 font-medium text-xs transition-colors shrink-0"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Retry Request</span>
        </button>
      </div>
    </div>
  );
};
