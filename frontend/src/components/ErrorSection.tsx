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
      className="bg-rose-50/90 rounded-xl border border-rose-300 p-6 sm:p-8 shadow-classic"
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="w-10 h-10 rounded-lg bg-rose-100 border border-rose-300 flex items-center justify-center shrink-0 text-rose-700 mt-0.5 sm:mt-0 shadow-xs">
            <IconComponent className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-serif font-bold text-rose-950 mb-1">
              {errorTitle}
            </h3>
            <p className="text-xs sm:text-sm font-serif text-rose-900/90 leading-relaxed max-w-xl">
              {message}
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onRetry}
          className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-white hover:bg-rose-100/60 text-rose-900 border border-rose-300 font-serif font-bold text-xs transition-colors shrink-0 shadow-classic"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Retry Transmission</span>
        </button>
      </div>
    </div>
  );
};
