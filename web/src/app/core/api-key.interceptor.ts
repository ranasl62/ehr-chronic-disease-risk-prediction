import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { AuthBannerService } from './auth-banner.service';

/** Optional X-API-Key from localStorage (research local use). Surfaces 401 clearly. */
export const apiKeyInterceptor: HttpInterceptorFn = (req, next) => {
  const banner = inject(AuthBannerService);
  const key = localStorage.getItem('ehr_api_key')?.trim();
  if (key) {
    req = req.clone({ setHeaders: { 'X-API-Key': key } });
  }
  return next(req).pipe(
    catchError((err: unknown) => {
      if (err instanceof HttpErrorResponse && err.status === 401) {
        const detail =
          typeof err.error?.detail === 'string'
            ? err.error.detail
            : 'Invalid or missing API key';
        banner.set(
          `${detail}. Set X-API-Key under Config (browser localStorage), matching the server API_KEY.`
        );
      }
      return throwError(() => err);
    })
  );
};
