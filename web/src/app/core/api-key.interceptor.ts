import { HttpInterceptorFn } from '@angular/common/http';

/** Optional X-API-Key from localStorage (research local use). */
export const apiKeyInterceptor: HttpInterceptorFn = (req, next) => {
  const key = localStorage.getItem('ehr_api_key')?.trim();
  if (key) {
    req = req.clone({ setHeaders: { 'X-API-Key': key } });
  }
  return next(req);
};
