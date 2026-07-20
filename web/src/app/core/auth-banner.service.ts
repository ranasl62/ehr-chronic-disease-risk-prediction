import { Injectable, signal } from '@angular/core';

/** Global banner for auth / API-key failures (set by interceptor). */
@Injectable({ providedIn: 'root' })
export class AuthBannerService {
  readonly message = signal<string | null>(null);

  set(msg: string | null): void {
    this.message.set(msg);
  }

  clear(): void {
    this.message.set(null);
  }
}
