import { TestBed } from '@angular/core/testing';
import { AuthBannerService } from './auth-banner.service';

describe('AuthBannerService', () => {
  let svc: AuthBannerService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    svc = TestBed.inject(AuthBannerService);
  });

  it('sets and clears banner message', () => {
    expect(svc.message()).toBeNull();
    svc.set('API key missing');
    expect(svc.message()).toBe('API key missing');
    svc.clear();
    expect(svc.message()).toBeNull();
  });
});
