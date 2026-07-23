import { TestBed } from '@angular/core/testing';
import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { apiKeyInterceptor } from './api-key.interceptor';
import { AuthBannerService } from './auth-banner.service';

describe('apiKeyInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;
  let banner: AuthBannerService;

  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([apiKeyInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
    banner = TestBed.inject(AuthBannerService);
  });

  afterEach(() => httpMock.verify());

  it('adds X-API-Key when stored', () => {
    localStorage.setItem('ehr_api_key', 'secret');
    http.get('/health').subscribe();
    const req = httpMock.expectOne('/health');
    expect(req.request.headers.get('X-API-Key')).toBe('secret');
    req.flush({});
  });

  it('surfaces 401 on banner', () => {
    http.get('/v1/meta').subscribe({ error: () => {} });
    const req = httpMock.expectOne('/v1/meta');
    req.flush({ detail: 'bad key' }, { status: 401, statusText: 'Unauthorized' });
    expect(banner.message()).toContain('bad key');
  });
});
