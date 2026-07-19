import { TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { routes } from './app.routes';

describe('App routes (UI pages)', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideRouter(routes)],
    });
  });

  it('registers all workbench pages', () => {
    const paths = routes.map((r) => r.path);
    for (const p of ['', 'datasets', 'train', 'results', 'analytics', 'config', 'predict', 'docs']) {
      expect(paths).toContain(p);
    }
  });

  it('navigates to analytics and predict', async () => {
    const router = TestBed.inject(Router);
    expect(await router.navigateByUrl('/analytics')).toBeTrue();
    expect(router.url).toBe('/analytics');
    expect(await router.navigateByUrl('/predict')).toBeTrue();
    expect(router.url).toBe('/predict');
  });
});
