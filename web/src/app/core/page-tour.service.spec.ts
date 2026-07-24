import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { Router, NavigationEnd } from '@angular/router';
import { Subject } from 'rxjs';
import { PageTourService } from './page-tour.service';

describe('PageTourService', () => {
  let tour: PageTourService;
  let events$: Subject<NavigationEnd>;
  let navigateByUrl: jasmine.Spy;

  beforeEach(() => {
    localStorage.clear();
    events$ = new Subject();
    navigateByUrl = jasmine.createSpy('navigateByUrl').and.resolveTo(true);
    TestBed.configureTestingModule({
      providers: [
        PageTourService,
        {
          provide: Router,
          useValue: {
            url: '/',
            events: events$.asObservable(),
            navigateByUrl,
          },
        },
      ],
    });
    tour = TestBed.inject(PageTourService);
  });

  it('starts home tour and advances steps', () => {
    tour.start('home', { force: true });
    expect(tour.active()).toBeTrue();
    expect(tour.currentStep()?.id).toBe('home-intro');
    tour.next();
    expect(tour.currentStep()?.id).toBe('home-actions');
    tour.back();
    expect(tour.currentStep()?.id).toBe('home-intro');
  });

  it('navigates when a step declares a route', fakeAsync(() => {
    tour.start('home', { force: true });
    // Skip to last home step which routes to /results
    while (tour.currentStep()?.id !== 'home-to-results') {
      tour.next();
    }
    expect(navigateByUrl).toHaveBeenCalledWith('/results');
  }));

  it('marks completed on skip and blocks auto-start', fakeAsync(() => {
    tour.start('analytics', { force: true });
    tour.skip();
    expect(tour.active()).toBeFalse();
    expect(tour.shouldAutoStart('analytics')).toBeFalse();
    const raw = JSON.parse(localStorage.getItem('ehr_page_tours_v1') || '{}');
    expect(raw.completed.analytics).toBeTrue();
  }));

  it('honors dont-show-again for auto tours', () => {
    tour.start('train', { force: true });
    tour.setDontShowAgain(true);
    tour.finish(true);
    expect(tour.shouldAutoStart('datasets')).toBeFalse();
  });

  it('maps routes to page ids', () => {
    expect(tour.pageFromUrl('/analytics?x=1')).toBe('analytics');
    expect(tour.pageFromUrl('/')).toBe('home');
    expect(tour.pageFromUrl('/nope')).toBeNull();
  });

  it('auto-starts once on NavigationEnd', fakeAsync(() => {
    events$.next(new NavigationEnd(1, '/predict', '/predict'));
    tick(500);
    expect(tour.active()).toBeTrue();
    expect(tour.activePage()).toBe('predict');
    tour.skip();
    events$.next(new NavigationEnd(2, '/predict', '/predict'));
    tick(500);
    expect(tour.active()).toBeFalse();
  }));
});
