import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { provideRouter, RouterLink } from '@angular/router';
import { of, throwError } from 'rxjs';
import { HomeComponent } from './home.component';
import { ApiService } from '../../core/api.service';

describe('HomeComponent', () => {
  let fixture: ComponentFixture<HomeComponent>;
  let api: jasmine.SpyObj<ApiService>;

  beforeEach(async () => {
    api = jasmine.createSpyObj('ApiService', ['workspaceStatus', 'resultsZipUrl']);
    api.resultsZipUrl.and.returnValue('/v1/reports/download.zip');
    api.workspaceStatus.and.returnValue(
      of({
        api_ok: true,
        model_ready: true,
        evaluation_present: true,
        leakage_audit_present: false,
        shap_present: false,
        calibration_present: false,
        demo_datasets_available: true,
        checklist: { demo_dataset: true, model: true },
        recent_jobs: [],
      })
    );

    await TestBed.configureTestingModule({
      imports: [HomeComponent],
      providers: [{ provide: ApiService, useValue: api }, provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(HomeComponent);
    fixture.detectChanges();
  });

  it('loads workspace status for dashboard', () => {
    expect(api.workspaceStatus).toHaveBeenCalled();
    expect(fixture.componentInstance.status()?.api_ok).toBeTrue();
    expect(fixture.nativeElement.textContent.length).toBeGreaterThan(10);
  });

  it('links Run demo to Train with demo=1 queryParams', () => {
    const links = fixture.debugElement.queryAll(By.directive(RouterLink));
    const demo = links.find((el) => (el.nativeElement.textContent || '').includes('Run demo'));
    expect(demo).toBeTruthy();
    const routerLink = demo!.injector.get(RouterLink) as RouterLink & {
      queryParams: { demo?: string };
    };
    expect(routerLink.queryParams).toEqual({ demo: '1' });
    expect(fixture.nativeElement.textContent).toContain('bundled longitudinal CSV');
  });

  it('surfaces API errors on refresh', () => {
    api.workspaceStatus.and.returnValue(throwError(() => ({ message: 'offline' })));
    fixture.componentInstance.refresh();
    expect(fixture.componentInstance.error()).toContain('offline');
    expect(fixture.componentInstance.loading()).toBeFalse();
  });
});
