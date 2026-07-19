import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ConfigComponent } from './config.component';
import { UiPrefsService } from '../../core/ui-prefs.service';

describe('ConfigComponent', () => {
  let fixture: ComponentFixture<ConfigComponent>;
  let http: HttpTestingController;

  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({
      imports: [ConfigComponent],
      providers: [provideHttpClient(), provideHttpClientTesting(), UiPrefsService],
    }).compileComponents();
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(ConfigComponent);
  });

  afterEach(() => http.verify());

  it('loads workspace config and events', () => {
    fixture.detectChanges();
    const cfgReq = http.expectOne('/v1/workspace/config');
    cfgReq.flush({
      config: { persona: 'researcher', windows_days: [7, 30, 180], ui: { theme: 'slate' } },
      effective_train: { model_kind: 'logreg' },
    });
    const evReq = http.expectOne((r) => r.url.startsWith('/v1/events'));
    evReq.flush({ events: [{ kind: 'train_queued', message: 'ok', ts: '2026-01-01' }] });
    fixture.detectChanges();
    expect(fixture.componentInstance.cfg['persona']).toBe('researcher');
    expect(fixture.componentInstance.eventRows.length).toBe(1);
    expect(fixture.nativeElement.textContent.toLowerCase()).toContain('config');
  });
});
