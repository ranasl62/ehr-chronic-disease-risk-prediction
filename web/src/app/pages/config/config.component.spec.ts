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

  it('maps event timestamps from alternate fields', () => {
    fixture.detectChanges();
    http.expectOne('/v1/workspace/config').flush({ config: {}, effective_train: {} });
    http.expectOne((r) => r.url.startsWith('/v1/events')).flush({
      events: [{ kind: 'train_queued', message: 'ok', created_at: '2026-02-01' }],
    });
    expect(fixture.componentInstance.eventRows[0]['ts']).toBe('2026-02-01');
  });

  it('handles localStorage read failure on init', () => {
    spyOn(localStorage, 'getItem').and.callFake((key: string) => {
      if (key === 'ehr_api_key') throw new Error('blocked');
      return null;
    });
    const f2 = TestBed.createComponent(ConfigComponent);
    f2.detectChanges();
    http.expectOne('/v1/workspace/config').flush({ config: {}, effective_train: {} });
    http.expectOne((r) => r.url.startsWith('/v1/events')).flush({ events: [] });
    expect(f2.componentInstance.apiKey).toBe('');
  });

  it('syncs compare model checkboxes from config', () => {
    fixture.detectChanges();
    http.expectOne('/v1/workspace/config').flush({
      config: { compare_models: ['logreg'], windows_days: [7, 30] },
      effective_train: {},
    });
    http.expectOne((r) => r.url.startsWith('/v1/events')).flush({ events: [] });
    expect(fixture.componentInstance.compareSelected['logreg']).toBeTrue();
    expect(fixture.componentInstance.compareSelected['xgboost']).toBeFalse();
  });

  it('reports config reload errors', () => {
    fixture.detectChanges();
    http.expectOne('/v1/workspace/config').flush({ detail: 'bad cfg' }, { status: 500, statusText: 'Err' });
    http.expectOne((r) => r.url.startsWith('/v1/events')).flush({ events: [] });
    expect(fixture.componentInstance.error()).toBeTruthy();
  });

  it('saves and clears API key', () => {
    const cmp = fixture.componentInstance;
    cmp.apiKey = 'secret';
    cmp.saveApiKey();
    expect(localStorage.getItem('ehr_api_key')).toBe('secret');
    expect(cmp.message()).toContain('saved');
    cmp.apiKey = '';
    cmp.saveApiKey();
    expect(localStorage.getItem('ehr_api_key')).toBeNull();
    expect(cmp.message()).toContain('cleared');
  });

  it('handles API key localStorage write failure', () => {
    spyOn(localStorage, 'setItem').and.throwError('quota');
    fixture.componentInstance.apiKey = 'x';
    fixture.componentInstance.saveApiKey();
    expect(fixture.componentInstance.error()).toContain('Could not write');
  });

  it('saves workspace config and reloads', () => {
    fixture.detectChanges();
    http.expectOne('/v1/workspace/config').flush({ config: {}, effective_train: {} });
    http.expectOne((r) => r.url.startsWith('/v1/events')).flush({ events: [] });
    const cmp = fixture.componentInstance;
    cmp.windowsText = 'abc';
    cmp.compareSelected = { logreg: true, random_forest: false, xgboost: false, lightgbm: false };
    cmp.save();
    const put = http.expectOne((r) => r.method === 'PUT');
    expect(put.request.body.windows_days).toEqual([7, 30, 180]);
    put.flush({});
    http.expectOne('/v1/workspace/config').flush({ config: {}, effective_train: {} });
    http.expectOne((r) => r.url.startsWith('/v1/events')).flush({ events: [] });
    expect(cmp.message()).toContain('saved');
  });

  it('reports save errors', () => {
    fixture.detectChanges();
    http.expectOne('/v1/workspace/config').flush({ config: {}, effective_train: {} });
    http.expectOne((r) => r.url.startsWith('/v1/events')).flush({ events: [] });
    fixture.componentInstance.save();
    http.expectOne((r) => r.method === 'PUT').flush({ detail: 'denied' }, { status: 403, statusText: 'Forbidden' });
    expect(fixture.componentInstance.error()).toBeTruthy();
  });

  it('resets UI prefs and patches page size', () => {
    const cmp = fixture.componentInstance;
    const ui = TestBed.inject(UiPrefsService);
    ui.patch({ theme: 'sand' });
    cmp.resetUi();
    expect(cmp.message()).toContain('reset');
    cmp.onPageSize(25);
    expect(ui.prefs().table_page_size).toBe(25);
  });
});
