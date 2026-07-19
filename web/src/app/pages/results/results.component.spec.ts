import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { ResultsComponent } from './results.component';
import { ApiService } from '../../core/api.service';
import { UiPrefsService } from '../../core/ui-prefs.service';

describe('ResultsComponent', () => {
  let fixture: ComponentFixture<ResultsComponent>;
  let cmp: ResultsComponent;
  let api: jasmine.SpyObj<ApiService>;

  beforeEach(async () => {
    localStorage.clear();
    api = jasmine.createSpyObj('ApiService', [
      'reportsSummary',
      'shap',
      'job',
      'reportFileUrl',
      'resultsZipUrl',
    ]);
    api.reportsSummary.and.returnValue(
      of({
        metrics: { roc_auc: 0.81, pr_auc: 0.62 },
        leakage_audit: { ok: true, notes: ['clean'] },
        feature_importance: { w7d_glucose: 0.4, w7d_age: -0.2 },
        model_comparison: {
          selected_model: 'xgboost',
          comparison: [
            { model: 'logreg', roc_auc: 0.7, pr_auc: 0.5, brier: 0.2, ece: 0.1, selected: false },
            { model: 'xgboost', roc_auc: 0.81, pr_auc: 0.62, brier: 0.18, ece: 0.08, selected: true },
          ],
        },
        files: [
          { name: 'calibration.png', bytes: 100, url: '/v1/reports/file/calibration.png' },
          { name: 'evaluation_report.json', bytes: 50, url: '/v1/reports/file/evaluation_report.json' },
        ],
      })
    );
    api.reportFileUrl.and.callFake((n: string) => `/v1/reports/file/${n}`);
    api.resultsZipUrl.and.returnValue('/v1/reports/download.zip');

    await TestBed.configureTestingModule({
      imports: [ResultsComponent],
      providers: [{ provide: ApiService, useValue: api }, UiPrefsService, provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
  });

  it('loads summary tables', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    expect(cmp.summary()?.model_comparison?.selected_model).toBe('xgboost');
    expect(cmp.compareRows.length).toBe(2);
    expect(cmp.metricRows.length).toBe(2);
    expect(cmp.importanceRows.length).toBe(2);
    expect(cmp.leakageRows.length).toBeGreaterThan(0);
    expect(fixture.nativeElement.textContent).toContain('Results');
  }));

  it('filters figures without rendering remote PNGs', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.figFilter = 'calibration';
    expect(cmp.filteredFigs().map((f) => f.name)).toEqual(['calibration.png']);
    cmp.figFilter = 'missing';
    expect(cmp.filteredFigs().length).toBe(0);
  }));

  it('filters metrics by name', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.metricFilter = 'roc';
    expect(cmp.filteredMetrics().length).toBe(1);
  }));
});
