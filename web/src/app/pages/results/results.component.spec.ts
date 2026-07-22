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
      'fairness',
      'fairnessReport',
      'thresholds',
      'runs',
      'runDetail',
      'promoteRun',
      'job',
      'reportFileUrl',
      'resultsZipUrl',
      'methodsMdUrl',
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
        fairness: {
          present: true,
          skipped: false,
          by_group: [
            { group: 'lt50', n: 10, prevalence: 0.2, accuracy: 0.7, tpr: 0.6, fpr: 0.3, mean_predicted_prob: 0.3 },
            { group: 'ge50', n: 12, prevalence: 0.25, accuracy: 0.75, tpr: 0.65, fpr: 0.28, mean_predicted_prob: 0.35 },
          ],
        },
        thresholds: {
          present: true,
          points: [
            { threshold: 0.3, precision: 0.4, recall: 0.8, f1: 0.53, accuracy: 0.55, positive_rate: 0.6 },
            { threshold: 0.5, precision: 0.6, recall: 0.5, f1: 0.55, accuracy: 0.7, positive_rate: 0.4 },
          ],
        },
        hpo: {
          note: 'Research-scoped light grid only; not clinical AutoML.',
          trials: [
            { trial: 0, params: { C: 0.1 }, roc_auc: null, pr_auc: null, brier: 0.2243, ece: 0.4697, f1: 0 },
            { trial: 1, params: { C: 1 }, roc_auc: null, pr_auc: null, brier: 0.2076, ece: 0.4416, f1: 0 },
          ],
          best: { trial: 0, params: { C: 0.1 }, roc_auc: null, pr_auc: null, brier: 0.2243, ece: 0.4697, f1: 0 },
        },
        files: [
          { name: 'other_plot.png', bytes: 80, url: '/v1/reports/file/other_plot.png' },
          { name: 'shap_summary.png', bytes: 120, url: '/v1/reports/file/shap_summary.png' },
          { name: 'calibration_holdout.png', bytes: 100, url: '/v1/reports/file/calibration_holdout.png' },
          { name: 'evaluation_report.json', bytes: 50, url: '/v1/reports/file/evaluation_report.json' },
        ],
      })
    );
    api.runs.and.returnValue(
      of({
        runs: [
          {
            run_id: 'run_a',
            path: 'reports/runs/run_a',
            has_model: true,
            model_kind: 'logreg',
            metrics: { roc_auc: 0.75 },
          },
        ],
      })
    );
    api.fairnessReport.and.returnValue(
      of({
        present: true,
        skipped: false,
        by_group: [
          { group: 'lt50', n: 10, prevalence: 0.2, accuracy: 0.7, tpr: 0.6, fpr: 0.3, mean_predicted_prob: 0.3 },
          { group: 'ge50', n: 12, prevalence: 0.25, accuracy: 0.75, tpr: 0.65, fpr: 0.28, mean_predicted_prob: 0.35 },
        ],
      })
    );
    api.thresholds.and.returnValue(
      of({
        present: true,
        points: [{ threshold: 0.5, precision: 0.6, recall: 0.5, f1: 0.55, accuracy: 0.7, positive_rate: 0.4 }],
      })
    );
    api.runDetail.and.returnValue(
      of({
        run_id: 'run_a',
        path: 'reports/runs/run_a',
        has_model: true,
        model_kind: 'logreg',
        metrics: { roc_auc: 0.75 },
        meta: { kind: 'train' },
      })
    );
    api.reportFileUrl.and.callFake((n: string) => `/v1/reports/file/${n}`);
    api.resultsZipUrl.and.returnValue('/v1/reports/download.zip');
    api.methodsMdUrl.and.returnValue('/v1/reports/methods.md');

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

  it('renders HPO null metrics as n/a without raw JSON', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);

    const text = fixture.nativeElement.textContent;
    expect(cmp.hpoRows.length).toBe(2);
    expect(cmp.hpoBest?.params).toEqual([{ name: 'C', value: '0.1' }]);
    expect(text).toContain('Best trial');
    expect(text).toContain('Trial #0');
    expect(text).toContain('n/a');
    expect(text).toContain('C = 0.1');
    expect(text).not.toContain('"roc_auc"');
    expect(text).not.toContain('null');
  }));

  it('renders defined HPO metrics with three decimal places', fakeAsync(() => {
    cmp.showCharts = false;
    api.reportsSummary.and.returnValue(
      of({
        files: [],
        hpo: {
          trials: [{ trial: 2, params: { learning_rate: 0.01234 }, roc_auc: 0.8123, pr_auc: 0.6234, brier: 0.1876, ece: 0.0421, f1: 0.7 }],
          best: { trial: 2, params: { learning_rate: 0.01234 }, roc_auc: 0.8123, pr_auc: 0.6234, brier: 0.1876, ece: 0.0421, f1: 0.7 },
        },
      })
    );
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);

    const text = fixture.nativeElement.textContent;
    expect(text).toContain('0.812');
    expect(text).toContain('0.0123');
    expect(text).not.toContain('n/a');
    expect(cmp.formatMetric(null)).toBe('n/a');
  }));

  it('filters figures without rendering remote PNGs', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.figFilter = 'calibration';
    expect(cmp.filteredFigs().map((f) => f.name)).toEqual(['calibration_holdout.png']);
    cmp.figFilter = 'missing';
    expect(cmp.filteredFigs().length).toBe(0);
  }));

  it('prioritizes calibration and SHAP figures', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    expect(cmp.filteredFigs().map((f) => f.name)).toEqual([
      'calibration_holdout.png',
      'shap_summary.png',
      'other_plot.png',
    ]);
  }));

  it('hides calibration figure controls when PNG is absent', fakeAsync(() => {
    cmp.showCharts = false;
    api.reportsSummary.and.returnValue(
      of({
        metrics: { roc_auc: 0.81 },
        files: [
          { name: 'shap_summary.png', bytes: 120, url: '/v1/reports/file/shap_summary.png' },
          { name: 'evaluation_report.json', bytes: 50, url: '/v1/reports/file/evaluation_report.json' },
        ],
      })
    );
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);

    expect(cmp.hasCalibrationFigure()).toBeFalse();
    expect(cmp.hasShapFigure()).toBeTrue();
    expect(cmp.hasReportPngs()).toBeTrue();
    expect(fixture.nativeElement.textContent).not.toContain('Calibration');
    expect(fixture.nativeElement.textContent).toContain('SHAP');
    expect(cmp.filteredFigs().map((f) => f.name)).toEqual(['shap_summary.png']);
  }));

  it('clears calibration filter when figure disappears on reload', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    expect(cmp.hasCalibrationFigure()).toBeTrue();
    cmp.figFilter = 'calibration';
    api.reportsSummary.and.returnValue(
      of({
        metrics: {},
        files: [{ name: 'other_plot.png', bytes: 80, url: '/v1/reports/file/other_plot.png' }],
      })
    );
    cmp.reload();
    tick(20);
    expect(cmp.hasCalibrationFigure()).toBeFalse();
    expect(cmp.figFilter).toBe('');
  }));

  it('exposes threshold and fairness chart data from reports JSON', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    expect(cmp.hasThresholdChartData()).toBeTrue();
    expect(cmp.hasFairnessChartData()).toBeTrue();
  }));

  it('links methods.md download', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    expect(cmp.methodsMdUrl()).toBe('/v1/reports/methods.md');
    expect(fixture.nativeElement.textContent).toContain('Download methods.md');
  }));

  it('filters metrics by name', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.metricFilter = 'roc';
    expect(cmp.filteredMetrics().length).toBe(1);
  }));

  it('loads experiment runs and fairness panel', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    expect(api.runs).toHaveBeenCalled();
    expect(cmp.runs().length).toBe(1);
    expect(cmp.fairnessRows.length).toBeGreaterThan(0);
    expect(fixture.nativeElement.textContent).toContain('Experiment runs');
    expect(fixture.nativeElement.textContent).toContain('Fairness');
  }));

  it('opens run detail', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.openRun('run_a');
    expect(api.runDetail).toHaveBeenCalledWith('run_a');
    expect(cmp.selectedRun()?.run_id).toBe('run_a');
  }));

  it('filters experiment runs client-side by visible fields', fakeAsync(() => {
    cmp.showCharts = false;
    api.runs.and.returnValue(
      of({
        runs: [
          {
            run_id: '20240101T000000Z_logreg',
            path: 'reports/runs/20240101T000000Z_logreg',
            has_model: true,
            has_evaluation: true,
            model_kind: 'logreg',
            metrics: { roc_auc: 0.75 },
            meta: { kind: 'train' },
          },
          {
            run_id: '20240102T000000Z_xgb',
            path: 'reports/runs/20240102T000000Z_xgb',
            has_model: false,
            has_evaluation: true,
            model_kind: 'xgboost',
            metrics: { roc_auc: 0.88 },
            meta: { kind: 'train' },
          },
        ],
      })
    );
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);

    expect(cmp.filteredRuns().length).toBe(2);
    cmp.onRunFilterChange('xgboost');
    expect(cmp.filteredRuns().map((r) => r.run_id)).toEqual(['20240102T000000Z_xgb']);
    cmp.onRunFilterChange('model');
    expect(cmp.filteredRuns().map((r) => r.run_id)).toEqual(['20240101T000000Z_logreg']);
    cmp.onRunFilterChange('0.88');
    expect(cmp.filteredRuns().length).toBe(1);
    cmp.onRunFilterChange('missing-xyz');
    expect(cmp.filteredRuns().length).toBe(0);
  }));

  it('paginates filtered experiment runs client-side', fakeAsync(() => {
    cmp.showCharts = false;
    const many = Array.from({ length: 12 }, (_, i) => ({
      run_id: `run_${String(i).padStart(2, '0')}`,
      path: `reports/runs/run_${String(i).padStart(2, '0')}`,
      has_model: i % 2 === 0,
      model_kind: i % 2 === 0 ? 'logreg' : 'xgboost',
      metrics: { roc_auc: 0.5 + i * 0.01 },
      meta: { kind: 'train' },
    }));
    api.runs.and.returnValue(of({ runs: many }));
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);

    cmp.onRunPageSize(5);
    expect(cmp.runTotalPages()).toBe(3);
    expect(cmp.pagedRuns().map((r) => r.run_id)).toEqual([
      'run_00',
      'run_01',
      'run_02',
      'run_03',
      'run_04',
    ]);
    cmp.goRunPage(2);
    expect(cmp.pagedRuns().map((r) => r.run_id)).toEqual([
      'run_05',
      'run_06',
      'run_07',
      'run_08',
      'run_09',
    ]);
    cmp.onRunFilterChange('logreg');
    expect(cmp.runPage).toBe(1);
    expect(cmp.filteredRuns().length).toBe(6);
    expect(cmp.runTotalPages()).toBe(2);
    expect(cmp.pagedRuns().every((r) => r.model_kind === 'logreg')).toBeTrue();
    cmp.goRunPage(99);
    expect(cmp.runPage).toBe(2);
  }));
});
