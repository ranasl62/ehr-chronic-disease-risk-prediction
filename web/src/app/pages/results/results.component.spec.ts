import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { Subject, of, throwError } from 'rxjs';
import { ResultsComponent } from './results.component';
import { ApiService, ReportsSummary } from '../../core/api.service';
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
      'externalValidate',
      'leakageAudit',
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
        has_leakage: true,
        has_shap: false,
        trust_complete: false,
        leakage_passed: true,
        trust: {
          has_model: true,
          has_evaluation: true,
          has_leakage: true,
          has_shap: false,
          has_calibration: true,
          leakage_passed: true,
          trust_complete: false,
        },
        model_kind: 'logreg',
        metrics: { roc_auc: 0.75 },
        meta: { kind: 'train' },
      })
    );
    api.reportFileUrl.and.callFake((n: string) => `/v1/reports/file/${n}`);
    api.resultsZipUrl.and.callFake((runId?: string | null) =>
      runId ? `/v1/reports/download.zip?run_id=${runId}` : '/v1/reports/download.zip'
    );
    api.methodsMdUrl.and.callFake((runId?: string | null) =>
      runId ? `/v1/reports/methods.md?run_id=${runId}` : '/v1/reports/methods.md'
    );
    api.externalValidate.and.returnValue(of({ id: 'j3', kind: 'external_validate', status: 'queued', message: '', result: {}, log_tail: [] }));
    api.shap.and.returnValue(of({ id: 'j1', kind: 'shap', status: 'queued', message: '', result: {}, log_tail: [] }));
    api.leakageAudit.and.returnValue(of({ id: 'j2', kind: 'leakage_audit', status: 'queued', message: '', result: {}, log_tail: [] }));

    await TestBed.configureTestingModule({
      imports: [ResultsComponent],
      providers: [{ provide: ApiService, useValue: api }, UiPrefsService, provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
  });

  it('tracks figure load errors', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.onFigError('shap_summary.png');
    expect(cmp.figLoadError()['shap_summary.png']).toBeTrue();
    cmp.onFigLoad('shap_summary.png');
    expect(cmp.figLoadError()['shap_summary.png']).toBeUndefined();
  }));

  it('shows loading then content without empty-state flash', fakeAsync(() => {
    const pending = new Subject<ReportsSummary>();
    api.reportsSummary.and.returnValue(pending.asObservable());
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    expect(cmp.summaryLoading()).toBeTrue();
    expect(fixture.nativeElement.querySelector('[data-tour="results-loading"]')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('[data-tour="results-empty"]')).toBeFalsy();
    expect(fixture.nativeElement.textContent).not.toContain('No reports yet');

    pending.next({
      metrics: { roc_auc: 0.9 },
      files: [{ name: 'evaluation_report.json', bytes: 10, url: '/v1/reports/file/evaluation_report.json' }],
    });
    pending.complete();
    tick();
    fixture.detectChanges();
    expect(cmp.summaryLoading()).toBeFalse();
    expect(cmp.summary()?.metrics?.['roc_auc']).toBe(0.9);
    expect(fixture.nativeElement.querySelector('[data-tour="results-loading"]')).toBeFalsy();
    expect(fixture.nativeElement.querySelector('[data-tour="results-empty"]')).toBeFalsy();
  }));

  it('shows empty state only after summary load fails', fakeAsync(() => {
    api.reportsSummary.and.returnValue(throwError(() => ({ message: 'down' })));
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    tick();
    fixture.detectChanges();
    expect(cmp.summaryLoading()).toBeFalse();
    expect(cmp.summary()).toBeNull();
    expect(fixture.nativeElement.querySelector('[data-tour="results-empty"]')).toBeTruthy();
    expect(fixture.nativeElement.textContent).toContain('No reports yet');
    expect(cmp.error()).toContain('down');
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

  it('shows trust checklist and passes run_id to SHAP/leakage', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.openRun('run_a');
    tick(20);
    fixture.detectChanges();
    const items = cmp.trustChecklist(cmp.selectedRun());
    expect(items.some((i) => i.label === 'Leakage audit' && i.ok === true)).toBeTrue();
    expect(fixture.nativeElement.textContent).toContain('Trust pack');
    expect(cmp.zipUrl()).toContain('run_id=run_a');
    expect(cmp.methodsMdUrl()).toContain('run_id=run_a');
    cmp.runShap();
    expect(api.shap).toHaveBeenCalledWith({ run_id: 'run_a' });
    cmp.runLeakageForSelected();
    expect(api.leakageAudit).toHaveBeenCalled();
    const leakArg = api.leakageAudit.calls.mostRecent().args[0] as Record<string, unknown>;
    expect(leakArg['run_id']).toBe('run_a');
  }));

  it('posts external validate job for selected run', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.openRun('run_a');
    tick(20);
    cmp.extValPath = 'data/demo/ehr_data.csv';
    cmp.runExternalValidate();
    expect(api.externalValidate).toHaveBeenCalledWith(
      jasmine.objectContaining({ data_path: 'data/demo/ehr_data.csv', run_id: 'run_a' })
    );
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

  it('renders charts when enabled', fakeAsync(() => {
    for (const name of ['compare', 'metric', 'importance', 'threshold', 'fairness']) {
      const c = document.createElement('canvas');
      c.setAttribute('data-chart', name);
      fixture.nativeElement.appendChild(c);
    }
    cmp.showCharts = true;
    fixture.detectChanges();
    tick(50);
    expect(cmp.chartsReady()).toBeTrue();
    cmp.onShowCharts(false);
    tick(20);
    fixture.destroy();
  }));

  it('promoteSelected loadThresholds runFairness and compare runs', fakeAsync(() => {
    cmp.showCharts = false;
    api.promoteRun.and.returnValue(of({ run_id: 'run_a', model_path: 'models/x.joblib' }));
    api.job.and.returnValue(
      of({ id: 'f1', kind: 'fairness', status: 'succeeded', message: '', result: {}, log_tail: [] })
    );
    api.fairness.and.returnValue(
      of({ id: 'f1', kind: 'fairness', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    fixture.detectChanges();
    tick(20);
    cmp.openRun('run_a');
    tick(20);
    cmp.promoteSelected();
    expect(api.promoteRun).toHaveBeenCalledWith('run_a');
    tick(20);

    cmp.toggleCompareRun('run_a', true);
    expect(cmp.runCompareRows.length).toBe(1);
    cmp.toggleCompareRun('run_a', false);
    expect(cmp.runCompareRows.length).toBe(0);

    cmp.loadThresholds();
    expect(api.thresholds).toHaveBeenCalled();
    tick(20);

    cmp.runFairness();
    tick(2000);
    expect(api.fairness).toHaveBeenCalled();
  }));

  it('handles reload and job errors', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    api.reportsSummary.and.returnValue(throwError(() => ({ status: 401 })));
    cmp.reload();
    tick(20);
    expect(cmp.error()).toContain('401');

    api.runDetail.and.returnValue(throwError(() => ({ error: { detail: 'missing run' } })));
    cmp.openRun('bad');
    expect(cmp.error()).toBe('missing run');

    api.promoteRun.and.returnValue(throwError(() => ({ error: { detail: { message: 'deny' } } })));
    cmp.selectedRun.set({
      run_id: 'run_a',
      path: 'x',
      has_model: true,
      trust: {},
    } as never);
    cmp.promoteSelected();
    expect(cmp.error()).toBe('deny');

    api.shap.and.returnValue(throwError(() => ({ error: { detail: ['a', 'b'] } })));
    cmp.runShap();
    expect(cmp.error()).toContain('a');

    api.thresholds.and.returnValue(throwError(() => ({ message: 'thr fail' })));
    cmp.loadThresholds();
    expect(cmp.error()).toBe('thr fail');
  }));

  it('covers helpers setFigFilter trustChecklist and formatting', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.setFigFilter('shap');
    expect(cmp.figFilter).toBe('shap');
    expect(cmp.trustChecklist(null)).toEqual([]);

    const run = {
      run_id: 'r1',
      path: 'p',
      has_model: true,
      has_evaluation: true,
      has_manifest: true,
      has_leakage: false,
      metrics: { roc_auc: null },
      meta: { note: null, nested: { a: 1 } },
    } as never;
    expect(cmp.runStatusLabels(run)).toContain('manifest');
    const priv = cmp as unknown as { runSearchText(r: unknown): string };
    expect(priv.runSearchText(run)).toContain('r1');

    const fmt = cmp as unknown as {
      formatParamValue(v: unknown): string;
      flattenAudit(o: Record<string, unknown>): unknown[];
      thresholdChartPoints(): unknown[];
    };
    expect(fmt.formatParamValue(true)).toBe('true');
    expect(fmt.formatParamValue(null)).toBe('n/a');
    expect(fmt.formatParamValue({ x: 1 })).toContain('x');
    expect(fmt.flattenAudit({ a: { b: 1 }, c: [1, 2] }).length).toBeGreaterThan(0);

    api.reportsSummary.and.returnValue(
      of({
        files: [],
        thresholds: {
          decision_curve: {
            points: [{ threshold: 0.2, net_benefit: 0.05 }, { threshold: 0.9, net_benefit: 0.01 }],
          },
        },
      })
    );
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    expect(cmp.hasThresholdChartData()).toBeTrue();

    cmp.onPageSize(50);
    expect(TestBed.inject(UiPrefsService).prefs().table_page_size).toBe(50);

    cmp.selectedRun.set(null);
    cmp.promoteSelected();
    expect(api.promoteRun).not.toHaveBeenCalled();

    api.shap.and.returnValue(
      of({ id: 'j9', kind: 'shap', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(throwError(() => ({ message: 'poll err' })));
    cmp.runShap();
    tick(2000);
    expect(cmp.error()).toContain('poll err');
  }));

  it('clears shap fig filter and handles empty summary helpers', fakeAsync(() => {
    cmp.showCharts = false;
    api.reportsSummary.and.returnValue(
      of({
        metrics: {},
        files: [{ name: 'calibration_holdout.png', bytes: 1, url: '/v1/reports/file/c.png' }],
      })
    );
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    cmp.figFilter = 'shap';
    cmp.reload();
    tick(20);
    expect(cmp.figFilter).toBe('');

    expect(cmp.formatParams(null)).toEqual([]);
    const priv = cmp as unknown as { thresholdChartPoints(): unknown[] };
    cmp.thresholds.set({
      decision_curve: { points: [{ threshold: Number.NaN, net_benefit: 1 }] },
    });
    expect(priv.thresholdChartPoints().length).toBe(0);

    cmp.summary.set(null);
    expect(cmp.filteredFigs()).toEqual([]);

    api.runs.and.returnValue(throwError(() => new Error('runs fail')));
    api.fairnessReport.and.returnValue(throwError(() => new Error('fairness fail')));
    cmp.reload();
    tick(20);
    fixture.destroy();
  }));

  it('renders decision-curve chart with net benefit only', fakeAsync(() => {
    api.reportsSummary.and.returnValue(
      of({
        metrics: {},
        files: [],
        thresholds: {
          decision_curve: {
            points: [{ threshold: 0.1, net_benefit: 0.02 }, { threshold: 0.5, net_benefit: 0.01 }],
          },
        },
      })
    );
    fixture = TestBed.createComponent(ResultsComponent);
    cmp = fixture.componentInstance;
    const c = document.createElement('canvas');
    c.setAttribute('data-chart', 'threshold');
    fixture.nativeElement.appendChild(c);
    cmp.showCharts = true;
    fixture.detectChanges();
    tick(50);
    expect(cmp.chartsReady()).toBeTrue();
    fixture.destroy();
  }));

  it('covers chart helper and destroy paths', fakeAsync(() => {
    cmp.showCharts = false;
    fixture.detectChanges();
    tick(20);
    const priv = cmp as unknown as {
      canvas(ref: unknown, name: string): HTMLCanvasElement | undefined;
      bar(el: HTMLCanvasElement | undefined, cfg: object): void;
      thresholdChartPoints(): unknown[];
    };
    const c = document.createElement('canvas');
    c.setAttribute('data-chart', 'compare');
    fixture.nativeElement.appendChild(c);
    priv.bar(undefined, { type: 'bar', data: { labels: [], datasets: [] } });
    priv.bar(c, { type: 'bar', data: { labels: ['a'], datasets: [{ data: [1] }] } });
    priv.bar(c, { type: 'bar', data: { labels: ['b'], datasets: [{ data: [2] }] } });
    expect(priv.canvas(undefined, 'compare')).toBeTruthy();
    cmp.summary.set({ files: [] });
    cmp.thresholds.set(null);
    expect(priv.thresholdChartPoints()).toEqual([]);
    fixture.destroy();
  }));

  it('clears pending redraw timer on destroy', fakeAsync(() => {
    cmp.showCharts = true;
    fixture.detectChanges();
    fixture.destroy();
  }));
});
