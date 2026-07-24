import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { AnalyticsComponent } from './analytics.component';
import { ApiService } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';
import { UiPrefsService } from '../../core/ui-prefs.service';

describe('AnalyticsComponent', () => {
  let fixture: ComponentFixture<AnalyticsComponent>;
  let cmp: AnalyticsComponent;
  let api: jasmine.SpyObj<ApiService>;

  beforeEach(async () => {
    localStorage.clear();
    api = jasmine.createSpyObj('ApiService', [
      'datasets',
      'datasetProfile',
      'workspaceStatus',
      'reportsSummary',
      'analysisPack',
      'analysisPackUrl',
    ]);
    api.datasets.and.returnValue(
      of({
        datasets: [
          {
            id: 'ehr_data',
            label: 'Demo',
            path: 'data/raw/ehr_data.csv',
            format: 'longitudinal',
            exists: true,
          },
        ],
      })
    );
    api.datasetProfile.and.returnValue(
      of({
        path: 'data/raw/ehr_data.csv',
        n_rows: 20,
        n_columns: 10,
        columns: ['patient_id', 'age', 'label', 'glucose'],
        n_patients: 10,
        label_counts: { '0': 6, '1': 4 },
        age_band_counts: { '50_59': 8, '40_49': 6 },
        missing_pct: { glucose: 5 },
        numeric_preview: { glucose: { mean: 110, std: 12 } },
        time_span: { min: '2020-01-01', max: '2021-01-01' },
      })
    );
    api.workspaceStatus.and.returnValue(
      of({
        api_ok: true,
        model_ready: true,
        evaluation_present: true,
        metrics: { roc_auc: 0.8, pr_auc: 0.6 },
        leakage_audit_present: false,
        shap_present: false,
        calibration_present: false,
        demo_datasets_available: true,
        checklist: {},
        recent_jobs: [],
      })
    );
    api.reportsSummary.and.returnValue(
      of({
        files: [],
        quality_note: 'ROC-AUC bootstrap note',
        curves: {
          roc: { fpr: [0, 0.5, 1], tpr: [0, 0.7, 1], thresholds: [1, 0.5, 0] },
          pr: { precision: [1, 0.6, 0.4], recall: [0, 0.5, 1], thresholds: [1, 0.5] },
          calibration: {
            bin_mid: [0.25, 0.75],
            frac_positive: [0.2, 0.7],
            mean_predicted: [0.3, 0.8],
            counts: [3, 4],
          },
          notes: [],
        },
        feature_importance: { w7d_glucose: 0.4, w7d_age: 0.2 },
        model_comparison: {
          selected_model: 'logreg',
          comparison: [
            { model: 'logreg', roc_auc: 0.8, pr_auc: 0.6, brier: 0.2, ece: 0.05, selected: true },
          ],
        },
      })
    );
    api.analysisPack.and.returnValue(
      of({
        n_patients: 10,
        n_rows: 20,
        label_prevalence: 0.4,
        time_span: '2020 → 2021',
        missingness: { glucose: 5 },
      })
    );
    api.analysisPackUrl.and.returnValue('/v1/reports/analysis-pack?path=data%2Fraw%2Fehr_data.csv');

    await TestBed.configureTestingModule({
      imports: [AnalyticsComponent],
      providers: [
        { provide: ApiService, useValue: api },
        WorkspaceState,
        UiPrefsService,
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AnalyticsComponent);
    cmp = fixture.componentInstance;
  });

  it('loads datasets and profile tables', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    expect(api.datasets).toHaveBeenCalled();
    expect(api.datasetProfile).toHaveBeenCalled();
    expect(api.analysisPack).toHaveBeenCalled();
    expect(cmp.profile()?.n_rows).toBe(20);
    expect(cmp.analysisPack()?.n_patients).toBe(10);
    expect(cmp.labelRows.length).toBe(2);
    expect(cmp.ageRows.length).toBe(2);
    expect(cmp.metricRows.length).toBe(2);
    expect(cmp.importanceRows.length).toBe(2);
    expect(fixture.nativeElement.textContent).toContain('Analytics');
    expect(fixture.nativeElement.textContent).toContain('Analysis pack');
  }));

  it('toggles view modes without error', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    cmp.setView('charts');
    expect(cmp.showCharts()).toBeTrue();
    expect(cmp.showTables()).toBeFalse();
    cmp.setView('tables');
    expect(cmp.showTables()).toBeTrue();
    cmp.setView('split');
    tick(20);
    expect(cmp.showCharts()).toBeTrue();
  }));

  it('scheduleRedraw after profile does not throw', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    expect(() => {
      cmp.load();
      tick(20);
    }).not.toThrow();
    expect(api.datasetProfile).toHaveBeenCalled();
  }));

  it('formats prevalence and exposes export/print helpers', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    expect(cmp.formatPct(0.402)).toBe('40.2%');
    expect(cmp.formatPct(null)).toBe('—');
    expect(cmp.analysisPackDownloadUrl()).toContain('analysis-pack');
    expect(() => cmp.exportChartPng('label', 'label_distribution')).not.toThrow();
    expect(() => cmp.exportAllChartsPng()).not.toThrow();
    const printSpy = spyOn(window, 'print');
    cmp.printReport();
    expect(printSpy).toHaveBeenCalled();
  }));

  it('builds sex rows from cohort when present', fakeAsync(() => {
    api.datasetProfile.and.returnValue(
      of({
        path: 'data/raw/ehr_data.csv',
        n_rows: 3,
        n_columns: 4,
        columns: ['patient_id', 'sex', 'label'],
        label_counts: { '0': 2, '1': 1 },
        age_band_counts: {},
        cohort_rows: [
          { patient_id: 'a', sex: 'F', label: '0' },
          { patient_id: 'b', sex: 'M', label: '1' },
          { patient_id: 'c', sex: 'F', label: '0' },
        ],
      })
    );
    fixture = TestBed.createComponent(AnalyticsComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
    tick();
    expect(cmp.sexRows.length).toBe(2);
    expect(cmp.sexRows.find((r) => r['sex'] === 'F')?.['count']).toBe(2);
  }));

  it('toggles sex and prevalence chart prefs', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    cmp.toggleChart('show_sex_chart', false);
    expect(cmp.prefs.show_sex_chart).toBeFalse();
    cmp.toggleChart('show_prevalence_chart', false);
    expect(cmp.prefs.show_prevalence_chart).toBeFalse();
  }));

  it('handles load errors and chart prefs', fakeAsync(() => {
    api.datasets.and.returnValue(throwError(() => ({ message: 'ds fail' })));
    fixture = TestBed.createComponent(AnalyticsComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
    tick();
    expect(cmp.error()).toContain('ds fail');

    api.datasets.and.returnValue(
      of({ datasets: [{ id: 'x', label: 'X', path: 'data/raw/x.csv', format: 'longitudinal', exists: true }] })
    );
    api.datasetProfile.and.returnValue(throwError(() => ({ error: { detail: 'profile fail' } })));
    fixture = TestBed.createComponent(AnalyticsComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
    tick();
    cmp.path = '';
    cmp.load();
    expect(cmp.error()).toContain('Select a dataset');

    api.datasetProfile.and.returnValue(
      of({
        path: 'x',
        n_rows: 1,
        n_columns: 1,
        columns: ['age_band'],
        label_counts: { '0': 1, '1': 1 },
        age_band_counts: { '40_49': 2 },
        missing_pct: { glucose: 10, bp: 20, hr: 5 },
        numeric_preview: { glucose: { mean: 1 } },
      })
    );
    api.analysisPack.and.returnValue(throwError(() => ({ message: 'pack fail' })));
    cmp.path = 'data/raw/x.csv';
    cmp.load();
    tick(20);

    cmp.onTopNChange(Number('bad' as unknown as number));
    cmp.onPageSize(25);
    expect(cmp.prefs.top_n_features).toBe(15);

    const ui = TestBed.inject(UiPrefsService);
    ui.patch({
      theme: 'slate',
      show_missing_chart: true,
      show_numeric_chart: true,
      show_importance_chart: true,
      show_compare_chart: true,
      show_metric_chart: true,
      metric_chart_type: 'bar',
      numeric_chart_type: 'line',
      show_age_chart: true,
      show_sex_chart: true,
      show_prevalence_chart: true,
      show_label_chart: true,
    });
    for (const name of ['label', 'age', 'missing', 'numeric', 'importance', 'compare', 'metric', 'sex', 'prevalence']) {
      const c = document.createElement('canvas');
      c.setAttribute('data-chart', name);
      fixture.nativeElement.appendChild(c);
    }
    ui.patch({ theme: 'sand' });
    cmp.setView('charts');
    tick(50);
    cmp.exportChartPng('missing', 'missingness');
    cmp.exportAllChartsPng();
    cmp.setView('tables');
    tick(20);
    fixture.destroy();
  }));

  it('destroys timers/charts and handles export without canvas', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    cmp.exportChartPng('missing', 'x');
    const ui = TestBed.inject(UiPrefsService);
    ui.patch({ theme: 'slate' });
    const priv = cmp as unknown as { palette(): string[]; bar(canvas: unknown, cfg: object): void };
    expect(priv.palette()[0]).toBe('#3d5a80');
    priv.bar(undefined, { type: 'bar', data: { labels: [], datasets: [] } });
    const c = document.createElement('canvas');
    fixture.nativeElement.appendChild(c);
    priv.bar(c, { type: 'bar', data: { labels: ['a'], datasets: [{ data: [1] }] } });
    priv.bar(c, { type: 'bar', data: { labels: ['b'], datasets: [{ data: [2] }] } });
    api.reportsSummary.and.returnValue(throwError(() => new Error('summary fail')));
    cmp.load();
    tick(20);
    cmp.setView('charts');
    fixture.destroy();
  }));

  it('clears pending redraw timer on destroy', () => {
    fixture.detectChanges();
    cmp.setView('charts');
    fixture.destroy();
  });

  it('covers chart render early-return branches', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    const ui = TestBed.inject(UiPrefsService);
    const priv = cmp as unknown as {
      renderImportance(r: object): void;
      renderCompare(r: object): void;
      renderMetrics(s: object): void;
      renderEvalCurves(r: object): void;
    };
    ui.patch({ show_importance_chart: false, show_compare_chart: false, show_metric_chart: false });
    priv.renderImportance({});
    priv.renderCompare({});
    priv.renderMetrics({ metrics: {} });
    ui.patch({ show_importance_chart: true });
    cmp.importanceRows = [];
    priv.renderImportance({ feature_importance: {} });
    priv.renderCompare({ model_comparison: { comparison: [] } });
    priv.renderMetrics({ metrics: { x: null } });
    ui.patch({ show_compare_chart: true, show_metric_chart: true });
    priv.renderCompare({ model_comparison: { comparison: [] } });
    priv.renderMetrics({ metrics: {} });
    priv.renderEvalCurves({});
    priv.renderEvalCurves({ curves: { roc: {}, pr: {}, calibration: {} } });
    for (const name of ['roc', 'pr', 'cal']) {
      const c = document.createElement('canvas');
      c.setAttribute('data-chart', name);
      fixture.nativeElement.appendChild(c);
    }
    priv.renderEvalCurves({
      curves: {
        roc: { fpr: [0, 1], tpr: [0, 1] },
        pr: { precision: [1, 0.5], recall: [0, 1] },
        calibration: { bin_mid: [0.5], frac_positive: [0.4], mean_predicted: undefined, counts: [2] },
      },
    });
    tick(20);
    expect(cmp.formatPct(undefined)).toBe('—');
  }));

  it('shows empty-state copy when ROC/PR points are unavailable', fakeAsync(() => {
    api.reportsSummary.and.returnValue(
      of({
        files: [],
        quality_note: 'Hold-out has a single class — ROC/PR-AUC and curve plots are n/a',
        curves: {
          roc: { fpr: [], tpr: [], thresholds: [] },
          pr: { precision: [], recall: [], thresholds: [] },
          calibration: { bin_mid: [0.5], frac_positive: [0], mean_predicted: [0.4], counts: [1] },
          notes: ['single_class_holdout_curves_unavailable'],
        },
      })
    );
    fixture.detectChanges();
    tick(20);
    expect(cmp.hasRocCurve()).toBeFalse();
    expect(cmp.hasPrCurve()).toBeFalse();
    expect(cmp.hasCalCurve()).toBeTrue();
    expect(cmp.curveEmpty().roc || '').toMatch(/single.class|paper_synthetic|larger/i);
    const text = fixture.nativeElement.textContent as string;
    expect(text).toMatch(/single class|larger cohort|n\/a/i);
  }));

  it('prompts retrain when curves key is absent', fakeAsync(() => {
    api.reportsSummary.and.returnValue(of({ files: [], metrics: { brier: 0.2 } }));
    fixture.detectChanges();
    tick(20);
    expect(cmp.hasRocCurve()).toBeFalse();
    expect(cmp.curveEmpty().roc || '').toMatch(/retrain/i);
  }));
});
