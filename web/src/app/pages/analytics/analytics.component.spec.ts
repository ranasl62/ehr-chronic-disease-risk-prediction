import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { of } from 'rxjs';
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
        feature_importance: { w7d_glucose: 0.4, w7d_age: 0.2 },
        model_comparison: {
          selected_model: 'logreg',
          comparison: [
            { model: 'logreg', roc_auc: 0.8, pr_auc: 0.6, brier: 0.2, ece: 0.05, selected: true },
          ],
        },
      })
    );

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
    expect(cmp.profile()?.n_rows).toBe(20);
    expect(cmp.labelRows.length).toBe(2);
    expect(cmp.ageRows.length).toBe(2);
    expect(cmp.metricRows.length).toBe(2);
    expect(cmp.importanceRows.length).toBe(2);
    expect(fixture.nativeElement.textContent).toContain('Analytics');
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
});
