import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ApiService } from './api.service';

describe('ApiService (UI → backend contract)', () => {
  let api: ApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    api = TestBed.inject(ApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('GET /health', () => {
    api.health().subscribe((r) => expect(r).toEqual({ status: 'ok' }));
    const req = http.expectOne('/health');
    expect(req.request.method).toBe('GET');
    req.flush({ status: 'ok' });
  });

  it('GET /v1/workspace/status', () => {
    api.workspaceStatus().subscribe((r) => expect(r.api_ok).toBeTrue());
    const req = http.expectOne('/v1/workspace/status');
    req.flush({
      api_ok: true,
      model_ready: true,
      evaluation_present: false,
      leakage_audit_present: false,
      shap_present: false,
      calibration_present: false,
      demo_datasets_available: true,
      checklist: {},
      recent_jobs: [],
    });
  });

  it('GET /v1/datasets', () => {
    api.datasets().subscribe((r) => expect(r.datasets.length).toBe(1));
    http.expectOne('/v1/datasets').flush({
      datasets: [{ id: 'ehr_data', label: 'Demo', path: 'data/demo/ehr_data.csv', format: 'longitudinal', exists: true }],
    });
  });

  it('GET /v1/datasets?include_demo=false', () => {
    api.datasets(false).subscribe((r) => expect(r.datasets.length).toBe(0));
    http.expectOne('/v1/datasets?include_demo=false').flush({ datasets: [], include_demo: false });
  });

  it('POST /v1/datasets/upload', () => {
    const file = new File(['a,b\n1,2\n'], 'demo.csv', { type: 'text/csv' });
    api.uploadDataset(file).subscribe((r) => expect(r).toEqual({ path: 'data/uploads/demo.csv' }));
    const req = http.expectOne('/v1/datasets/upload');
    expect(req.request.method).toBe('POST');
    expect(req.request.body instanceof FormData).toBeTrue();
    req.flush({ path: 'data/uploads/demo.csv' });
  });

  it('POST /v1/datasets/from-form', () => {
    api.importForm([{ patient_id: 1 }], 'x.csv').subscribe();
    const req = http.expectOne('/v1/datasets/from-form');
    expect(req.request.body).toEqual({ name: 'x.csv', rows: [{ patient_id: 1 }] });
    req.flush({ path: 'data/uploads/x.csv' });
  });

  it('POST /v1/datasets/from-sql', () => {
    api.importSql('select 1', 'sqlite://', 's.csv').subscribe();
    const req = http.expectOne('/v1/datasets/from-sql');
    expect(req.request.body).toEqual({
      sql: 'select 1',
      connection_url: 'sqlite://',
      name: 's.csv',
    });
    req.flush({ path: 'data/uploads/s.csv' });
  });

  it('GET /v1/datasets/profile?path=', () => {
    api.datasetProfile('data/raw/ehr_data.csv', { age_band: '50_59' }).subscribe((p) => expect(p.n_rows).toBe(20));
    const req = http.expectOne((r) => r.url === '/v1/datasets/profile');
    expect(req.request.params.get('path')).toBe('data/raw/ehr_data.csv');
    expect(req.request.params.get('age_band')).toBe('50_59');
    req.flush({
      path: 'data/raw/ehr_data.csv',
      n_rows: 20,
      n_columns: 10,
      columns: ['patient_id', 'age'],
      cohort_rows: [],
    });
  });

  it('GET /v1/datasets/health?path=', () => {
    api.datasetHealth('data/raw/ehr_data.csv').subscribe((h) => expect(h.health.ready_for_training).toBeTrue());
    const req = http.expectOne((r) => r.url === '/v1/datasets/health');
    expect(req.request.params.get('path')).toBe('data/raw/ehr_data.csv');
    req.flush({
      path: 'data/raw/ehr_data.csv',
      n_rows: 20,
      n_columns: 10,
      health: { ready_for_training: true, blockers: [], warnings: [] },
    });
  });

  it('GET /v1/datasets/health with task_id', () => {
    api.datasetHealth('data/demo/ehr_data.csv', 'readmission_30d').subscribe((h) => {
      expect(h.health.blockers?.length).toBeGreaterThan(0);
    });
    const req = http.expectOne((r) => r.url === '/v1/datasets/health');
    expect(req.request.params.get('task_id')).toBe('readmission_30d');
    req.flush({
      path: 'data/demo/ehr_data.csv',
      n_rows: 20,
      n_columns: 10,
      health: { ready_for_training: false, blockers: ['task_required_columns: missing'], warnings: [] },
    });
  });

  it('POST /v1/jobs/train', () => {
    api
      .train({
        data_path: 'data/raw/ehr_data.csv',
        data_format: 'longitudinal',
        model_kind: 'logreg',
        calibrate: false,
        split_by_patient: true,
        temporal_split: false,
        windows_days: [7, 30, 180],
        window_days: 180,
        horizon_days: null,
        index_strategy: 'last_event',
        index_time_col: null,
        feature_inclusive: true,
      })
      .subscribe((j) => expect(j.id).toBe('job-1'));
    const req = http.expectOne('/v1/jobs/train');
    expect(req.request.method).toBe('POST');
    req.flush({ id: 'job-1', kind: 'train', status: 'queued', message: '', result: {}, log_tail: [] });
  });

  it('POST /v1/jobs/compare', () => {
    api
      .compare({
        data_path: 'data/raw/ehr_data.csv',
        data_format: 'longitudinal',
        calibrate: false,
        split_by_patient: true,
        temporal_split: false,
        windows_days: [7, 30, 180],
        window_days: 180,
        horizon_days: null,
        index_strategy: 'last_event',
        index_time_col: null,
        feature_inclusive: true,
        promote_best: true,
      })
      .subscribe((j) => expect(j.kind).toBe('compare'));
    http.expectOne('/v1/jobs/compare').flush({
      id: 'c1',
      kind: 'compare',
      status: 'queued',
      message: '',
      result: {},
      log_tail: [],
    });
  });

  it('GET /v1/tasks', () => {
    api.tasks().subscribe((r) => expect(r.tasks.length).toBe(1));
    http.expectOne('/v1/tasks').flush({ tasks: [{ id: 'diabetes', name: 'Diabetes' }] });
  });

  it('POST /v1/jobs/leakage-audit and /v1/jobs/shap', () => {
    api.leakageAudit().subscribe((j) => expect(j.kind).toBe('leakage_audit'));
    http.expectOne('/v1/jobs/leakage-audit').flush({
      id: 'l1',
      kind: 'leakage_audit',
      status: 'queued',
      message: '',
      result: {},
      log_tail: [],
    });

    api.shap({ run_id: 'run_x' }).subscribe((j) => expect(j.kind).toBe('shap'));
    const shapReq = http.expectOne('/v1/jobs/shap');
    expect(shapReq.request.body).toEqual({ run_id: 'run_x' });
    shapReq.flush({
      id: 's1',
      kind: 'shap',
      status: 'queued',
      message: '',
      result: {},
      log_tail: [],
    });
  });

  it('POST /v1/jobs/external-validate and GET analysis-pack', () => {
    api.externalValidate({ data_path: 'data/demo/ehr_data.csv' }).subscribe((j) => {
      expect(j.kind).toBe('external_validate');
    });
    http.expectOne('/v1/jobs/external-validate').flush({
      id: 'e1',
      kind: 'external_validate',
      status: 'queued',
      message: '',
      result: {},
      log_tail: [],
    });

    api.analysisPack('data/demo/ehr_data.csv').subscribe((p) => expect(p.n_rows).toBe(10));
    const req = http.expectOne((r) => r.url === '/v1/reports/analysis-pack');
    expect(req.request.params.get('path')).toBe('data/demo/ehr_data.csv');
    req.flush({ n_rows: 10, n_patients: 5 });
  });

  it('resultsZipUrl and methodsMdUrl accept run_id', () => {
    expect(api.resultsZipUrl('run_a')).toContain('run_id=run_a');
    expect(api.methodsMdUrl('run_a')).toContain('run_id=run_a');
  });

  it('GET /v1/jobs/:id', () => {
    api.job('abc').subscribe((j) => expect(j.status).toBe('succeeded'));
    http.expectOne('/v1/jobs/abc').flush({
      id: 'abc',
      kind: 'train',
      status: 'succeeded',
      message: 'ok',
      result: {},
      log_tail: [],
    });
  });

  it('GET /v1/jobs, POST cancel, runs, fairness, hpo', () => {
    api.jobs().subscribe((r) => expect(r.jobs.length).toBe(1));
    http.expectOne('/v1/jobs').flush({
      jobs: [{ id: 'j1', kind: 'train', status: 'succeeded', message: '', result: {}, log_tail: [] }],
    });

    api.cancelJob('j1').subscribe((j) => expect(j.status).toBe('cancelled'));
    http.expectOne('/v1/jobs/j1/cancel').flush({
      id: 'j1',
      kind: 'train',
      status: 'cancelled',
      message: 'cancelled',
      result: {},
      log_tail: [],
    });

    api.runs(10).subscribe((r) => expect(r.runs.length).toBe(1));
    const runsReq = http.expectOne((r) => r.url === '/v1/runs');
    expect(runsReq.request.params.get('limit')).toBe('10');
    runsReq.flush({ runs: [{ run_id: 'r1', path: 'reports/runs/r1', has_model: true }] });

    api.fairness().subscribe((j) => expect(j.kind).toBe('fairness'));
    http.expectOne('/v1/jobs/fairness').flush({
      id: 'f1',
      kind: 'fairness',
      status: 'queued',
      message: '',
      result: {},
      log_tail: [],
    });

    api.hpo({
      data_path: 'data/raw/ehr_data.csv',
      data_format: 'longitudinal',
      model_kind: 'logreg',
      calibrate: false,
      split_by_patient: true,
      temporal_split: false,
      windows_days: [7, 30, 180],
      window_days: 180,
      horizon_days: null,
      index_strategy: 'last_event',
      index_time_col: null,
      feature_inclusive: true,
      max_trials: 2,
    }).subscribe((j) => expect(j.kind).toBe('hpo'));
    http.expectOne('/v1/jobs/hpo').flush({
      id: 'h1',
      kind: 'hpo',
      status: 'queued',
      message: '',
      result: {},
      log_tail: [],
    });
  });

  it('GET schema/metrics/meta and POST predict', () => {
    api.schema().subscribe((s) => expect(s.feature_columns).toEqual(['a']));
    http.expectOne('/v1/model/schema').flush({
      feature_columns: ['a'],
      model_kind: 'logreg',
      calibrated: false,
    });

    api.metrics().subscribe();
    http.expectOne('/v1/model/metrics').flush({ roc_auc: 0.8 });

    api.meta().subscribe();
    http.expectOne('/v1/meta').flush({ clinical_use: 'research' });

    api.predict({ a: 1 }, true).subscribe((p) => expect(p.risk_level).toBe('medium'));
    const req = http.expectOne('/v1/predict');
    expect(req.request.body).toEqual({ features: { a: 1 }, include_explanation: true });
    req.flush({ risk_probability: 0.5, risk_level: 'medium' });
  });

  it('DELETE dataset, run detail, promote, thresholds, fairness report', () => {
    api.deleteDataset('data/uploads/x.csv').subscribe((r) => expect(r.deleted).toBeTrue());
    http.expectOne((r) => r.url === '/v1/datasets' && r.method === 'DELETE').flush({ deleted: true, path: 'x' });

    api.runDetail('run_a').subscribe((d) => expect(d.run_id).toBe('run_a'));
    http.expectOne('/v1/runs/run_a').flush({ run_id: 'run_a', path: 'reports/runs/run_a', has_model: true });

    api.promoteRun('run_a').subscribe((r) => expect(r.model_path).toContain('models'));
    http.expectOne('/v1/runs/run_a/promote').flush({ run_id: 'run_a', model_path: 'models/x.joblib' });

    api.thresholds().subscribe((t) => expect(t.present).toBeTrue());
    http.expectOne('/v1/reports/thresholds').flush({ present: true, points: [] });

    api.fairnessReport().subscribe((f) => expect(f.present).toBeTrue());
    http.expectOne('/v1/reports/fairness').flush({ present: true, by_group: [] });

    api.reportsSummary().subscribe((r) => expect(r.files.length).toBe(1));
    http.expectOne('/v1/reports/summary').flush({
      files: [{ name: 'evaluation_report.json', bytes: 10, url: '/v1/reports/file/evaluation_report.json' }],
    });
    expect(api.reportFileUrl('x.png')).toBe('/v1/reports/file/x.png');
    expect(api.resultsZipUrl()).toBe('/v1/reports/download.zip');
  });

  it('covers analysisPackUrl and profile filter params', () => {
    expect(api.analysisPackUrl('data/raw/x.csv')).toContain('path=data%2Fraw%2Fx.csv');
    api.datasetProfile('data/raw/x.csv', { label: '1', patient_id: 'p1' }).subscribe();
    const req = http.expectOne((r) => r.url === '/v1/datasets/profile');
    expect(req.request.params.get('label')).toBe('1');
    expect(req.request.params.get('patient_id')).toBe('p1');
    req.flush({ path: 'x', n_rows: 1, n_columns: 1, columns: [], cohort_rows: [] });

    api.importSql('select 1').subscribe();
    const sqlReq = http.expectOne('/v1/datasets/from-sql');
    expect(sqlReq.request.body.connection_url).toBeUndefined();
    sqlReq.flush({ path: 'data/uploads/s.csv' });
  });
});
