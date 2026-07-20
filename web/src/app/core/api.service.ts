import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiUrl;

  health(): Observable<unknown> {
    return this.http.get(`${this.base}/health`);
  }

  workspaceStatus(): Observable<WorkspaceStatus> {
    return this.http.get<WorkspaceStatus>(`${this.base}/v1/workspace/status`);
  }

  datasets(): Observable<{ datasets: DatasetInfo[] }> {
    return this.http.get<{ datasets: DatasetInfo[] }>(`${this.base}/v1/datasets`);
  }

  uploadDataset(file: File): Observable<unknown> {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return this.http.post(`${this.base}/v1/datasets/upload`, fd);
  }

  importForm(rows: Record<string, unknown>[], name = 'form_import.csv'): Observable<unknown> {
    return this.http.post(`${this.base}/v1/datasets/from-form`, { name, rows });
  }

  importSql(sql: string, connection_url?: string, name = 'sql_import.csv'): Observable<unknown> {
    return this.http.post(`${this.base}/v1/datasets/from-sql`, { sql, connection_url, name });
  }

  datasetProfile(
    path: string,
    filters?: { age_band?: string; label?: string; patient_id?: string }
  ): Observable<DatasetProfile> {
    const params: Record<string, string> = { path };
    if (filters?.age_band) params['age_band'] = filters.age_band;
    if (filters?.label) params['label'] = filters.label;
    if (filters?.patient_id) params['patient_id'] = filters.patient_id;
    return this.http.get<DatasetProfile>(`${this.base}/v1/datasets/profile`, { params });
  }

  resultsZipUrl(): string {
    return `${this.base}/v1/reports/download.zip`;
  }

  train(body: TrainBody): Observable<JobInfo> {
    return this.http.post<JobInfo>(`${this.base}/v1/jobs/train`, body);
  }

  compare(body: CompareBody): Observable<JobInfo> {
    return this.http.post<JobInfo>(`${this.base}/v1/jobs/compare`, body);
  }

  tasks(): Observable<{ tasks: TaskInfo[] }> {
    return this.http.get<{ tasks: TaskInfo[] }>(`${this.base}/v1/tasks`);
  }

  datasetHealth(path: string): Observable<DatasetHealth> {
    return this.http.get<DatasetHealth>(`${this.base}/v1/datasets/health`, {
      params: { path },
    });
  }

  leakageAudit(body: Record<string, unknown> = { use_artifact: true }): Observable<JobInfo> {
    return this.http.post<JobInfo>(`${this.base}/v1/jobs/leakage-audit`, body);
  }

  shap(): Observable<JobInfo> {
    return this.http.post<JobInfo>(`${this.base}/v1/jobs/shap`, {});
  }

  job(id: string): Observable<JobInfo> {
    return this.http.get<JobInfo>(`${this.base}/v1/jobs/${id}`);
  }

  jobs(): Observable<{ jobs: JobInfo[] }> {
    return this.http.get<{ jobs: JobInfo[] }>(`${this.base}/v1/jobs`);
  }

  cancelJob(id: string): Observable<JobInfo> {
    return this.http.post<JobInfo>(`${this.base}/v1/jobs/${id}/cancel`, {});
  }

  fairness(body: { groups_path?: string | null; group_column?: string } = {}): Observable<JobInfo> {
    return this.http.post<JobInfo>(`${this.base}/v1/jobs/fairness`, body);
  }

  fairnessReport(): Observable<FairnessReport> {
    return this.http.get<FairnessReport>(`${this.base}/v1/reports/fairness`);
  }

  thresholds(): Observable<ThresholdReport> {
    return this.http.get<ThresholdReport>(`${this.base}/v1/reports/thresholds`);
  }

  hpo(body: HpoBody): Observable<JobInfo> {
    return this.http.post<JobInfo>(`${this.base}/v1/jobs/hpo`, body);
  }

  runs(limit = 30): Observable<{ runs: RunSummary[] }> {
    return this.http.get<{ runs: RunSummary[] }>(`${this.base}/v1/runs`, {
      params: { limit: String(limit) },
    });
  }

  runDetail(runId: string): Observable<RunDetail> {
    return this.http.get<RunDetail>(`${this.base}/v1/runs/${encodeURIComponent(runId)}`);
  }

  promoteRun(runId: string): Observable<{ run_id: string; model_path: string }> {
    return this.http.post<{ run_id: string; model_path: string }>(
      `${this.base}/v1/runs/${encodeURIComponent(runId)}/promote`,
      {}
    );
  }

  reportsSummary(): Observable<ReportsSummary> {
    return this.http.get<ReportsSummary>(`${this.base}/v1/reports/summary`);
  }

  reportFileUrl(name: string): string {
    return `${this.base}/v1/reports/file/${name}`;
  }

  schema(): Observable<ModelSchema> {
    return this.http.get<ModelSchema>(`${this.base}/v1/model/schema`);
  }

  metrics(): Observable<unknown> {
    return this.http.get(`${this.base}/v1/model/metrics`);
  }

  predict(features: Record<string, number>, includeExplanation = true): Observable<PredictResult> {
    return this.http.post<PredictResult>(`${this.base}/v1/predict`, {
      features,
      include_explanation: includeExplanation,
    });
  }

  meta(): Observable<unknown> {
    return this.http.get(`${this.base}/v1/meta`);
  }
}

export interface WorkspaceStatus {
  api_ok: boolean;
  model_ready: boolean;
  evaluation_present: boolean;
  metrics?: Record<string, number | null>;
  leakage_audit_present: boolean;
  shap_present: boolean;
  calibration_present: boolean;
  demo_datasets_available: boolean;
  checklist: Record<string, boolean>;
  recent_jobs: JobInfo[];
}

export interface DatasetInfo {
  id: string;
  label: string;
  path: string;
  format: string;
  exists: boolean;
  bundled?: boolean;
  source_type?: string;
  suggested?: {
    horizon_days?: number;
    index_strategy?: string;
    index_time_col?: string;
    windows_days?: number[];
  };
}

export interface TrainBody {
  data_path: string;
  data_format: string;
  model_kind: string;
  calibrate: boolean;
  split_by_patient: boolean;
  temporal_split: boolean;
  windows_days: number[] | null;
  window_days: number;
  horizon_days: number | null;
  index_strategy: string;
  index_time_col: string | null;
  feature_inclusive: boolean;
  label_col?: string | null;
  task_id?: string | null;
}

export interface CompareBody {
  data_path: string;
  data_format: string;
  calibrate: boolean;
  split_by_patient: boolean;
  temporal_split: boolean;
  windows_days: number[] | null;
  window_days: number;
  horizon_days: number | null;
  index_strategy: string;
  index_time_col: string | null;
  feature_inclusive: boolean;
  label_col?: string | null;
  task_id?: string | null;
  promote_best?: boolean;
}

export interface HpoBody {
  data_path: string;
  data_format: string;
  model_kind: string;
  calibrate: boolean;
  split_by_patient: boolean;
  temporal_split: boolean;
  windows_days: number[] | null;
  window_days: number;
  horizon_days: number | null;
  index_strategy: string;
  index_time_col: string | null;
  feature_inclusive: boolean;
  label_col?: string | null;
  task_id?: string | null;
  promote_best?: boolean;
  max_trials?: number;
}

export interface TaskInfo {
  id: string;
  name: string;
  description?: string;
  target_column?: string | null;
  horizon_days?: number | null;
  index_strategy?: string;
  index_time_col?: string | null;
  windows_days?: number[];
  window_days?: number;
  data_format?: string;
  suggested_path?: string | null;
  model_kind?: string;
  calibrate?: boolean;
  split_by_patient?: boolean;
  temporal_split?: boolean;
  required_columns?: string[];
}

export interface DatasetHealth {
  path: string;
  n_rows: number;
  n_columns: number;
  n_patients?: number;
  health: {
    patients?: number;
    features?: number;
    missing_pct_overall?: number;
    temporal_integrity?: string;
    leakage_risk?: string;
    ready_for_training?: boolean;
    blockers?: string[];
    warnings?: string[];
    checks?: { name: string; ok: boolean; detail: string }[];
    leakage_notes?: string[];
  };
}

export interface JobInfo {
  id: string;
  kind: string;
  status: string;
  message: string;
  result: Record<string, unknown>;
  log_tail: string[];
  created_at?: string;
  finished_at?: string | null;
}

export interface RunSummary {
  run_id: string;
  path: string;
  has_model: boolean;
  has_evaluation?: boolean;
  has_manifest?: boolean;
  meta?: Record<string, unknown>;
  metrics?: Record<string, number | null> | null;
  model_kind?: string | null;
}

export interface RunDetail extends RunSummary {
  evaluation?: Record<string, unknown> | null;
  manifest?: Record<string, unknown> | null;
  feature_importance?: Record<string, unknown> | null;
  files?: { name: string; bytes: number }[];
}

export interface FairnessReport {
  present?: boolean;
  skipped?: boolean;
  reason?: string;
  group_column?: string;
  by_group?: Record<string, unknown>[];
}

export interface ThresholdReport {
  present?: boolean;
  threshold?: number;
  points?: {
    threshold: number;
    precision: number;
    recall: number;
    f1: number;
    accuracy: number;
    positive_rate: number;
  }[];
  note?: string;
}

export interface ReportsSummary {
  metrics?: Record<string, number | null>;
  threshold?: number;
  leakage_audit?: Record<string, unknown>;
  feature_importance?: Record<string, number> | { importance?: Record<string, number> };
  model_comparison?: {
    selected_model?: string;
    comparison?: {
      model: string;
      roc_auc?: number;
      pr_auc?: number;
      brier?: number;
      ece?: number;
      selected?: boolean;
    }[];
  };
  fairness?: FairnessReport | null;
  hpo?: {
    model_kind?: string;
    n_trials?: number;
    trials?: Record<string, unknown>[];
    best?: Record<string, unknown>;
    note?: string;
  } | null;
  thresholds?: ThresholdReport | null;
  files: { name: string; bytes: number; url: string }[];
  download_zip?: string;
}

export interface ModelSchema {
  feature_columns: string[];
  model_kind: string;
  calibrated: boolean;
  input_stats?: Record<string, { median?: number; p05?: number; p95?: number }>;
}

export interface DatasetProfile {
  path: string;
  n_rows: number;
  n_columns: number;
  columns: string[];
  n_patients?: number;
  label_column?: string;
  label_counts?: Record<string, number>;
  age_band_counts?: Record<string, number>;
  missing_pct?: Record<string, number>;
  numeric_preview?: Record<string, { mean?: number; std?: number }>;
  time_span?: { min?: string; max?: string };
  filters?: { age_band?: string | null; label?: string | null; patient_id?: string | null };
  cohort_rows?: {
    patient_id: string;
    age?: number | null;
    age_band?: string | null;
    sex?: string | null;
    label?: string | null;
    glucose_mean?: number | null;
  }[];
  filter_options?: { age_bands?: string[]; labels?: string[]; sexes?: string[] };
}

export interface PredictResult {
  risk_probability: number;
  risk_level: string;
  explanation?: unknown;
}
