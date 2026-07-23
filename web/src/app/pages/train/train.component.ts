import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { ApiService, CompareBody, HpoBody, JobInfo, TaskInfo, TrainBody } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';
import { Subscription, interval, switchMap, takeWhile } from 'rxjs';

@Component({
  selector: 'app-train',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './train.component.html',
  styleUrl: './train.component.css',
})
export class TrainComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly state = inject(WorkspaceState);
  private readonly route = inject(ActivatedRoute);
  private pollSub?: Subscription;
  demoBanner = signal(false);

  tasks = signal<TaskInfo[]>([]);
  taskId = '';
  modelKind = 'logreg';
  calibrate = false;
  splitByPatient = true;
  temporalSplit = false;
  useMultiWindow = true;
  windowsDays: number[] = [7, 30, 180];
  horizonDays: number | null = null;
  indexStrategy = 'last_event';
  indexTimeCol: string | null = null;
  labelCol: string | null = null;
  dataPath = 'data/demo/ehr_data.csv';
  dataFormat = 'longitudinal';
  /** Optional research-scoped light HPO grid. */
  enableHpo = false;
  hpoPromoteBest = false;

  job = signal<JobInfo | null>(null);
  recentJobs = signal<JobInfo[]>([]);
  error = signal<string | null>(null);
  busy = signal(false);

  ngOnInit(): void {
    const demo = this.route.snapshot.queryParamMap.get('demo') === '1';
    const d = this.state.selectedDataset();
    if (!demo && d) {
      this.dataPath = d.path;
      this.dataFormat = d.format;
      const s = d.suggested;
      if (s) {
        this.horizonDays = s.horizon_days ?? null;
        this.indexStrategy = s.index_strategy || 'last_event';
        this.indexTimeCol = s.index_time_col || null;
        this.useMultiWindow = !!(s.windows_days && s.windows_days.length > 1);
        if (s.windows_days) this.windowsDays = s.windows_days;
      }
    }
    this.api.tasks().subscribe({
      next: (r) => {
        this.tasks.set(r.tasks);
        if (demo) this.applyDemoPreset();
      },
    });
    this.refreshJobs();
  }

  /** Bundled CSV + custom task — research/education demo only. */
  applyDemoPreset(): void {
    this.demoBanner.set(true);
    this.taskId = 'custom';
    const t = this.selectedTask();
    if (t) {
      this.applyTask();
    } else {
      this.dataPath = 'data/demo/ehr_data.csv';
      this.dataFormat = 'longitudinal';
      this.modelKind = 'logreg';
      this.calibrate = false;
      this.splitByPatient = true;
      this.temporalSplit = false;
      this.useMultiWindow = true;
      this.windowsDays = [7, 30, 180];
      this.horizonDays = null;
      this.indexStrategy = 'last_event';
      this.indexTimeCol = null;
      this.labelCol = 'label';
    }
  }

  ngOnDestroy(): void {
    this.pollSub?.unsubscribe();
  }

  selectedTask(): TaskInfo | null {
    return this.tasks().find((x) => x.id === this.taskId) || null;
  }

  taskHealthBlockers = signal<string[]>([]);

  checkTaskHealth(): void {
    const t = this.selectedTask();
    if (!this.dataPath) return;
    this.api.datasetHealth(this.dataPath, t?.id || null).subscribe({
      next: (h) => {
        const blockers = (h.health?.blockers || []) as string[];
        this.taskHealthBlockers.set(blockers);
        if (!blockers.length) this.error.set(null);
        else this.error.set(`Task health blockers: ${blockers.join('; ')}`);
      },
      error: (e) => this.error.set(this.fmtErr(e)),
    });
  }

  applyTask(): void {
    const t = this.selectedTask();
    if (!t) return;
    if (t.suggested_path) this.dataPath = t.suggested_path;
    if (t.data_format) this.dataFormat = t.data_format;
    if (t.model_kind) this.modelKind = t.model_kind;
    this.calibrate = !!t.calibrate;
    this.splitByPatient = t.split_by_patient !== false;
    this.temporalSplit = !!t.temporal_split;
    this.horizonDays = t.horizon_days ?? null;
    this.indexStrategy = t.index_strategy || 'last_event';
    this.indexTimeCol = t.index_time_col || null;
    this.labelCol = t.target_column || null;
    if (t.windows_days?.length) {
      this.windowsDays = t.windows_days;
      this.useMultiWindow = t.windows_days.length > 1;
    }
  }

  private bodyBase(): TrainBody {
    return {
      data_path: this.dataPath,
      data_format: this.dataFormat,
      model_kind: this.modelKind,
      calibrate: this.calibrate,
      split_by_patient: this.splitByPatient && !this.temporalSplit,
      temporal_split: this.temporalSplit,
      windows_days: this.dataFormat === 'longitudinal' && this.useMultiWindow ? this.windowsDays : null,
      window_days: 180,
      horizon_days: this.horizonDays,
      index_strategy: this.indexStrategy,
      index_time_col: this.indexTimeCol,
      feature_inclusive: true,
      label_col: this.labelCol,
      task_id: this.taskId || null,
    };
  }

  startTrain(): void {
    this.error.set(null);
    this.busy.set(true);
    this.api.train(this.bodyBase()).subscribe({
      next: (j) => {
        this.state.lastJobId.set(j.id);
        this.poll(j.id);
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  startCompare(): void {
    this.error.set(null);
    this.busy.set(true);
    const b = this.bodyBase();
    const body: CompareBody = {
      data_path: b.data_path,
      data_format: b.data_format,
      calibrate: b.calibrate,
      split_by_patient: b.split_by_patient,
      temporal_split: b.temporal_split,
      windows_days: b.windows_days,
      window_days: b.window_days,
      horizon_days: b.horizon_days,
      index_strategy: b.index_strategy,
      index_time_col: b.index_time_col,
      feature_inclusive: b.feature_inclusive,
      label_col: b.label_col,
      task_id: b.task_id,
      promote_best: true,
    };
    this.api.compare(body).subscribe({
      next: (j) => this.poll(j.id),
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  startHpo(): void {
    this.error.set(null);
    this.busy.set(true);
    const b = this.bodyBase();
    const body: HpoBody = {
      data_path: b.data_path,
      data_format: b.data_format,
      model_kind: b.model_kind,
      calibrate: b.calibrate,
      split_by_patient: b.split_by_patient,
      temporal_split: b.temporal_split,
      windows_days: b.windows_days,
      window_days: b.window_days,
      horizon_days: b.horizon_days,
      index_strategy: b.index_strategy,
      index_time_col: b.index_time_col,
      feature_inclusive: b.feature_inclusive,
      label_col: b.label_col,
      task_id: b.task_id,
      promote_best: this.hpoPromoteBest,
      max_trials: 6,
    };
    this.api.hpo(body).subscribe({
      next: (j) => this.poll(j.id),
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  startAudit(): void {
    this.error.set(null);
    this.busy.set(true);
    this.api.leakageAudit({ use_artifact: true }).subscribe({
      next: (j) => this.poll(j.id),
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  cancelActive(): void {
    const j = this.job();
    if (!j || (j.status !== 'queued' && j.status !== 'running')) return;
    this.api.cancelJob(j.id).subscribe({
      next: (x) => {
        this.job.set(x);
        this.busy.set(false);
        this.refreshJobs();
      },
      error: (e) => this.error.set(this.fmtErr(e)),
    });
  }

  refreshJobs(): void {
    this.api.jobs().subscribe({
      next: (r) => this.recentJobs.set(r.jobs || []),
      error: () => undefined,
    });
  }

  private poll(id: string): void {
    this.pollSub?.unsubscribe();
    this.pollSub = interval(800)
      .pipe(
        switchMap(() => this.api.job(id)),
        takeWhile((j) => j.status === 'queued' || j.status === 'running', true)
      )
      .subscribe({
        next: (j) => {
          this.job.set(j);
          if (j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled') {
            this.busy.set(false);
            this.refreshJobs();
          }
        },
        error: (e) => {
          this.busy.set(false);
          this.error.set(this.fmtErr(e));
        },
      });
  }

  private fmtErr(e: unknown): string {
    const any = e as { error?: { detail?: unknown }; message?: string; status?: number };
    if (any?.status === 401) {
      return 'API key required or invalid (401). Set it under Config.';
    }
    const d = any?.error?.detail;
    if (typeof d === 'string') return d;
    if (d && typeof d === 'object' && 'message' in d) {
      const msg = String((d as { message: string }).message);
      const blockers = (d as { blockers?: string[] }).blockers;
      return blockers?.length ? `${msg}: ${blockers.join('; ')}` : msg;
    }
    if (Array.isArray(d)) {
      return d
        .map((x) => (typeof x === 'object' && x && 'msg' in x ? String((x as { msg: string }).msg) : String(x)))
        .join('; ');
    }
    return any?.message || 'Request failed';
  }
}
