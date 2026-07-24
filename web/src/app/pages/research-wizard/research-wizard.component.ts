import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import {
  ApiService,
  DatasetInfo,
  JobInfo,
  TaskInfo,
  WorkspaceStatus,
} from '../../core/api.service';

type WizardStepId =
  | 'data'
  | 'health'
  | 'train'
  | 'trust'
  | 'leakage'
  | 'external'
  | 'export';

const PAPER_SYNTHETIC = 'data/raw/paper_synthetic_cohort.csv';
const DEMO_TINY = 'data/demo/ehr_data.csv';
const DEFAULT_TASK = 'horizon_detection_30d';

@Component({
  selector: 'app-research-wizard',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './research-wizard.component.html',
  styleUrl: './research-wizard.component.css',
})
export class ResearchWizardComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  step = signal<WizardStepId>('data');
  datasets = signal<DatasetInfo[]>([]);
  tasks = signal<TaskInfo[]>([]);
  status = signal<WorkspaceStatus | null>(null);
  /** Horizon tasks need index_time — default to paper synthetic, not tiny demo. */
  path = PAPER_SYNTHETIC;
  taskId = DEFAULT_TASK;
  modelKind = 'logreg';
  runId = signal<string | null>(null);
  job = signal<JobInfo | null>(null);
  health = signal<Record<string, unknown> | null>(null);
  trustNote = signal<string | null>(null);
  qualityNote = signal<string | null>(null);
  error = signal<string | null>(null);
  busy = signal(false);
  log = signal<string[]>([]);
  showHealthDebug = signal(false);

  readonly steps: { id: WizardStepId; label: string }[] = [
    { id: 'data', label: '1. Data & task' },
    { id: 'health', label: '2. Health' },
    { id: 'train', label: '3. Train' },
    { id: 'trust', label: '4. Trust pack' },
    { id: 'leakage', label: '5. Leakage' },
    { id: 'external', label: '6. External val' },
    { id: 'export', label: '7. Export' },
  ];

  ngOnInit(): void {
    this.api.datasets().subscribe({
      next: (r) => {
        this.datasets.set(r.datasets.filter((d) => d.exists));
        const paper = r.datasets.find((d) => d.path.includes('paper_synthetic'));
        const demo = r.datasets.find((d) => d.path.includes('demo/ehr_data'));
        if (paper?.exists) this.path = paper.path;
        else if (demo?.exists) this.path = demo.path;
        this.applyTaskSuggestion();
      },
      error: (e) => this.error.set(this.fmt(e)),
    });
    this.api.tasks().subscribe({
      next: (r) => {
        this.tasks.set(r.tasks || []);
        this.applyTaskSuggestion();
      },
      error: () => undefined,
    });
    this.refreshStatus();
  }

  ngOnDestroy(): void {
    this.clearPoll();
  }

  go(id: WizardStepId): void {
    if (id === 'train' && !this.canProceedToTrain()) {
      this.error.set(
        'Run a successful health check first. Fix blockers (often missing index_time) before training.'
      );
      this.step.set('health');
      return;
    }
    this.step.set(id);
    this.error.set(null);
  }

  onTaskChange(): void {
    this.applyTaskSuggestion();
    this.invalidateHealth();
  }

  onDatasetChange(): void {
    this.invalidateHealth();
    this.warnDatasetTaskMismatch();
  }

  usePaperSynthetic(): void {
    const paper = this.datasets().find((d) => d.path.includes('paper_synthetic') && d.exists);
    this.path = paper?.path || PAPER_SYNTHETIC;
    this.invalidateHealth();
    this.push(`Switched dataset to ${this.path}`);
  }

  useCustomTask(): void {
    const custom = this.tasks().find((t) => t.id === 'custom');
    if (custom) this.taskId = 'custom';
    else this.taskId = 'custom';
    this.invalidateHealth();
    this.push('Switched task to custom (last_event-friendly for tiny demo)');
  }

  refreshStatus(): void {
    this.api.workspaceStatus().subscribe({
      next: (s) => this.status.set(s),
      error: () => undefined,
    });
    this.api.reportsSummary().subscribe({
      next: (r) => this.qualityNote.set(r.quality_note || null),
      error: () => undefined,
    });
  }

  runHealth(): void {
    this.busy.set(true);
    this.error.set(null);
    this.api.datasetHealth(this.path, this.taskId).subscribe({
      next: (h) => {
        this.health.set(h as unknown as Record<string, unknown>);
        this.busy.set(false);
        const hh = (h as { health?: { ready_for_training?: boolean; blockers?: string[] } })?.health;
        const ready = hh?.ready_for_training;
        const blockers = hh?.blockers || [];
        if (ready) {
          this.push(`Health OK — ready for training (${this.taskId} @ ${this.path})`);
          this.go('train');
        } else {
          this.push(`Health blocked — ${blockers.length || 0} blocker(s)`);
          const msg = blockers.length
            ? `Not ready for training: ${blockers.join('; ')}`
            : 'Not ready for training — see health checks below.';
          this.error.set(msg);
        }
      },
      error: (e) => {
        this.busy.set(false);
        this.health.set(null);
        this.error.set(this.fmt(e));
      },
    });
  }

  startTrain(): void {
    if (!this.canProceedToTrain()) {
      this.error.set('Health check must pass before training. Fix blockers or switch dataset/task.');
      this.step.set('health');
      return;
    }
    this.busy.set(true);
    this.error.set(null);
    this.api
      .train({
        data_path: this.path,
        data_format: 'longitudinal',
        model_kind: this.modelKind,
        task_id: this.taskId,
        split_by_patient: true,
        temporal_split: false,
        calibrate: false,
        windows_days: [7, 30, 180],
        window_days: 180,
        horizon_days: null,
        index_strategy: 'last_event',
        index_time_col: null,
        feature_inclusive: true,
      })
      .subscribe({
        next: (j) => {
          this.job.set(j);
          this.push(`Train job ${j.id} queued`);
          this.pollJob(j.id, () => {
            const done = this.job();
            const rid = (done?.result as { run_id?: string } | undefined)?.run_id || null;
            if (rid) this.runId.set(rid);
            this.push(rid ? `Train finished · run ${rid}` : 'Train finished');
            this.refreshStatus();
            this.go('trust');
            this.loadTrust();
          });
        },
        error: (e) => {
          this.busy.set(false);
          this.error.set(this.fmt(e));
        },
      });
  }

  loadTrust(): void {
    const rid = this.runId();
    if (!rid) {
      this.trustNote.set('No run_id yet — open Results after a train, or retrain from this wizard.');
      return;
    }
    this.api.runDetail(rid).subscribe({
      next: (d) => {
        const flags = (d as { trust?: Record<string, unknown>; trust_pack?: Record<string, unknown> })
          ?.trust_pack || (d as { trust?: Record<string, unknown> })?.trust || d;
        this.trustNote.set(JSON.stringify(flags, null, 2).slice(0, 1200));
        this.push('Trust / run detail loaded');
      },
      error: (e) => this.trustNote.set(this.fmt(e)),
    });
  }

  runLeakage(): void {
    this.busy.set(true);
    this.error.set(null);
    this.api.leakageAudit({ use_artifact: true, run_id: this.runId() }).subscribe({
      next: (j) => {
        this.push(`Leakage job ${j.id}`);
        this.pollJob(j.id, () => {
          this.push('Leakage audit finished');
          this.refreshStatus();
          this.go('external');
        });
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmt(e));
      },
    });
  }

  runExternal(): void {
    this.busy.set(true);
    this.error.set(null);
    this.api
      .externalValidate({
        data_path: this.path,
        data_format: 'longitudinal',
        run_id: this.runId(),
      })
      .subscribe({
        next: (j) => {
          this.push(`External validate job ${j.id}`);
          this.pollJob(j.id, () => {
            this.push('External validation finished');
            this.refreshStatus();
            this.go('export');
          });
        },
        error: (e) => {
          this.busy.set(false);
          this.error.set(this.fmt(e));
        },
      });
  }

  zipUrl(): string {
    return this.api.resultsZipUrl(this.runId());
  }

  methodsUrl(): string {
    return this.api.methodsMdUrl(this.runId());
  }

  healthBlockers(): string[] {
    const h = this.health() as { health?: { blockers?: string[] } } | null;
    return h?.health?.blockers || [];
  }

  healthWarnings(): string[] {
    const h = this.health() as { health?: { warnings?: string[] } } | null;
    return h?.health?.warnings || [];
  }

  healthChecks(): { name: string; ok: boolean; detail: string; blocking?: boolean }[] {
    const h = this.health() as {
      health?: { checks?: { name: string; ok: boolean; detail: string; blocking?: boolean }[] };
    } | null;
    return h?.health?.checks || [];
  }

  healthReady(): boolean | null {
    const h = this.health() as { health?: { ready_for_training?: boolean } } | null;
    if (!h?.health) return null;
    return !!h.health.ready_for_training;
  }

  canProceedToTrain(): boolean {
    return this.healthReady() === true;
  }

  needsIndexTime(): boolean {
    const t = this.tasks().find((x) => x.id === this.taskId);
    if (!t) return false;
    return (
      t.index_strategy === 'column' ||
      (t.required_columns || []).includes('index_time') ||
      !!t.index_time_col
    );
  }

  private invalidateHealth(): void {
    this.health.set(null);
    this.showHealthDebug.set(false);
  }

  private warnDatasetTaskMismatch(): void {
    if (!this.needsIndexTime()) return;
    if (this.path === DEMO_TINY || this.path.endsWith('/ehr_data.csv')) {
      this.error.set(
        'Tiny demo lacks index_time for this task. Switch to paper_synthetic, or use the custom task.'
      );
    }
  }

  private applyTaskSuggestion(): void {
    const t = this.tasks().find((x) => x.id === this.taskId);
    if (!t) return;
    const suggested = (t.suggested_path || '').trim();
    if (suggested) {
      const exists = this.datasets().some((d) => d.path === suggested && d.exists);
      if (exists || suggested.includes('paper_synthetic') || suggested.includes('demo/')) {
        this.path = suggested;
      }
    }
    // Tiny demo cannot satisfy column/index_time tasks — nudge to paper synthetic.
    const needsIndex =
      t.index_strategy === 'column' ||
      (t.required_columns || []).includes('index_time') ||
      !!t.index_time_col;
    if (needsIndex && (this.path === DEMO_TINY || this.path.endsWith('/ehr_data.csv'))) {
      const paper = this.datasets().find((d) => d.path.includes('paper_synthetic') && d.exists);
      if (paper) this.path = paper.path;
      else this.path = PAPER_SYNTHETIC;
    }
  }

  private pollJob(id: string, onDone: () => void): void {
    this.clearPoll();
    this.pollTimer = setInterval(() => {
      this.api.job(id).subscribe({
        next: (j) => {
          this.job.set(j);
          if (j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled') {
            this.clearPoll();
            this.busy.set(false);
            if (j.status !== 'succeeded') {
              this.error.set((j as { error?: string }).error || `Job ${j.status}`);
              return;
            }
            onDone();
          }
        },
        error: (e) => {
          this.clearPoll();
          this.busy.set(false);
          this.error.set(this.fmt(e));
        },
      });
    }, 800);
  }

  private clearPoll(): void {
    if (this.pollTimer != null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private push(msg: string): void {
    this.log.update((rows) => [`${new Date().toISOString().slice(11, 19)} ${msg}`, ...rows].slice(0, 40));
  }

  private fmt(e: unknown): string {
    const err = e as {
      error?: {
        detail?:
          | string
          | { message?: string; blockers?: string[]; warnings?: string[]; hint?: string };
      };
      message?: string;
    };
    const detail = err?.error?.detail;
    if (detail && typeof detail === 'object') {
      const blockers = detail.blockers?.length ? ` — ${detail.blockers.join('; ')}` : '';
      const hint = detail.hint ? ` ${detail.hint}` : '';
      return `${detail.message || 'Request failed'}${blockers}${hint}`;
    }
    if (typeof detail === 'string') return detail;
    return err?.message || String(e);
  }
}
