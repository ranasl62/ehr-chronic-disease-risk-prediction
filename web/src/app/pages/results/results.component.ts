import {
  AfterViewInit,
  ChangeDetectorRef,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
  inject,
  signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import {
  ApiService,
  FairnessReport,
  JobInfo,
  ReportsSummary,
  RunDetail,
  RunSummary,
  ThresholdReport,
} from '../../core/api.service';
import { UiPrefsService } from '../../core/ui-prefs.service';
import { DataTableColumn, DataTableComponent } from '../../shared/data-table.component';
import { interval, switchMap, takeWhile } from 'rxjs';

Chart.register(...registerables);

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, DataTableComponent],
  templateUrl: './results.component.html',
  styleUrl: './results.component.css',
})
export class ResultsComponent implements OnInit, AfterViewInit, OnDestroy {
  private readonly api = inject(ApiService);
  readonly ui = inject(UiPrefsService);
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly host = inject(ElementRef<HTMLElement>);

  @ViewChild('compareChart') compareChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('metricChart') metricChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('importanceChart') importanceChartRef?: ElementRef<HTMLCanvasElement>;

  summary = signal<ReportsSummary | null>(null);
  runs = signal<RunSummary[]>([]);
  selectedRun = signal<RunDetail | null>(null);
  fairness = signal<FairnessReport | null>(null);
  thresholds = signal<ThresholdReport | null>(null);
  compareRunIds: string[] = [];
  error = signal<string | null>(null);
  job = signal<JobInfo | null>(null);
  busy = signal(false);
  figFilter = '';
  metricFilter = '';
  showCharts = true;
  chartsReady = signal(false);

  compareRows: Record<string, unknown>[] = [];
  metricRows: Record<string, unknown>[] = [];
  fileRows: Record<string, unknown>[] = [];
  leakageRows: Record<string, unknown>[] = [];
  importanceRows: Record<string, unknown>[] = [];
  fairnessRows: Record<string, unknown>[] = [];
  thresholdRows: Record<string, unknown>[] = [];
  hpoRows: Record<string, unknown>[] = [];
  runCompareRows: Record<string, unknown>[] = [];

  compareCols: DataTableColumn[] = [
    { key: 'model', label: 'Model' },
    { key: 'roc_auc', label: 'ROC-AUC', numeric: true, format: 'number' },
    { key: 'pr_auc', label: 'PR-AUC', numeric: true, format: 'number' },
    { key: 'brier', label: 'Brier', numeric: true, format: 'number' },
    { key: 'ece', label: 'ECE', numeric: true, format: 'number' },
    { key: 'selected', label: 'Selected' },
  ];
  metricCols: DataTableColumn[] = [
    { key: 'metric', label: 'Metric' },
    { key: 'value', label: 'Value', numeric: true, format: 'number' },
  ];
  fileCols: DataTableColumn[] = [
    { key: 'name', label: 'File' },
    { key: 'bytes', label: 'Bytes', numeric: true, format: 'number', digits: '1.0-0' },
    { key: 'kind', label: 'Kind' },
  ];
  leakageCols: DataTableColumn[] = [
    { key: 'key', label: 'Check' },
    { key: 'value', label: 'Value' },
  ];
  importanceCols: DataTableColumn[] = [
    { key: 'feature', label: 'Feature' },
    { key: 'importance', label: 'Importance', numeric: true, format: 'number' },
    { key: 'abs', label: '|Importance|', numeric: true, format: 'number' },
  ];
  fairnessCols: DataTableColumn[] = [
    { key: 'group', label: 'Group' },
    { key: 'n', label: 'N', numeric: true, format: 'number', digits: '1.0-0' },
    { key: 'prevalence', label: 'Prevalence', numeric: true, format: 'number' },
    { key: 'accuracy', label: 'Accuracy', numeric: true, format: 'number' },
    { key: 'mean_pred', label: 'Mean pred', numeric: true, format: 'number' },
    { key: 'tpr', label: 'TPR', numeric: true, format: 'number' },
    { key: 'fpr', label: 'FPR', numeric: true, format: 'number' },
  ];
  thresholdCols: DataTableColumn[] = [
    { key: 'threshold', label: 'Threshold', numeric: true, format: 'number' },
    { key: 'precision', label: 'Precision', numeric: true, format: 'number' },
    { key: 'recall', label: 'Recall', numeric: true, format: 'number' },
    { key: 'f1', label: 'F1', numeric: true, format: 'number' },
    { key: 'accuracy', label: 'Accuracy', numeric: true, format: 'number' },
    { key: 'positive_rate', label: 'Pos rate', numeric: true, format: 'number' },
  ];
  hpoCols: DataTableColumn[] = [
    { key: 'trial', label: 'Trial', numeric: true, format: 'number', digits: '1.0-0' },
    { key: 'params', label: 'Params' },
    { key: 'roc_auc', label: 'ROC-AUC', numeric: true, format: 'number' },
    { key: 'pr_auc', label: 'PR-AUC', numeric: true, format: 'number' },
    { key: 'brier', label: 'Brier', numeric: true, format: 'number' },
  ];
  runCompareCols: DataTableColumn[] = [
    { key: 'run_id', label: 'Run' },
    { key: 'model_kind', label: 'Model' },
    { key: 'roc_auc', label: 'ROC-AUC', numeric: true, format: 'number' },
    { key: 'pr_auc', label: 'PR-AUC', numeric: true, format: 'number' },
    { key: 'brier', label: 'Brier', numeric: true, format: 'number' },
  ];

  private charts: Chart[] = [];
  private lastSummary: ReportsSummary | null = null;
  private redrawTimer: ReturnType<typeof setTimeout> | null = null;

  ngOnInit(): void {
    this.reload();
  }

  ngAfterViewInit(): void {
    this.scheduleRedraw();
  }

  ngOnDestroy(): void {
    if (this.redrawTimer != null) clearTimeout(this.redrawTimer);
    this.destroyCharts();
  }

  get prefs() {
    return this.ui.prefs();
  }

  reload(): void {
    this.error.set(null);
    this.api.reportsSummary().subscribe({
      next: (s) => {
        this.summary.set(s);
        this.lastSummary = s;
        this.buildTables(s);
        this.scheduleRedraw();
      },
      error: (e) => this.error.set(this.fmtErr(e)),
    });
    this.api.runs(40).subscribe({
      next: (r) => this.runs.set(r.runs || []),
      error: () => undefined,
    });
    this.api.fairnessReport().subscribe({
      next: (f) => {
        this.fairness.set(f);
        this.buildFairnessRows(f);
      },
      error: () => undefined,
    });
  }

  onShowCharts(v: boolean): void {
    this.showCharts = v;
    this.scheduleRedraw();
  }

  filteredFigs(): { name: string; bytes: number; url: string }[] {
    const s = this.summary();
    if (!s) return [];
    const q = this.figFilter.trim().toLowerCase();
    return (s.files || []).filter(
      (f) => f.name.endsWith('.png') && (!q || f.name.toLowerCase().includes(q))
    );
  }

  filteredMetrics(): Record<string, unknown>[] {
    const q = this.metricFilter.trim().toLowerCase();
    if (!q) return this.metricRows;
    return this.metricRows.filter((r) => String(r['metric'] ?? '').toLowerCase().includes(q));
  }

  openRun(runId: string): void {
    this.error.set(null);
    this.api.runDetail(runId).subscribe({
      next: (d) => this.selectedRun.set(d),
      error: (e) => this.error.set(this.fmtErr(e)),
    });
  }

  promoteSelected(): void {
    const d = this.selectedRun();
    if (!d?.run_id) return;
    this.busy.set(true);
    this.api.promoteRun(d.run_id).subscribe({
      next: () => {
        this.busy.set(false);
        this.reload();
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  toggleCompareRun(runId: string, checked: boolean): void {
    if (checked) {
      if (!this.compareRunIds.includes(runId)) this.compareRunIds = [...this.compareRunIds, runId].slice(-4);
    } else {
      this.compareRunIds = this.compareRunIds.filter((id) => id !== runId);
    }
    this.buildRunCompare();
  }

  private buildRunCompare(): void {
    const map = new Map(this.runs().map((r) => [r.run_id, r]));
    this.runCompareRows = this.compareRunIds
      .map((id) => map.get(id))
      .filter((r): r is RunSummary => !!r)
      .map((r) => ({
        run_id: r.run_id,
        model_kind: r.model_kind || (r.meta?.['model_kind'] as string) || '—',
        roc_auc: r.metrics?.['roc_auc'] ?? null,
        pr_auc: r.metrics?.['pr_auc'] ?? null,
        brier: r.metrics?.['brier'] ?? null,
      }));
  }

  runShap(): void {
    this.startJob(() => this.api.shap());
  }

  runFairness(): void {
    this.startJob(() => this.api.fairness({ group_column: 'age_band' }));
  }

  loadThresholds(): void {
    this.busy.set(true);
    this.api.thresholds().subscribe({
      next: (t) => {
        this.busy.set(false);
        this.thresholds.set(t);
        this.thresholdRows = (t.points || []).map((p) => ({ ...p }));
        this.reload();
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  private startJob(fn: () => ReturnType<ApiService['shap']>): void {
    this.busy.set(true);
    fn().subscribe({
      next: (j) => {
        interval(1000)
          .pipe(
            switchMap(() => this.api.job(j.id)),
            takeWhile((x) => x.status === 'queued' || x.status === 'running', true)
          )
          .subscribe({
            next: (x) => {
              this.job.set(x);
              if (x.status === 'succeeded' || x.status === 'failed' || x.status === 'cancelled') {
                this.busy.set(false);
                this.reload();
              }
            },
            error: (e) => {
              this.busy.set(false);
              this.error.set(this.fmtErr(e));
            },
          });
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  fileUrl(name: string): string {
    return this.api.reportFileUrl(name);
  }

  zipUrl(): string {
    return this.api.resultsZipUrl();
  }

  onPageSize(n: number): void {
    this.ui.patch({ table_page_size: n });
  }

  private fmtErr(e: unknown): string {
    const any = e as { error?: { detail?: unknown }; message?: string; status?: number };
    if (any?.status === 401) {
      return 'API key required or invalid (401). Set it under Config.';
    }
    const d = any?.error?.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d
        .map((x) => (typeof x === 'object' && x && 'msg' in x ? String((x as { msg: string }).msg) : String(x)))
        .join('; ');
    }
    if (d && typeof d === 'object' && 'message' in d) return String((d as { message: string }).message);
    return any?.message || 'Request failed';
  }

  private scheduleRedraw(): void {
    this.cdr.detectChanges();
    if (this.redrawTimer != null) clearTimeout(this.redrawTimer);
    this.redrawTimer = setTimeout(() => {
      this.redrawTimer = null;
      requestAnimationFrame(() => this.renderCharts(this.lastSummary));
    }, 0);
  }

  private buildFairnessRows(f: FairnessReport | null): void {
    const rows = f?.by_group || [];
    this.fairnessRows = rows.map((row) => {
      const r = row as Record<string, unknown>;
      return {
        group: r['group'] ?? r['subgroup'] ?? r['age_band'] ?? '—',
        n: r['n'] ?? r['count'] ?? null,
        prevalence: r['prevalence'] ?? r['positive_rate'] ?? null,
        accuracy: r['accuracy'] ?? null,
        mean_pred: r['mean_pred'] ?? r['mean_predicted_prob'] ?? null,
        tpr: r['tpr'] ?? r['recall'] ?? null,
        fpr: r['fpr'] ?? null,
      };
    });
  }

  private buildTables(s: ReportsSummary): void {
    this.compareRows = (s.model_comparison?.comparison || []).map((row) => ({
      model: row.model,
      roc_auc: row.roc_auc,
      pr_auc: row.pr_auc,
      brier: row.brier,
      ece: row.ece,
      selected: row.selected ? 'yes' : '',
    }));
    const m = s.metrics || {};
    this.metricRows = Object.entries(m).map(([metric, value]) => ({ metric, value }));
    this.fileRows = (s.files || []).map((f) => ({
      name: f.name,
      bytes: f.bytes,
      kind: f.name.includes('.') ? f.name.split('.').pop() : '',
    }));
    this.leakageRows = this.flattenAudit(s.leakage_audit || {});
    const raw = s.feature_importance;
    const nested = raw && typeof raw === 'object' && 'importance' in raw ? raw.importance : undefined;
    const map: Record<string, number> =
      nested && typeof nested === 'object' ? nested : ((raw as Record<string, number>) || {});
    this.importanceRows = Object.entries(map)
      .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
      .map(([feature, importance]) => ({ feature, importance, abs: Math.abs(importance) }));

    if (s.fairness) {
      this.fairness.set(s.fairness);
      this.buildFairnessRows(s.fairness);
    }
    if (s.thresholds?.points?.length) {
      this.thresholds.set(s.thresholds);
      this.thresholdRows = s.thresholds.points.map((p) => ({ ...p }));
    }
    this.hpoRows = (s.hpo?.trials || []).map((t) => ({
      trial: t['trial'],
      params: JSON.stringify(t['params'] ?? {}),
      roc_auc: t['roc_auc'],
      pr_auc: t['pr_auc'],
      brier: t['brier'],
    }));
  }

  private flattenAudit(obj: Record<string, unknown>, prefix = ''): Record<string, unknown>[] {
    const out: Record<string, unknown>[] = [];
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        out.push(...this.flattenAudit(v as Record<string, unknown>, key));
      } else {
        out.push({ key, value: Array.isArray(v) ? JSON.stringify(v) : String(v) });
      }
    }
    return out;
  }

  private destroyCharts(): void {
    for (const c of this.charts) {
      try {
        c.destroy();
      } catch {
        /* ignore */
      }
    }
    this.charts = [];
  }

  private canvas(ref: ElementRef<HTMLCanvasElement> | undefined, name: string): HTMLCanvasElement | undefined {
    if (ref?.nativeElement) return ref.nativeElement;
    return (
      this.host.nativeElement.querySelector(`canvas[data-chart="${name}"]`) as HTMLCanvasElement | null
    ) ?? undefined;
  }

  private bar(el: HTMLCanvasElement | undefined, cfg: ChartConfiguration): void {
    if (!el) return;
    const existing = Chart.getChart(el);
    if (existing) existing.destroy();
    cfg.options = {
      ...(cfg.options || {}),
      animation: this.prefs.chart_animation ? undefined : false,
      responsive: true,
      maintainAspectRatio: false,
    };
    this.charts.push(new Chart(el, cfg));
  }

  private renderCharts(s: ReportsSummary | null): void {
    this.destroyCharts();
    this.chartsReady.set(false);
    if (!this.showCharts || !s) return;

    const rows = s.model_comparison?.comparison || [];
    if (rows.length) {
      this.bar(this.canvas(this.compareChartRef, 'compare'), {
        type: 'bar',
        data: {
          labels: rows.map((x) => x.model),
          datasets: [
            { label: 'ROC-AUC', data: rows.map((x) => x.roc_auc ?? 0), backgroundColor: '#2a6b5a' },
            { label: 'PR-AUC', data: rows.map((x) => x.pr_auc ?? 0), backgroundColor: '#c4a35a' },
            { label: 'Brier', data: rows.map((x) => x.brier ?? 0), backgroundColor: '#6b8f9e' },
            { label: 'ECE', data: rows.map((x) => x.ece ?? 0), backgroundColor: '#3d5a80' },
          ],
        },
        options: { plugins: { title: { display: true, text: 'Model comparison' } }, scales: { y: { beginAtZero: true } } },
      });
    }

    const metrics = s.metrics || {};
    const keys = Object.keys(metrics).filter((k) => typeof metrics[k] === 'number');
    if (keys.length) {
      this.bar(this.canvas(this.metricChartRef, 'metric'), {
        type: 'radar',
        data: {
          labels: keys,
          datasets: [
            {
              label: 'Hold-out',
              data: keys.map((k) => Number(metrics[k])),
              borderColor: '#2a6b5a',
              backgroundColor: 'rgba(42,107,90,0.2)',
            },
          ],
        },
        options: { plugins: { title: { display: true, text: 'Metrics radar' } } },
      });
    }

    const top = [...this.importanceRows].sort((a, b) => Number(b['abs']) - Number(a['abs'])).slice(0, this.prefs.top_n_features);
    if (top.length) {
      this.bar(this.canvas(this.importanceChartRef, 'importance'), {
        type: 'bar',
        data: {
          labels: top.map((r) => String(r['feature'])),
          datasets: [{ label: '|importance|', data: top.map((r) => Number(r['abs'])), backgroundColor: '#2a6b5a' }],
        },
        options: {
          indexAxis: 'y',
          plugins: { title: { display: true, text: 'Feature importance' } },
          scales: { x: { beginAtZero: true } },
        },
      });
    }
    this.chartsReady.set(this.charts.length > 0);
  }
}
