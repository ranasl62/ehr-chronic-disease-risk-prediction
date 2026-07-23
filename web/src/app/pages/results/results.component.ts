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

interface HpoParam {
  name: string;
  value: string;
}

interface HpoTrialRow {
  trial: number | null;
  params: HpoParam[];
  roc_auc: number | null;
  pr_auc: number | null;
  brier: number | null;
  ece: number | null;
  f1: number | null;
  isBest: boolean;
}

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
  @ViewChild('thresholdChart') thresholdChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('fairnessChart') fairnessChartRef?: ElementRef<HTMLCanvasElement>;

  summary = signal<ReportsSummary | null>(null);
  runs = signal<RunSummary[]>([]);
  selectedRun = signal<RunDetail | null>(null);
  fairness = signal<FairnessReport | null>(null);
  thresholds = signal<ThresholdReport | null>(null);
  compareRunIds: string[] = [];
  error = signal<string | null>(null);
  job = signal<JobInfo | null>(null);
  busy = signal(false);
  /** Empty = show all PNGs, calibration/SHAP first when present. */
  figFilter = '';
  metricFilter = '';
  /** Client-side filter for Experiment runs table. */
  runFilter = '';
  runPage = 1;
  runPageSize = 10;
  readonly runPageSizeOptions = [10, 15, 25] as const;
  showCharts = true;
  chartsReady = signal(false);

  compareRows: Record<string, unknown>[] = [];
  metricRows: Record<string, unknown>[] = [];
  fileRows: Record<string, unknown>[] = [];
  leakageRows: Record<string, unknown>[] = [];
  importanceRows: Record<string, unknown>[] = [];
  fairnessRows: Record<string, unknown>[] = [];
  thresholdRows: Record<string, unknown>[] = [];
  hpoRows: HpoTrialRow[] = [];
  hpoBest: HpoTrialRow | null = null;
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
        if (this.figFilter === 'calibration' && !this.hasCalibrationFigure()) this.figFilter = '';
        if (this.figFilter === 'shap' && !this.hasShapFigure()) this.figFilter = '';
        this.scheduleRedraw();
      },
      error: (e) => this.error.set(this.fmtErr(e)),
    });
    this.api.runs(40).subscribe({
      next: (r) => {
        this.runs.set(r.runs || []);
        this.runPage = 1;
        this.buildRunCompare();
      },
      error: () => undefined,
    });
    this.api.fairnessReport().subscribe({
      next: (f) => {
        this.fairness.set(f);
        this.buildFairnessRows(f);
        this.scheduleRedraw();
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
    const pngs = (s.files || []).filter(
      (f) => f.name.endsWith('.png') && (!q || f.name.toLowerCase().includes(q))
    );
    return [...pngs].sort((a, b) => this.figPriority(a.name) - this.figPriority(b.name));
  }

  /** True when reports include at least one PNG figure. */
  hasReportPngs(): boolean {
    return (this.summary()?.files || []).some((f) => f.name.endsWith('.png'));
  }

  /** Hide calibration UI when calibration_holdout.png (or any calibration*.png) is absent. */
  hasCalibrationFigure(): boolean {
    return (this.summary()?.files || []).some((f) => {
      const n = f.name.toLowerCase();
      return n.endsWith('.png') && n.includes('calibration');
    });
  }

  hasShapFigure(): boolean {
    return (this.summary()?.files || []).some((f) => {
      const n = f.name.toLowerCase();
      return n.endsWith('.png') && n.includes('shap');
    });
  }

  setFigFilter(q: string): void {
    this.figFilter = q;
  }

  methodsMdUrl(): string {
    return this.api.methodsMdUrl(this.selectedRun()?.run_id);
  }

  zipUrl(): string {
    return this.api.resultsZipUrl(this.selectedRun()?.run_id);
  }

  trustChecklist(sr: RunDetail | RunSummary | null | undefined): { label: string; ok: boolean | null }[] {
    if (!sr) return [];
    const flags = (sr.trust || (sr as RunDetail).trust_pack?.['flags'] || {}) as Record<string, unknown>;
    const bool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
    return [
      { label: 'Model', ok: bool(flags['has_model'] ?? sr.has_model) },
      { label: 'Evaluation', ok: bool(flags['has_evaluation'] ?? sr.has_evaluation) },
      { label: 'Leakage audit', ok: bool(flags['has_leakage'] ?? sr.has_leakage) },
      { label: 'Leakage passed', ok: bool(flags['leakage_passed'] ?? sr.leakage_passed) },
      { label: 'SHAP', ok: bool(flags['has_shap'] ?? sr.has_shap) },
      { label: 'Calibration', ok: bool(flags['has_calibration'] ?? sr.has_calibration) },
      { label: 'Trust complete', ok: bool(flags['trust_complete'] ?? sr.trust_complete) },
    ];
  }

  /** Prefer calibration and SHAP figures when listing without a text filter. */
  private figPriority(name: string): number {
    const n = name.toLowerCase();
    if (n.includes('calibration')) return 0;
    if (n.includes('shap')) return 1;
    return 2;
  }

  filteredMetrics(): Record<string, unknown>[] {
    const q = this.metricFilter.trim().toLowerCase();
    if (!q) return this.metricRows;
    return this.metricRows.filter((r) => String(r['metric'] ?? '').toLowerCase().includes(q));
  }

  /** Client-side filter over visible run fields (id, model, status, path, meta, metrics). */
  filteredRuns(): RunSummary[] {
    const q = this.runFilter.trim().toLowerCase();
    const all = this.runs();
    if (!q) return all;
    return all.filter((r) => this.runSearchText(r).includes(q));
  }

  pagedRuns(): RunSummary[] {
    const rows = this.filteredRuns();
    const start = (this.runPage - 1) * this.runPageSize;
    return rows.slice(start, start + this.runPageSize);
  }

  runTotalPages(): number {
    return Math.max(1, Math.ceil(this.filteredRuns().length / this.runPageSize) || 1);
  }

  onRunFilterChange(q: string): void {
    this.runFilter = q;
    this.runPage = 1;
  }

  goRunPage(p: number): void {
    const max = this.runTotalPages();
    this.runPage = Math.min(Math.max(1, p), max);
  }

  onRunPageSize(n: number): void {
    this.runPageSize = n;
    this.runPage = 1;
  }

  runKind(r: RunSummary): string {
    const kind = r.meta?.['kind'];
    return typeof kind === 'string' && kind ? kind : '—';
  }

  runStatusLabels(r: RunSummary): string[] {
    const tags: string[] = [];
    if (r.has_model) tags.push('model');
    if (r.has_evaluation) tags.push('eval');
    if (r.has_manifest) tags.push('manifest');
    return tags;
  }

  private runSearchText(r: RunSummary): string {
    const parts: string[] = [
      r.run_id,
      r.path,
      r.model_kind || '',
      this.runKind(r),
      ...this.runStatusLabels(r),
    ];
    if (r.metrics) {
      for (const [k, v] of Object.entries(r.metrics)) {
        parts.push(k, v == null ? '' : String(v));
      }
    }
    if (r.meta) {
      for (const [k, v] of Object.entries(r.meta)) {
        if (v == null) continue;
        parts.push(k, typeof v === 'object' ? JSON.stringify(v) : String(v));
      }
    }
    return parts.join(' ').toLowerCase();
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
    const runId = this.selectedRun()?.run_id;
    this.startJob(() => this.api.shap(runId ? { run_id: runId } : {}));
  }

  runLeakageForSelected(): void {
    const runId = this.selectedRun()?.run_id;
    this.startJob(() =>
      this.api.leakageAudit({ use_artifact: true, ...(runId ? { run_id: runId } : {}) })
    );
  }

  extValPath = 'data/demo/ehr_data.csv';
  extValFormat = 'longitudinal';
  extValRows: Record<string, unknown>[] = [];
  extValCols: DataTableColumn[] = [
    { key: 'metric', label: 'Metric' },
    { key: 'value', label: 'Value' },
  ];

  runExternalValidate(): void {
    const runId = this.selectedRun()?.run_id;
    this.startJob(() =>
      this.api.externalValidate({
        data_path: this.extValPath,
        data_format: this.extValFormat,
        run_id: runId || null,
      })
    );
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
        this.scheduleRedraw();
        this.reload();
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  /** True when threshold points or decision-curve points exist for charting. */
  hasThresholdChartData(): boolean {
    return this.thresholdChartPoints().length > 0;
  }

  hasFairnessChartData(): boolean {
    return this.fairnessRows.some(
      (r) =>
        typeof r['accuracy'] === 'number' ||
        typeof r['tpr'] === 'number' ||
        typeof r['fpr'] === 'number'
    );
  }

  /**
   * Operating-point or decision-curve rows already present in reports JSON.
   * Does not invent net-benefit values.
   */
  private thresholdChartPoints(): {
    threshold: number;
    precision?: number;
    recall?: number;
    f1?: number;
    net_benefit?: number;
  }[] {
    const thr = this.thresholds() || this.summary()?.thresholds || null;
    const points = thr?.points || [];
    if (points.length) {
      return points.map((p) => ({
        threshold: Number(p.threshold),
        precision: typeof p.precision === 'number' ? p.precision : undefined,
        recall: typeof p.recall === 'number' ? p.recall : undefined,
        f1: typeof p.f1 === 'number' ? p.f1 : undefined,
        net_benefit:
          typeof (p as { net_benefit?: number }).net_benefit === 'number'
            ? (p as { net_benefit: number }).net_benefit
            : undefined,
      }));
    }
    const dc = thr?.decision_curve;
    const dcp = dc?.points;
    if (Array.isArray(dcp) && dcp.length) {
      const out: {
        threshold: number;
        precision?: number;
        recall?: number;
        f1?: number;
        net_benefit?: number;
      }[] = [];
      for (const row of dcp) {
        const r = row as Record<string, unknown>;
        const threshold = Number(r['threshold']);
        if (!Number.isFinite(threshold)) continue;
        out.push({
          threshold,
          net_benefit: typeof r['net_benefit'] === 'number' ? r['net_benefit'] : undefined,
        });
      }
      return out;
    }
    return [];
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

  onPageSize(n: number): void {
    this.ui.patch({ table_page_size: n });
  }

  formatMetric(value: number | null | undefined): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : 'n/a';
  }

  formatParams(params: unknown): HpoParam[] {
    if (!params || typeof params !== 'object' || Array.isArray(params)) return [];
    return Object.entries(params as Record<string, unknown>).map(([name, value]) => ({
      name,
      value: this.formatParamValue(value),
    }));
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
    const best = s.hpo?.best;
    const bestTrial = best ? this.hpoTrial(best, null, true) : null;
    this.hpoBest = bestTrial;
    this.hpoRows = (s.hpo?.trials || []).map((trial, index) =>
      this.hpoTrial(trial, index, bestTrial?.trial === this.hpoTrialNumber(trial))
    );
  }

  private hpoTrial(raw: Record<string, unknown>, fallbackTrial: number | null, isBest: boolean): HpoTrialRow {
    return {
      trial: this.hpoTrialNumber(raw) ?? fallbackTrial,
      params: this.formatParams(raw['params']),
      roc_auc: this.hpoMetric(raw['roc_auc']),
      pr_auc: this.hpoMetric(raw['pr_auc']),
      brier: this.hpoMetric(raw['brier']),
      ece: this.hpoMetric(raw['ece']),
      f1: this.hpoMetric(raw['f1']),
      isBest,
    };
  }

  private hpoTrialNumber(raw: Record<string, unknown> | undefined): number | null {
    const trial = raw?.['trial'];
    return typeof trial === 'number' && Number.isFinite(trial) ? trial : null;
  }

  private hpoMetric(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  private formatParamValue(value: unknown): string {
    if (typeof value === 'number' && Number.isFinite(value)) return String(Number(value.toFixed(4)));
    if (typeof value === 'string' || typeof value === 'boolean') return String(value);
    if (value == null) return 'n/a';
    return JSON.stringify(value);
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

    const thrPts = this.thresholdChartPoints();
    if (thrPts.length) {
      const labels = thrPts.map((p) => String(p.threshold));
      const datasets: ChartConfiguration['data']['datasets'] = [];
      if (thrPts.some((p) => typeof p.precision === 'number')) {
        datasets.push({
          label: 'Precision',
          data: thrPts.map((p) => p.precision ?? null),
          borderColor: '#2a6b5a',
          backgroundColor: 'rgba(42,107,90,0.15)',
          tension: 0.2,
        });
      }
      if (thrPts.some((p) => typeof p.recall === 'number')) {
        datasets.push({
          label: 'Recall',
          data: thrPts.map((p) => p.recall ?? null),
          borderColor: '#c4a35a',
          backgroundColor: 'rgba(196,163,90,0.15)',
          tension: 0.2,
        });
      }
      if (thrPts.some((p) => typeof p.f1 === 'number')) {
        datasets.push({
          label: 'F1',
          data: thrPts.map((p) => p.f1 ?? null),
          borderColor: '#6b8f9e',
          backgroundColor: 'rgba(107,143,158,0.15)',
          tension: 0.2,
        });
      }
      if (thrPts.some((p) => typeof p.net_benefit === 'number')) {
        datasets.push({
          label: 'Net benefit',
          data: thrPts.map((p) => p.net_benefit ?? null),
          borderColor: '#3d5a80',
          backgroundColor: 'rgba(61,90,128,0.15)',
          tension: 0.2,
        });
      }
      if (datasets.length) {
        this.bar(this.canvas(this.thresholdChartRef, 'threshold'), {
          type: 'line',
          data: { labels, datasets },
          options: {
            plugins: {
              title: {
                display: true,
                text: thrPts.some((p) => typeof p.net_benefit === 'number' && p.precision == null)
                  ? 'Decision curve (from report)'
                  : 'Threshold operating points',
              },
            },
            scales: { y: { beginAtZero: true } },
          },
        });
      }
    }

    if (this.hasFairnessChartData()) {
      const labels = this.fairnessRows.map((r) => String(r['group'] ?? '—'));
      const datasets: ChartConfiguration['data']['datasets'] = [];
      if (this.fairnessRows.some((r) => typeof r['accuracy'] === 'number')) {
        datasets.push({
          label: 'Accuracy',
          data: this.fairnessRows.map((r) => (typeof r['accuracy'] === 'number' ? r['accuracy'] : null)),
          backgroundColor: '#2a6b5a',
        });
      }
      if (this.fairnessRows.some((r) => typeof r['tpr'] === 'number')) {
        datasets.push({
          label: 'TPR',
          data: this.fairnessRows.map((r) => (typeof r['tpr'] === 'number' ? r['tpr'] : null)),
          backgroundColor: '#c4a35a',
        });
      }
      if (this.fairnessRows.some((r) => typeof r['fpr'] === 'number')) {
        datasets.push({
          label: 'FPR',
          data: this.fairnessRows.map((r) => (typeof r['fpr'] === 'number' ? r['fpr'] : null)),
          backgroundColor: '#6b8f9e',
        });
      }
      if (datasets.length) {
        this.bar(this.canvas(this.fairnessChartRef, 'fairness'), {
          type: 'bar',
          data: { labels, datasets },
          options: {
            plugins: { title: { display: true, text: 'Fairness by group' } },
            scales: { y: { beginAtZero: true, max: 1 } },
          },
        });
      }
    }
    this.chartsReady.set(this.charts.length > 0);
  }
}
