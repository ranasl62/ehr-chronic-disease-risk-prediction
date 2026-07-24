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
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import {
  ApiService,
  AnalysisPack,
  DatasetInfo,
  DatasetProfile,
  ReportsSummary,
  WorkspaceStatus,
} from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';
import { UiPrefs, UiPrefsService } from '../../core/ui-prefs.service';
import { DataTableColumn, DataTableComponent } from '../../shared/data-table.component';

Chart.register(...registerables);

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule, FormsModule, DataTableComponent],
  templateUrl: './analytics.component.html',
  styleUrl: './analytics.component.css',
})
export class AnalyticsComponent implements OnInit, AfterViewInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly state = inject(WorkspaceState);
  readonly ui = inject(UiPrefsService);
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly host = inject(ElementRef<HTMLElement>);
  private redrawTimer: ReturnType<typeof setTimeout> | null = null;

  @ViewChild('labelChart') labelChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('ageChart') ageChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('sexChart') sexChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('prevalenceChart') prevalenceChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('missChart') missChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('numericChart') numericChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('metricChart') metricChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('importanceChart') importanceChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('compareChart') compareChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('rocChart') rocChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('prChart') prChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('calChart') calChartRef?: ElementRef<HTMLCanvasElement>;

  datasets = signal<DatasetInfo[]>([]);
  path = '';
  qualityNote = signal<string | null>(null);
  filterAgeBand = '';
  filterLabel = '';
  filterPatientId = '';
  filterOptions = signal<{ age_bands: string[]; labels: string[]; sexes: string[] }>({
    age_bands: [],
    labels: [],
    sexes: [],
  });
  cohortRows: Record<string, unknown>[] = [];
  cohortCols: DataTableColumn[] = [
    { key: 'patient_id', label: 'Patient' },
    { key: 'age', label: 'Age', numeric: true, format: 'number', digits: '1.0-0' },
    { key: 'age_band', label: 'Age band' },
    { key: 'sex', label: 'Sex' },
    { key: 'label', label: 'Label' },
    { key: 'glucose_mean', label: 'Glucose μ', numeric: true, format: 'number' },
  ];
  profile = signal<DatasetProfile | null>(null);
  analysisPack = signal<AnalysisPack | null>(null);
  status = signal<WorkspaceStatus | null>(null);
  reports = signal<ReportsSummary | null>(null);
  error = signal<string | null>(null);
  chartsReady = signal(false);
  /** Empty-state copy when ROC/PR/calibration points are unavailable. */
  curveEmpty = signal<{ roc: string | null; pr: string | null; cal: string | null }>({
    roc: null,
    pr: null,
    cal: null,
  });
  hasRocCurve = signal(false);
  hasPrCurve = signal(false);
  hasCalCurve = signal(false);

  numericRows: Record<string, unknown>[] = [];
  missingRows: Record<string, unknown>[] = [];
  labelRows: Record<string, unknown>[] = [];
  ageRows: Record<string, unknown>[] = [];
  sexRows: Record<string, unknown>[] = [];
  importanceRows: Record<string, unknown>[] = [];
  compareRows: Record<string, unknown>[] = [];
  columnRows: Record<string, unknown>[] = [];
  metricRows: Record<string, unknown>[] = [];

  numericCols: DataTableColumn[] = [
    { key: 'feature', label: 'Feature' },
    { key: 'mean', label: 'Mean', numeric: true, format: 'number' },
    { key: 'std', label: 'Std', numeric: true, format: 'number' },
  ];
  missingCols: DataTableColumn[] = [
    { key: 'column', label: 'Column' },
    { key: 'missing_pct', label: 'Missing %', numeric: true, format: 'percent', digits: '1.1-1' },
  ];
  labelCols: DataTableColumn[] = [
    { key: 'label', label: 'Label' },
    { key: 'count', label: 'Count', numeric: true, format: 'number', digits: '1.0-0' },
  ];
  ageCols: DataTableColumn[] = [
    { key: 'band', label: 'Age band' },
    { key: 'count', label: 'Count', numeric: true, format: 'number', digits: '1.0-0' },
  ];
  sexCols: DataTableColumn[] = [
    { key: 'sex', label: 'Sex' },
    { key: 'count', label: 'Count', numeric: true, format: 'number', digits: '1.0-0' },
  ];
  importanceCols: DataTableColumn[] = [
    { key: 'feature', label: 'Feature' },
    { key: 'importance', label: 'Importance', numeric: true, format: 'number' },
    { key: 'abs', label: '|Importance|', numeric: true, format: 'number' },
  ];
  compareCols: DataTableColumn[] = [
    { key: 'model', label: 'Model' },
    { key: 'roc_auc', label: 'ROC-AUC', numeric: true, format: 'number' },
    { key: 'pr_auc', label: 'PR-AUC', numeric: true, format: 'number' },
    { key: 'brier', label: 'Brier', numeric: true, format: 'number' },
    { key: 'ece', label: 'ECE', numeric: true, format: 'number' },
    { key: 'selected', label: 'Selected' },
  ];
  columnCols: DataTableColumn[] = [
    { key: 'name', label: 'Column' },
    { key: 'index', label: '#', numeric: true, format: 'number', digits: '1.0-0' },
  ];
  metricCols: DataTableColumn[] = [
    { key: 'metric', label: 'Metric' },
    { key: 'value', label: 'Value', numeric: true, format: 'number' },
  ];

  private charts: Chart[] = [];
  private lastProfile: DatasetProfile | null = null;
  private lastStatus: WorkspaceStatus | null = null;
  private lastReports: ReportsSummary | null = null;

  ngOnInit(): void {
    this.api.datasets().subscribe({
      next: (r) => {
        this.datasets.set(r.datasets.filter((d) => d.exists));
        const sel = this.state.selectedDataset();
        this.path =
          sel?.path ||
          r.datasets.find((d) => d.id === 'paper_synthetic')?.path ||
          r.datasets.find((d) => d.exists)?.path ||
          'data/raw/ehr_data.csv';
        this.load();
      },
      error: (e) => this.error.set(e?.error?.detail || e.message),
    });
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

  showCharts(): boolean {
    const v = this.prefs.analytics_view;
    return v === 'charts' || v === 'split';
  }

  showTables(): boolean {
    const v = this.prefs.analytics_view;
    return v === 'tables' || v === 'split';
  }

  setView(v: 'charts' | 'tables' | 'split'): void {
    this.ui.patch({ analytics_view: v });
    this.scheduleRedraw();
  }

  onTopNChange(n: number): void {
    this.ui.patch({ top_n_features: Number(n) || 15 });
    this.scheduleRedraw();
  }

  onPageSize(n: number): void {
    this.ui.patch({ table_page_size: n });
  }

  toggleChart(key: keyof UiPrefs, value: boolean): void {
    this.ui.patch({ [key]: value } as Partial<UiPrefs>);
    this.scheduleRedraw();
  }

  analysisPackDownloadUrl(): string {
    return this.api.analysisPackUrl(this.path || 'data/demo/ehr_data.csv');
  }

  formatPct(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(Number(v))) return '—';
    return `${(Number(v) * 100).toFixed(1)}%`;
  }

  exportChartPng(chartName: string, fileBase: string): void {
    const el = this.canvas(undefined, chartName);
    if (!el) return;
    const chart = Chart.getChart(el);
    const href = chart ? chart.toBase64Image('image/png', 1) : el.toDataURL('image/png');
    this.downloadDataUrl(href, `${fileBase}.png`);
  }

  exportAllChartsPng(): void {
    const names: Array<[string, string]> = [
      ['label', 'label_distribution'],
      ['age', 'age_bands'],
      ['sex', 'sex_distribution'],
      ['prevalence', 'label_prevalence'],
      ['miss', 'missingness'],
      ['numeric', 'numeric_means'],
      ['metric', 'holdout_metrics'],
      ['importance', 'feature_importance'],
      ['compare', 'model_comparison'],
      ['roc', 'roc_curve'],
      ['pr', 'pr_curve'],
      ['cal', 'calibration_curve'],
    ];
    for (const [name, file] of names) {
      const el = this.canvas(undefined, name);
      if (!el || !Chart.getChart(el)) continue;
      this.exportChartPng(name, file);
    }
  }

  printReport(): void {
    if (typeof window !== 'undefined') window.print();
  }

  private downloadDataUrl(href: string, filename: string): void {
    const a = document.createElement('a');
    a.href = href;
    a.download = filename;
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  load(): void {
    if (!this.path?.trim()) {
      this.error.set('Select a dataset to load analytics.');
      return;
    }
    this.error.set(null);
    this.destroyCharts();
    this.chartsReady.set(false);

    this.api
      .datasetProfile(this.path, {
        age_band: this.filterAgeBand || undefined,
        label: this.filterLabel || undefined,
        patient_id: this.filterPatientId || undefined,
      })
      .subscribe({
      next: (p) => {
        this.profile.set(p);
        this.lastProfile = p;
        this.buildProfileTables(p);
        this.cohortRows = (p.cohort_rows || []) as Record<string, unknown>[];
        this.filterOptions.set({
          age_bands: p.filter_options?.age_bands || [],
          labels: p.filter_options?.labels || [],
          sexes: p.filter_options?.sexes || [],
        });
        this.scheduleRedraw();
      },
      error: (e) => this.error.set(e?.error?.detail || e.message),
    });
    this.api.analysisPack(this.path).subscribe({
      next: (pack) => this.analysisPack.set(pack),
      error: () => this.analysisPack.set(null),
    });
    this.api.workspaceStatus().subscribe({
      next: (s) => {
        this.status.set(s);
        this.lastStatus = s;
        this.buildMetricTable(s);
        this.scheduleRedraw();
      },
    });
    this.api.reportsSummary().subscribe({
      next: (r) => {
        this.reports.set(r);
        this.lastReports = r;
        this.qualityNote.set(r.quality_note || null);
        this.updateCurveAvailability(r);
        this.buildReportTables(r);
        this.scheduleRedraw();
      },
      error: () => {
        this.updateCurveAvailability(null);
      },
    });
  }

  private updateCurveAvailability(r: ReportsSummary | null): void {
    const retrain =
      'No ROC/PR/calibration points yet. Retrain on paper_synthetic (or a larger two-class cohort) so evaluation_report.json includes curves.';
    const single =
      'Hold-out is single-class or too small — ROC/PR are unavailable. Retrain on paper_synthetic or a stratified cohort with both outcomes.';
    const mismatch =
      'Curve arrays missing or mismatched in the evaluation report. Retrain to regenerate curves (prefer paper_synthetic).';
    const calHint =
      'Calibration bins unavailable (need enough scored samples). Retrain on a larger cohort such as paper_synthetic.';
    if (!r?.curves) {
      this.hasRocCurve.set(false);
      this.hasPrCurve.set(false);
      this.hasCalCurve.set(false);
      this.curveEmpty.set({ roc: retrain, pr: retrain, cal: retrain });
      return;
    }
    const notes = (r.curves.notes || []).join(' ');
    const qn = (r.quality_note || '').toLowerCase();
    let rocPrHint = retrain;
    if (notes.includes('single_class') || qn.includes('single class')) {
      rocPrHint = single;
    } else if (notes.includes('mismatched') || notes.includes('empty_or_mismatched')) {
      rocPrHint = mismatch;
    } else if (notes.includes('unavailable') || notes.includes('empty')) {
      rocPrHint = single;
    }
    const rocOk = (r.curves.roc?.fpr?.length || 0) > 0 && (r.curves.roc?.tpr?.length || 0) > 0;
    const prOk =
      (r.curves.pr?.precision?.length || 0) > 0 && (r.curves.pr?.recall?.length || 0) > 0;
    const calOk = (r.curves.calibration?.bin_mid?.length || 0) > 0;
    this.hasRocCurve.set(rocOk);
    this.hasPrCurve.set(prOk);
    this.hasCalCurve.set(calOk);
    this.curveEmpty.set({
      roc: rocOk ? null : rocPrHint,
      pr: prOk ? null : rocPrHint,
      cal: calOk ? null : calHint,
    });
  }

  /**
   * Paint canvases after Angular updates the DOM.
   * Do not use afterNextRender here: detectChanges() often finishes the current
   * render first, so afterNextRender would wait forever for a later cycle.
   */
  private scheduleRedraw(): void {
    this.cdr.detectChanges();
    if (this.redrawTimer != null) clearTimeout(this.redrawTimer);
    this.redrawTimer = setTimeout(() => {
      this.redrawTimer = null;
      requestAnimationFrame(() => this.redraw());
    }, 0);
  }

  private redraw(): void {
    if (!this.showCharts() || !this.lastProfile) {
      this.destroyCharts();
      this.chartsReady.set(false);
      return;
    }
    this.destroyCharts();
    this.renderCharts(this.lastProfile);
    if (this.lastStatus) this.renderMetrics(this.lastStatus);
    if (this.lastReports) {
      this.renderImportance(this.lastReports);
      this.renderCompare(this.lastReports);
      this.renderEvalCurves(this.lastReports);
    }
    this.chartsReady.set(this.charts.length > 0);
  }

  private canvas(ref: ElementRef<HTMLCanvasElement> | undefined, name: string): HTMLCanvasElement | undefined {
    const fromRef = ref?.nativeElement;
    if (fromRef) return fromRef;
    return (
      this.host.nativeElement.querySelector(`canvas[data-chart="${name}"]`) as HTMLCanvasElement | null
    ) ?? undefined;
  }

  private buildProfileTables(p: DatasetProfile): void {
    const preview = p.numeric_preview || {};
    this.numericRows = Object.keys(preview).map((k) => ({
      feature: k,
      mean: preview[k]?.mean,
      std: preview[k]?.std,
    }));
    const miss = p.missing_pct || {};
    this.missingRows = Object.entries(miss).map(([column, missing_pct]) => ({
      column,
      missing_pct,
    }));
    const labels = p.label_counts || {};
    this.labelRows = Object.entries(labels).map(([label, count]) => ({ label, count }));
    const ages = p.age_band_counts || {};
    this.ageRows = Object.entries(ages).map(([band, count]) => ({ band, count }));
    const sexCounts: Record<string, number> = {};
    for (const row of p.cohort_rows || []) {
      const sex = row.sex;
      const key = sex == null || sex === '' ? 'unknown' : String(sex);
      sexCounts[key] = (sexCounts[key] || 0) + 1;
    }
    this.sexRows = Object.entries(sexCounts).map(([sex, count]) => ({ sex, count }));
    this.columnRows = (p.columns || []).map((name, index) => ({ name, index: index + 1 }));
  }

  private buildMetricTable(s: WorkspaceStatus): void {
    const m = s.metrics || {};
    this.metricRows = Object.entries(m)
      .filter(([, v]) => typeof v === 'number' && Number.isFinite(v as number))
      .map(([metric, value]) => ({ metric, value }));
  }

  private buildReportTables(r: ReportsSummary): void {
    const raw = r.feature_importance;
    const nested = raw && typeof raw === 'object' && 'importance' in raw ? raw.importance : undefined;
    const map: Record<string, number> =
      nested && typeof nested === 'object' ? nested : ((raw as Record<string, number>) || {});
    this.importanceRows = Object.entries(map)
      .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
      .map(([feature, importance]) => ({
        feature,
        importance,
        abs: Math.abs(importance),
      }));
    this.compareRows = (r.model_comparison?.comparison || []).map((row) => ({
      model: row.model,
      roc_auc: row.roc_auc,
      pr_auc: row.pr_auc,
      brier: row.brier,
      ece: row.ece,
      selected: row.selected ? 'yes' : '',
    }));
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

  private bar(canvas: HTMLCanvasElement | undefined, cfg: ChartConfiguration): void {
    if (!canvas) return;
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    const anim = this.prefs.chart_animation;
    cfg.options = {
      ...(cfg.options || {}),
      animation: anim ? undefined : false,
      responsive: true,
      maintainAspectRatio: false,
    };
    this.charts.push(new Chart(canvas, cfg));
  }

  private palette(): string[] {
    const t = this.prefs.theme;
    if (t === 'slate') return ['#3d5a80', '#98c1d9', '#ee6c4d', '#293241', '#6b8f9e'];
    if (t === 'sand') return ['#8c6a3d', '#c4a35a', '#6b8f9e', '#a33', '#2a6b5a'];
    return ['#2a6b5a', '#c4a35a', '#6b8f9e', '#a33', '#3d5a80'];
  }

  private renderCharts(p: DatasetProfile): void {
    const colors = this.palette();
    const topN = this.prefs.top_n_features;

    if (this.prefs.show_label_chart) {
      const labels = p.label_counts || {};
      const keys = Object.keys(labels);
      if (keys.length) {
        this.bar(this.canvas(this.labelChartRef, 'label'), {
          type: this.prefs.label_chart_type,
          data: {
            labels: keys,
            datasets: [{ data: keys.map((k) => labels[k]), backgroundColor: colors }],
          },
          options: { plugins: { title: { display: true, text: 'Label distribution' } } },
        });
      }
    }

    if (this.prefs.show_age_chart) {
      const ages = p.age_band_counts || {};
      const keys = Object.keys(ages);
      if (keys.length) {
        this.bar(this.canvas(this.ageChartRef, 'age'), {
          type: 'bar',
          data: {
            labels: keys,
            datasets: [{ label: 'Patients/rows', data: keys.map((k) => ages[k]), backgroundColor: colors[0] }],
          },
          options: {
            plugins: { title: { display: true, text: 'Age bands' } },
            scales: { y: { beginAtZero: true } },
          },
        });
      }
    }

    if (this.prefs.show_sex_chart && this.sexRows.length) {
      this.bar(this.canvas(this.sexChartRef, 'sex'), {
        type: 'doughnut',
        data: {
          labels: this.sexRows.map((r) => String(r['sex'])),
          datasets: [
            {
              data: this.sexRows.map((r) => Number(r['count'])),
              backgroundColor: colors,
            },
          ],
        },
        options: { plugins: { title: { display: true, text: 'Sex distribution (cohort)' } } },
      });
    }

    if (this.prefs.show_prevalence_chart) {
      const labels = p.label_counts || {};
      const keys = Object.keys(labels);
      const total = keys.reduce((s, k) => s + Number(labels[k] || 0), 0);
      if (keys.length && total > 0) {
        this.bar(this.canvas(this.prevalenceChartRef, 'prevalence'), {
          type: 'bar',
          data: {
            labels: keys,
            datasets: [
              {
                label: 'Share of rows',
                data: keys.map((k) => Number(labels[k]) / total),
                backgroundColor: colors[1],
              },
            ],
          },
          options: {
            plugins: { title: { display: true, text: 'Label prevalence (share)' } },
            scales: { y: { beginAtZero: true, max: 1 } },
          },
        });
      }
    }

    if (this.prefs.show_missing_chart) {
      const miss = p.missing_pct || {};
      const entries = Object.entries(miss)
        .sort((a, b) => b[1] - a[1])
        .slice(0, topN);
      if (entries.length) {
        this.bar(this.canvas(this.missChartRef, 'miss'), {
          type: 'bar',
          data: {
            labels: entries.map(([k]) => k),
            datasets: [{ label: 'Missing %', data: entries.map(([, v]) => v), backgroundColor: colors[2] }],
          },
          options: {
            indexAxis: 'y',
            plugins: { title: { display: true, text: `Missingness (top ${topN})` } },
            scales: { x: { beginAtZero: true, max: 100 } },
          },
        });
      }
    }

    if (this.prefs.show_numeric_chart) {
      const preview = p.numeric_preview || {};
      const nKeys = Object.keys(preview)
        .filter((k) => preview[k]?.mean != null)
        .slice(0, topN);
      if (nKeys.length) {
        this.bar(this.canvas(this.numericChartRef, 'numeric'), {
          type: this.prefs.numeric_chart_type,
          data: {
            labels: nKeys,
            datasets: [
              {
                label: 'Mean',
                data: nKeys.map((k) => Number(preview[k].mean)),
                backgroundColor: colors[1],
                borderColor: colors[0],
                fill: false,
                tension: 0.25,
              },
            ],
          },
          options: {
            plugins: { title: { display: true, text: `Numeric feature means (top ${topN})` } },
            scales: { y: { beginAtZero: true } },
          },
        });
      }
    }
  }

  private renderImportance(_r: ReportsSummary): void {
    if (!this.prefs.show_importance_chart) return;
    const topN = this.prefs.top_n_features;
    const entries = [...this.importanceRows]
      .sort((a, b) => Number(b['abs']) - Number(a['abs']))
      .slice(0, topN);
    if (!entries.length) return;
    this.bar(this.canvas(this.importanceChartRef, 'importance'), {
      type: 'bar',
      data: {
        labels: entries.map((e) => String(e['feature'])),
        datasets: [
          {
            label: '|importance|',
            data: entries.map((e) => Number(e['abs'])),
            backgroundColor: this.palette()[0],
          },
        ],
      },
      options: {
        indexAxis: 'y',
        plugins: { title: { display: true, text: `Top feature importances (${topN})` } },
        scales: { x: { beginAtZero: true } },
      },
    });
  }

  private renderCompare(r: ReportsSummary): void {
    if (!this.prefs.show_compare_chart) return;
    const rows = r.model_comparison?.comparison || [];
    if (!rows.length) return;
    const colors = this.palette();
    this.bar(this.canvas(this.compareChartRef, 'compare'), {
      type: 'bar',
      data: {
        labels: rows.map((x) => x.model),
        datasets: [
          { label: 'ROC-AUC', data: rows.map((x) => x.roc_auc ?? 0), backgroundColor: colors[0] },
          { label: 'PR-AUC', data: rows.map((x) => x.pr_auc ?? 0), backgroundColor: colors[1] },
          { label: 'Brier', data: rows.map((x) => x.brier ?? 0), backgroundColor: colors[2] },
          { label: 'ECE', data: rows.map((x) => x.ece ?? 0), backgroundColor: colors[3] },
        ],
      },
      options: {
        plugins: { title: { display: true, text: 'Model comparison metrics' } },
        scales: { y: { beginAtZero: true } },
      },
    });
  }

  private renderEvalCurves(r: ReportsSummary): void {
    const curves = r.curves;
    if (!curves) return;
    const colors = this.palette();
    const fpr = curves.roc?.fpr || [];
    const tpr = curves.roc?.tpr || [];
    if (fpr.length && tpr.length) {
      this.bar(this.canvas(this.rocChartRef, 'roc'), {
        type: 'line',
        data: {
          labels: fpr.map((x) => Number(x).toFixed(2)),
          datasets: [
            {
              label: 'ROC',
              data: tpr.map((y, i) => ({ x: fpr[i], y })),
              borderColor: colors[0],
              backgroundColor: 'transparent',
              pointRadius: 0,
              tension: 0.15,
            },
            {
              label: 'Chance',
              data: [
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ],
              borderColor: '#999',
              borderDash: [4, 4],
              pointRadius: 0,
            },
          ],
        },
        options: {
          parsing: false,
          plugins: { title: { display: true, text: 'ROC curve (hold-out)' } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'FPR' } },
            y: { min: 0, max: 1, title: { display: true, text: 'TPR' } },
          },
        },
      });
    }
    const prec = curves.pr?.precision || [];
    const rec = curves.pr?.recall || [];
    if (prec.length && rec.length) {
      this.bar(this.canvas(this.prChartRef, 'pr'), {
        type: 'line',
        data: {
          labels: rec.map((x) => Number(x).toFixed(2)),
          datasets: [
            {
              label: 'Precision-Recall',
              data: prec.map((y, i) => ({ x: rec[i], y })),
              borderColor: colors[1],
              backgroundColor: 'transparent',
              pointRadius: 0,
              tension: 0.15,
            },
          ],
        },
        options: {
          parsing: false,
          plugins: { title: { display: true, text: 'Precision–Recall curve' } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Recall' } },
            y: { min: 0, max: 1, title: { display: true, text: 'Precision' } },
          },
        },
      });
    }
    const mid = curves.calibration?.bin_mid || [];
    const frac = curves.calibration?.frac_positive || [];
    const meanP = curves.calibration?.mean_predicted || [];
    if (mid.length && frac.length) {
      this.bar(this.canvas(this.calChartRef, 'cal'), {
        type: 'line',
        data: {
          labels: mid.map((x) => Number(x).toFixed(2)),
          datasets: [
            {
              label: 'Observed',
              data: frac.map((y, i) => ({ x: meanP[i] ?? mid[i], y })),
              borderColor: colors[0],
              backgroundColor: 'transparent',
              showLine: true,
            },
            {
              label: 'Ideal',
              data: [
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ],
              borderColor: '#999',
              borderDash: [4, 4],
              pointRadius: 0,
            },
          ],
        },
        options: {
          parsing: false,
          plugins: { title: { display: true, text: 'Calibration (reliability)' } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Mean predicted' } },
            y: { min: 0, max: 1, title: { display: true, text: 'Fraction positive' } },
          },
        },
      });
    }
  }

  private renderMetrics(s: WorkspaceStatus): void {
    if (!this.prefs.show_metric_chart) return;
    const m = s.metrics || {};
    const keys = Object.keys(m).filter((k) => typeof m[k] === 'number' && Number.isFinite(m[k] as number));
    if (!keys.length) return;
    const colors = this.palette();
    const el = this.canvas(this.metricChartRef, 'metric');
    if (this.prefs.metric_chart_type === 'bar') {
      this.bar(el, {
        type: 'bar',
        data: {
          labels: keys,
          datasets: [{ label: 'Hold-out', data: keys.map((k) => Number(m[k])), backgroundColor: colors[0] }],
        },
        options: {
          plugins: { title: { display: true, text: 'Model metrics' } },
          scales: { y: { beginAtZero: true } },
        },
      });
      return;
    }
    this.bar(el, {
      type: 'radar',
      data: {
        labels: keys,
        datasets: [
          {
            label: 'Hold-out metrics',
            data: keys.map((k) => Number(m[k])),
            borderColor: colors[0],
            backgroundColor: 'rgba(42,107,90,0.2)',
          },
        ],
      },
      options: { plugins: { title: { display: true, text: 'Model metrics (where available)' } } },
    });
  }
}
