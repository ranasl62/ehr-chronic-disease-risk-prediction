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
import { ApiService, ModelSchema, PredictResult } from '../../core/api.service';

Chart.register(...registerables);

interface Driver {
  feature: string;
  shap_value: number;
  abs_contribution: number;
}

@Component({
  selector: 'app-predict',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './predict.component.html',
  styleUrl: './predict.component.css',
})
export class PredictComponent implements OnInit, AfterViewInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly host = inject(ElementRef<HTMLElement>);

  @ViewChild('riskChart') riskChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('shapChart') shapChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('vsMedianChart') vsMedianChartRef?: ElementRef<HTMLCanvasElement>;

  schema = signal<ModelSchema | null>(null);
  values: Record<string, number> = {};
  result = signal<PredictResult | null>(null);
  error = signal<string | null>(null);
  loading = signal(true);
  includeExplanation = true;
  reuseAsNextInput = true;
  featureQuery = '';
  showRawJson = false;

  private charts: Chart[] = [];
  private redrawTimer: ReturnType<typeof setTimeout> | null = null;

  ngOnInit(): void {
    this.api.schema().subscribe({
      next: (s) => {
        this.schema.set(s);
        this.fillMedians();
        this.loading.set(false);
      },
      error: (e) => {
        this.loading.set(false);
        this.error.set(this.fmtErr(e));
      },
    });
  }

  ngAfterViewInit(): void {
    this.scheduleCharts();
  }

  ngOnDestroy(): void {
    if (this.redrawTimer != null) clearTimeout(this.redrawTimer);
    this.destroyCharts();
  }

  featureGroups(): { window: string; cols: string[] }[] {
    const s = this.schema();
    if (!s) return [];
    const q = this.featureQuery.trim().toLowerCase();
    const cols = s.feature_columns.filter((c) => !q || c.toLowerCase().includes(q));
    const groups = new Map<string, string[]>();
    for (const c of cols) {
      const m = c.match(/^(w\d+d)_/);
      const key = m ? m[1] : 'other';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(c);
    }
    return [...groups.entries()].map(([window, cols]) => ({ window, cols }));
  }

  statsHint(col: string): string {
    const st = this.schema()?.input_stats?.[col];
    if (!st) return '';
    const parts: string[] = [];
    if (st.p05 != null) parts.push(`p05 ${st.p05}`);
    if (st.median != null) parts.push(`med ${st.median}`);
    if (st.p95 != null) parts.push(`p95 ${st.p95}`);
    return parts.join(' · ');
  }

  submit(): void {
    this.error.set(null);
    this.api.predict({ ...this.values }, this.includeExplanation).subscribe({
      next: (r) => {
        this.result.set(r);
        if (!this.reuseAsNextInput) {
          this.fillMedians();
        }
        this.scheduleCharts();
      },
      error: (e) => this.error.set(this.fmtErr(e)),
    });
  }

  fillMedians(): void {
    const s = this.schema();
    if (!s) return;
    for (const c of s.feature_columns) {
      const st = s.input_stats?.[c];
      this.values[c] = st?.median ?? 0;
    }
  }

  exportSessionJson(): void {
    const payload = {
      generated_at_utc: new Date().toISOString(),
      disclaimer: 'For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future.',
      model: this.schema()
        ? {
            model_kind: this.schema()!.model_kind,
            calibrated: this.schema()!.calibrated,
            n_features: this.schema()!.feature_columns.length,
          }
        : null,
      features: { ...this.values },
      prediction: this.result(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predict_session.json';
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  drivers(): Driver[] {
    const expl = this.result()?.explanation as
      | { top_positive_risk_drivers?: Driver[]; shap_vector?: Record<string, number> }
      | undefined;
    if (!expl) return [];
    if (expl.top_positive_risk_drivers?.length) {
      return [...expl.top_positive_risk_drivers].sort(
        (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
      );
    }
    const vec = expl.shap_vector || {};
    return Object.entries(vec)
      .map(([feature, shap_value]) => ({
        feature,
        shap_value,
        abs_contribution: Math.abs(shap_value),
      }))
      .sort((a, b) => b.abs_contribution - a.abs_contribution)
      .slice(0, 12);
  }

  riskPct(): number {
    const r = this.result();
    return r ? r.risk_probability * 100 : 0;
  }

  private fmtErr(e: unknown): string {
    const any = e as { error?: { detail?: unknown }; message?: string };
    const d = any?.error?.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d
        .map((x) => (typeof x === 'object' && x && 'msg' in x ? String((x as { msg: string }).msg) : String(x)))
        .join('; ');
    }
    return any?.message || 'Request failed — is a model trained?';
  }

  private scheduleCharts(): void {
    this.cdr.detectChanges();
    if (this.redrawTimer != null) clearTimeout(this.redrawTimer);
    this.redrawTimer = setTimeout(() => {
      this.redrawTimer = null;
      requestAnimationFrame(() => this.drawCharts());
    }, 0);
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

  private push(el: HTMLCanvasElement | undefined, cfg: ChartConfiguration): void {
    if (!el) return;
    const existing = Chart.getChart(el);
    if (existing) existing.destroy();
    cfg.options = { ...(cfg.options || {}), responsive: true, maintainAspectRatio: false, animation: false };
    this.charts.push(new Chart(el, cfg));
  }

  private drawCharts(): void {
    this.destroyCharts();
    const r = this.result();
    if (!r) return;

    const pct = r.risk_probability;
    this.push(this.canvas(this.riskChartRef, 'risk'), {
      type: 'doughnut',
      data: {
        labels: ['Risk', 'Remainder'],
        datasets: [
          {
            data: [pct, Math.max(0, 1 - pct)],
            backgroundColor: ['#2a6b5a', '#d5e0db'],
            borderWidth: 0,
          },
        ],
      },
      options: {
        plugins: {
          legend: { display: false },
          title: { display: true, text: `${(pct * 100).toFixed(1)}% · ${r.risk_level}` },
        },
      },
    } as ChartConfiguration);

    const drv = this.drivers().slice(0, 10);
    if (drv.length) {
      this.push(this.canvas(this.shapChartRef, 'shap'), {
        type: 'bar',
        data: {
          labels: drv.map((d) => d.feature),
          datasets: [
            {
              label: 'SHAP',
              data: drv.map((d) => d.shap_value),
              backgroundColor: drv.map((d) => (d.shap_value >= 0 ? '#a33' : '#2a6b5a')),
            },
          ],
        },
        options: {
          indexAxis: 'y',
          plugins: { title: { display: true, text: 'Top explanation drivers' } },
        },
      });
    }

    const s = this.schema();
    if (s) {
      const cols = s.feature_columns.slice(0, 8);
      this.push(this.canvas(this.vsMedianChartRef, 'vs'), {
        type: 'bar',
        data: {
          labels: cols,
          datasets: [
            {
              label: 'Your input',
              data: cols.map((c) => Number(this.values[c] ?? 0)),
              backgroundColor: '#2a6b5a',
            },
            {
              label: 'Training median',
              data: cols.map((c) => Number(s.input_stats?.[c]?.median ?? 0)),
              backgroundColor: '#c4a35a',
            },
          ],
        },
        options: {
          plugins: { title: { display: true, text: 'Input vs training medians (sample features)' } },
        },
      });
    }
  }
}
