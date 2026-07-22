import { Injectable, signal, effect } from '@angular/core';

export type UiDensity = 'comfortable' | 'compact';
export type UiTheme = 'forest' | 'slate' | 'sand';
export type AnalyticsView = 'charts' | 'tables' | 'split';

export interface UiPrefs {
  density: UiDensity;
  theme: UiTheme;
  analytics_view: AnalyticsView;
  chart_animation: boolean;
  table_page_size: number;
  top_n_features: number;
  show_label_chart: boolean;
  show_age_chart: boolean;
  show_missing_chart: boolean;
  show_numeric_chart: boolean;
  show_metric_chart: boolean;
  show_importance_chart: boolean;
  show_compare_chart: boolean;
  /** When false, Datasets browse hides bundled demos (data/demo) and shows uploads only. */
  show_demo_datasets: boolean;
  label_chart_type: 'doughnut' | 'pie' | 'bar';
  metric_chart_type: 'radar' | 'bar';
  numeric_chart_type: 'bar' | 'line';
}

const STORAGE_KEY = 'ehr_ui_prefs_v1';

export const DEFAULT_UI_PREFS: UiPrefs = {
  density: 'comfortable',
  theme: 'forest',
  analytics_view: 'split',
  chart_animation: true,
  table_page_size: 10,
  top_n_features: 15,
  show_label_chart: true,
  show_age_chart: true,
  show_missing_chart: true,
  show_numeric_chart: true,
  show_metric_chart: true,
  show_importance_chart: true,
  show_compare_chart: true,
  show_demo_datasets: true,
  label_chart_type: 'doughnut',
  metric_chart_type: 'radar',
  numeric_chart_type: 'bar',
};

@Injectable({ providedIn: 'root' })
export class UiPrefsService {
  readonly prefs = signal<UiPrefs>(this.loadLocal());

  constructor() {
    effect(() => {
      const p = this.prefs();
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
      } catch {
        /* ignore quota */
      }
      this.applyDom(p);
    });
  }

  patch(partial: Partial<UiPrefs>): void {
    this.prefs.update((cur) => ({ ...cur, ...partial }));
  }

  reset(): void {
    this.prefs.set({ ...DEFAULT_UI_PREFS });
  }

  /** Merge server workspace `ui` block without wiping local chart-type toggles unless provided. */
  mergeFromWorkspace(ui: Record<string, unknown> | null | undefined): void {
    if (!ui || typeof ui !== 'object') return;
    const next: Partial<UiPrefs> = {};
    for (const key of Object.keys(DEFAULT_UI_PREFS) as (keyof UiPrefs)[]) {
      if (key in ui && ui[key] !== undefined && ui[key] !== null) {
        (next as Record<string, unknown>)[key] = ui[key];
      }
    }
    this.patch(next);
  }

  toWorkspaceUi(): Record<string, unknown> {
    const p = this.prefs();
    return {
      density: p.density,
      theme: p.theme,
      analytics_view: p.analytics_view,
      chart_animation: p.chart_animation,
      table_page_size: p.table_page_size,
      top_n_features: p.top_n_features,
      show_label_chart: p.show_label_chart,
      show_age_chart: p.show_age_chart,
      show_missing_chart: p.show_missing_chart,
      show_numeric_chart: p.show_numeric_chart,
      show_metric_chart: p.show_metric_chart,
      show_importance_chart: p.show_importance_chart,
      show_compare_chart: p.show_compare_chart,
      show_demo_datasets: p.show_demo_datasets,
    };
  }

  private loadLocal(): UiPrefs {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { ...DEFAULT_UI_PREFS };
      return { ...DEFAULT_UI_PREFS, ...JSON.parse(raw) };
    } catch {
      return { ...DEFAULT_UI_PREFS };
    }
  }

  private applyDom(p: UiPrefs): void {
    if (typeof document === 'undefined') return;
    const root = document.documentElement;
    root.dataset['density'] = p.density;
    root.dataset['theme'] = p.theme;
  }
}
