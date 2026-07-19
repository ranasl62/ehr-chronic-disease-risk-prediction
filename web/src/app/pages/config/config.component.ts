import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';
import { UiPrefs, UiPrefsService } from '../../core/ui-prefs.service';
import { DataTableColumn, DataTableComponent } from '../../shared/data-table.component';

const MODEL_OPTS = ['logreg', 'random_forest', 'xgboost', 'lightgbm'] as const;

@Component({
  selector: 'app-config',
  standalone: true,
  imports: [CommonModule, FormsModule, DataTableComponent],
  templateUrl: './config.component.html',
  styleUrl: './config.component.css',
})
export class ConfigComponent implements OnInit {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiUrl;
  readonly ui = inject(UiPrefsService);

  cfg: Record<string, unknown> = {};
  effective: Record<string, unknown> = {};
  windowsText = '7, 30, 180';
  compareSelected: Record<string, boolean> = {
    logreg: true,
    random_forest: true,
    xgboost: true,
    lightgbm: false,
  };
  events = signal<Record<string, unknown>[]>([]);
  eventRows: Record<string, unknown>[] = [];
  message = signal<string | null>(null);
  error = signal<string | null>(null);
  apiKey = '';

  modelOpts = MODEL_OPTS;
  eventCols: DataTableColumn[] = [
    { key: 'kind', label: 'Kind' },
    { key: 'message', label: 'Message' },
    { key: 'ts', label: 'When' },
  ];

  ngOnInit(): void {
    try {
      this.apiKey = localStorage.getItem('ehr_api_key') || '';
    } catch {
      this.apiKey = '';
    }
    this.reload();
  }

  get prefs(): UiPrefs {
    return this.ui.prefs();
  }

  reload(): void {
    this.http
      .get<{ config: Record<string, unknown>; effective_train: Record<string, unknown> }>(
        `${this.base}/v1/workspace/config`
      )
      .subscribe({
        next: (r) => {
          this.cfg = { ...r.config };
          this.effective = r.effective_train || {};
          const w = r.config['windows_days'];
          if (Array.isArray(w)) this.windowsText = w.join(', ');
          const cm = (r.config['compare_models'] as string[]) || [];
          for (const m of MODEL_OPTS) {
            this.compareSelected[m] = cm.includes(m);
          }
          const uiBlock = r.config['ui'];
          if (uiBlock && typeof uiBlock === 'object') {
            this.ui.mergeFromWorkspace(uiBlock as Record<string, unknown>);
          }
        },
        error: (e) => this.error.set(e?.error?.detail || e.message),
      });
    this.http
      .get<{ events: Record<string, unknown>[] }>(`${this.base}/v1/events`, {
        params: { limit: 50 },
      })
      .subscribe({
        next: (r) => {
          this.events.set(r.events || []);
          this.eventRows = (r.events || []).map((e) => ({
            kind: e['kind'],
            message: e['message'],
            ts: e['ts'] || e['at'] || e['created_at'] || '',
          }));
        },
      });
  }

  saveApiKey(): void {
    try {
      const v = this.apiKey.trim();
      if (v) localStorage.setItem('ehr_api_key', v);
      else localStorage.removeItem('ehr_api_key');
      this.message.set(v ? 'API key saved for this browser' : 'API key cleared');
    } catch {
      this.error.set('Could not write API key to localStorage');
    }
  }

  save(): void {
    this.error.set(null);
    const windows = this.windowsText
      .split(/[, ]+/)
      .map((x) => Number(x.trim()))
      .filter((n) => Number.isFinite(n) && n > 0);
    const compare_models = MODEL_OPTS.filter((m) => this.compareSelected[m]);
    const body = {
      ...this.cfg,
      windows_days: windows.length ? windows : [7, 30, 180],
      compare_models,
      ui: this.ui.toWorkspaceUi(),
    };
    this.http.put(`${this.base}/v1/workspace/config`, body).subscribe({
      next: () => {
        this.message.set('Config + UI preferences saved');
        this.reload();
      },
      error: (e) => this.error.set(e?.error?.detail || e.message),
    });
  }

  resetUi(): void {
    this.ui.reset();
    this.message.set('UI preferences reset to defaults (save to persist to workspace.yaml)');
  }

  onPageSize(n: number): void {
    this.ui.patch({ table_page_size: n });
  }
}
