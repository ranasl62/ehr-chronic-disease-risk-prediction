import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { ApiService, DatasetHealth, DatasetInfo } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';
import { UiPrefsService } from '../../core/ui-prefs.service';

type DeleteResult = {
  deleted: boolean;
  already_absent?: boolean;
  path: string;
  error?: string;
};

@Component({
  selector: 'app-datasets',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './datasets.component.html',
  styleUrl: './datasets.component.css',
})
export class DatasetsComponent implements OnInit {
  private readonly api = inject(ApiService);
  private readonly state = inject(WorkspaceState);
  private readonly router = inject(Router);
  readonly prefs = inject(UiPrefsService);

  tab = signal<'browse' | 'file' | 'form' | 'sql'>('browse');
  datasets = signal<DatasetInfo[]>([]);
  /** Active row for health / train (single). */
  selected = signal<DatasetInfo | null>(null);
  /** Checkbox selection for bulk delete. */
  checkedIds = signal<Set<string>>(new Set());
  health = signal<DatasetHealth | null>(null);
  message = signal<string | null>(null);
  error = signal<string | null>(null);
  deleting = signal(false);

  tableRows = computed(() => this.datasets());
  checkedCount = computed(() => this.checkedIds().size);
  allVisibleChecked = computed(() => {
    const rows = this.tableRows();
    if (!rows.length) return false;
    const ids = this.checkedIds();
    return rows.every((d) => ids.has(d.id));
  });

  formJson = `[
  {"patient_id": 1, "timestamp": "2023-01-01", "glucose": 100, "blood_pressure": 120, "age": 45, "label": 0},
  {"patient_id": 1, "timestamp": "2023-06-01", "glucose": 110, "blood_pressure": 122, "age": 45, "label": 0},
  {"patient_id": 2, "timestamp": "2023-01-01", "glucose": 145, "blood_pressure": 140, "age": 61, "label": 1}
]`;
  formName = 'form_import.csv';
  sqlText = 'SELECT * FROM patients LIMIT 100';
  sqlUrl = '';
  sqlName = 'sql_import.csv';

  ngOnInit(): void {
    this.reload();
  }

  setTab(t: 'browse' | 'file' | 'form' | 'sql'): void {
    this.tab.set(t);
  }

  showDemoDatasets(): boolean {
    return this.prefs.prefs().show_demo_datasets !== false;
  }

  toggleShowDemo(on: boolean): void {
    this.prefs.patch({ show_demo_datasets: on });
    const sel = this.selected();
    if (!on && sel && (sel.bundled || sel.category === 'demo')) {
      this.selected.set(null);
      this.state.selectedDataset.set(null);
      this.health.set(null);
    }
    this.checkedIds.set(new Set());
    this.reload();
  }

  reload(): void {
    this.api.datasets(this.showDemoDatasets()).subscribe({
      next: (r) => {
        this.datasets.set(r.datasets);
        const alive = new Set(r.datasets.map((d) => d.id));
        this.checkedIds.update((cur) => new Set([...cur].filter((id) => alive.has(id))));
      },
      error: (e) => this.error.set(String(e.message || e)),
    });
  }

  select(d: DatasetInfo): void {
    this.selected.set(d);
    this.state.selectedDataset.set(d);
    this.health.set(null);
  }

  isChecked(id: string): boolean {
    return this.checkedIds().has(id);
  }

  toggleCheck(d: DatasetInfo, ev: Event): void {
    ev.stopPropagation();
    const on = (ev.target as HTMLInputElement).checked;
    this.checkedIds.update((cur) => {
      const next = new Set(cur);
      if (on) next.add(d.id);
      else next.delete(d.id);
      return next;
    });
  }

  toggleCheckAll(ev: Event): void {
    const on = (ev.target as HTMLInputElement).checked;
    if (!on) {
      this.checkedIds.set(new Set());
      return;
    }
    this.checkedIds.set(new Set(this.tableRows().map((d) => d.id)));
  }

  kindLabel(d: DatasetInfo): string {
    if (d.bundled) return d.source_type || 'demo';
    return 'your data';
  }

  formatBytes(n: number | undefined): string {
    if (n == null || !Number.isFinite(n)) return '—';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  runHealth(): void {
    const d = this.selected();
    if (!d) return;
    this.error.set(null);
    this.api.datasetHealth(d.path).subscribe({
      next: (h) => this.health.set(h),
      error: (e) => this.error.set(e?.error?.detail || e.message),
    });
  }

  /** Delete checked rows (1..n). If none checked, delete the active row. */
  deleteSelected(): void {
    const checked = this.checkedIds();
    let targets = this.tableRows().filter((d) => checked.has(d.id));
    if (!targets.length && this.selected()) {
      targets = [this.selected()!];
    }
    if (!targets.length) return;

    const n = targets.length;
    const list = targets.map((d) => `• ${d.path}`).join('\n');
    const ok = window.confirm(
      `Delete ${n} dataset${n === 1 ? '' : 's'} permanently?\n\n${list}\n\nThis cannot be undone.`
    );
    if (!ok) return;

    this.error.set(null);
    this.message.set(null);
    this.deleting.set(true);

    forkJoin(
      targets.map((d) =>
        this.api.deleteDataset(d.path).pipe(
          catchError((err) =>
            of<DeleteResult>({
              deleted: false,
              path: d.path,
              error: err?.error?.detail || err.message || 'Request failed',
            })
          )
        )
      )
    ).subscribe({
      next: (results) => {
        const failed = results.filter((r) => !r.deleted);
        const okCount = results.length - failed.length;
        const alreadyAbsent = results.filter((r) => r.deleted && r.already_absent).length;
        if (okCount) {
          const suffix = alreadyAbsent
            ? ` (${alreadyAbsent} already removed)`
            : '';
          this.message.set(`Deleted ${okCount} dataset${okCount === 1 ? '' : 's'}${suffix}`);
        }
        if (failed.length) {
          this.error.set(
            `Failed to delete ${failed.length} dataset${failed.length === 1 ? '' : 's'}: ` +
              failed.map((f) => `${f.path} (${f.error || 'Request failed'})`).join(', ')
          );
        }
        const removed = new Set(targets.map((d) => d.id));
        if (this.selected() && removed.has(this.selected()!.id)) {
          this.selected.set(null);
          this.state.selectedDataset.set(null);
          this.health.set(null);
        }
        this.checkedIds.set(new Set());
        this.deleting.set(false);
        this.reload();
      },
      error: this.handleDeleteJoinError.bind(this),
    });
  }

  private handleDeleteJoinError(e: { error?: { detail?: string }; message?: string }): void {
    this.deleting.set(false);
    this.error.set(String(e?.error?.detail || e.message || 'Request failed'));
  }

  continueTrain(): void {
    if (this.selected() && this.canTrain()) {
      this.router.navigate(['/train']);
    }
  }

  canTrain(): boolean {
    const h = this.health();
    if (!this.selected()) return false;
    if (!h) return false;
    return !!h.health?.ready_for_training;
  }

  onFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    this.message.set('Importing…');
    this.error.set(null);
    this.api.uploadDataset(file).subscribe({
      next: () => {
        this.message.set(`Imported ${file.name} → data/uploads/`);
        this.reload();
        this.tab.set('browse');
      },
      error: (e) => {
        this.error.set(e?.error?.detail || e.message);
        this.message.set(null);
      },
    });
  }

  submitForm(): void {
    this.error.set(null);
    try {
      const rows = JSON.parse(this.formJson);
      if (!Array.isArray(rows)) throw new Error('JSON must be an array of row objects');
      this.api.importForm(rows, this.formName).subscribe({
        next: () => {
          this.message.set('Form import saved → data/uploads/');
          this.reload();
          this.tab.set('browse');
        },
        error: (e) => this.error.set(e?.error?.detail || e.message),
      });
    } catch (e: unknown) {
      this.error.set(e instanceof Error ? e.message : String(e));
    }
  }

  submitSql(): void {
    this.error.set(null);
    this.api.importSql(this.sqlText, this.sqlUrl || undefined, this.sqlName).subscribe({
      next: () => {
        this.message.set('SQL import saved → data/uploads/');
        this.reload();
        this.tab.set('browse');
      },
      error: (e) => this.error.set(e?.error?.detail || e.message),
    });
  }
}
