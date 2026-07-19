import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { ApiService, DatasetHealth, DatasetInfo } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';

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

  tab = signal<'browse' | 'file' | 'form' | 'sql'>('browse');
  datasets = signal<DatasetInfo[]>([]);
  selected = signal<DatasetInfo | null>(null);
  health = signal<DatasetHealth | null>(null);
  message = signal<string | null>(null);
  error = signal<string | null>(null);

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

  reload(): void {
    this.api.datasets().subscribe({
      next: (r) => this.datasets.set(r.datasets),
      error: (e) => this.error.set(String(e.message || e)),
    });
  }

  select(d: DatasetInfo): void {
    this.selected.set(d);
    this.state.selectedDataset.set(d);
    this.health.set(null);
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
        this.message.set(`Imported ${file.name}`);
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
          this.message.set('Form import saved');
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
        this.message.set('SQL import saved');
        this.reload();
        this.tab.set('browse');
      },
      error: (e) => this.error.set(e?.error?.detail || e.message),
    });
  }
}
