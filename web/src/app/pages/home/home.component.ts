import { Component, OnInit, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ApiService, WorkspaceStatus } from '../../core/api.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
})
export class HomeComponent implements OnInit {
  private readonly api = inject(ApiService);
  status = signal<WorkspaceStatus | null>(null);
  error = signal<string | null>(null);
  loading = signal(true);
  readonly zipUrl = this.api.resultsZipUrl();

  readonly steps = [
    { key: 'api_healthy', label: 'API healthy', fix: null },
    { key: 'demo_dataset', label: 'Demo dataset available', fix: '/datasets' },
    { key: 'model_trained', label: 'Model trained', fix: '/train' },
    { key: 'metrics_available', label: 'Metrics available', fix: '/results' },
    { key: 'leakage_audited', label: 'Leakage audit present', fix: '/train' },
    { key: 'shap_available', label: 'SHAP summary present', fix: '/results' },
  ];

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.workspaceStatus().subscribe({
      next: (s) => {
        this.status.set(s);
        this.loading.set(false);
      },
      error: (e) => {
        this.error.set(e?.message || 'Cannot reach API. Is uvicorn running on :8000?');
        this.loading.set(false);
      },
    });
  }
}
