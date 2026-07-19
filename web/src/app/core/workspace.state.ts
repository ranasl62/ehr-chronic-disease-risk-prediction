import { Injectable, signal } from '@angular/core';
import { DatasetInfo } from './api.service';

/** Shared selection across wizard / train pages. */
@Injectable({ providedIn: 'root' })
export class WorkspaceState {
  readonly selectedDataset = signal<DatasetInfo | null>(null);
  readonly lastJobId = signal<string | null>(null);
}
