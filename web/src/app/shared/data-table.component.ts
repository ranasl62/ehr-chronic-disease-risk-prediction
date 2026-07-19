import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

export interface DataTableColumn {
  key: string;
  label: string;
  numeric?: boolean;
  format?: 'number' | 'text' | 'percent';
  digits?: string;
}

@Component({
  selector: 'app-data-table',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="dt">
      <div class="dt-toolbar">
        <input
          type="search"
          class="dt-search"
          placeholder="Filter rows…"
          [(ngModel)]="query"
          (ngModelChange)="recompute()"
          name="dt_q"
        />
        <span class="dt-meta">{{ filtered.length }} / {{ rows.length }} rows</span>
        @if (exportable) {
          <button type="button" class="btn dt-btn" (click)="exportCsv()">Export CSV</button>
        }
      </div>
      <div class="dt-scroll">
        <table>
          <thead>
            <tr>
              @for (c of columns; track c.key) {
                <th (click)="sortBy(c.key)" [class.num]="c.numeric">
                  {{ c.label }}
                  @if (sortKey === c.key) {
                    <span class="sort">{{ sortDir === 'asc' ? '↑' : '↓' }}</span>
                  }
                </th>
              }
            </tr>
          </thead>
          <tbody>
            @for (row of pageRows; track trackRow($index, row)) {
              <tr>
                @for (c of columns; track c.key) {
                  <td [class.num]="c.numeric">
                    @if (c.format === 'number' || c.numeric) {
                      {{ asNumber(row[c.key]) | number: (c.digits || '1.2-3') }}
                    } @else if (c.format === 'percent') {
                      {{ asNumber(row[c.key]) | number: '1.1-1' }}%
                    } @else {
                      {{ row[c.key] }}
                    }
                  </td>
                }
              </tr>
            } @empty {
              <tr><td [attr.colspan]="columns.length" class="empty">No matching rows</td></tr>
            }
          </tbody>
        </table>
      </div>
      <div class="dt-pager">
        <button type="button" class="btn dt-btn" [disabled]="page <= 1" (click)="go(page - 1)">Prev</button>
        <span>Page {{ page }} / {{ totalPages }}</span>
        <button type="button" class="btn dt-btn" [disabled]="page >= totalPages" (click)="go(page + 1)">Next</button>
        <label class="page-size">Rows
          <select [(ngModel)]="pageSize" (ngModelChange)="onPageSize()" name="dt_ps">
            <option [ngValue]="5">5</option>
            <option [ngValue]="10">10</option>
            <option [ngValue]="25">25</option>
            <option [ngValue]="50">50</option>
            <option [ngValue]="100">100</option>
          </select>
        </label>
      </div>
    </div>
  `,
  styles: [
    `
      .dt { display: flex; flex-direction: column; gap: 0.5rem; }
      .dt-toolbar { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
      .dt-search {
        flex: 1; min-width: 12rem; padding: 0.4rem 0.6rem;
        border: 1px solid var(--line); border-radius: 4px; font: inherit; background: var(--surface);
      }
      .dt-meta { color: var(--muted); font-size: 0.85rem; }
      .dt-scroll { overflow: auto; max-height: 28rem; border: 1px solid var(--line); border-radius: 6px; }
      table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
      th, td { padding: 0.4rem 0.55rem; border-bottom: 1px solid var(--line); text-align: left; }
      th { position: sticky; top: 0; background: var(--surface-2); cursor: pointer; user-select: none; white-space: nowrap; }
      th:hover { color: var(--accent); }
      th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
      .sort { margin-left: 0.25rem; color: var(--accent); }
      .empty { text-align: center; color: var(--muted); padding: 1rem; }
      .dt-pager { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; font-size: 0.85rem; }
      .dt-btn { padding: 0.3rem 0.65rem; font-size: 0.85rem; }
      .page-size { display: flex; gap: 0.35rem; align-items: center; margin-left: auto; }
      .page-size select { padding: 0.25rem; border: 1px solid var(--line); border-radius: 4px; font: inherit; }
    `,
  ],
})
export class DataTableComponent implements OnChanges {
  @Input() columns: DataTableColumn[] = [];
  @Input() rows: Record<string, unknown>[] = [];
  @Input() pageSize = 10;
  @Input() exportable = true;
  @Input() exportName = 'table.csv';
  @Output() pageSizeChange = new EventEmitter<number>();

  query = '';
  sortKey = '';
  sortDir: 'asc' | 'desc' = 'asc';
  page = 1;
  filtered: Record<string, unknown>[] = [];
  pageRows: Record<string, unknown>[] = [];

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['rows'] || changes['pageSize'] || changes['columns']) {
      this.recompute();
    }
  }

  get totalPages(): number {
    return Math.max(1, Math.ceil(this.filtered.length / Math.max(1, this.pageSize)));
  }

  trackRow(i: number, row: Record<string, unknown>): string {
    return String(row['__id'] ?? i);
  }

  asNumber(v: unknown): number {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }

  sortBy(key: string): void {
    if (this.sortKey === key) {
      this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortKey = key;
      this.sortDir = 'asc';
    }
    this.recompute();
  }

  go(p: number): void {
    this.page = Math.min(this.totalPages, Math.max(1, p));
    this.slicePage();
  }

  onPageSize(): void {
    this.page = 1;
    this.pageSizeChange.emit(this.pageSize);
    this.slicePage();
  }

  recompute(): void {
    const q = this.query.trim().toLowerCase();
    let rows = [...this.rows];
    if (q) {
      rows = rows.filter((r) =>
        this.columns.some((c) => String(r[c.key] ?? '').toLowerCase().includes(q))
      );
    }
    if (this.sortKey) {
      const key = this.sortKey;
      const col = this.columns.find((c) => c.key === key);
      const dir = this.sortDir === 'asc' ? 1 : -1;
      rows.sort((a, b) => {
        const av = a[key];
        const bv = b[key];
        if (col?.numeric || col?.format === 'number' || col?.format === 'percent') {
          return (Number(av) - Number(bv)) * dir;
        }
        return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
      });
    }
    this.filtered = rows;
    if (this.page > this.totalPages) this.page = 1;
    this.slicePage();
  }

  exportCsv(): void {
    const cols = this.columns;
    const esc = (v: unknown) => {
      const s = String(v ?? '');
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [
      cols.map((c) => esc(c.label)).join(','),
      ...this.filtered.map((r) => cols.map((c) => esc(r[c.key])).join(',')),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = this.exportName;
    a.click();
    URL.revokeObjectURL(url);
  }

  private slicePage(): void {
    const start = (this.page - 1) * this.pageSize;
    this.pageRows = this.filtered.slice(start, start + this.pageSize);
  }
}
