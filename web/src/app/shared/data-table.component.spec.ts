import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DataTableComponent } from './data-table.component';

describe('DataTableComponent', () => {
  let fixture: ComponentFixture<DataTableComponent>;
  let cmp: DataTableComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DataTableComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(DataTableComponent);
    cmp = fixture.componentInstance;
    cmp.columns = [
      { key: 'name', label: 'Name' },
      { key: 'age', label: 'Age', numeric: true, format: 'number', digits: '1.0-0' },
    ];
    cmp.rows = [
      { name: 'alice', age: 40 },
      { name: 'bob', age: 55 },
      { name: 'cara', age: 33 },
    ];
    cmp.pageSize = 2;
    cmp.recompute();
    fixture.detectChanges();
  });

  it('creates and paginates', () => {
    expect(cmp).toBeTruthy();
    expect(cmp.filtered.length).toBe(3);
    expect(cmp.pageRows.length).toBe(2);
    expect(cmp.totalPages).toBe(2);
  });

  it('filters rows by query', () => {
    cmp.query = 'bo';
    cmp.recompute();
    expect(cmp.filtered.length).toBe(1);
    expect(cmp.pageRows[0]['name']).toBe('bob');
  });

  it('sorts numeric columns', () => {
    cmp.sortBy('age');
    expect(cmp.filtered.map((r) => r['age'])).toEqual([33, 40, 55]);
    cmp.sortBy('age');
    expect(cmp.filtered.map((r) => r['age'])).toEqual([55, 40, 33]);
  });

  it('exports csv blob click', () => {
    const click = spyOn(HTMLAnchorElement.prototype, 'click');
    cmp.exportCsv();
    expect(click).toHaveBeenCalled();
  });

  it('paginates and sorts text columns', () => {
    cmp.go(2);
    expect(cmp.page).toBe(2);
    cmp.sortBy('name');
    cmp.recompute();
    cmp.onPageSize();
    expect(cmp.pageRows.length).toBeLessThanOrEqual(cmp.pageSize);
    cmp.rows = [];
    cmp.recompute();
    expect(cmp.pageRows.length).toBe(0);
  });

  it('formats percent columns and tracks rows', () => {
    cmp.columns = [{ key: 'pct', label: 'Pct', format: 'percent' }];
    cmp.rows = [{ pct: 42.5, __id: 'x' }];
    cmp.recompute();
    expect(cmp.trackRow(0, cmp.rows[0])).toBe('x');
    expect(cmp.asNumber('nope')).toBe(0);
    fixture.detectChanges();
  });

  it('resets page when filtered rows shrink', () => {
    cmp.go(2);
    cmp.query = 'zzzzz';
    cmp.recompute();
    expect(cmp.page).toBe(1);
  });
});
