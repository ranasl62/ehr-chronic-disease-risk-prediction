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
});
