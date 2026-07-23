import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { of, throwError } from 'rxjs';
import { DatasetsComponent } from './datasets.component';
import { ApiService } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';
import { UiPrefsService } from '../../core/ui-prefs.service';

describe('DatasetsComponent', () => {
  let fixture: ComponentFixture<DatasetsComponent>;
  let cmp: DatasetsComponent;
  let api: jasmine.SpyObj<ApiService>;
  let prefs: UiPrefsService;

  beforeEach(async () => {
    api = jasmine.createSpyObj('ApiService', [
      'datasets',
      'datasetHealth',
      'uploadDataset',
      'deleteDataset',
      'importForm',
      'importSql',
    ]);
    api.datasets.and.callFake((includeDemo = true) =>
      of({
        datasets: includeDemo
          ? [
              {
                id: 'ehr_data',
                label: 'Demo',
                path: 'data/demo/ehr_data.csv',
                format: 'longitudinal',
                exists: true,
                bytes: 100,
                bundled: true,
                category: 'demo',
                source_type: 'demo',
              },
              {
                id: 'upload:mine.csv',
                label: 'Upload: mine.csv',
                path: 'data/uploads/mine.csv',
                format: 'longitudinal',
                exists: true,
                bytes: 200,
                bundled: false,
                category: 'user',
                source_type: 'byo',
              },
            ]
          : [
              {
                id: 'upload:mine.csv',
                label: 'Upload: mine.csv',
                path: 'data/uploads/mine.csv',
                format: 'longitudinal',
                exists: true,
                bytes: 200,
                bundled: false,
                category: 'user',
                source_type: 'byo',
              },
            ],
        include_demo: includeDemo,
      })
    );
    api.datasetHealth.and.returnValue(
      of({
        path: 'data/demo/ehr_data.csv',
        n_rows: 20,
        n_columns: 10,
        health: { ready_for_training: true, blockers: [], warnings: [] },
      })
    );

    await TestBed.configureTestingModule({
      imports: [DatasetsComponent],
      providers: [
        { provide: ApiService, useValue: api },
        WorkspaceState,
        UiPrefsService,
        provideRouter([]),
      ],
    }).compileComponents();

    prefs = TestBed.inject(UiPrefsService);
    prefs.patch({ show_demo_datasets: true });
    fixture = TestBed.createComponent(DatasetsComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('lists datasets in a table on init', () => {
    expect(api.datasets).toHaveBeenCalledWith(true);
    expect(cmp.tableRows().length).toBe(2);
    expect(fixture.nativeElement.querySelectorAll('.ds-table tbody tr').length).toBe(2);
  });

  it('hides demos when toggle is off', () => {
    cmp.toggleShowDemo(false);
    expect(prefs.prefs().show_demo_datasets).toBeFalse();
    expect(api.datasets).toHaveBeenCalledWith(false);
    expect(cmp.tableRows().length).toBe(1);
  });

  it('loads health for selected dataset', () => {
    cmp.select(cmp.datasets()[0]);
    cmp.runHealth();
    expect(api.datasetHealth).toHaveBeenCalledWith('data/demo/ehr_data.csv');
    expect(cmp.health()?.health.ready_for_training).toBeTrue();
  });

  it('imports form rows', () => {
    api.importForm.and.returnValue(of({ path: 'data/uploads/x.csv' }));
    cmp.formJson = '[{"patient_id":1,"label":0}]';
    cmp.submitForm();
    expect(api.importForm).toHaveBeenCalled();
  });

  it('deletes checked datasets (multi)', () => {
    spyOn(window, 'confirm').and.returnValue(true);
    api.deleteDataset.and.returnValue(of({ deleted: true, path: 'x' }));
    cmp.checkedIds.set(new Set(['ehr_data', 'upload:mine.csv']));
    cmp.deleteSelected();
    expect(api.deleteDataset).toHaveBeenCalledTimes(2);
    expect(api.deleteDataset).toHaveBeenCalledWith('data/demo/ehr_data.csv');
    expect(api.deleteDataset).toHaveBeenCalledWith('data/uploads/mine.csv');
  });

  it('deletes active row when nothing checked', () => {
    spyOn(window, 'confirm').and.returnValue(true);
    api.deleteDataset.and.returnValue(of({ deleted: true, path: 'data/uploads/mine.csv' }));
    cmp.select(cmp.datasets()[1]);
    cmp.deleteSelected();
    expect(api.deleteDataset).toHaveBeenCalledWith('data/uploads/mine.csv');
  });

  it('treats an already absent dataset as a successful delete', () => {
    spyOn(window, 'confirm').and.returnValue(true);
    api.deleteDataset.and.returnValue(
      of({ deleted: true, already_absent: true, path: 'data/uploads/mine.csv' })
    );

    cmp.select(cmp.datasets()[1]);
    cmp.deleteSelected();

    expect(cmp.message()).toBe('Deleted 1 dataset (1 already removed)');
    expect(cmp.error()).toBeNull();
    expect(api.datasets).toHaveBeenCalledTimes(2);
  });

  it('reports paths and server details for partial delete failures', () => {
    spyOn(window, 'confirm').and.returnValue(true);
    api.deleteDataset.and.callFake((path: string) =>
      path.includes('ehr_data')
        ? of({ deleted: true, path })
        : of({ deleted: false, path, error: 'Protected file' })
    );
    cmp.checkedIds.set(new Set(['ehr_data', 'upload:mine.csv']));

    cmp.deleteSelected();

    expect(cmp.message()).toBe('Deleted 1 dataset');
    expect(cmp.error()).toBe(
      'Failed to delete 1 dataset: data/uploads/mine.csv (Protected file)'
    );
  });

  it('switches tabs and formats bytes', () => {
    cmp.setTab('form');
    expect(cmp.tab()).toBe('form');
    expect(cmp.formatBytes(500)).toBe('500 B');
    expect(cmp.formatBytes(2048)).toBe('2.0 KB');
    expect(cmp.formatBytes(2 * 1024 * 1024)).toBe('2.0 MB');
    expect(cmp.formatBytes(undefined)).toBe('—');
    expect(cmp.kindLabel(cmp.datasets()[1])).toBe('your data');
  });

  it('toggleCheck and toggleCheckAll', () => {
    const d = cmp.datasets()[0];
    cmp.toggleCheck(d, { stopPropagation: () => undefined, target: { checked: true } } as unknown as Event);
    expect(cmp.isChecked(d.id)).toBeTrue();
    cmp.toggleCheckAll({ target: { checked: false } } as unknown as Event);
    expect(cmp.checkedCount()).toBe(0);
    cmp.toggleCheckAll({ target: { checked: true } } as unknown as Event);
    expect(cmp.allVisibleChecked()).toBeTrue();
  });

  it('clears demo selection when hiding demos', () => {
    cmp.select(cmp.datasets()[0]);
    cmp.toggleShowDemo(false);
    expect(cmp.selected()).toBeNull();
  });

  it('continueTrain navigates when health ready', () => {
    const router = TestBed.inject(Router);
    spyOn(router, 'navigate');
    cmp.select(cmp.datasets()[0]);
    cmp.runHealth();
    expect(cmp.canTrain()).toBeTrue();
    cmp.continueTrain();
    expect(router.navigate).toHaveBeenCalledWith(['/train']);
  });

  it('canTrain returns false without selection or health', () => {
    expect(cmp.canTrain()).toBeFalse();
    cmp.select(cmp.datasets()[0]);
    expect(cmp.canTrain()).toBeFalse();
  });

  it('uploads file via onFile', () => {
    api.uploadDataset.and.returnValue(of({ path: 'data/uploads/new.csv' }));
    const file = new File(['a'], 'new.csv');
    cmp.onFile({ target: { files: [file] } } as unknown as Event);
    expect(api.uploadDataset).toHaveBeenCalled();
    expect(cmp.message()).toContain('Imported');
  });

  it('handles upload and reload errors', () => {
    api.uploadDataset.and.returnValue(throwError(() => ({ error: { detail: 'upload fail' } })));
    cmp.onFile({ target: { files: [new File(['a'], 'x.csv')] } } as unknown as Event);
    expect(cmp.error()).toBe('upload fail');
    api.datasets.and.returnValue(throwError(() => ({ message: 'list fail' })));
    cmp.reload();
    expect(cmp.error()).toContain('list fail');
  });

  it('submitForm rejects invalid JSON and API errors', () => {
    cmp.formJson = '{bad';
    cmp.submitForm();
    expect(cmp.error()).toContain('JSON');
    cmp.formJson = '{"not":"array"}';
    cmp.submitForm();
    expect(cmp.error()).toContain('array');
    cmp.formJson = '[{"patient_id":1}]';
    api.importForm.and.returnValue(throwError(() => ({ error: { detail: 'form err' } })));
    cmp.submitForm();
    expect(cmp.error()).toBe('form err');
  });

  it('submitSql imports and handles errors', () => {
    api.importSql.and.returnValue(of({ path: 'data/uploads/sql.csv' }));
    cmp.submitSql();
    expect(api.importSql).toHaveBeenCalled();
    api.importSql.and.returnValue(throwError(() => ({ message: 'sql fail' })));
    cmp.submitSql();
    expect(cmp.error()).toBe('sql fail');
  });

  it('runHealth errors and delete guards', () => {
    cmp.runHealth();
    expect(api.datasetHealth).not.toHaveBeenCalled();
    cmp.select(cmp.datasets()[0]);
    api.datasetHealth.and.returnValue(throwError(() => ({ error: { detail: 'health err' } })));
    cmp.runHealth();
    expect(cmp.error()).toBe('health err');
    spyOn(window, 'confirm').and.returnValue(false);
    cmp.checkedIds.set(new Set(['ehr_data']));
    cmp.deleteSelected();
    expect(api.deleteDataset).not.toHaveBeenCalled();
  });

  it('deleteSelected handles forkJoin error', () => {
    spyOn(window, 'confirm').and.returnValue(true);
    api.deleteDataset.and.returnValue(throwError(() => ({ error: { detail: 'boom' } })));
    cmp.select(cmp.datasets()[1]);
    cmp.deleteSelected();
    expect(cmp.error()).toContain('Failed to delete');
  });

  it('onFile ignores empty file selection', () => {
    cmp.onFile({ target: { files: [] } } as unknown as Event);
    expect(api.uploadDataset).not.toHaveBeenCalled();
  });

  it('allVisibleChecked is false when table empty', () => {
    cmp.toggleShowDemo(false);
    api.datasets.and.returnValue(of({ datasets: [], include_demo: false }));
    cmp.reload();
    expect(cmp.allVisibleChecked()).toBeFalse();
  });

  it('kindLabel uses source_type for bundled rows', () => {
    const d = { ...cmp.datasets()[0], bundled: true, source_type: 'teaching' };
    expect(cmp.kindLabel(d)).toBe('teaching');
  });

  it('unchecks rows and prunes stale checked ids on reload', () => {
    const d = cmp.datasets()[0];
    cmp.toggleCheck(d, { stopPropagation: () => undefined, target: { checked: true } } as unknown as Event);
    cmp.toggleCheck(d, { stopPropagation: () => undefined, target: { checked: false } } as unknown as Event);
    expect(cmp.isChecked(d.id)).toBeFalse();
    cmp.checkedIds.set(new Set(['ehr_data']));
    api.datasets.and.returnValue(
      of({
        datasets: [
          {
            id: 'upload:mine.csv',
            label: 'Upload: mine.csv',
            path: 'data/uploads/mine.csv',
            format: 'longitudinal',
            exists: true,
            bytes: 200,
            bundled: false,
            category: 'user',
            source_type: 'byo',
          },
        ],
        include_demo: false,
      })
    );
    cmp.reload();
    expect(cmp.checkedIds().size).toBe(0);
  });

  it('deleteSelected returns when nothing to delete', () => {
    cmp.checkedIds.set(new Set());
    cmp.selected.set(null);
    cmp.deleteSelected();
    expect(api.deleteDataset).not.toHaveBeenCalled();
  });

  it('handleDeleteJoinError sets error state', () => {
    const priv = cmp as unknown as { handleDeleteJoinError(e: { message: string }): void };
    priv.handleDeleteJoinError({ message: 'join fail' });
    expect(cmp.error()).toBe('join fail');
    expect(cmp.deleting()).toBeFalse();
  });
});
