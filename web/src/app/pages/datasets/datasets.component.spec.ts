import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
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
});
