import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { DatasetsComponent } from './datasets.component';
import { ApiService } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';

describe('DatasetsComponent', () => {
  let fixture: ComponentFixture<DatasetsComponent>;
  let cmp: DatasetsComponent;
  let api: jasmine.SpyObj<ApiService>;

  beforeEach(async () => {
    api = jasmine.createSpyObj('ApiService', [
      'datasets',
      'datasetHealth',
      'uploadDataset',
      'importForm',
      'importSql',
    ]);
    api.datasets.and.returnValue(
      of({
        datasets: [
          {
            id: 'ehr_data',
            label: 'Demo',
            path: 'data/raw/ehr_data.csv',
            format: 'longitudinal',
            exists: true,
          },
        ],
      })
    );
    api.datasetHealth.and.returnValue(
      of({
        path: 'data/raw/ehr_data.csv',
        n_rows: 20,
        n_columns: 10,
        health: { ready_for_training: true, blockers: [], warnings: [] },
      })
    );

    await TestBed.configureTestingModule({
      imports: [DatasetsComponent],
      providers: [{ provide: ApiService, useValue: api }, WorkspaceState, provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetsComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('lists datasets on init', () => {
    expect(api.datasets).toHaveBeenCalled();
    expect(cmp.datasets().length).toBe(1);
  });

  it('loads health for selected dataset', () => {
    cmp.select(cmp.datasets()[0]);
    cmp.runHealth();
    expect(api.datasetHealth).toHaveBeenCalledWith('data/raw/ehr_data.csv');
    expect(cmp.health()?.health.ready_for_training).toBeTrue();
  });

  it('imports form rows', () => {
    api.importForm.and.returnValue(of({ path: 'data/uploads/x.csv' }));
    cmp.formJson = '[{"patient_id":1,"label":0}]';
    cmp.submitForm();
    expect(api.importForm).toHaveBeenCalled();
  });
});
