import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { TrainComponent } from './train.component';
import { ApiService } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';

describe('TrainComponent', () => {
  let fixture: ComponentFixture<TrainComponent>;
  let cmp: TrainComponent;
  let api: jasmine.SpyObj<ApiService>;
  let queryParams: Record<string, string> = {};

  beforeEach(async () => {
    queryParams = {};
    api = jasmine.createSpyObj('ApiService', [
      'tasks',
      'train',
      'compare',
      'hpo',
      'leakageAudit',
      'job',
      'jobs',
      'cancelJob',
    ]);
    api.tasks.and.returnValue(
      of({
        tasks: [
          {
            id: 'custom',
            name: 'Custom Chronic Risk',
            description: 'Generic longitudinal chronic-disease label.',
            required_columns: ['patient_id', 'timestamp', 'label'],
            data_format: 'longitudinal',
            suggested_path: 'data/demo/ehr_data.csv',
            model_kind: 'logreg',
            target_column: 'label',
            windows_days: [7, 30, 180],
            index_strategy: 'last_event',
          },
          {
            id: 'readmission_30d',
            name: '30-Day Readmission',
            description: 'Binary risk after index_time.',
            required_columns: ['patient_id', 'timestamp', 'label', 'index_time'],
            data_format: 'longitudinal',
          },
        ],
      })
    );
    api.train.and.returnValue(
      of({ id: 't1', kind: 'train', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.compare.and.returnValue(
      of({ id: 'c1', kind: 'compare', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.hpo.and.returnValue(
      of({ id: 'h1', kind: 'hpo', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.jobs.and.returnValue(
      of({
        jobs: [{ id: 't1', kind: 'train', status: 'succeeded', message: 'ok', result: {}, log_tail: [] }],
      })
    );
    api.cancelJob.and.returnValue(
      of({ id: 't1', kind: 'train', status: 'cancelled', message: 'cancelled', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(
      of({ id: 't1', kind: 'train', status: 'succeeded', message: 'done', result: {}, log_tail: [] })
    );

    await TestBed.configureTestingModule({
      imports: [TrainComponent],
      providers: [
        { provide: ApiService, useValue: api },
        WorkspaceState,
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useFactory: () => ({
            snapshot: { queryParamMap: convertToParamMap(queryParams) },
          }),
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(TrainComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('loads tasks', () => {
    expect(api.tasks).toHaveBeenCalled();
    expect(cmp.tasks().length).toBeGreaterThan(0);
    expect(cmp.dataPath).toBe('data/demo/ehr_data.csv');
  });

  it('applies demo preset when ?demo=1 with labelCol=label', async () => {
    queryParams = { demo: '1' };
    await TestBed.resetTestingModule();
    api.tasks.and.returnValue(
      of({
        tasks: [
          {
            id: 'custom',
            name: 'Custom Chronic Risk',
            description: 'Generic longitudinal chronic-disease label.',
            required_columns: ['patient_id', 'timestamp', 'label'],
            data_format: 'longitudinal',
            suggested_path: 'data/demo/ehr_data.csv',
            model_kind: 'logreg',
            target_column: 'label',
            windows_days: [7, 30, 180],
            index_strategy: 'last_event',
          },
        ],
      })
    );
    api.jobs.and.returnValue(of({ jobs: [] }));
    await TestBed.configureTestingModule({
      imports: [TrainComponent],
      providers: [
        { provide: ApiService, useValue: api },
        WorkspaceState,
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { queryParamMap: convertToParamMap({ demo: '1' }) } },
        },
      ],
    }).compileComponents();
    const demoFixture = TestBed.createComponent(TrainComponent);
    const demoCmp = demoFixture.componentInstance;
    demoFixture.detectChanges();
    expect(demoCmp.demoBanner()).toBeTrue();
    expect(demoCmp.taskId).toBe('custom');
    expect(demoCmp.dataPath).toContain('ehr_data.csv');
    expect(demoCmp.labelCol).toBe('label');
    expect(demoFixture.nativeElement.textContent).toContain('Demo preset loaded');
  });

  it('sends label_col=label on train when demo preset applied', fakeAsync(() => {
    cmp.taskId = 'custom';
    cmp.applyTask();
    expect(cmp.labelCol).toBe('label');
    cmp.dataPath = 'data/demo/ehr_data.csv';
    cmp.startTrain();
    expect(api.train).toHaveBeenCalled();
    const payload = api.train.calls.mostRecent().args[0] as { label_col?: string | null };
    expect(payload.label_col).toBe('label');
    tick(1000);
  }));

  it('starts train job and polls', fakeAsync(() => {
    cmp.dataPath = 'data/demo/ehr_data.csv';
    cmp.startTrain();
    expect(api.train).toHaveBeenCalled();
    tick(1000);
    expect(api.job).toHaveBeenCalledWith('t1');
    expect(cmp.job()?.status).toBe('succeeded');
    expect(cmp.busy()).toBeFalse();
  }));

  it('starts compare job', () => {
    cmp.dataPath = 'data/demo/ehr_data.csv';
    cmp.startCompare();
    expect(api.compare).toHaveBeenCalled();
  });

  it('surfaces task description and required columns', () => {
    cmp.taskId = 'readmission_30d';
    cmp.applyTask();
    expect(cmp.selectedTask()?.required_columns).toContain('index_time');
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('index_time');
  });

  it('starts light HPO when enabled', () => {
    cmp.enableHpo = true;
    cmp.dataPath = 'data/demo/ehr_data.csv';
    cmp.startHpo();
    expect(api.hpo).toHaveBeenCalled();
  });

  it('loads recent jobs', () => {
    expect(api.jobs).toHaveBeenCalled();
    expect(cmp.recentJobs().length).toBeGreaterThan(0);
  });
});
