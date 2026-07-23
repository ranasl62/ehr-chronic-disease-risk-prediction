import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of, throwError } from 'rxjs';
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
      'datasetHealth',
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
          {
            id: 'horizon_detection_30d',
            name: 'Horizon Outcome Detection (30d)',
            description: 'Research detection within 30 days of index.',
            required_columns: ['patient_id', 'timestamp', 'label', 'index_time'],
            data_format: 'longitudinal',
            suggested_path: 'data/demo/ehr_data.csv',
            horizon_days: 30,
            index_strategy: 'column',
            index_time_col: 'index_time',
          },
          {
            id: 'teaching_leaky_contrast',
            name: 'Teaching — Leaky ICD Contrast',
            description: 'Education-only leaky ICD demo.',
            required_columns: ['patient_id', 'timestamp', 'label', 'index_time'],
            data_format: 'longitudinal',
            suggested_path: 'data/demo/teaching_leaky_contrast.csv',
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

  it('applies horizon detection task and checks task health', fakeAsync(() => {
    api.datasetHealth.and.returnValue(
      of({
        path: 'data/demo/ehr_data.csv',
        n_rows: 10,
        n_columns: 5,
        health: {
          ready_for_training: false,
          blockers: ['task_required_columns: missing index_time'],
          warnings: [],
        },
      })
    );
    fixture.detectChanges();
    cmp.taskId = 'horizon_detection_30d';
    cmp.applyTask();
    fixture.detectChanges();
    expect(cmp.selectedTask()?.id).toBe('horizon_detection_30d');
    expect(fixture.nativeElement.textContent).toContain('Research detection');
    cmp.checkTaskHealth();
    expect(api.datasetHealth).toHaveBeenCalledWith('data/demo/ehr_data.csv', 'horizon_detection_30d');
    expect(cmp.taskHealthBlockers().length).toBeGreaterThan(0);
  }));

  it('lists teaching leaky contrast task', fakeAsync(() => {
    fixture.detectChanges();
    expect(cmp.tasks().some((t) => t.id === 'teaching_leaky_contrast')).toBeTrue();
  }));

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

  it('hydrates from workspace state when not demo', async () => {
    await TestBed.resetTestingModule();
    const state = new WorkspaceState();
    state.selectedDataset.set({
      id: 'x',
      label: 'X',
      path: 'data/uploads/x.csv',
      format: 'tabular',
      exists: true,
      suggested: {
        horizon_days: 30,
        windows_days: [7, 14],
        index_strategy: 'column',
        index_time_col: 'index_time',
      },
    });
    await TestBed.configureTestingModule({
      imports: [TrainComponent],
      providers: [
        { provide: ApiService, useValue: api },
        { provide: WorkspaceState, useValue: state },
        provideRouter([]),
        { provide: ActivatedRoute, useValue: { snapshot: { queryParamMap: convertToParamMap({}) } } },
      ],
    }).compileComponents();
    const f = TestBed.createComponent(TrainComponent);
    const c = f.componentInstance;
    f.detectChanges();
    expect(c.dataPath).toBe('data/uploads/x.csv');
    expect(c.dataFormat).toBe('tabular');
    expect(c.horizonDays).toBe(30);
    expect(c.indexTimeCol).toBe('index_time');
  });

  it('applyDemoPreset uses fallback when custom task missing', async () => {
    queryParams = { demo: '1' };
    await TestBed.resetTestingModule();
    api.tasks.and.returnValue(of({ tasks: [] }));
    api.jobs.and.returnValue(of({ jobs: [] }));
    await TestBed.configureTestingModule({
      imports: [TrainComponent],
      providers: [
        { provide: ApiService, useValue: api },
        WorkspaceState,
        provideRouter([]),
        { provide: ActivatedRoute, useValue: { snapshot: { queryParamMap: convertToParamMap({ demo: '1' }) } } },
      ],
    }).compileComponents();
    const f = TestBed.createComponent(TrainComponent);
    const c = f.componentInstance;
    f.detectChanges();
    expect(c.labelCol).toBe('label');
    expect(c.dataPath).toContain('ehr_data.csv');
  });

  it('checkTaskHealth no-ops without data path', () => {
    cmp.dataPath = '';
    cmp.checkTaskHealth();
    expect(api.datasetHealth).not.toHaveBeenCalled();
  });

  it('applyTask no-ops for unknown task', () => {
    cmp.taskId = 'missing';
    cmp.dataPath = 'unchanged';
    cmp.applyTask();
    expect(cmp.dataPath).toBe('unchanged');
  });

  it('startAudit and cancelActive', fakeAsync(() => {
    api.leakageAudit.and.returnValue(
      of({ id: 'a1', kind: 'leakage_audit', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(
      of({ id: 'a1', kind: 'leakage_audit', status: 'succeeded', message: '', result: {}, log_tail: [] })
    );
    cmp.startAudit();
    tick(1000);
    expect(api.leakageAudit).toHaveBeenCalled();
    cmp.job.set({ id: 'a1', kind: 'leakage_audit', status: 'running', message: '', result: {}, log_tail: [] });
    cmp.cancelActive();
    expect(api.cancelJob).toHaveBeenCalledWith('a1');
  }));

  it('cancelActive skips when no active job', () => {
    cmp.job.set(null);
    cmp.cancelActive();
    expect(api.cancelJob).not.toHaveBeenCalled();
  });

  it('surfaces errors from train compare hpo and health', fakeAsync(() => {
    api.train.and.returnValue(throwError(() => ({ status: 401 })));
    cmp.startTrain();
    expect(cmp.error()).toContain('401');

    api.compare.and.returnValue(throwError(() => ({ error: { detail: 'compare fail' } })));
    cmp.startCompare();
    expect(cmp.error()).toBe('compare fail');

    api.hpo.and.returnValue(throwError(() => ({ error: { detail: { message: 'hpo', blockers: ['x'] } } })));
    cmp.startHpo();
    expect(cmp.error()).toContain('hpo');

    api.datasetHealth.and.returnValue(throwError(() => ({ error: { detail: [{ msg: 'bad' }] } })));
    cmp.dataPath = 'data/demo/ehr_data.csv';
    cmp.checkTaskHealth();
    expect(cmp.error()).toContain('bad');

    api.jobs.and.returnValue(throwError(() => new Error('jobs')));
    cmp.refreshJobs();

    api.train.and.returnValue(
      of({ id: 't2', kind: 'train', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(throwError(() => ({ message: 'poll fail' })));
    cmp.startTrain();
    tick(1000);
    expect(cmp.error()).toContain('poll fail');
  }));

  it('clears error when health has no blockers', () => {
    cmp.error.set('old');
    api.datasetHealth.and.returnValue(
      of({ path: 'x', n_rows: 1, n_columns: 1, health: { ready_for_training: true, blockers: [], warnings: [] } })
    );
    cmp.dataPath = 'data/demo/ehr_data.csv';
    cmp.checkTaskHealth();
    expect(cmp.error()).toBeNull();
  });

  it('cancelActive surfaces errors and poll marks failed jobs', fakeAsync(() => {
    api.cancelJob.and.returnValue(throwError(() => ({ error: { detail: 'cancel fail' } })));
    cmp.job.set({ id: 't1', kind: 'train', status: 'running', message: '', result: {}, log_tail: [] });
    cmp.cancelActive();
    expect(cmp.error()).toBe('cancel fail');

    api.train.and.returnValue(
      of({ id: 't3', kind: 'train', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(
      of({ id: 't3', kind: 'train', status: 'failed', message: 'bad', result: {}, log_tail: [] })
    );
    cmp.startTrain();
    tick(1000);
    expect(cmp.busy()).toBeFalse();
    expect(cmp.job()?.status).toBe('failed');
  }));

  it('startAudit surfaces API errors', () => {
    api.leakageAudit.and.returnValue(throwError(() => ({ error: { detail: 'audit fail' } })));
    cmp.startAudit();
    expect(cmp.error()).toBe('audit fail');
    expect(cmp.busy()).toBeFalse();
  });
});
