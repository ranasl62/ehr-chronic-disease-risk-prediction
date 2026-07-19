import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { TrainComponent } from './train.component';
import { ApiService } from '../../core/api.service';
import { WorkspaceState } from '../../core/workspace.state';

describe('TrainComponent', () => {
  let fixture: ComponentFixture<TrainComponent>;
  let cmp: TrainComponent;
  let api: jasmine.SpyObj<ApiService>;

  beforeEach(async () => {
    api = jasmine.createSpyObj('ApiService', ['tasks', 'train', 'compare', 'leakageAudit', 'job']);
    api.tasks.and.returnValue(
      of({ tasks: [{ id: 'demo', name: 'Demo task', data_format: 'longitudinal' }] })
    );
    api.train.and.returnValue(
      of({ id: 't1', kind: 'train', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.compare.and.returnValue(
      of({ id: 'c1', kind: 'compare', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(
      of({ id: 't1', kind: 'train', status: 'succeeded', message: 'done', result: {}, log_tail: [] })
    );

    await TestBed.configureTestingModule({
      imports: [TrainComponent],
      providers: [{ provide: ApiService, useValue: api }, WorkspaceState, provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(TrainComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('loads tasks', () => {
    expect(api.tasks).toHaveBeenCalled();
    expect(cmp.tasks().length).toBeGreaterThan(0);
  });

  it('starts train job and polls', fakeAsync(() => {
    cmp.dataPath = 'data/raw/ehr_data.csv';
    cmp.startTrain();
    expect(api.train).toHaveBeenCalled();
    tick(1000);
    expect(api.job).toHaveBeenCalledWith('t1');
    expect(cmp.job()?.status).toBe('succeeded');
    expect(cmp.busy()).toBeFalse();
  }));

  it('starts compare job', () => {
    cmp.dataPath = 'data/raw/ehr_data.csv';
    cmp.startCompare();
    expect(api.compare).toHaveBeenCalled();
  });
});
