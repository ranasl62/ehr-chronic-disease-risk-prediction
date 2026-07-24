import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of, throwError } from 'rxjs';
import { ResearchWizardComponent } from './research-wizard.component';
import { ApiService } from '../../core/api.service';

describe('ResearchWizardComponent', () => {
  let fixture: ComponentFixture<ResearchWizardComponent>;
  let cmp: ResearchWizardComponent;
  let api: jasmine.SpyObj<ApiService>;

  beforeEach(async () => {
    api = jasmine.createSpyObj('ApiService', [
      'datasets',
      'tasks',
      'workspaceStatus',
      'reportsSummary',
      'datasetHealth',
      'train',
      'job',
      'runDetail',
      'leakageAudit',
      'externalValidate',
      'resultsZipUrl',
      'methodsMdUrl',
    ]);
    api.datasets.and.returnValue(
      of({
        datasets: [
          {
            id: 'demo',
            label: 'Demo',
            path: 'data/demo/ehr_data.csv',
            format: 'longitudinal',
            exists: true,
          },
          {
            id: 'paper_synthetic',
            label: 'Paper synthetic',
            path: 'data/raw/paper_synthetic_cohort.csv',
            format: 'longitudinal',
            exists: true,
          },
        ],
      })
    );
    api.tasks.and.returnValue(
      of({
        tasks: [
          {
            id: 'horizon_detection_30d',
            name: 'H30',
            index_strategy: 'column',
            index_time_col: 'index_time',
            suggested_path: 'data/raw/paper_synthetic_cohort.csv',
            required_columns: ['patient_id', 'timestamp', 'label', 'index_time'],
          },
        ],
      })
    );
    api.workspaceStatus.and.returnValue(
      of({
        api_ok: true,
        model_ready: false,
        evaluation_present: false,
        leakage_audit_present: false,
        shap_present: false,
        calibration_present: false,
        demo_datasets_available: true,
        checklist: {},
        recent_jobs: [],
      })
    );
    api.reportsSummary.and.returnValue(of({ files: [], quality_note: 'note' }));
    api.datasetHealth.and.returnValue(
      of({
        path: 'data/demo/ehr_data.csv',
        n_rows: 20,
        n_columns: 10,
        health: { ready_for_training: true, blockers: [], warnings: [] },
      })
    );
    api.train.and.returnValue(
      of({ id: 'j1', kind: 'train', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(
      of({
        id: 'j1',
        kind: 'train',
        status: 'succeeded',
        message: 'ok',
        result: { run_id: 'run_a' },
        log_tail: [],
      })
    );
    api.runDetail.and.returnValue(
      of({
        run_id: 'run_a',
        path: 'reports/runs/run_a',
        has_model: true,
        trust_pack: { trust_complete: true },
      } as any)
    );
    api.leakageAudit.and.returnValue(
      of({ id: 'j2', kind: 'leakage', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.externalValidate.and.returnValue(
      of({ id: 'j3', kind: 'external', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.resultsZipUrl.and.returnValue('/zip');
    api.methodsMdUrl.and.returnValue('/methods');

    await TestBed.configureTestingModule({
      imports: [ResearchWizardComponent],
      providers: [{ provide: ApiService, useValue: api }, provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(ResearchWizardComponent);
    cmp = fixture.componentInstance;
  });

  it('loads datasets and navigates steps', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    expect(api.datasets).toHaveBeenCalled();
    // Horizon task pairs with paper_synthetic (needs index_time), not tiny demo.
    expect(cmp.path).toContain('paper_synthetic');
    expect(cmp.taskId).toBe('horizon_detection_30d');
    cmp.go('health');
    expect(cmp.step()).toBe('health');
    cmp.runHealth();
    tick();
    expect(api.datasetHealth).toHaveBeenCalled();
    expect(cmp.step()).toBe('train');
  }));

  it('surfaces health blockers without advancing', fakeAsync(() => {
    api.datasetHealth.and.returnValue(
      of({
        path: 'data/demo/ehr_data.csv',
        n_rows: 20,
        n_columns: 10,
        health: {
          ready_for_training: false,
          blockers: ['missing required columns for task: index_time'],
          warnings: [],
        },
      })
    );
    fixture.detectChanges();
    tick();
    cmp.path = 'data/demo/ehr_data.csv';
    cmp.go('health');
    cmp.runHealth();
    tick();
    expect(cmp.step()).toBe('health');
    expect(cmp.healthReady()).toBeFalse();
    expect(cmp.healthBlockers().length).toBeGreaterThan(0);
    expect(cmp.error() || '').toMatch(/Not ready|index_time/i);
    expect(cmp.canProceedToTrain()).toBeFalse();
  }));

  it('keeps Next: train disabled until health is ready', fakeAsync(() => {
    api.datasetHealth.and.returnValue(
      of({
        path: 'data/demo/ehr_data.csv',
        n_rows: 20,
        n_columns: 10,
        health: { ready_for_training: false, blockers: ['need more rows'], warnings: [] },
      })
    );
    fixture.detectChanges();
    tick();
    cmp.go('health');
    fixture.detectChanges();
    const nextBtn = fixture.nativeElement.querySelector('[data-tour="wizard-next-train"]') as HTMLButtonElement;
    expect(nextBtn).toBeTruthy();
    expect(nextBtn.disabled).toBeTrue();
    cmp.runHealth();
    tick();
    fixture.detectChanges();
    expect(cmp.step()).toBe('health');
    expect(nextBtn.disabled).toBeTrue();
    cmp.go('train');
    expect(cmp.step()).toBe('health');
    expect(cmp.error() || '').toMatch(/health check/i);
  }));

  it('offers paper_synthetic and custom recovery actions', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    cmp.path = 'data/demo/ehr_data.csv';
    cmp.taskId = 'horizon_detection_30d';
    cmp.go('health');
    api.datasetHealth.and.returnValue(
      of({
        path: 'data/demo/ehr_data.csv',
        n_rows: 20,
        n_columns: 10,
        health: {
          ready_for_training: false,
          blockers: ['missing required columns for task: index_time'],
          warnings: ['tiny_cohort'],
          checks: [{ name: 'index_time', ok: false, detail: 'missing', blocking: true }],
        },
      })
    );
    cmp.runHealth();
    tick();
    fixture.detectChanges();
    expect(cmp.needsIndexTime()).toBeTrue();
    expect(cmp.healthWarnings().length).toBeGreaterThan(0);
    expect(cmp.healthChecks().some((c) => c.name === 'index_time')).toBeTrue();
    cmp.usePaperSynthetic();
    expect(cmp.path).toContain('paper_synthetic');
    expect(cmp.health()).toBeNull();
    cmp.useCustomTask();
    expect(cmp.taskId).toBe('custom');
  }));

  it('defaults model options to API-accepted kinds', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    cmp.go('data');
    fixture.detectChanges();
    const opts = Array.from(
      fixture.nativeElement.querySelectorAll('select[name="mk"] option') as NodeListOf<HTMLOptionElement>
    ).map((o) => o.value);
    expect(opts).toContain('logreg');
    expect(opts).toContain('random_forest');
    expect(opts).toContain('xgboost');
    expect(opts).not.toContain('rf');
  }));

  it('trains and loads trust', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    // Simulate passed health so train gate opens.
    cmp.health.set({
      health: { ready_for_training: true, blockers: [], warnings: [] },
    });
    cmp.go('train');
    cmp.startTrain();
    tick(900);
    expect(api.train).toHaveBeenCalled();
    expect(cmp.runId()).toBe('run_a');
    expect(cmp.step()).toBe('trust');
    cmp.loadTrust();
    tick();
    expect(api.runDetail).toHaveBeenCalledWith('run_a');
    expect(cmp.zipUrl()).toBe('/zip');
    expect(cmp.methodsUrl()).toBe('/methods');
  }));

  it('runs leakage and external jobs', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    cmp.runId.set('run_a');
    let calls = 0;
    api.job.and.callFake(() => {
      calls += 1;
      return of({
        id: 'jx',
        kind: 'x',
        status: calls === 1 ? 'running' : 'succeeded',
        message: '',
        result: {},
        log_tail: [],
      });
    });
    cmp.runLeakage();
    tick(900);
    tick(900);
    expect(api.leakageAudit).toHaveBeenCalled();
    expect(cmp.step()).toBe('external');

    calls = 0;
    cmp.runExternal();
    tick(900);
    tick(900);
    expect(api.externalValidate).toHaveBeenCalled();
    expect(cmp.step()).toBe('export');
  }));

  it('surfaces train errors', fakeAsync(() => {
    api.train.and.returnValue(throwError(() => ({ error: { detail: 'busy' } })));
    fixture.detectChanges();
    tick();
    cmp.health.set({ health: { ready_for_training: true, blockers: [], warnings: [] } });
    cmp.startTrain();
    tick();
    expect(cmp.error()).toContain('busy');
  }));

  it('covers health/train/leakage/external/trust edge paths', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    api.datasets.and.returnValue(throwError(() => ({ message: 'ds fail' })));
    api.tasks.and.returnValue(throwError(() => ({ message: 't fail' })));
    api.workspaceStatus.and.returnValue(throwError(() => ({ message: 'ws' })));
    api.reportsSummary.and.returnValue(throwError(() => ({ message: 'rs' })));
    cmp.ngOnInit();
    tick();
    expect(cmp.error()).toContain('ds fail');

    api.datasetHealth.and.returnValue(
      of({
        path: 'x',
        n_rows: 1,
        n_columns: 1,
        health: { ready_for_training: false, blockers: ['x'], warnings: [] },
      })
    );
    cmp.runHealth();
    tick();
    expect(cmp.step()).not.toBe('train');

    api.datasetHealth.and.returnValue(throwError(() => ({ error: { detail: 'health bad' } })));
    cmp.runHealth();
    tick();
    expect(cmp.error()).toContain('health bad');

    cmp.runId.set(null);
    cmp.loadTrust();
    expect(cmp.trustNote() || '').toContain('No run_id');

    cmp.runId.set('run_a');
    api.runDetail.and.returnValue(of({ run_id: 'run_a', path: 'p', has_model: true, trust: { ok: 1 } } as any));
    cmp.loadTrust();
    tick();
    expect(cmp.trustNote() || '').toContain('ok');

    api.runDetail.and.returnValue(throwError(() => ({ message: 'trust fail' })));
    cmp.loadTrust();
    tick();
    expect(cmp.trustNote() || '').toContain('trust fail');

    cmp.health.set({ health: { ready_for_training: true, blockers: [], warnings: [] } });
    api.job.and.returnValue(
      of({ id: 'j', kind: 't', status: 'cancelled', message: 'x', result: {}, log_tail: [] })
    );
    cmp.startTrain();
    tick(900);
    expect(cmp.error() || '').toMatch(/cancelled/i);

    expect(cmp['fmt']('plain')).toBe('plain');
    expect(cmp['fmt']({ message: 'm' })).toBe('m');
    expect(
      cmp['fmt']({
        error: { detail: { message: 'blocked', blockers: ['a'], hint: 'use paper' } },
      })
    ).toMatch(/blocked.*a.*paper/i);

    api.train.and.returnValue(
      of({ id: 'j9', kind: 'train', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    api.job.and.returnValue(
      of({ id: 'j9', kind: 'train', status: 'succeeded', message: '', result: {}, log_tail: [] })
    );
    cmp.startTrain();
    tick(900);
    expect(cmp.log().some((l) => l.includes('Train finished'))).toBeTrue();

    api.leakageAudit.and.returnValue(throwError(() => ({ message: 'leak fail' })));
    cmp.runLeakage();
    tick();
    expect(cmp.error()).toContain('leak fail');

    api.externalValidate.and.returnValue(throwError(() => ({ message: 'ext fail' })));
    cmp.runExternal();
    tick();
    expect(cmp.error()).toContain('ext fail');

    api.job.and.returnValue(throwError(() => ({ message: 'poll fail' })));
    api.leakageAudit.and.returnValue(
      of({ id: 'jl', kind: 'leakage', status: 'queued', message: '', result: {}, log_tail: [] })
    );
    cmp.runLeakage();
    tick(900);
    expect(cmp.error()).toContain('poll fail');

    fixture.destroy();
  }));
});
