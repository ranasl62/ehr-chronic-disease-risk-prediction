import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of, throwError } from 'rxjs';
import { PredictComponent } from './predict.component';
import { ApiService, ModelSchema, PredictResult } from '../../core/api.service';

describe('PredictComponent', () => {
  let fixture: ComponentFixture<PredictComponent>;
  let cmp: PredictComponent;
  let api: jasmine.SpyObj<ApiService>;

  const schema: ModelSchema = {
    feature_columns: ['w7d_glucose', 'w7d_age'],
    model_kind: 'logreg',
    calibrated: false,
    input_stats: {
      w7d_glucose: { median: 120, p05: 90, p95: 160 },
      w7d_age: { median: 55 },
    },
  };

  beforeEach(async () => {
    api = jasmine.createSpyObj('ApiService', ['schema', 'predict']);
    api.schema.and.returnValue(of(schema));
    await TestBed.configureTestingModule({
      imports: [PredictComponent],
      providers: [{ provide: ApiService, useValue: api }, provideRouter([])],
    }).compileComponents();
    fixture = TestBed.createComponent(PredictComponent);
    cmp = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('loads schema and fills median defaults', () => {
    expect(cmp.schema()?.feature_columns.length).toBe(2);
    expect(cmp.values['w7d_glucose']).toBe(120);
    expect(cmp.values['w7d_age']).toBe(55);
    expect(fixture.nativeElement.textContent).toContain('Predict');
  });

  it('submits prediction and shows risk', () => {
    const result: PredictResult = {
      risk_probability: 0.42,
      risk_level: 'medium',
      explanation: {
        method: 'shap',
        top_positive_risk_drivers: [
          { feature: 'w7d_age', shap_value: 0.1, abs_contribution: 0.1 },
        ],
      },
    };
    api.predict.and.returnValue(of(result));
    cmp.submit();
    expect(api.predict).toHaveBeenCalled();
    expect(cmp.result()?.risk_level).toBe('medium');
    expect(cmp.drivers().length).toBe(1);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('medium');
  });

  it('surfaces predict errors with train CTA when no schema', () => {
    api.schema.and.returnValue(throwError(() => ({ error: { detail: 'no model' }, message: 'err' })));
    const f2 = TestBed.createComponent(PredictComponent);
    f2.detectChanges();
    expect(f2.componentInstance.error()).toContain('no model');
  });

  it('groups features by window', () => {
    expect(cmp.featureGroups().some((g) => g.window === 'w7d')).toBeTrue();
  });

  it('fillMedians restores stats', () => {
    cmp.values['w7d_glucose'] = 999;
    cmp.fillMedians();
    expect(cmp.values['w7d_glucose']).toBe(120);
  });

  it('statsHint and empty feature groups without schema', () => {
    expect(cmp.statsHint('w7d_glucose')).toContain('med 120');
    expect(cmp.statsHint('missing')).toBe('');
    cmp.schema.set(null);
    expect(cmp.featureGroups()).toEqual([]);
    cmp.fillMedians();
    expect(cmp.drivers()).toEqual([]);
  });

  it('drivers from shap_vector fallback', () => {
    cmp.result.set({
      risk_probability: 0.5,
      risk_level: 'medium',
      explanation: { shap_vector: { w7d_age: 0.2, w7d_glucose: -0.1 } },
    });
    const drv = cmp.drivers();
    expect(drv.length).toBe(2);
    expect(drv[0].feature).toBe('w7d_age');
  });

  it('submit restores medians when reuseAsNextInput false', () => {
    api.predict.and.returnValue(
      of({ risk_probability: 0.1, risk_level: 'low', explanation: {} })
    );
    cmp.reuseAsNextInput = false;
    cmp.values['w7d_glucose'] = 999;
    cmp.submit();
    expect(cmp.values['w7d_glucose']).toBe(120);
  });

  it('submit and schema errors use fmtErr branches', () => {
    api.predict.and.returnValue(throwError(() => ({ error: { detail: [{ msg: 'bad feat' }] } })));
    cmp.submit();
    expect(cmp.error()).toContain('bad feat');
    api.schema.and.returnValue(throwError(() => ({ message: 'fallback' })));
    const f2 = TestBed.createComponent(PredictComponent);
    f2.detectChanges();
    expect(f2.componentInstance.error()).toContain('fallback');
  });

  it('renders charts after prediction', fakeAsync(() => {
    api.predict.and.returnValue(
      of({
        risk_probability: 0.42,
        risk_level: 'medium',
        explanation: {
          top_positive_risk_drivers: [{ feature: 'w7d_age', shap_value: 0.1, abs_contribution: 0.1 }],
        },
      })
    );
    for (const name of ['risk', 'shap', 'vs']) {
      const c = document.createElement('canvas');
      c.setAttribute('data-chart', name);
      fixture.nativeElement.appendChild(c);
    }
    cmp.submit();
    tick(50);
    fixture.destroy();
  }));

  it('riskPct returns zero without result', () => {
    cmp.result.set(null);
    expect(cmp.riskPct()).toBe(0);
  });

  it('filters feature groups by query', () => {
    cmp.featureQuery = 'glucose';
    const groups = cmp.featureGroups();
    expect(groups.every((g) => g.cols.every((c) => c.includes('glucose')))).toBeTrue();
  });

  it('sorts drivers by absolute shap contribution', () => {
    cmp.result.set({
      risk_probability: 0.5,
      risk_level: 'medium',
      explanation: {
        top_positive_risk_drivers: [
          { feature: 'a', shap_value: 0.1, abs_contribution: 0.1 },
          { feature: 'b', shap_value: -0.3, abs_contribution: 0.3 },
        ],
      },
    });
    expect(cmp.drivers()[0].feature).toBe('b');
  });

  it('covers chart helper edge paths', fakeAsync(() => {
    const priv = cmp as unknown as {
      canvas(ref: unknown, name: string): HTMLCanvasElement | undefined;
      push(el: HTMLCanvasElement | undefined, cfg: object): void;
      drawCharts(): void;
    };
    const c = document.createElement('canvas');
    c.setAttribute('data-chart', 'risk');
    fixture.nativeElement.appendChild(c);
    expect(priv.canvas(undefined, 'risk')).toBeTruthy();
    priv.push(undefined, { type: 'bar', data: { labels: [], datasets: [] } });
    priv.push(c, { type: 'bar', data: { labels: ['a'], datasets: [{ data: [1] }] } });
    priv.push(c, { type: 'bar', data: { labels: ['b'], datasets: [{ data: [2] }] } });
    cmp.result.set(null);
    priv.drawCharts();
    tick(10);
  }));
});
