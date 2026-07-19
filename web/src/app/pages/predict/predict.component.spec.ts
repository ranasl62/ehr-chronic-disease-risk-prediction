import { ComponentFixture, TestBed } from '@angular/core/testing';
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
});
