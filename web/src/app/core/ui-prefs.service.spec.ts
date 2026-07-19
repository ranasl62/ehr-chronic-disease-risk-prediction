import { TestBed } from '@angular/core/testing';
import { UiPrefsService, DEFAULT_UI_PREFS } from './ui-prefs.service';

describe('UiPrefsService', () => {
  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({});
  });

  it('loads defaults when storage empty', () => {
    const ui = TestBed.inject(UiPrefsService);
    expect(ui.prefs().theme).toBe(DEFAULT_UI_PREFS.theme);
    expect(ui.prefs().analytics_view).toBe('split');
  });

  it('patches and persists prefs', () => {
    const ui = TestBed.inject(UiPrefsService);
    ui.patch({ theme: 'slate', top_n_features: 20 });
    expect(ui.prefs().theme).toBe('slate');
    expect(ui.prefs().top_n_features).toBe(20);
    TestBed.flushEffects();
    const raw = JSON.parse(localStorage.getItem('ehr_ui_prefs_v1') || '{}');
    expect(raw.theme).toBe('slate');
  });

  it('reset restores defaults', () => {
    const ui = TestBed.inject(UiPrefsService);
    ui.patch({ analytics_view: 'tables' });
    ui.reset();
    expect(ui.prefs().analytics_view).toBe('split');
  });

  it('mergeFromWorkspace applies known keys only', () => {
    const ui = TestBed.inject(UiPrefsService);
    ui.mergeFromWorkspace({ theme: 'sand', unknown: 1 });
    expect(ui.prefs().theme).toBe('sand');
  });

  it('toWorkspaceUi exports chart toggles', () => {
    const ui = TestBed.inject(UiPrefsService);
    const out = ui.toWorkspaceUi();
    expect(out['show_label_chart']).toBeTrue();
    expect(out['density']).toBe('comfortable');
  });
});
