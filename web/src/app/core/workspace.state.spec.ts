import { TestBed } from '@angular/core/testing';
import { WorkspaceState } from './workspace.state';

describe('WorkspaceState', () => {
  let state: WorkspaceState;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    state = TestBed.inject(WorkspaceState);
  });

  it('tracks dataset and job selection', () => {
    expect(state.selectedDataset()).toBeNull();
    expect(state.lastJobId()).toBeNull();
    state.selectedDataset.set({ id: 'x', label: 'X', path: 'p', format: 'longitudinal', exists: true });
    state.lastJobId.set('job-1');
    expect(state.selectedDataset()?.id).toBe('x');
    expect(state.lastJobId()).toBe('job-1');
  });
});
