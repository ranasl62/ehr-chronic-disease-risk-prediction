import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { AppComponent } from './app.component';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideRouter([]), provideHttpClient()],
    }).compileComponents();
  });

  it('should create', () => {
    const fixture = TestBed.createComponent(AppComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('shows brand text and logo mark', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.brand')).toBeTruthy();
    expect(el.querySelector('.brand-logo')).toBeTruthy();
    expect(el.querySelector('.brand-text')?.textContent).toContain('EHR Risk Workbench');
    expect(el.textContent).toContain('research and education');
  });

  it('exposes Take tour and starts page tour', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const cmp = fixture.componentInstance;
    spyOn(cmp.tour, 'startForCurrentRoute');
    fixture.detectChanges();
    const btn = (fixture.nativeElement as HTMLElement).querySelector('.tour-launch') as HTMLButtonElement;
    expect(btn?.textContent).toContain('Take tour');
    btn.click();
    expect(cmp.tour.startForCurrentRoute).toHaveBeenCalledWith(true);
  });
});
