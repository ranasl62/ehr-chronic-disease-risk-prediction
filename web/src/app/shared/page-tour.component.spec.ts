import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { PageTourComponent } from './page-tour.component';
import { PageTourService } from '../core/page-tour.service';

describe('PageTourComponent', () => {
  let fixture: ComponentFixture<PageTourComponent>;
  let tour: PageTourService;

  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({
      imports: [PageTourComponent],
      providers: [provideRouter([])],
    }).compileComponents();
    fixture = TestBed.createComponent(PageTourComponent);
    tour = TestBed.inject(PageTourService);
    fixture.detectChanges();
  });

  it('renders tip when tour is active', () => {
    const el = document.createElement('h1');
    el.setAttribute('data-tour', 'home-title');
    document.body.appendChild(el);
    tour.start('home', { force: true });
    fixture.detectChanges();
    const tip = fixture.nativeElement.querySelector('.tour-tip');
    expect(tip).toBeTruthy();
    expect(tip.textContent).toContain('Workbench home');
    const next = tip.querySelector('.primary') as HTMLButtonElement;
    next.click();
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Primary actions');
    el.remove();
  });

  it('closes on Escape', () => {
    tour.start('docs', { force: true });
    fixture.detectChanges();
    expect(tour.active()).toBeTrue();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    fixture.detectChanges();
    expect(tour.active()).toBeFalse();
  });

  it('hides when inactive', () => {
    expect(fixture.nativeElement.querySelector('.tour-root')).toBeFalsy();
  });
});
