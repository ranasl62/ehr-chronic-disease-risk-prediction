import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DocsComponent } from './docs.component';

describe('DocsComponent', () => {
  let fixture: ComponentFixture<DocsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DocsComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(DocsComponent);
    fixture.detectChanges();
  });

  it('renders documentation panel', () => {
    expect(fixture.componentInstance).toBeTruthy();
    expect(fixture.nativeElement.textContent.length).toBeGreaterThan(20);
  });
});
