import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DocsComponent, DOCS_SITE } from './docs.component';

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

  it('points guides at the documentation website, not GitHub Markdown blobs', () => {
    const html: string = fixture.nativeElement.innerHTML;
    expect(html).toContain(DOCS_SITE);
    expect(html).toContain(`${DOCS_SITE}/help/`);
    expect(html).toContain(`${DOCS_SITE}/limits/`);
    expect(html).toContain(`${DOCS_SITE}/data/`);
    expect(html).not.toContain('github.com/ranasl62/ehr-chronic-disease-risk-prediction/blob/');
    expect(html).not.toContain('LIMITATIONS.md');
    expect(html).not.toContain('help.html');
  });

  it('builds docs URLs with trailing slashes for Pages paths', () => {
    const cmp = fixture.componentInstance;
    expect(cmp.docsUrl('/')).toBe(`${DOCS_SITE}/`);
    expect(cmp.docsUrl('/help/')).toBe(`${DOCS_SITE}/help/`);
    expect(cmp.docsUrl('quickstart')).toBe(`${DOCS_SITE}/quickstart/`);
  });
});
