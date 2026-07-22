import { Component } from '@angular/core';

/** Published documentation site (https://ehr.larucare.com). */
export const DOCS_SITE = 'https://ehr.larucare.com';

@Component({
  selector: 'app-docs',
  standalone: true,
  templateUrl: './docs.component.html',
  styleUrl: './docs.component.css',
})
export class DocsComponent {
  readonly docsSite = DOCS_SITE;

  /** Map in-app guide labels to documentation website paths. */
  readonly guides: ReadonlyArray<{ label: string; path: string; blurb: string }> = [
    { label: 'Documentation home', path: '/', blurb: 'Overview and getting started' },
    { label: 'Help library', path: '/help/', blurb: 'Full guide index matching this page' },
    { label: 'Blog & tutorials', path: '/blog/', blurb: 'Leakage guide and risk-model quickstart' },
    { label: 'How it helps / feedback', path: '/cite/', blurb: 'Citation and feedback' },
    { label: 'Architecture', path: '/architecture/', blurb: 'Components, jobs, config' },
    { label: 'Diagrams', path: '/diagrams/', blurb: 'Data flow and temporal splits' },
    { label: 'Quickstart', path: '/quickstart/', blurb: 'First successful train loop' },
    { label: 'Install & Docker', path: '/docker-images/', blurb: 'Compose and Hub images' },
    { label: 'Commands', path: '/commands/', blurb: 'Make, curl, and CLI recipes' },
    { label: 'Why this framework', path: '/why/', blurb: 'Audience and research fit' },
    { label: 'Features', path: '/features/', blurb: 'Capability catalog' },
    { label: 'Workbench UI tour', path: '/workbench/', blurb: 'Screenshots of every route' },
    { label: 'Compare vs notebooks', path: '/compare/vs-ad-hoc-notebooks/', blurb: 'When to use the workbench' },
    { label: 'Limitations & model card', path: '/limits/', blurb: 'Honest non-goals and safety' },
    { label: 'Data guide & schema', path: '/data/', blurb: 'Demo paths, integrity, columns' },
    { label: 'Fine-tuning', path: '/fine-tuning/', blurb: 'Compare, promote, iterate' },
    { label: 'API overview', path: '/api/', blurb: 'Researcher endpoints' },
    { label: 'Cite', path: '/cite/', blurb: 'Citation metadata' },
    { label: 'Full sitemap', path: '/sitemap/', blurb: 'Every public docs page' },
  ];

  docsUrl(path: string): string {
    if (!path || path === '/') {
      return `${this.docsSite}/`;
    }
    const normalized = path.startsWith('/') ? path : `/${path}`;
    return `${this.docsSite}${normalized.endsWith('/') ? normalized : `${normalized}/`}`;
  }
}
