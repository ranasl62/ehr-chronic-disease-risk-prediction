import { Component } from '@angular/core';

/** Published documentation site (https://ehr.larucare.com). */
export const DOCS_SITE = 'https://ehr.larucare.com';

export type DocsGuide = Readonly<{ label: string; path: string; blurb: string }>;

export type DocsGuideGroup = Readonly<{ title: string; items: ReadonlyArray<DocsGuide> }>;

@Component({
  selector: 'app-docs',
  standalone: true,
  templateUrl: './docs.component.html',
  styleUrl: './docs.component.css',
})
export class DocsComponent {
  readonly docsSite = DOCS_SITE;

  /** Primary shortcuts shown as buttons (labels only — no raw paths). */
  readonly shortcuts: ReadonlyArray<DocsGuide> = [
    { label: 'How it works A–Z', path: '/guide/', blurb: 'Full walkthrough with screenshots' },
    { label: 'Docs home', path: '/', blurb: 'Overview and getting started' },
    { label: 'Quickstart', path: '/quickstart/', blurb: 'First successful train loop' },
    { label: 'UI tour', path: '/workbench/', blurb: 'Screenshots of every route' },
  ];

  /** Grouped guides — link text is the label; paths stay in href only. */
  readonly guideGroups: ReadonlyArray<DocsGuideGroup> = [
    {
      title: 'Get started',
      items: [
        {
          label: 'How it works A–Z',
          path: '/guide/',
          blurb: 'Detailed walkthrough with every screenshot — start here',
        },
        { label: 'Quickstart', path: '/quickstart/', blurb: 'Docker → first train → predict' },
        { label: 'Install & Docker images', path: '/docker-images/', blurb: 'Compose and Hub pulls' },
        { label: 'Commands', path: '/commands/', blurb: 'Make, curl, and CLI recipes' },
        { label: 'Workbench UI tour', path: '/workbench/', blurb: 'Every in-app screen with screenshots' },
      ],
    },
    {
      title: 'Concepts & methods',
      items: [
        { label: 'Why this framework', path: '/why/', blurb: 'Audience and research fit' },
        { label: 'Features', path: '/features/', blurb: 'Capability catalog from ingest to serve' },
        { label: 'Architecture', path: '/architecture/', blurb: 'Components, jobs, and config' },
        { label: 'Diagrams', path: '/diagrams/', blurb: 'Data flow and temporal splits' },
        {
          label: 'Prevent data leakage',
          path: '/blog/prevent-data-leakage-clinical-ai/',
          blurb: 'Index time, splits, and audits',
        },
        { label: 'Compare vs notebooks', path: '/compare/vs-ad-hoc-notebooks/', blurb: 'When to use the workbench' },
      ],
    },
    {
      title: 'Data, training & results',
      items: [
        { label: 'Data guide & schema', path: '/data/', blurb: 'Demo paths, integrity, columns' },
        { label: 'Fine-tuning', path: '/fine-tuning/', blurb: 'Compare, promote, and iterate' },
        { label: 'API overview', path: '/api/', blurb: 'Researcher endpoints' },
        { label: 'Blog & tutorials', path: '/blog/', blurb: 'Leakage guide and risk-model quickstart' },
      ],
    },
    {
      title: 'Reference',
      items: [
        { label: 'Help', path: '/help/', blurb: 'Full documentation library index' },
        { label: 'Limitations & model card', path: '/limits/', blurb: 'Honest non-goals and safety' },
        { label: 'Cite & feedback', path: '/cite/', blurb: 'Citation metadata and how to reach us' },
        { label: 'Sitemap', path: '/sitemap/', blurb: 'Every public docs page' },
      ],
    },
  ];

  docsUrl(path: string): string {
    if (!path || path === '/') {
      return `${this.docsSite}/`;
    }
    const normalized = path.startsWith('/') ? path : `/${path}`;
    return `${this.docsSite}${normalized.endsWith('/') ? normalized : `${normalized}/`}`;
  }
}
