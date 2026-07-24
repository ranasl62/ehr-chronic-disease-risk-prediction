/** Guided tour copy + selectors for each workbench page. */

export type TourPageId =
  | 'home'
  | 'research'
  | 'datasets'
  | 'train'
  | 'results'
  | 'analytics'
  | 'config'
  | 'predict'
  | 'docs';

export interface TourStep {
  id: string;
  title: string;
  body: string;
  /** CSS selector for spotlight; omit for centered tip. */
  selector?: string;
  /** Navigate here before highlighting (same or other page). */
  route?: string;
  placement?: 'top' | 'bottom' | 'left' | 'right';
}

const DISCLAIMER =
  'For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future.';

export const TOUR_ROUTE_TO_PAGE: Record<string, TourPageId> = {
  '/': 'home',
  '/research': 'research',
  '/datasets': 'datasets',
  '/train': 'train',
  '/results': 'results',
  '/analytics': 'analytics',
  '/config': 'config',
  '/predict': 'predict',
  '/docs': 'docs',
};

export const PAGE_TOURS: Record<TourPageId, TourStep[]> = {
  home: [
    {
      id: 'home-intro',
      title: 'Workbench home',
      body: `Check readiness: API, demo data, model.pkl, metrics, leakage audit, and a real SHAP PNG (stubs don’t count). ${DISCLAIMER}`,
      selector: '[data-tour="home-title"]',
      route: '/',
      placement: 'bottom',
    },
    {
      id: 'home-actions',
      title: 'Primary actions',
      body: 'Start with Research wizard for a full methods loop, or Run demo train for a quick smoke path. ZIP packs reports for your appendix.',
      selector: '[data-tour="home-actions"]',
      placement: 'bottom',
    },
    {
      id: 'home-wizard',
      title: 'Research wizard',
      body: 'Guided path: dataset → health → train → trust → leakage → external val → export. Same jobs as Train / Results.',
      selector: '[data-tour="home-wizard"]',
      placement: 'bottom',
    },
    {
      id: 'home-demo',
      title: 'Demo train',
      body: 'Tiny cohort smoke path — expect limited ROC/PR. Prefer paper_synthetic later for paper-style curves.',
      selector: '[data-tour="home-demo"]',
      placement: 'bottom',
    },
    {
      id: 'home-to-results',
      title: 'Where metrics land',
      body: 'After train: Results for tables & figures; Analytics for cohort charts and ROC/PR when curves exist.',
      route: '/results',
      selector: '[data-tour="results-title"]',
      placement: 'bottom',
    },
  ],
  research: [
    {
      id: 'research-intro',
      title: 'Research wizard',
      body: `Walk a methods-style study step by step — no endpoint hunting. ${DISCLAIMER}`,
      selector: '[data-tour="research-title"]',
      route: '/research',
      placement: 'bottom',
    },
    {
      id: 'research-steps',
      title: 'Stepper',
      body: 'Pick data & task → health (must pass) → train → trust → leakage → external → ZIP / Analytics.',
      selector: '[data-tour="research-steps"]',
      placement: 'bottom',
    },
    {
      id: 'research-health-next',
      title: 'Health → train',
      body: 'Run health check first. Blockers (often missing index_time) stay loud until fixed — Next: train stays disabled. Use paper_synthetic for horizon tasks.',
      selector: '[data-tour="wizard-next-train"]',
      placement: 'top',
    },
    {
      id: 'research-artifacts',
      title: 'Artifacts you create',
      body: 'Train writes reports/runs/. Trust pack hashes flags. SHAP & leakage update the Home checklist.',
      route: '/results',
      selector: '[data-tour="results-metrics"]',
      placement: 'bottom',
    },
  ],
  datasets: [
    {
      id: 'datasets-intro',
      title: 'Datasets',
      body: `Pick bundled demos or import BYO CSV. paper_synthetic includes index_time for horizon tasks. ${DISCLAIMER}`,
      selector: '[data-tour="datasets-title"]',
      route: '/datasets',
      placement: 'bottom',
    },
    {
      id: 'datasets-list',
      title: 'Catalog',
      body: 'ehr_data = tiny smoke test. paper_synthetic = larger cohort with usable ROC/PR. Select a row to send into Train.',
      selector: '[data-tour="datasets-list"]',
      placement: 'top',
    },
    {
      id: 'datasets-to-train',
      title: 'Next: configure train',
      body: 'Open Train (or Research wizard) for model, windows, and index strategy. Health catches missing columns early.',
      route: '/train',
      selector: '[data-tour="train-title"]',
      placement: 'bottom',
    },
  ],
  train: [
    {
      id: 'train-intro',
      title: 'Train & audit',
      body: `Queue train / compare / HPO / leakage jobs. Outputs land in named runs and reports/. ${DISCLAIMER}`,
      selector: '[data-tour="train-title"]',
      route: '/train',
      placement: 'bottom',
    },
    {
      id: 'train-index',
      title: 'Index strategy',
      body: 'last_event / before_last for demos without index_time. column needs index_time (paper_synthetic).',
      selector: '[data-tour="train-index"]',
      placement: 'bottom',
    },
    {
      id: 'train-jobs',
      title: 'Job history',
      body: 'Watch queued → running → done. On success, promote if needed, then Generate SHAP on Results.',
      selector: '[data-tour="train-jobs"]',
      placement: 'top',
    },
    {
      id: 'train-to-results',
      title: 'Inspect outputs',
      body: 'Results: metrics & figures. Analytics: ROC/PR when evaluation curves exist.',
      route: '/results',
      selector: '[data-tour="results-title"]',
      placement: 'bottom',
    },
  ],
  results: [
    {
      id: 'results-intro',
      title: 'Results',
      body: `Hold-out metrics, runs, fairness, HPO, and PNG figures. ZIP / methods.md for reporting. ${DISCLAIMER}`,
      selector: '[data-tour="results-title"]',
      route: '/results',
      placement: 'bottom',
    },
    {
      id: 'results-metrics',
      title: 'Actions & runs',
      body: 'Refresh summary, Generate SHAP (real PNG only), fairness / thresholds, and open a run for trust flags.',
      selector: '[data-tour="results-metrics"]',
      placement: 'bottom',
    },
    {
      id: 'results-figures',
      title: 'Figures gallery',
      body: 'Only valid PNGs appear. Corrupt stubs are omitted — Generate SHAP to regenerate. Retry if an image fails to load.',
      selector: '[data-tour="results-figures"]',
      placement: 'top',
    },
    {
      id: 'results-to-analytics',
      title: 'Paper curves',
      body: 'Analytics charts ROC / PR / calibration. Empty → retrain on paper_synthetic (both classes in hold-out).',
      route: '/analytics',
      selector: '[data-tour="analytics-roc"]',
      placement: 'top',
    },
  ],
  analytics: [
    {
      id: 'analytics-intro',
      title: 'Analytics dashboard',
      body: `Cohort exploration plus hold-out charts for research reporting. Export PNG or print. ${DISCLAIMER}`,
      selector: '[data-tour="analytics-title"]',
      route: '/analytics',
      placement: 'bottom',
    },
    {
      id: 'analytics-curves',
      title: 'ROC curve',
      body: 'Needs both classes and curves in evaluation_report.json. Empty message → retrain on paper_synthetic.',
      selector: '[data-tour="analytics-roc"]',
      placement: 'top',
    },
    {
      id: 'analytics-pr',
      title: 'PR & calibration',
      body: 'PR is prevalence-aware; calibration needs enough positives. Quality notes explain tiny/single-class demos.',
      selector: '[data-tour="analytics-pr"]',
      placement: 'top',
    },
    {
      id: 'analytics-to-predict',
      title: 'Optional: score a patient',
      body: 'Predict uses the promoted model for research scores only — feature columns must match training.',
      route: '/predict',
      selector: '[data-tour="predict-title"]',
      placement: 'bottom',
    },
  ],
  config: [
    {
      id: 'config-intro',
      title: 'Config Center',
      body: `Workspace defaults, optional API key, theme, density. Check “Don’t auto-start tours” in any tour tip. ${DISCLAIMER}`,
      selector: '[data-tour="config-title"]',
      route: '/config',
      placement: 'bottom',
    },
    {
      id: 'config-to-home',
      title: 'Back to checklist',
      body: 'After saving defaults or an API key, return Home and Refresh status.',
      route: '/',
      selector: '[data-tour="home-title"]',
      placement: 'bottom',
    },
  ],
  predict: [
    {
      id: 'predict-intro',
      title: 'Predict',
      body: `Score one patient or a batch against the promoted model — research probabilities only. ${DISCLAIMER}`,
      selector: '[data-tour="predict-title"]',
      route: '/predict',
      placement: 'bottom',
    },
    {
      id: 'predict-form',
      title: 'Inputs',
      body: 'Fill expected features or paste a JSON row. Schema mismatches return clear errors.',
      selector: '[data-tour="predict-form"]',
      placement: 'top',
    },
    {
      id: 'predict-to-docs',
      title: 'Methods & docs',
      body: 'In-app Docs cover workflow and honesty notes. Prefer Research wizard for a first full pass.',
      route: '/docs',
      selector: '[data-tour="docs-title"]',
      placement: 'bottom',
    },
  ],
  docs: [
    {
      id: 'docs-intro',
      title: 'In-app docs',
      body: `Guide, research workflow, OpenAPI — methods language, not a clinical protocol. ${DISCLAIMER}`,
      selector: '[data-tour="docs-title"]',
      route: '/docs',
      placement: 'bottom',
    },
    {
      id: 'docs-to-wizard',
      title: 'Hands-on next',
      body: 'Start Research wizard for a guided loop, or Home for the setup checklist.',
      route: '/research',
      selector: '[data-tour="research-title"]',
      placement: 'bottom',
    },
  ],
};
