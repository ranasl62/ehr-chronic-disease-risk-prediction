/**
 * Lean Playwright smoke (optional). Requires UI on :8080 (docker compose) or :4200.
 *
 *   cd web && npm i -D @playwright/test && npx playwright install chromium
 *   EHR_UI_BASE=http://127.0.0.1:8080 npx playwright test
 */
import { test, expect } from '@playwright/test';

const base = process.env['EHR_UI_BASE'] || 'http://127.0.0.1:8080';

test.describe('Workbench smoke', () => {
  test.beforeEach(async ({ page }) => {
    // Disable auto-start page tours so the backdrop never intercepts nav clicks.
    await page.addInitScript(() => {
      localStorage.setItem(
        'ehr_page_tours_v1',
        JSON.stringify({ completed: {}, disable_auto: true })
      );
    });
  });

  test('home, research wizard, analytics, predict routes render', async ({ page }) => {
    await page.goto(base + '/');
    await expect(page.getByRole('heading', { name: /Researcher workbench/i })).toBeVisible();
    await page.getByRole('link', { name: /Research/i }).first().click();
    await expect(page.getByRole('heading', { name: /Research wizard/i })).toBeVisible();
    await page.goto(base + '/analytics');
    await expect(page.getByRole('heading', { name: /Analytics/i })).toBeVisible();
    await page.goto(base + '/predict');
    await expect(page.getByRole('heading', { name: /Predict/i })).toBeVisible();
    await page.goto(base + '/results');
    await expect(page.getByRole('heading', { name: /Results/i })).toBeVisible();
  });

  test('results figures and analytics quality/curve notes', async ({ page }) => {
    await page.goto(base + '/results');
    await expect(page.getByRole('heading', { name: /Results/i })).toBeVisible();
    // SHAP figure may or may not exist; if listed, img or error alert must be present.
    const figs = page.locator('.figs figure, [data-tour="results-figures"] figure');
    if ((await figs.count()) > 0) {
      const first = figs.first();
      const img = first.locator('img');
      const err = first.locator('.fig-err, [role="alert"]');
      await expect(img.or(err).first()).toBeVisible({ timeout: 10000 });
    }

    await page.goto(base + '/analytics');
    await expect(page.getByRole('heading', { name: /Analytics/i })).toBeVisible();
    const quality = page.locator('[data-tour="analytics-quality"], [data-tour="analytics-curve-hint"], .curve-empty');
    // At least one quality / empty-curve / hint signal for researchers.
    await expect(quality.first()).toBeVisible({ timeout: 15000 });
  });
});
