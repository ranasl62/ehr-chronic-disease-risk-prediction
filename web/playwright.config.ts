import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  use: {
    baseURL: process.env['EHR_UI_BASE'] || 'http://127.0.0.1:8080',
    headless: true,
  },
  reporter: 'list',
});
