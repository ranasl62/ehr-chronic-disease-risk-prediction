import { Injectable, inject, signal } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs/operators';
import {
  PAGE_TOURS,
  TOUR_ROUTE_TO_PAGE,
  TourPageId,
  TourStep,
} from './page-tour.definitions';

const STORAGE_KEY = 'ehr_page_tours_v1';

interface TourPrefs {
  /** Pages where the user finished or skipped the tour. */
  completed: Partial<Record<TourPageId, boolean>>;
  /** Disable first-visit auto-start globally. */
  disable_auto: boolean;
}

const DEFAULT_PREFS: TourPrefs = { completed: {}, disable_auto: false };

@Injectable({ providedIn: 'root' })
export class PageTourService {
  private readonly router = inject(Router);
  private prefs: TourPrefs = this.load();

  readonly activePage = signal<TourPageId | null>(null);
  readonly stepIndex = signal(0);
  readonly active = signal(false);
  readonly dontShowAgain = signal(false);

  constructor() {
    this.router.events.pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd)).subscribe((e) => {
      const page = this.pageFromUrl(e.urlAfterRedirects);
      if (page && this.shouldAutoStart(page)) {
        // Defer so the destination view can paint targets.
        setTimeout(() => this.start(page, { auto: true }), 400);
      }
    });
  }

  steps(): TourStep[] {
    const page = this.activePage();
    return page ? PAGE_TOURS[page] || [] : [];
  }

  currentStep(): TourStep | null {
    const list = this.steps();
    const i = this.stepIndex();
    return list[i] || null;
  }

  start(page: TourPageId, opts?: { auto?: boolean; force?: boolean }): void {
    const list = PAGE_TOURS[page];
    if (!list?.length) return;
    if (opts?.auto && !opts.force && !this.shouldAutoStart(page)) return;
    this.activePage.set(page);
    this.stepIndex.set(0);
    this.dontShowAgain.set(false);
    this.active.set(true);
    void this.ensureRoute(list[0]);
  }

  startForCurrentRoute(force = true): void {
    const page = this.pageFromUrl(this.router.url);
    if (page) this.start(page, { force });
  }

  next(): void {
    const list = this.steps();
    const i = this.stepIndex();
    if (i >= list.length - 1) {
      this.finish(true);
      return;
    }
    const next = list[i + 1];
    this.stepIndex.set(i + 1);
    void this.ensureRoute(next);
  }

  back(): void {
    const i = this.stepIndex();
    if (i <= 0) return;
    const prev = this.steps()[i - 1];
    this.stepIndex.set(i - 1);
    void this.ensureRoute(prev);
  }

  skip(): void {
    this.finish(true);
  }

  finish(markCompleted: boolean): void {
    const page = this.activePage();
    if (markCompleted && page) {
      this.prefs.completed[page] = true;
    }
    if (this.dontShowAgain()) {
      this.prefs.disable_auto = true;
    }
    this.save();
    this.active.set(false);
    this.activePage.set(null);
    this.stepIndex.set(0);
  }

  setDontShowAgain(v: boolean): void {
    this.dontShowAgain.set(v);
  }

  shouldAutoStart(page: TourPageId): boolean {
    if (this.prefs.disable_auto) return false;
    if (this.prefs.completed[page]) return false;
    if (this.active()) return false;
    return true;
  }

  pageFromUrl(url: string): TourPageId | null {
    const path = (url.split('?')[0] || '/').replace(/\/+$/, '') || '/';
    return TOUR_ROUTE_TO_PAGE[path] || null;
  }

  private async ensureRoute(step: TourStep | null | undefined): Promise<void> {
    if (!step?.route) return;
    const cur = (this.router.url.split('?')[0] || '/').replace(/\/+$/, '') || '/';
    const want = step.route.replace(/\/+$/, '') || '/';
    if (cur !== want) {
      await this.router.navigateByUrl(step.route);
    }
  }

  private load(): TourPrefs {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { ...DEFAULT_PREFS, completed: {} };
      const parsed = JSON.parse(raw) as Partial<TourPrefs>;
      return {
        disable_auto: !!parsed.disable_auto,
        completed: { ...(parsed.completed || {}) },
      };
    } catch {
      return { ...DEFAULT_PREFS, completed: {} };
    }
  }

  private save(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.prefs));
    } catch {
      /* ignore quota */
    }
  }
}
