import {
  AfterViewChecked,
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  ViewChild,
  effect,
  inject,
  signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NavigationEnd, Router } from '@angular/router';
import { Subscription, filter } from 'rxjs';
import { PageTourService } from '../core/page-tour.service';
import { TourStep } from '../core/page-tour.definitions';

@Component({
  selector: 'app-page-tour',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './page-tour.component.html',
  styleUrl: './page-tour.component.css',
})
export class PageTourComponent implements OnDestroy, AfterViewChecked {
  readonly tour = inject(PageTourService);
  private readonly host = inject(ElementRef<HTMLElement>);
  private readonly router = inject(Router);
  private repositionTimer: ReturnType<typeof setTimeout> | null = null;
  private needsFocus = false;
  private readonly navSub: Subscription;

  @ViewChild('tipPanel') tipPanel?: ElementRef<HTMLElement>;
  @ViewChild('primaryBtn') primaryBtn?: ElementRef<HTMLButtonElement>;

  readonly hole = signal<{ top: number; left: number; width: number; height: number } | null>(null);
  readonly tip = signal<{ top: number; left: number }>({ top: 80, left: 24 });
  readonly step = signal<TourStep | null>(null);

  constructor() {
    this.navSub = this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe(() => {
        if (this.tour.active()) {
          this.needsFocus = true;
          this.schedulePosition(120);
        }
      });
    effect(() => {
      if (!this.tour.active()) {
        this.hole.set(null);
        this.step.set(null);
        return;
      }
      const s = this.tour.currentStep();
      this.step.set(s);
      this.needsFocus = true;
      this.schedulePosition();
    });
  }

  ngAfterViewChecked(): void {
    if (!this.needsFocus || !this.tour.active()) return;
    this.needsFocus = false;
    const btn = this.primaryBtn?.nativeElement;
    if (btn) {
      try {
        btn.focus({ preventScroll: true });
      } catch {
        btn.focus();
      }
    }
  }

  ngOnDestroy(): void {
    if (this.repositionTimer != null) clearTimeout(this.repositionTimer);
    this.navSub.unsubscribe();
  }

  @HostListener('window:resize')
  @HostListener('window:scroll')
  onViewport(): void {
    if (this.tour.active()) this.schedulePosition();
  }

  @HostListener('document:keydown', ['$event'])
  onKeydown(ev: KeyboardEvent): void {
    if (!this.tour.active()) return;
    if (ev.key === 'Escape') {
      ev.preventDefault();
      this.tour.skip();
      return;
    }
    if (ev.key !== 'Tab') return;
    const root = this.tipPanel?.nativeElement;
    if (!root) return;
    const focusable = Array.from(
      root.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
    ).filter((el) => el.offsetParent !== null || el === document.activeElement);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement as HTMLElement | null;
    if (ev.shiftKey && active === first) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && active === last) {
      ev.preventDefault();
      first.focus();
    }
  }

  get stepLabel(): string {
    const n = this.tour.steps().length;
    const i = this.tour.stepIndex() + 1;
    return n ? `${i} / ${n}` : '';
  }

  onDontShow(v: boolean): void {
    this.tour.setDontShowAgain(v);
  }

  private schedulePosition(delayMs = 50): void {
    if (this.repositionTimer != null) clearTimeout(this.repositionTimer);
    this.repositionTimer = setTimeout(() => {
      this.repositionTimer = null;
      this.position();
    }, delayMs);
  }

  private position(): void {
    const s = this.tour.currentStep();
    if (!s) return;
    const pad = 8;
    let target: DOMRect | null = null;
    if (s.selector) {
      const el = document.querySelector(s.selector) as HTMLElement | null;
      if (el) {
        el.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
        target = el.getBoundingClientRect();
      }
    }
    if (target && target.width > 0 && target.height > 0) {
      this.hole.set({
        top: Math.max(0, target.top - pad),
        left: Math.max(0, target.left - pad),
        width: target.width + pad * 2,
        height: target.height + pad * 2,
      });
      const tipW = 360;
      const tipH = 240;
      let top = target.bottom + 12;
      let left = Math.min(window.innerWidth - tipW - 16, Math.max(16, target.left));
      const place = s.placement || 'bottom';
      if (place === 'top') top = Math.max(16, target.top - tipH - 12);
      if (place === 'left') {
        top = Math.max(16, target.top);
        left = Math.max(16, target.left - tipW - 12);
      }
      if (place === 'right') {
        top = Math.max(16, target.top);
        left = Math.min(window.innerWidth - tipW - 16, target.right + 12);
      }
      if (top + tipH > window.innerHeight - 16) top = Math.max(16, window.innerHeight - tipH - 16);
      this.tip.set({ top, left });
    } else {
      this.hole.set(null);
      this.tip.set({
        top: Math.max(72, window.innerHeight * 0.15),
        left: Math.max(16, (window.innerWidth - 360) / 2),
      });
    }
    this.host.nativeElement.setAttribute('data-tour-active', '1');
  }
}
