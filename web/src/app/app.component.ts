import { Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthBannerService } from './core/auth-banner.service';
import { UiPrefsService } from './core/ui-prefs.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  title = 'EHR Risk Workbench';
  /** Eager-apply theme/density from localStorage on boot. */
  private readonly ui = inject(UiPrefsService);
  readonly auth = inject(AuthBannerService);

  dismissAuth(): void {
    this.auth.clear();
  }
}
