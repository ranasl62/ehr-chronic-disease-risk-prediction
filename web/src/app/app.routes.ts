import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { DatasetsComponent } from './pages/datasets/datasets.component';
import { TrainComponent } from './pages/train/train.component';
import { ResultsComponent } from './pages/results/results.component';
import { PredictComponent } from './pages/predict/predict.component';
import { DocsComponent } from './pages/docs/docs.component';

import { AnalyticsComponent } from './pages/analytics/analytics.component';
import { ConfigComponent } from './pages/config/config.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'datasets', component: DatasetsComponent },
  { path: 'train', component: TrainComponent },
  { path: 'results', component: ResultsComponent },
  { path: 'analytics', component: AnalyticsComponent },
  { path: 'config', component: ConfigComponent },
  { path: 'predict', component: PredictComponent },
  { path: 'docs', component: DocsComponent },
  { path: '**', redirectTo: '' },
];
