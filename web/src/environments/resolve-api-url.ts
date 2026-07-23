import { BUILD_API_ENDPOINT } from './api-endpoint.generated';

/**
 * Resolve the API base URL.
 * Prefer build-time ``API_ENDPOINT`` (baked into ``BUILD_API_ENDPOINT``);
 * otherwise use ``fallback`` (typically ``environment.apiUrl``, often '').
 * Empty string keeps same-origin calls for Docker nginx / ``ng serve`` proxy.
 */
export function resolveApiUrl(fallback = '', buildEndpoint: string = BUILD_API_ENDPOINT): string {
  const fromBuild = (buildEndpoint || '').trim().replace(/\/+$/, '');
  if (fromBuild) {
    return fromBuild;
  }
  return (fallback || '').trim().replace(/\/+$/, '');
}
