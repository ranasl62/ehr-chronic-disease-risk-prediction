import { resolveApiUrl } from './resolve-api-url';

describe('resolveApiUrl', () => {
  it('returns empty when no build endpoint and empty fallback', () => {
    // BUILD_API_ENDPOINT is '' in the committed generated stub
    expect(resolveApiUrl('')).toBe('');
  });

  it('uses fallback when build endpoint is empty', () => {
    expect(resolveApiUrl('https://example.test')).toBe('https://example.test');
  });

  it('strips trailing slashes from fallback', () => {
    expect(resolveApiUrl('https://example.test/')).toBe('https://example.test');
  });
});
