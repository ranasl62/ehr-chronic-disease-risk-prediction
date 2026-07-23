import { resolveApiUrl } from './resolve-api-url';

describe('resolveApiUrl', () => {
  it('returns trimmed fallback when build endpoint empty', () => {
    expect(resolveApiUrl('')).toBe('');
    expect(resolveApiUrl('  /api/  ')).toBe('/api');
  });

  it('prefers build endpoint when provided', () => {
    expect(resolveApiUrl('', 'https://api.example.com/')).toBe('https://api.example.com');
  });
});
