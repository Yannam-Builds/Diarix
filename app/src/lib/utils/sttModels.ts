import type { ModelStatus } from '@/lib/api/types';

const LANGUAGE_NAME_OVERRIDES: Record<string, string> = {
  fil: 'Filipino',
  jw: 'Javanese',
  nb: 'Norwegian Bokmål',
  yue: 'Cantonese',
};

const languageNames = new Intl.DisplayNames(['en'], { type: 'language' });

export interface SttLanguageOption {
  value: string;
  label: string;
}

export function sttLanguageLabel(code: string): string {
  return LANGUAGE_NAME_OVERRIDES[code] ?? languageNames.of(code) ?? code.toUpperCase();
}

export function sttLanguageOptions(
  model?: Pick<ModelStatus, 'capabilities' | 'languages'>,
): SttLanguageOption[] {
  if (!model) return [];

  const seen = new Set<string>();
  const supported = model.languages
    .map((code) => code.trim().toLowerCase())
    .filter((code) => {
      if (!code || code === 'auto' || code === 'multilingual' || seen.has(code)) return false;
      seen.add(code);
      return true;
    })
    .map((code) => ({ value: code, label: sttLanguageLabel(code) }));

  return model.capabilities.includes('language_detection')
    ? [{ value: 'auto', label: 'Detect automatically' }, ...supported]
    : supported;
}

export function normalizedSttLanguage(
  model: Pick<ModelStatus, 'capabilities' | 'languages'> | undefined,
  current: string,
): string {
  const options = sttLanguageOptions(model);
  return options.some((option) => option.value === current) ? current : (options[0]?.value ?? current);
}

export function modelRuntimeLabel(runtimeGroup: string): string | null {
  if (runtimeGroup === 'core') return null;
  if (runtimeGroup === 'native-asr') return 'Native GGUF';
  if (runtimeGroup === 'advanced-asr') return 'Advanced ASR';
  return runtimeGroup
    .split('-')
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(' ');
}
