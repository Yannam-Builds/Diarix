import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import {
  CheckCircle2,
  FileAudio2,
  FileVideo2,
  FolderOpen,
  Gauge,
  Languages,
  Loader2,
  Play,
  Square,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { type ChangeEvent, type DragEvent, useEffect, useMemo, useRef, useState } from 'react';
import { WaveField } from '@/components/SpeechMotion/WaveField';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { TranscriptionJob } from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';
import { sttLanguageOptions } from '@/lib/utils/sttModels';
import { usePlatform } from '@/platform/PlatformContext';

const ACCEPTED_MEDIA =
  '.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.mp4,.mov,.webm,audio/*,video/mp4,video/quicktime,video/webm';
const SUPPORTED_EXTENSIONS = new Set([
  'wav',
  'mp3',
  'm4a',
  'aac',
  'flac',
  'ogg',
  'opus',
  'mp4',
  'mov',
  'webm',
]);
const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'webm']);
const ACTIVE_JOB_STORAGE_KEY = 'diarix.transcription.activeJob';
const TERMINAL_STATUSES = new Set([
  'complete',
  'completed',
  'error',
  'failed',
  'cancelled',
  'canceled',
]);

interface QueuedFile {
  id: string;
  file: File;
  relativePath: string;
}

interface LegacyEntry {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  file?: (success: (file: File) => void, failure?: (error: DOMException) => void) => void;
  createReader?: () => {
    readEntries: (
      success: (entries: LegacyEntry[]) => void,
      failure?: (error: DOMException) => void,
    ) => void;
  };
}

function extensionOf(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() ?? '';
}

function isSupported(file: File): boolean {
  return SUPPORTED_EXTENSIONS.has(extensionOf(file.name));
}

function queuedFile(file: File, relativePath?: string): QueuedFile {
  const path = relativePath || file.webkitRelativePath || file.name;
  return {
    id: [path, file.size, file.lastModified].join(':'),
    file,
    relativePath: path,
  };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function readEntryFile(entry: LegacyEntry): Promise<File> {
  return new Promise((resolve, reject) => {
    if (!entry.file) {
      reject(new Error('The dropped file could not be read.'));
      return;
    }
    entry.file(resolve, reject);
  });
}

async function readDirectory(
  reader: NonNullable<ReturnType<NonNullable<LegacyEntry['createReader']>>>,
) {
  const entries: LegacyEntry[] = [];
  while (true) {
    const chunk = await new Promise<LegacyEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (chunk.length === 0) return entries;
    entries.push(...chunk);
  }
}

async function collectEntry(entry: LegacyEntry, parent = ''): Promise<QueuedFile[]> {
  if (entry.isFile) {
    const file = await readEntryFile(entry);
    return isSupported(file) ? [queuedFile(file, parent + file.name)] : [];
  }
  if (!entry.isDirectory || !entry.createReader) return [];

  const children = await readDirectory(entry.createReader());
  const directory = parent + entry.name + '/';
  const nested = await Promise.all(children.map((child) => collectEntry(child, directory)));
  return nested.flat();
}

function readStoredTaskId(): string | null {
  try {
    return window.localStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
  } catch {
    return null;
  }
}

function statusLabel(status: string): string {
  if (!status) return 'Idle';
  return status.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function isTerminal(status?: string): boolean {
  return Boolean(status && TERMINAL_STATUSES.has(status.toLowerCase()));
}

export function TranscriptionDashboard() {
  const platform = usePlatform();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const recentlyFinishedTaskIdRef = useRef<string | null>(null);
  const partialTextRef = useRef<HTMLDivElement>(null);
  const [files, setFiles] = useState<QueuedFile[]>([]);
  const [dragging, setDragging] = useState(false);
  const [modelName, setModelName] = useState('');
  const [language, setLanguage] = useState('auto');
  const [precision, setPrecision] = useState('default');
  const [outputSuffix, setOutputSuffix] = useState('_transcript');
  const [outputDirectory, setOutputDirectory] = useState('');
  const [exportFormats, setExportFormats] = useState<string[]>([]);
  const [silenceParagraphs, setSilenceParagraphs] = useState(false);
  const [job, setJob] = useState<TranscriptionJob | null>(null);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(readStoredTaskId);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);

  const { data: modelCatalog, isLoading: catalogLoading } = useQuery({
    queryKey: ['modelCatalog'],
    queryFn: () => apiClient.getModelCatalog(),
    staleTime: 30_000,
  });
  const { data: modelStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => apiClient.getModelStatus(),
    staleTime: 10_000,
  });
  const { data: captures } = useQuery({
    queryKey: ['captures', 'dashboard-recent'],
    queryFn: () => apiClient.listCaptures(5, 0),
    staleTime: 10_000,
  });
  const { data: activeTasks } = useQuery({
    queryKey: ['activeTasks'],
    queryFn: () => apiClient.getActiveTasks(),
    refetchInterval: 2_000,
  });

  const sttModels = useMemo(() => {
    const statusByName = new Map(
      modelStatus?.models.map((model) => [model.model_name, model] as const) ?? [],
    );
    return (
      modelCatalog?.models
        .filter((model) => model.modality === 'stt')
        .map((model) => ({ ...model, ...(statusByName.get(model.model_name) ?? {}) })) ?? []
    );
  }, [modelCatalog, modelStatus]);
  const selectedModel = sttModels.find((model) => model.model_name === modelName);
  const precisionOptions =
    selectedModel?.precision_options && selectedModel.precision_options.length > 0
      ? selectedModel.precision_options
      : ['default'];
  // Timestamped exports and silence-based paragraphs need real segment
  // timestamps. Only the CTranslate2 engines report them (segment_timestamps
  // on Faster-Whisper, alignment on WhisperX) — NeMo's word_timestamps
  // capability is not surfaced by its adapter, so it does not count here.
  const supportsTimestamps = Boolean(
    selectedModel?.capabilities.some((capability) =>
      ['segment_timestamps', 'alignment'].includes(capability),
    ),
  );
  const languageOptions = useMemo(() => {
    return sttLanguageOptions(selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    if (
      statusLoading ||
      sttModels.length === 0 ||
      sttModels.some((model) => model.model_name === modelName)
    ) {
      return;
    }
    const preferred =
      sttModels.find((model) => model.recommended && model.downloaded) ??
      sttModels.find((model) => model.downloaded) ??
      sttModels.find((model) => model.recommended) ??
      sttModels[0];
    setModelName(preferred.model_name);
  }, [modelName, statusLoading, sttModels]);

  useEffect(() => {
    if (!selectedModel) return;
    const options =
      selectedModel.precision_options.length > 0 ? selectedModel.precision_options : ['default'];
    if (!options.includes(precision)) {
      setPrecision(
        selectedModel.default_precision && options.includes(selectedModel.default_precision)
          ? selectedModel.default_precision
          : options[0],
      );
    }
  }, [precision, selectedModel]);

  useEffect(() => {
    if (languageOptions.length === 0) return;
    if (!languageOptions.some((option) => option.value === language)) {
      setLanguage(languageOptions[0].value);
    }
  }, [language, languageOptions]);

  useEffect(() => {
    if (activeTaskId) return;
    const active = activeTasks?.transcriptions.find(
      (task) => !isTerminal(task.status) && task.task_id !== recentlyFinishedTaskIdRef.current,
    );
    if (!active) return;

    setJob(active);
    setActiveTaskId(active.task_id);
    try {
      window.localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, active.task_id);
    } catch {
      // Storage can be unavailable in hardened webviews.
    }
  }, [activeTaskId, activeTasks]);

  useEffect(() => {
    const node = partialTextRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [job?.partial_text]);

  useEffect(() => {
    if (!activeTaskId) return;

    let disposed = false;
    let finished = false;
    let source: EventSource | null = null;
    let poll: ReturnType<typeof setInterval> | null = null;

    const finish = () => {
      if (finished) return;
      finished = true;
      recentlyFinishedTaskIdRef.current = activeTaskId;
      try {
        window.localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
      } catch {
        // Storage can be unavailable in hardened webviews.
      }
      source?.close();
      if (poll) clearInterval(poll);
      void queryClient.invalidateQueries({ queryKey: ['captures'] });
      void queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
      if (!disposed) setActiveTaskId(null);
    };

    const consume = (update: Partial<TranscriptionJob>) => {
      if (disposed) return;
      setJob((current) => {
        return { ...(current ?? {}), ...update } as TranscriptionJob;
      });
      if (isTerminal(update.status)) finish();
    };

    const refresh = async () => {
      if (finished) return;
      try {
        consume(await apiClient.getTranscriptionJob(activeTaskId));
        setJobError(null);
      } catch (error) {
        if (!disposed) {
          setJobError(error instanceof Error ? error.message : 'Could not read job status.');
          setJob(null);
          finish();
        }
      }
    };

    const handleProgress = (event: MessageEvent<string>) => {
      try {
        const payload = JSON.parse(event.data) as Partial<TranscriptionJob> & {
          task?: Partial<TranscriptionJob>;
        };
        consume(payload.task ?? payload);
      } catch {
        void refresh();
      }
    };

    void refresh();
    source = new EventSource(apiClient.getTranscriptionJobProgressUrl(activeTaskId));
    source.onmessage = handleProgress;
    source.addEventListener('progress', handleProgress as EventListener);
    source.addEventListener('completed', handleProgress as EventListener);
    source.addEventListener('failed', handleProgress as EventListener);
    source.addEventListener('cancelled', handleProgress as EventListener);
    source.onerror = () => {
      void refresh();
    };
    poll = setInterval(refresh, 2_000);

    return () => {
      disposed = true;
      source?.close();
      if (poll) clearInterval(poll);
    };
  }, [activeTaskId, queryClient]);

  const addFiles = (incoming: QueuedFile[]) => {
    const supported = incoming.filter((item) => isSupported(item.file));
    const rejected = incoming.length - supported.length;
    setFiles((current) => {
      const known = new Set(current.map((item) => item.id));
      return [...current, ...supported.filter((item) => !known.has(item.id))];
    });
    if (rejected > 0) {
      toast({
        title: 'Some files were skipped',
        description: rejected + ' unsupported file' + (rejected === 1 ? '' : 's') + ' ignored.',
      });
    }
  };

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    addFiles(Array.from(event.target.files ?? []).map((file) => queuedFile(file)));
    event.target.value = '';
  };

  const handleDrop = async (event: DragEvent<HTMLFieldSetElement>) => {
    event.preventDefault();
    setDragging(false);
    const items = Array.from(event.dataTransfer.items);
    const entries = items
      .map((item) => {
        const getter = (item as DataTransferItem & { webkitGetAsEntry?: () => LegacyEntry | null })
          .webkitGetAsEntry;
        return getter?.call(item) ?? null;
      })
      .filter((entry): entry is LegacyEntry => entry !== null);

    if (entries.length > 0) {
      const collected = await Promise.all(entries.map((entry) => collectEntry(entry)));
      addFiles(collected.flat());
      return;
    }
    addFiles(Array.from(event.dataTransfer.files).map((file) => queuedFile(file)));
  };

  const chooseOutputDirectory = async () => {
    const selected = await platform.filesystem.pickDirectory('Choose transcript output folder');
    if (selected) setOutputDirectory(selected);
  };

  const openPath = async (path: string) => {
    try {
      await platform.filesystem.openPath(path);
    } catch (error) {
      toast({
        title: 'Could not open output',
        description: error instanceof Error ? error.message : 'The path is unavailable.',
        variant: 'destructive',
      });
    }
  };

  const startTranscription = async () => {
    if (files.length === 0 || !modelName) return;
    if (!selectedModel?.downloaded) {
      toast({
        title: 'Model download required',
        description: 'Open Models and download this transcription model before starting.',
      });
      return;
    }
    setSubmitting(true);
    setJobError(null);
    try {
      const response = await apiClient.createTranscriptionJob({
        files: files.map((item) => item.file),
        model: modelName,
        language,
        precision,
        output_suffix: outputSuffix.trim() || '_transcript',
        output_dir: outputDirectory,
        export_formats:
          supportsTimestamps && exportFormats.length > 0
            ? ['txt', ...exportFormats].join(',')
            : undefined,
        silence_paragraphs: supportsTimestamps && silenceParagraphs,
      });
      const placeholder: TranscriptionJob = {
        task_id: response.task_id,
        status: response.status,
        model_name: modelName,
        language,
        precision,
        output_dir: outputDirectory,
        total_files: files.length,
        completed_files: 0,
        current_file: null,
        stage: 'queued',
        progress: 0,
        error: null,
        partial_text: '',
        results: [],
      };
      recentlyFinishedTaskIdRef.current = null;
      setJob(placeholder);
      setActiveTaskId(response.task_id);
      try {
        window.localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, response.task_id);
      } catch {
        // The live task stays connected even if persistence is unavailable.
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'The transcription job failed to start.';
      setJobError(message);
      toast({
        title: 'Could not start transcription',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setSubmitting(false);
    }
  };

  const cancelJob = async () => {
    const taskId = activeTaskId ?? job?.task_id;
    if (!taskId) return;
    setCancelling(true);
    try {
      const cancelled = await apiClient.cancelTranscriptionJob(taskId);
      setJob(cancelled);
      if (isTerminal(cancelled.status)) {
        try {
          window.localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
        } catch {
          // Storage can be unavailable in hardened webviews.
        }
        setActiveTaskId(null);
      }
    } catch (error) {
      toast({
        title: 'Could not cancel job',
        description: error instanceof Error ? error.message : 'Cancellation failed.',
        variant: 'destructive',
      });
    } finally {
      setCancelling(false);
    }
  };

  const running = submitting || Boolean(activeTaskId) || Boolean(job && !isTerminal(job.status));
  const progress = Math.max(0, Math.min(100, Math.round(job?.progress ?? 0)));
  const selectedBytes = files.reduce((total, item) => total + item.file.size, 0);
  const normalizationSummary = selectedModel
    ? [
        selectedModel.audio_sample_rate
          ? Math.round(selectedModel.audio_sample_rate / 1000) + ' kHz'
          : null,
        selectedModel.audio_channels
          ? selectedModel.audio_channels === 1
            ? 'mono'
            : selectedModel.audio_channels + ' channels'
          : null,
        selectedModel.audio_format?.toUpperCase(),
      ]
        .filter(Boolean)
        .join(' · ')
    : '';

  return (
    <div className="h-full overflow-y-auto py-6 pr-1">
      <header className="relative isolate min-h-[176px] overflow-hidden border-b border-border">
        <WaveField variant="dashboard" className="z-0 left-[30%] right-[-6%] opacity-90" />
        <div className="relative z-10 flex min-h-[176px] items-center justify-between gap-8 py-7">
          <div className="max-w-xl">
            <h1 className="text-3xl font-semibold tracking-[-0.03em] text-balance">
              Transcribe audio and video
            </h1>
            <p className="mt-3 max-w-[62ch] text-sm leading-6 text-muted-foreground text-pretty">
              Add media, choose a speech model, and keep every output predictable. Originals stay
              untouched while Diarix prepares model-ready audio.
            </p>
            <p className="mt-4 text-xs font-medium text-muted-foreground">
              WAV · MP3 · M4A · AAC · FLAC · OGG · Opus · MP4 · MOV · WebM
            </p>
          </div>
          <div className="relative z-20 flex min-w-[220px] items-center justify-center">
            <Button
              size="lg"
              className="relative z-20 isolate min-w-[196px] bg-accent text-accent-foreground shadow-sm ring-[8px] ring-background hover:bg-accent/90 disabled:bg-accent disabled:text-accent-foreground disabled:opacity-100"
              onClick={startTranscription}
              disabled={running || files.length === 0 || !modelName || !selectedModel?.downloaded}
            >
              {submitting ? <Loader2 className="animate-spin" /> : running ? <Gauge /> : <Play />}
              {submitting ? 'Starting…' : running ? 'Job running' : 'Start transcription'}
            </Button>
            {!running && (
              <div className="absolute top-[calc(50%+30px)] left-1/2 -translate-x-1/2 whitespace-nowrap text-center">
                {!selectedModel?.downloaded && selectedModel ? (
                  <Link
                    to="/models"
                    className="text-xs font-medium text-accent underline-offset-4 hover:underline"
                  >
                    Download {selectedModel.display_name} in Models
                  </Link>
                ) : (
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {files.length === 0
                      ? 'Add media to begin'
                      : files.length +
                        ' file' +
                        (files.length === 1 ? '' : 's') +
                        ' · ' +
                        formatBytes(selectedBytes)}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 py-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <main className="min-w-0 space-y-6">
          <section className="overflow-hidden rounded-lg border border-border bg-card/40">
            <fieldset
              aria-label="Media drop zone"
              className={cn(
                'm-4 flex min-h-[132px] items-center justify-center rounded-md border border-dashed p-5 transition-colors',
                dragging ? 'border-accent bg-accent/5' : 'border-border bg-background/40',
              )}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false);
              }}
              onDrop={handleDrop}
            >
              <div className="text-center">
                <Upload className="mx-auto h-6 w-6 text-accent" />
                <p className="mt-3 text-sm font-medium">Drop files or folders here</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Audio and video are inspected with FFprobe before transcription.
                </p>
                <div className="mt-4 flex justify-center gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={running}
                  >
                    <FileAudio2 />
                    Add files
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => folderInputRef.current?.click()}
                    disabled={running}
                  >
                    <FolderOpen />
                    Add folder
                  </Button>
                </div>
              </div>
            </fieldset>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept={ACCEPTED_MEDIA}
              multiple
              onChange={handleFileInput}
            />
            <input
              ref={folderInputRef}
              type="file"
              className="hidden"
              accept={ACCEPTED_MEDIA}
              multiple
              onChange={handleFileInput}
              {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
            />

            {files.length > 0 && (
              <div className="border-t border-border">
                <div className="flex items-center justify-between px-4 py-3">
                  <span className="text-sm font-medium">Input queue</span>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    className="h-8 text-xs text-muted-foreground"
                    onClick={() => setFiles([])}
                    disabled={running}
                  >
                    <Trash2 />
                    Clear
                  </Button>
                </div>
                <div className="max-h-56 overflow-y-auto border-t border-border/70">
                  {files.map((item) => {
                    const video = VIDEO_EXTENSIONS.has(extensionOf(item.file.name));
                    const MediaIcon = video ? FileVideo2 : FileAudio2;
                    return (
                      <div
                        key={item.id}
                        className="flex items-center gap-3 border-b border-border/60 px-4 py-2.5 last:border-b-0"
                      >
                        <MediaIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
                        <span className="min-w-0 flex-1 truncate text-sm" title={item.relativePath}>
                          {item.relativePath}
                        </span>
                        <span className="text-xs tabular-nums text-muted-foreground">
                          {formatBytes(item.file.size)}
                        </span>
                        <button
                          type="button"
                          className="rounded-full p-1 text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                          aria-label={'Remove ' + item.relativePath}
                          onClick={() =>
                            setFiles((current) => current.filter((file) => file.id !== item.id))
                          }
                          disabled={running}
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </section>

          <section className="rounded-lg border border-border bg-card/40 p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold">Transcription setup</h2>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  Precision follows the selected model. Language detection is available where the
                  engine supports it.
                </p>
              </div>
              <div className="flex items-center gap-2">
                {selectedModel?.recommended && (
                  <span className="rounded-full bg-accent/12 px-2.5 py-1 text-[11px] font-medium text-accent">
                    Recommended
                  </span>
                )}
                {selectedModel && (
                  <span
                    className={cn(
                      'rounded-full px-2.5 py-1 text-[11px] font-medium',
                      selectedModel.downloaded
                        ? 'bg-green-500/10 text-green-700 dark:text-green-400'
                        : 'bg-muted text-muted-foreground',
                    )}
                  >
                    {selectedModel.downloaded ? 'Downloaded' : 'Download required'}
                  </span>
                )}
              </div>
            </div>

            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Speech model</span>
                <Select value={modelName} onValueChange={setModelName} disabled={running}>
                  <SelectTrigger>
                    <SelectValue
                      placeholder={
                        catalogLoading || statusLoading ? 'Loading models…' : 'Choose a model'
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {sttModels.map((model) => (
                      <SelectItem key={model.model_name} value={model.model_name}>
                        {model.display_name}
                        {model.recommended ? ' · Recommended' : ''}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Precision</span>
                <Select
                  value={precision}
                  onValueChange={setPrecision}
                  disabled={running || precisionOptions.length === 1}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {precisionOptions.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option === 'default' ? 'Model default' : option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  <Languages className="h-3.5 w-3.5" />
                  Language
                </span>
                <Select value={language} onValueChange={setLanguage} disabled={running}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {languageOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <label className="space-y-2" htmlFor="transcription-output-suffix">
                <span className="text-xs font-medium text-muted-foreground">Filename suffix</span>
                <Input
                  id="transcription-output-suffix"
                  value={outputSuffix}
                  onChange={(event) => setOutputSuffix(event.target.value)}
                  placeholder="_transcript"
                  disabled={running}
                />
              </label>
            </div>

            <div className="mt-4 space-y-2">
              <span className="text-xs font-medium text-muted-foreground">Output folder</span>
              <div className="flex gap-2">
                <Input
                  value={outputDirectory}
                  readOnly
                  placeholder="Use the Diarix default output folder"
                  className="min-w-0"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={chooseOutputDirectory}
                  disabled={running}
                >
                  <FolderOpen />
                  Browse
                </Button>
              </div>
            </div>

            {supportsTimestamps && (
              <div className="mt-4 space-y-3">
                <div className="space-y-2">
                  <span className="text-xs font-medium text-muted-foreground">
                    Additional export formats
                  </span>
                  <div className="flex flex-wrap gap-4">
                    {['srt', 'vtt', 'json'].map((format) => (
                      <label
                        key={format}
                        className="flex cursor-pointer items-center gap-2 text-xs font-medium uppercase"
                        htmlFor={`export-format-${format}`}
                      >
                        <Checkbox
                          id={`export-format-${format}`}
                          checked={exportFormats.includes(format)}
                          disabled={running}
                          onCheckedChange={(checked) =>
                            setExportFormats((current) =>
                              checked
                                ? [...current, format]
                                : current.filter((item) => item !== format),
                            )
                          }
                        />
                        {format}
                      </label>
                    ))}
                  </div>
                </div>
                <label
                  className="flex cursor-pointer items-start gap-2"
                  htmlFor="silence-paragraphs"
                >
                  <Checkbox
                    id="silence-paragraphs"
                    checked={silenceParagraphs}
                    disabled={running}
                    onCheckedChange={setSilenceParagraphs}
                  />
                  <span className="text-xs leading-4">
                    <span className="font-medium">Paragraph breaks on silence</span>
                    <span className="mt-0.5 block text-muted-foreground">
                      Start a new paragraph when the recording pauses for more than a second.
                    </span>
                  </span>
                </label>
              </div>
            )}

            <div className="mt-5 flex items-start gap-3 border-t border-border pt-4">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
              <div>
                <p className="text-xs font-medium">Automatic media normalization</p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  {normalizationSummary
                    ? 'This model receives ' + normalizationSummary + ' audio.'
                    : 'Diarix converts each source to the audio format required by the model.'}{' '}
                  Source files are never modified.
                </p>
              </div>
            </div>
          </section>

          {job?.results && job.results.length > 0 && (
            <section className="rounded-lg border border-border bg-card/40">
              <div className="flex items-center justify-between px-5 py-4">
                <div>
                  <h2 className="text-base font-semibold">Job results</h2>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {job.results.length} transcript{job.results.length === 1 ? '' : 's'} written
                  </p>
                </div>
                {job.output_dir && (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => openPath(job.output_dir)}
                  >
                    <FolderOpen />
                    Open folder
                  </Button>
                )}
              </div>
              <div className="border-t border-border">
                {job.results.map((result) => (
                  <div
                    key={result.output_path || result.filename}
                    className="flex items-start gap-4 border-b border-border/60 px-5 py-4 last:border-b-0"
                  >
                    <FileAudio2 className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{result.filename}</p>
                      <p className="mt-1 line-clamp-2 text-xs leading-5 text-muted-foreground">
                        {result.text || 'Transcript written successfully.'}
                      </p>
                    </div>
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      onClick={() => openPath(result.output_path)}
                    >
                      Open
                    </Button>
                  </div>
                ))}
              </div>
            </section>
          )}
        </main>

        <aside className="space-y-6">
          <section className="rounded-lg border border-border bg-card/40 p-5">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-base font-semibold">Current job</h2>
              <span
                className={cn(
                  'rounded-full px-2.5 py-1 text-[11px] font-medium',
                  job?.status === 'completed' || job?.status === 'complete'
                    ? 'bg-green-500/10 text-green-700 dark:text-green-400'
                    : job?.status === 'failed' || job?.status === 'error'
                      ? 'bg-destructive/10 text-destructive'
                      : running
                        ? 'bg-accent/12 text-accent'
                        : 'bg-muted text-muted-foreground',
                )}
              >
                {statusLabel(job?.status ?? 'idle')}
              </span>
            </div>

            {job ? (
              <div className="mt-5 space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="font-medium">{statusLabel(job.stage || job.status)}</span>
                    <span className="tabular-nums text-muted-foreground">{progress}%</span>
                  </div>
                  <div
                    className="h-1.5 overflow-hidden rounded-full bg-muted"
                    role="progressbar"
                    aria-label="Transcription progress"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={progress}
                  >
                    <div
                      className="h-full rounded-full bg-accent transition-[width] duration-200 motion-reduce:transition-none"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                <dl className="space-y-3 text-xs">
                  <div className="flex items-center justify-between gap-4">
                    <dt className="text-muted-foreground">Files</dt>
                    <dd className="tabular-nums font-medium">
                      {job.completed_files} / {job.total_files}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <dt className="text-muted-foreground">Model</dt>
                    <dd className="max-w-[210px] truncate font-medium">
                      {selectedModel?.display_name ?? job.model_name}
                    </dd>
                  </div>
                  {job.current_file && (
                    <div className="space-y-1">
                      <dt className="text-muted-foreground">Current file</dt>
                      <dd className="truncate font-medium" title={job.current_file}>
                        {job.current_file}
                      </dd>
                    </div>
                  )}
                </dl>

                {running && job.partial_text && (
                  <div className="space-y-1.5">
                    <span className="text-xs text-muted-foreground">Live transcript</span>
                    <div
                      ref={partialTextRef}
                      className="max-h-32 overflow-y-auto rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-xs leading-5 text-muted-foreground"
                      aria-live="polite"
                    >
                      {job.partial_text}
                    </div>
                  </div>
                )}

                {(job.error || jobError) && (
                  <p className="rounded-md bg-destructive/10 px-3 py-2 text-xs leading-5 text-destructive">
                    {job.error || jobError}
                  </p>
                )}

                {running && (
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full"
                    onClick={cancelJob}
                    disabled={cancelling}
                  >
                    {cancelling ? <Loader2 className="animate-spin" /> : <Square />}
                    {cancelling ? 'Cancelling…' : 'Cancel job'}
                  </Button>
                )}
              </div>
            ) : (
              <div className="mt-5 py-6 text-center">
                <Gauge className="mx-auto h-6 w-6 text-muted-foreground/60" />
                <p className="mt-3 text-sm font-medium">Ready for media</p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  Real stage, file, and progress details appear here once the task starts.
                </p>
                {jobError && <p className="mt-3 text-xs text-destructive">{jobError}</p>}
              </div>
            )}
          </section>

          <section className="rounded-lg border border-border bg-card/40">
            <div className="flex items-center justify-between px-5 py-4">
              <div>
                <h2 className="text-base font-semibold">Recent transcripts</h2>
                <p className="mt-1 text-xs text-muted-foreground">Captures and imported media</p>
              </div>
              <Button asChild type="button" size="sm" variant="ghost">
                <Link to="/history">View all</Link>
              </Button>
            </div>
            <div className="border-t border-border">
              {captures?.items.length ? (
                captures.items.map((capture) => (
                  <Link
                    key={capture.id}
                    to="/history"
                    className="block border-b border-border/60 px-5 py-3 transition-colors last:border-b-0 hover:bg-muted/30"
                  >
                    <p className="line-clamp-2 text-xs leading-5">
                      {capture.transcript_refined || capture.transcript_raw || 'Transcript pending'}
                    </p>
                    <p className="mt-1.5 text-[11px] text-muted-foreground">
                      {new Date(capture.created_at).toLocaleString()}
                    </p>
                  </Link>
                ))
              ) : (
                <div className="px-5 py-8 text-center text-xs leading-5 text-muted-foreground">
                  Completed captures will stay within reach here.
                </div>
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
