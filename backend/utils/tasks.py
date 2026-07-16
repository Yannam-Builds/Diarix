"""
Task tracking for active downloads and generations.
"""

from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field
from copy import deepcopy
import threading


@dataclass
class DownloadTask:
    """Represents an active download task."""
    model_name: str
    status: str = "downloading"  # downloading, extracting, complete, error
    started_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


@dataclass
class GenerationTask:
    """Represents an active generation task."""
    task_id: str
    profile_id: str
    text_preview: str  # First 50 chars of text
    started_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TranscriptionResult:
    """One completed file in a batch transcription task."""

    filename: str
    output_path: str
    text: str
    duration: float
    model_name: str
    extra_outputs: List[str] = field(default_factory=list)


@dataclass
class TranscriptionTask:
    """Durable in-memory state exposed by the transcription job API."""

    task_id: str
    status: str
    model_name: str
    language: Optional[str]
    precision: str
    output_dir: str
    total_files: int
    completed_files: int = 0
    current_file: Optional[str] = None
    stage: str = "queued"
    progress: float = 0.0
    error: Optional[str] = None
    partial_text: str = ""
    results: List[TranscriptionResult] = field(default_factory=list)
    work_dir: Optional[str] = field(default=None, repr=False)
    started_at: datetime = field(default_factory=datetime.utcnow, repr=False)


class TaskManager:
    """Manages active downloads and generations."""
    
    def __init__(self):
        self._active_downloads: Dict[str, DownloadTask] = {}
        self._active_generations: Dict[str, GenerationTask] = {}
        self._transcription_tasks: Dict[str, TranscriptionTask] = {}
        self._transcription_lock = threading.RLock()
    
    def start_download(self, model_name: str) -> None:
        """Mark a download as started."""
        self._active_downloads[model_name] = DownloadTask(
            model_name=model_name,
            status="downloading",
        )
    
    def complete_download(self, model_name: str) -> None:
        """Mark a download as complete."""
        if model_name in self._active_downloads:
            del self._active_downloads[model_name]
    
    def error_download(self, model_name: str, error: str) -> None:
        """Mark a download as failed."""
        if model_name in self._active_downloads:
            self._active_downloads[model_name].status = "error"
            self._active_downloads[model_name].error = error
    
    def start_generation(self, task_id: str, profile_id: str, text: str) -> None:
        """Mark a generation as started."""
        text_preview = text[:50] + "..." if len(text) > 50 else text
        self._active_generations[task_id] = GenerationTask(
            task_id=task_id,
            profile_id=profile_id,
            text_preview=text_preview,
        )
    
    def complete_generation(self, task_id: str) -> None:
        """Mark a generation as complete."""
        if task_id in self._active_generations:
            del self._active_generations[task_id]
    
    def get_active_downloads(self) -> List[DownloadTask]:
        """Get all active downloads."""
        return list(self._active_downloads.values())
    
    def get_active_generations(self) -> List[GenerationTask]:
        """Get all active generations."""
        return list(self._active_generations.values())

    def start_transcription(
        self,
        *,
        task_id: str,
        model_name: str,
        language: Optional[str],
        precision: str,
        output_dir: str,
        total_files: int,
        work_dir: str,
    ) -> TranscriptionTask:
        """Create and retain a transcription task until task history is cleared."""
        task = TranscriptionTask(
            task_id=task_id,
            status="queued",
            model_name=model_name,
            language=language,
            precision=precision,
            output_dir=output_dir,
            total_files=total_files,
            work_dir=work_dir,
        )
        with self._transcription_lock:
            self._transcription_tasks[task_id] = task
            return deepcopy(task)

    def update_transcription(self, task_id: str, **changes) -> Optional[TranscriptionTask]:
        """Apply an atomic state update and return a detached snapshot."""
        with self._transcription_lock:
            task = self._transcription_tasks.get(task_id)
            if task is None:
                return None
            for key, value in changes.items():
                if not hasattr(task, key) or key in {"task_id", "results"}:
                    raise ValueError(f"Unsupported transcription task field: {key}")
                setattr(task, key, value)
            task.progress = min(100.0, max(0.0, float(task.progress)))
            return deepcopy(task)

    def append_transcription_result(
        self, task_id: str, result: TranscriptionResult
    ) -> Optional[TranscriptionTask]:
        """Append one completed result without exposing mutable shared state."""
        with self._transcription_lock:
            task = self._transcription_tasks.get(task_id)
            if task is None:
                return None
            task.results.append(result)
            task.completed_files = len(task.results)
            return deepcopy(task)

    def get_transcription_task(self, task_id: str) -> Optional[TranscriptionTask]:
        with self._transcription_lock:
            task = self._transcription_tasks.get(task_id)
            return deepcopy(task) if task else None

    def get_transcription_tasks(self) -> List[TranscriptionTask]:
        with self._transcription_lock:
            return deepcopy(list(self._transcription_tasks.values()))
    
    def cancel_download(self, model_name: str) -> bool:
        """Cancel/dismiss a download task (removes it from active list)."""
        return self._active_downloads.pop(model_name, None) is not None

    def clear_all(self) -> None:
        """Clear all download and generation tasks."""
        self._active_downloads.clear()
        self._active_generations.clear()
        with self._transcription_lock:
            self._transcription_tasks.clear()

    def is_download_active(self, model_name: str) -> bool:
        """Check if a download is active."""
        return model_name in self._active_downloads
    
    def is_generation_active(self, task_id: str) -> bool:
        """Check if a generation is active."""
        return task_id in self._active_generations


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
