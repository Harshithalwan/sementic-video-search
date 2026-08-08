from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class SystemInfo:
    hostname: str
    platform: str
    platform_version: str
    processor: str
    cpu_count: int
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_devices: list[str]
    mps_available: bool
    torch_device: str
    machine_id: str


def collect_system_info() -> SystemInfo:
    hostname = socket.gethostname()
    mps_available = False
    torch_device = "cpu"
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_devices = []
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                try:
                    cuda_devices.append(torch.cuda.get_device_name(i))
                except Exception:
                    cuda_devices.append(f"cuda:{i}")
        mps_available = torch.backends.mps.is_available()
        if cuda_available:
            torch_device = "cuda"
        elif mps_available:
            torch_device = "mps"
    except Exception:
        torch_version = "unknown"
        cuda_available = False
        cuda_devices = []

    try:
        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = 0

    return SystemInfo(
        hostname=hostname,
        platform=platform.system(),
        platform_version=platform.version(),
        processor=platform.processor(),
        cpu_count=cpu_count,
        python_version=sys.version,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_devices=cuda_devices,
        mps_available=mps_available,
        torch_device=torch_device,
        machine_id=hostname,
    )


def _system_info_to_dict(info: SystemInfo) -> dict[str, Any]:
    return {
        "hostname": info.hostname,
        "platform": info.platform,
        "platform_version": info.platform_version,
        "processor": info.processor,
        "cpu_count": info.cpu_count,
        "python_version": info.python_version,
        "torch_version": info.torch_version,
        "cuda_available": info.cuda_available,
        "cuda_devices": info.cuda_devices,
        "mps_available": info.mps_available,
        "torch_device": info.torch_device,
        "machine_id": info.machine_id,
    }


class LatencyLogger:
    def __init__(
        self,
        log_dir: str = "latency_logs",
        model_type: str = "",
        model_id: str = "",
        max_new_tokens: int | None = None,
        source: str | None = None,
        caption_interval: float | None = None,
    ) -> None:
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.max_new_tokens = max_new_tokens
        self.source = source
        self.caption_interval = caption_interval
        sys_info = collect_system_info()
        self._sys_info = sys_info
        self._system = _system_info_to_dict(sys_info)

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        filename = f"{sys_info.hostname}_{self._session_id}_{model_type}.jsonl"
        self._file = open(log_path / filename, "a", encoding="utf-8")
        self._write("system_info", {
            "model_type": model_type,
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
            "source": source,
            "caption_interval": caption_interval,
            **self._system,
        })

    def _write(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "event": event,
            "session_id": self._session_id,
            "hostname": self._sys_info.hostname,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def log_ssim(
        self,
        elapsed_ms: float,
        score: float,
        active: bool,
        frame_index: int = -1,
        model_type: str = "",
    ) -> None:
        self._write("ssim", {
            "elapsed_ms": round(elapsed_ms, 3),
            "score": round(score, 6),
            "active": active,
            "frame_index": frame_index,
            "model_type": model_type,
        })

    def log_yolo(
        self,
        elapsed_ms: float,
        classes: list[str],
        frame_index: int = -1,
    ) -> None:
        self._write("yolo", {
            "elapsed_ms": round(elapsed_ms, 3),
            "classes": classes,
            "num_objects": len(classes),
            "frame_index": frame_index,
        })

    def log_caption(
        self,
        elapsed_ms: float,
        caption: str,
        model_type: str = "",
        model_id: str = "",
        frame_index: int = -1,
    ) -> None:
        self._write("caption", {
            "elapsed_ms": round(elapsed_ms, 3),
            "caption_length": len(caption),
            "max_new_tokens": self.max_new_tokens,
            "model_type": model_type,
            "model_id": model_id,
            "frame_index": frame_index,
        })

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


__all__ = ["LatencyLogger", "collect_system_info", "SystemInfo"]
