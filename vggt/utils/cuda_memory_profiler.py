"""Lightweight CUDA phase profiler that also preserves results after OOM."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any

import torch


_enabled = False
_output_path: str | None = None
_records: list[dict[str, Any]] = []
_previous_allocated = 0
_previous_reserved = 0


def configure_cuda_memory_profiler(enabled: bool, output_path: str | None = None) -> None:
    global _enabled, _output_path, _records, _previous_allocated, _previous_reserved
    _enabled = bool(enabled and torch.cuda.is_available())
    _output_path = output_path
    _records = []
    _previous_allocated = 0
    _previous_reserved = 0
    if _enabled:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mark_cuda_memory("profiler.start")


def cuda_memory_profiler_enabled() -> bool:
    return _enabled


def _mib(value: int | float) -> float:
    return float(value) / (1024.0**2)


def mark_cuda_memory(stage: str, **metadata: Any) -> dict[str, Any] | None:
    global _previous_allocated, _previous_reserved
    if not _enabled:
        return None

    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    record = {
        "stage": stage,
        "allocated_mib": round(_mib(allocated), 2),
        "reserved_mib": round(_mib(reserved), 2),
        "interval_peak_allocated_mib": round(_mib(peak_allocated), 2),
        "interval_peak_reserved_mib": round(_mib(peak_reserved), 2),
        "allocated_delta_mib": round(_mib(allocated - _previous_allocated), 2),
        "reserved_delta_mib": round(_mib(reserved - _previous_reserved), 2),
        "driver_used_mib": round(_mib(total - free), 2),
        "driver_free_mib": round(_mib(free), 2),
        **metadata,
    }
    _records.append(record)
    _previous_allocated = allocated
    _previous_reserved = reserved
    print(
        "[CUDA memory] "
        f"{stage}: allocated={record['allocated_mib']:.2f} MiB, "
        f"reserved={record['reserved_mib']:.2f} MiB, "
        f"interval_peak={record['interval_peak_allocated_mib']:.2f} MiB, "
        f"driver_used={record['driver_used_mib']:.2f} MiB"
    )
    torch.cuda.reset_peak_memory_stats(device)
    flush_cuda_memory_profile()
    return record


def flush_cuda_memory_profile() -> None:
    if not _enabled or not _output_path:
        return
    path = Path(_output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(_records, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def get_cuda_memory_records() -> list[dict[str, Any]]:
    return list(_records)


atexit.register(flush_cuda_memory_profile)

