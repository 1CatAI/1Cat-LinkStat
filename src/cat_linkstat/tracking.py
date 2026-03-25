from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

from .models import GpuSnapshot, NvLinkSnapshot, SystemSnapshot


def _format_rate_mb_s(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1000.0:.3f} GB/s"


@dataclass(frozen=True)
class ChangeEvent:
    timestamp: float
    message: str

    def render(self) -> str:
        stamp = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return f"{stamp} {self.message}"


class NvLinkChangeTracker:
    def __init__(self, max_events: int = 6) -> None:
        self._previous: SystemSnapshot | None = None
        self._events: deque[ChangeEvent] = deque(maxlen=max_events)

    def update(self, snapshot: SystemSnapshot) -> list[ChangeEvent]:
        if self._previous is None:
            self._previous = snapshot
            return list(self._events)

        previous_gpus = {gpu.index: gpu for gpu in self._previous.gpus}
        current_gpus = {gpu.index: gpu for gpu in snapshot.gpus}
        for gpu_index in sorted(set(previous_gpus) & set(current_gpus)):
            self._track_gpu(snapshot.timestamp, previous_gpus[gpu_index], current_gpus[gpu_index])

        self._previous = snapshot
        return list(self._events)

    def _push(self, timestamp: float, message: str) -> None:
        self._events.append(ChangeEvent(timestamp=timestamp, message=message))

    def _track_gpu(self, timestamp: float, previous: GpuSnapshot, current: GpuSnapshot) -> None:
        if previous.active_nvlink_count != current.active_nvlink_count:
            self._push(
                timestamp,
                f"GPU {current.index} active NVLink channels {previous.active_nvlink_count} -> {current.active_nvlink_count}",
            )

        if previous.total_nvlink_rate_mb_s != current.total_nvlink_rate_mb_s:
            self._push(
                timestamp,
                f"GPU {current.index} aggregate NVLink rate {_format_rate_mb_s(previous.total_nvlink_rate_mb_s)} -> {_format_rate_mb_s(current.total_nvlink_rate_mb_s)}",
            )

        if previous.nvlink_metrics_source != current.nvlink_metrics_source:
            self._push(
                timestamp,
                f"GPU {current.index} NVLink live source {previous.nvlink_metrics_source} -> {current.nvlink_metrics_source}",
            )

        previous_links = {link.link_id: link for link in previous.nvlinks}
        current_links = {link.link_id: link for link in current.nvlinks}
        for link_id in sorted(set(previous_links) | set(current_links)):
            before = previous_links.get(link_id)
            after = current_links.get(link_id)
            if before is None or after is None:
                continue
            self._track_link(timestamp, current.index, before, after)

    def _track_link(self, timestamp: float, gpu_index: int, previous: NvLinkSnapshot, current: NvLinkSnapshot) -> None:
        if previous.active != current.active:
            before = "up" if previous.active else "down"
            after = "up" if current.active else "down"
            self._push(timestamp, f"GPU {gpu_index} NVLink L{current.link_id} {before} -> {after}")

        if previous.speed_mb_s != current.speed_mb_s:
            self._push(
                timestamp,
                f"GPU {gpu_index} NVLink L{current.link_id} rate {_format_rate_mb_s(previous.speed_mb_s)} -> {_format_rate_mb_s(current.speed_mb_s)}",
            )
