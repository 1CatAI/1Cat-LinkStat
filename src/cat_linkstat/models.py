from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NvLinkSnapshot:
    link_id: int
    active: bool
    speed_mb_s: int | None
    version: int | None
    rx_bytes_per_s: float | None = None
    tx_bytes_per_s: float | None = None


@dataclass(frozen=True)
class GpuProcessSnapshot:
    pid: int
    gpu_index: int
    process_type: str | None
    username: str | None
    gpu_memory_mib: int | None
    gpu_sm_pct: int | None
    cpu_pct: float | None
    host_memory_mib: float | None
    command: str


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    uuid: str
    nvlink_metrics_source: str
    gpu_utilization_pct: int | None
    memory_utilization_pct: int | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    temperature_c: int | None
    power_usage_w: float | None
    power_limit_w: float | None
    fan_speed_pct: int | None
    sm_clock_mhz: int | None
    mem_clock_mhz: int | None
    graphics_clock_mhz: int | None
    nvlinks: list[NvLinkSnapshot]
    pcie_gen: int | None = None
    pcie_width: int | None = None
    pcie_rx_kib_s: int | None = None
    pcie_tx_kib_s: int | None = None
    nvlink_total_rx_bytes_per_s: float | None = None
    nvlink_total_tx_bytes_per_s: float | None = None

    @property
    def active_nvlinks(self) -> list[NvLinkSnapshot]:
        return [link for link in self.nvlinks if link.active]

    @property
    def active_nvlink_count(self) -> int:
        return len(self.active_nvlinks)

    @property
    def total_nvlink_rate_mb_s(self) -> int | None:
        active = [link.speed_mb_s for link in self.active_nvlinks if link.speed_mb_s is not None]
        if not active:
            return None
        return sum(active)

    @property
    def per_link_nvlink_rate_mb_s(self) -> int | None:
        active = [link.speed_mb_s for link in self.active_nvlinks if link.speed_mb_s is not None]
        if not active:
            return None
        speeds = set(active)
        if len(speeds) == 1:
            return active[0]
        return max(active)

    @property
    def dominant_nvlink_version(self) -> int | None:
        active = [link.version for link in self.active_nvlinks if link.version is not None]
        if not active:
            return None
        return max(active)

    @property
    def total_nvlink_rx_mib_s(self) -> float | None:
        if self.nvlink_total_rx_bytes_per_s is not None:
            return self.nvlink_total_rx_bytes_per_s / (1024**2)
        active = [link.rx_bytes_per_s for link in self.active_nvlinks if link.rx_bytes_per_s is not None]
        if not active:
            return None
        return sum(active) / (1024**2)

    @property
    def total_nvlink_tx_mib_s(self) -> float | None:
        if self.nvlink_total_tx_bytes_per_s is not None:
            return self.nvlink_total_tx_bytes_per_s / (1024**2)
        active = [link.tx_bytes_per_s for link in self.active_nvlinks if link.tx_bytes_per_s is not None]
        if not active:
            return None
        return sum(active) / (1024**2)

    @property
    def has_live_nvlink_throughput(self) -> bool:
        return self.total_nvlink_rx_mib_s is not None or self.total_nvlink_tx_mib_s is not None

    @property
    def total_nvlink_bandwidth_gb_s(self) -> float | None:
        if self.total_nvlink_rate_mb_s is None:
            return None
        return self.total_nvlink_rate_mb_s / 1000.0

    @property
    def total_nvlink_rx_gb_s(self) -> float | None:
        if self.total_nvlink_rx_mib_s is None:
            return None
        return (self.total_nvlink_rx_mib_s * (1024**2)) / 1_000_000_000

    @property
    def total_nvlink_tx_gb_s(self) -> float | None:
        if self.total_nvlink_tx_mib_s is None:
            return None
        return (self.total_nvlink_tx_mib_s * (1024**2)) / 1_000_000_000

    @property
    def nvlink_rx_pct(self) -> int | None:
        if self.total_nvlink_rx_gb_s is None or not self.total_nvlink_bandwidth_gb_s:
            return None
        return round(min(100.0, (self.total_nvlink_rx_gb_s / self.total_nvlink_bandwidth_gb_s) * 100.0))

    @property
    def nvlink_tx_pct(self) -> int | None:
        if self.total_nvlink_tx_gb_s is None or not self.total_nvlink_bandwidth_gb_s:
            return None
        return round(min(100.0, (self.total_nvlink_tx_gb_s / self.total_nvlink_bandwidth_gb_s) * 100.0))


@dataclass(frozen=True)
class SystemSnapshot:
    hostname: str
    driver_version: str | None
    timestamp: float
    gpus: list[GpuSnapshot]
    processes: list[GpuProcessSnapshot] = field(default_factory=list)
    nvlink_matrix: dict[int, dict[int, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
