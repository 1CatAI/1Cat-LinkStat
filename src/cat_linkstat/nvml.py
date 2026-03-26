from __future__ import annotations

import ctypes
import ctypes.util
import re
import socket
import shutil
import subprocess
import time
from dataclasses import dataclass

from .models import GpuProcessSnapshot, GpuSnapshot, NvLinkSnapshot, SystemSnapshot

NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4

NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_GPU = 0
NVML_PCIE_UTIL_TX_BYTES = 0
NVML_PCIE_UTIL_RX_BYTES = 1
NVML_NVLINK_MAX_LINKS = 18
NVML_FI_DEV_NVLINK_GET_SPEED = 164
NVML_FI_DEV_NVLINK_GET_STATE = 165
NVML_FI_DEV_NVLINK_GET_VERSION = 166

IGNORED_RETURN_CODES = {NVML_ERROR_NOT_SUPPORTED, NVML_ERROR_NO_PERMISSION}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class NvmlError(RuntimeError):
    pass


class NvmlValue(ctypes.Union):
    _fields_ = [
        ("dVal", ctypes.c_double),
        ("uiVal", ctypes.c_uint),
        ("ulVal", ctypes.c_ulong),
        ("ullVal", ctypes.c_ulonglong),
        ("sllVal", ctypes.c_longlong),
    ]


class NvmlFieldValue(ctypes.Structure):
    _fields_ = [
        ("fieldId", ctypes.c_uint),
        ("scopeId", ctypes.c_uint),
        ("timestamp", ctypes.c_longlong),
        ("latencyUsec", ctypes.c_longlong),
        ("valueType", ctypes.c_int),
        ("nvmlReturn", ctypes.c_int),
        ("value", NvmlValue),
    ]


class NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


@dataclass
class NvmlFunctionSet:
    init: ctypes._CFuncPtr
    shutdown: ctypes._CFuncPtr
    error_string: ctypes._CFuncPtr
    get_count: ctypes._CFuncPtr
    get_handle_by_index: ctypes._CFuncPtr
    get_name: ctypes._CFuncPtr
    get_uuid: ctypes._CFuncPtr
    get_utilization_rates: ctypes._CFuncPtr
    get_memory_info: ctypes._CFuncPtr
    get_temperature: ctypes._CFuncPtr
    get_power_usage: ctypes._CFuncPtr
    get_power_limit: ctypes._CFuncPtr
    get_fan_speed: ctypes._CFuncPtr
    get_clock_info: ctypes._CFuncPtr
    get_pcie_link_gen: ctypes._CFuncPtr
    get_pcie_link_width: ctypes._CFuncPtr
    get_pcie_throughput: ctypes._CFuncPtr
    get_compute_running_processes: ctypes._CFuncPtr
    get_graphics_running_processes: ctypes._CFuncPtr
    get_field_values: ctypes._CFuncPtr
    system_get_driver_version: ctypes._CFuncPtr


@dataclass(frozen=True)
class NvLinkLiveSample:
    total_rx_bytes_per_s: float | None
    total_tx_bytes_per_s: float | None
    link_throughput_bytes_per_s: dict[int, tuple[float | None, float | None]]


class DcgmiNvLinkSampler:
    def __init__(self, sample_delay_ms: int = 200) -> None:
        self._dcgmi = shutil.which("dcgmi")
        self._sample_delay_ms = max(50, sample_delay_ms)
        self._sample_count = 2
        self._supported = False
        if self._dcgmi is not None:
            self._supported = self._probe_support()

    @property
    def source(self) -> str:
        return "dcgm-total"

    def _probe_support(self) -> bool:
        try:
            completed = subprocess.run(
                [self._dcgmi, "profile", "-l", "-i", "0"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if completed.returncode != 0:
            return False
        text = completed.stdout
        return "1011" in text and "1012" in text

    def sample(self, gpu_indices: list[int]) -> dict[int, NvLinkLiveSample]:
        if not self._supported or not gpu_indices:
            return {}
        command = [
            self._dcgmi,
            "dmon",
            "-i",
            ",".join(str(index) for index in gpu_indices),
            "-e",
            "1011,1012",
            "-d",
            str(self._sample_delay_ms),
            "-c",
            str(self._sample_count),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}
        return self.parse_dmon_output(completed.stdout)

    @staticmethod
    def parse_dmon_output(raw_text: str) -> dict[int, NvLinkLiveSample]:
        samples: dict[int, NvLinkLiveSample] = {}
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line == "ID":
                continue
            parts = line.split()
            if len(parts) < 4 or parts[0] != "GPU":
                continue
            try:
                gpu_index = int(parts[1])
                tx_bytes_per_s = float(parts[2])
                rx_bytes_per_s = float(parts[3])
            except ValueError:
                continue
            # dcgmi dmon often emits an initial warm-up row with zeroed counters.
            # Keep overwriting so the latest sample wins when multiple rows are returned.
            samples[gpu_index] = NvLinkLiveSample(
                total_rx_bytes_per_s=rx_bytes_per_s,
                total_tx_bytes_per_s=tx_bytes_per_s,
                link_throughput_bytes_per_s={},
            )
        return samples


class NvidiaProcessSampler:
    def __init__(self) -> None:
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._ps = shutil.which("ps")

    def sample(self, process_memory: dict[tuple[int, int], dict[str, object]]) -> list[GpuProcessSnapshot]:
        if self._nvidia_smi is None:
            return []

        pmon_rows = self._read_pmon()
        if not pmon_rows and not process_memory:
            return []

        pids = sorted({pid for (_, pid) in pmon_rows} | {pid for (_, pid) in process_memory})
        ps_rows = self._read_ps(pids)

        keys = sorted(set(pmon_rows) | set(process_memory))
        processes: list[GpuProcessSnapshot] = []
        for gpu_index, pid in keys:
            pmon_row = pmon_rows.get((gpu_index, pid), {})
            memory_row = process_memory.get((gpu_index, pid), {})
            ps_row = ps_rows.get(pid, {})
            command = (
                ps_row.get("command")
                or str(memory_row.get("process_name") or "")
                or str(pmon_row.get("command") or "")
                or f"pid {pid}"
            )
            processes.append(
                GpuProcessSnapshot(
                    pid=pid,
                    gpu_index=gpu_index,
                    process_type=self._optional_str(memory_row.get("process_type")) or self._optional_str(pmon_row.get("process_type")),
                    username=self._optional_str(ps_row.get("user")),
                    gpu_memory_mib=self._optional_int(memory_row.get("gpu_memory_mib")),
                    gpu_sm_pct=self._optional_int(pmon_row.get("sm_pct")),
                    cpu_pct=self._optional_float(ps_row.get("cpu_pct")),
                    host_memory_mib=self._optional_float(ps_row.get("host_memory_mib")),
                    command=command,
                )
            )
        processes.sort(
            key=lambda item: (
                item.gpu_memory_mib or -1,
                item.gpu_sm_pct or -1,
                item.cpu_pct or -1.0,
                -item.gpu_index,
                -item.pid,
            ),
            reverse=True,
        )
        return processes

    def _run(self, command: list[str], timeout: int = 8) -> str:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if completed.returncode != 0 and not completed.stdout:
            return ""
        return completed.stdout

    def _read_pmon(self) -> dict[tuple[int, int], dict[str, object]]:
        stdout = self._run([self._nvidia_smi, "pmon", "-s", "u", "-c", "2"], timeout=12)
        rows: dict[tuple[int, int], dict[str, object]] = {}
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 9)
            if len(parts) < 10:
                continue
            gpu_text, pid_text, process_type, sm_text, _, _, _, _, _, command = parts
            if pid_text == "-":
                continue
            try:
                gpu_index = int(gpu_text)
                pid = int(pid_text)
            except ValueError:
                continue
            rows[(gpu_index, pid)] = {
                "process_type": process_type,
                "sm_pct": None if sm_text == "-" else int(sm_text),
                "command": command.strip(),
            }
        return rows

    def _read_ps(self, pids: list[int]) -> dict[int, dict[str, object]]:
        if not pids or self._ps is None:
            return {}
        stdout = self._run([self._ps, "-o", "pid=,user=,%cpu=,rss=,args=", "-p", ",".join(str(pid) for pid in pids)])
        rows: dict[int, dict[str, object]] = {}
        for raw_line in stdout.splitlines():
            parts = raw_line.strip().split(None, 4)
            if len(parts) < 5:
                continue
            pid_text, user, cpu_text, rss_text, command = parts
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            try:
                cpu_pct = float(cpu_text)
            except ValueError:
                cpu_pct = None
            try:
                host_memory_mib = int(rss_text) / 1024.0
            except ValueError:
                host_memory_mib = None
            rows[pid] = {
                "user": user,
                "cpu_pct": cpu_pct,
                "host_memory_mib": host_memory_mib,
                "command": command,
            }
        return rows

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class NvidiaSmiNvLinkSampler:
    def __init__(self) -> None:
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._gpm_supported = False
        if self._nvidia_smi is not None:
            self._gpm_supported = self._probe_gpm_support()

    @property
    def source(self) -> str:
        return "gpm-dmon" if self._gpm_supported else "state-poll"

    def _probe_gpm_support(self) -> bool:
        rows = self._run_dmon([60, 61])
        if not rows:
            return False
        for row in rows.values():
            if self._parse_float(row.get("nvlrx")) is not None:
                return True
            if self._parse_float(row.get("nvltx")) is not None:
                return True
        return False

    def sample(self, active_links_by_gpu: dict[int, list[int]]) -> dict[int, NvLinkLiveSample]:
        if not self._gpm_supported:
            return {}

        metric_ids = [60, 61]
        unique_links = sorted({link_id for link_ids in active_links_by_gpu.values() for link_id in link_ids})
        for link_id in unique_links:
            metric_ids.extend([62 + link_id * 2, 63 + link_id * 2])

        rows = self._run_dmon(metric_ids)
        samples: dict[int, NvLinkLiveSample] = {}
        for gpu_index, row in rows.items():
            per_link: dict[int, tuple[float | None, float | None]] = {}
            for link_id in active_links_by_gpu.get(gpu_index, []):
                per_link[link_id] = (
                    self._mib_to_bytes(self._parse_float(row.get(f"nvl{link_id}rx"))),
                    self._mib_to_bytes(self._parse_float(row.get(f"nvl{link_id}tx"))),
                )
            samples[gpu_index] = NvLinkLiveSample(
                total_rx_bytes_per_s=self._mib_to_bytes(self._parse_float(row.get("nvlrx"))),
                total_tx_bytes_per_s=self._mib_to_bytes(self._parse_float(row.get("nvltx"))),
                link_throughput_bytes_per_s=per_link,
            )
        return samples

    def _run_dmon(self, metric_ids: list[int]) -> dict[int, dict[str, str]]:
        if self._nvidia_smi is None:
            return {}
        command = [
            self._nvidia_smi,
            "dmon",
            "--gpm-metrics",
            ",".join(str(metric_id) for metric_id in metric_ids),
            "-c",
            "1",
            "--format",
            "csv,nounit",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}

        header: list[str] | None = None
        rows: dict[int, dict[str, str]] = {}
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = [part.strip().lstrip("#").strip().lower() for part in line.split(",")]
                continue
            if header is None:
                continue
            values = [part.strip() for part in line.split(",")]
            if len(values) != len(header):
                continue
            row = dict(zip(header, values))
            gpu_index = row.get("gpu")
            if gpu_index is None:
                continue
            try:
                rows[int(gpu_index)] = row
            except ValueError:
                continue
        return rows

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        if value is None:
            return None
        value = value.strip()
        if not value or value == "-":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _mib_to_bytes(value: float | None) -> float | None:
        if value is None:
            return None
        return value * (1024**2)


class NvidiaTopologySampler:
    def __init__(self) -> None:
        self._nvidia_smi = shutil.which("nvidia-smi")

    def sample(self, gpu_count: int) -> dict[int, dict[int, str]]:
        if self._nvidia_smi is None or gpu_count <= 0:
            return {}
        try:
            completed = subprocess.run(
                [self._nvidia_smi, "topo", "-m"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}
        return self.parse_matrix(completed.stdout, gpu_count)

    @staticmethod
    def parse_matrix(raw_text: str, gpu_count: int) -> dict[int, dict[int, str]]:
        if gpu_count <= 0:
            return {}

        lines = [ANSI_RE.sub("", line).rstrip() for line in raw_text.splitlines()]
        lines = [line for line in lines if line.strip()]
        if not lines:
            return {}

        header_labels: list[int] = []
        for line in lines:
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            gpu_tokens = [part for part in parts if part.startswith("GPU") and part[3:].isdigit()]
            if gpu_tokens:
                header_labels = [int(token[3:]) for token in gpu_tokens]
                break
        if not header_labels:
            return {}

        matrix: dict[int, dict[int, str]] = {}
        for line in lines:
            parts = [part.strip() for part in line.split("\t")]
            if not parts:
                continue
            row_label = parts[0].strip()
            if not (row_label.startswith("GPU") and row_label[3:].isdigit()):
                continue
            row_index = int(row_label[3:])
            if row_index >= gpu_count:
                continue
            cells = parts[1 : 1 + len(header_labels)]
            if len(cells) < len(header_labels):
                continue
            row_entries: dict[int, str] = {}
            for column_index, cell in zip(header_labels, cells):
                if column_index >= gpu_count:
                    continue
                row_entries[column_index] = cell or "?"
            if row_entries:
                matrix[row_index] = row_entries
        return matrix


def _bind(lib: ctypes.CDLL, name: str, restype: type, argtypes: list[type]) -> ctypes._CFuncPtr:
    func = getattr(lib, name)
    func.restype = restype
    func.argtypes = argtypes
    return func


class NvmlMonitor:
    def __init__(self) -> None:
        path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        try:
            self._lib = ctypes.CDLL(path)
        except OSError as exc:
            raise NvmlError(f"Unable to load NVML library: {path}") from exc

        self._funcs = NvmlFunctionSet(
            init=_bind(self._lib, "nvmlInit_v2", ctypes.c_int, []),
            shutdown=_bind(self._lib, "nvmlShutdown", ctypes.c_int, []),
            error_string=_bind(self._lib, "nvmlErrorString", ctypes.c_char_p, [ctypes.c_int]),
            get_count=_bind(self._lib, "nvmlDeviceGetCount_v2", ctypes.c_int, [ctypes.POINTER(ctypes.c_uint)]),
            get_handle_by_index=_bind(
                self._lib,
                "nvmlDeviceGetHandleByIndex_v2",
                ctypes.c_int,
                [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            ),
            get_name=_bind(
                self._lib,
                "nvmlDeviceGetName",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
            ),
            get_uuid=_bind(
                self._lib,
                "nvmlDeviceGetUUID",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
            ),
            get_utilization_rates=_bind(
                self._lib,
                "nvmlDeviceGetUtilizationRates",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(NvmlUtilization)],
            ),
            get_memory_info=_bind(
                self._lib,
                "nvmlDeviceGetMemoryInfo",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(NvmlMemory)],
            ),
            get_temperature=_bind(
                self._lib,
                "nvmlDeviceGetTemperature",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_power_usage=_bind(
                self._lib,
                "nvmlDeviceGetPowerUsage",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_power_limit=_bind(
                self._lib,
                "nvmlDeviceGetPowerManagementLimit",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_fan_speed=_bind(
                self._lib,
                "nvmlDeviceGetFanSpeed",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_clock_info=_bind(
                self._lib,
                "nvmlDeviceGetClockInfo",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_pcie_link_gen=_bind(
                self._lib,
                "nvmlDeviceGetCurrPcieLinkGeneration",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_pcie_link_width=_bind(
                self._lib,
                "nvmlDeviceGetCurrPcieLinkWidth",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_pcie_throughput=_bind(
                self._lib,
                "nvmlDeviceGetPcieThroughput",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
            ),
            get_compute_running_processes=_bind(
                self._lib,
                "nvmlDeviceGetComputeRunningProcesses_v3",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(NvmlProcessInfo)],
            ),
            get_graphics_running_processes=_bind(
                self._lib,
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(NvmlProcessInfo)],
            ),
            get_field_values=_bind(
                self._lib,
                "nvmlDeviceGetFieldValues",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(NvmlFieldValue)],
            ),
            system_get_driver_version=_bind(
                self._lib,
                "nvmlSystemGetDriverVersion",
                ctypes.c_int,
                [ctypes.c_char_p, ctypes.c_uint],
            ),
        )
        self._dcgmi_sampler = DcgmiNvLinkSampler()
        self._nvlink_sampler = NvidiaSmiNvLinkSampler()
        self._topology_sampler = NvidiaTopologySampler()
        self._process_sampler = NvidiaProcessSampler()
        self._check(self._funcs.init(), "nvmlInit_v2")

    def close(self) -> None:
        self._check(self._funcs.shutdown(), "nvmlShutdown", ignore=IGNORED_RETURN_CODES)

    def _error_string(self, rc: int) -> str:
        raw = self._funcs.error_string(rc)
        if not raw:
            return f"Unknown NVML error {rc}"
        return raw.decode("utf-8", "replace")

    def _check(self, rc: int, func_name: str, ignore: set[int] | None = None) -> int:
        if rc == NVML_SUCCESS:
            return rc
        if ignore and rc in ignore:
            return rc
        raise NvmlError(f"{func_name} failed: {self._error_string(rc)}")

    def _get_string(
        self,
        func: ctypes._CFuncPtr,
        handle: ctypes.c_void_p | None = None,
        length: int = 96,
    ) -> str | None:
        buffer = ctypes.create_string_buffer(length)
        if handle is None:
            rc = func(buffer, length)
        else:
            rc = func(handle, buffer, length)
        if rc in IGNORED_RETURN_CODES:
            return None
        self._check(rc, func.__name__)
        return buffer.value.decode("utf-8", "replace")

    def _get_uint(
        self,
        func: ctypes._CFuncPtr,
        *args: object,
        scale: float = 1.0,
    ) -> int | float | None:
        value = ctypes.c_uint()
        rc = func(*args, ctypes.byref(value))
        if rc in IGNORED_RETURN_CODES:
            return None
        self._check(rc, func.__name__)
        if scale != 1.0:
            return value.value / scale
        return value.value

    def _get_memory(self, handle: ctypes.c_void_p) -> NvmlMemory | None:
        value = NvmlMemory()
        rc = self._funcs.get_memory_info(handle, ctypes.byref(value))
        if rc in IGNORED_RETURN_CODES:
            return None
        self._check(rc, "nvmlDeviceGetMemoryInfo")
        return value

    def _get_utilization(self, handle: ctypes.c_void_p) -> NvmlUtilization | None:
        value = NvmlUtilization()
        rc = self._funcs.get_utilization_rates(handle, ctypes.byref(value))
        if rc in IGNORED_RETURN_CODES:
            return None
        self._check(rc, "nvmlDeviceGetUtilizationRates")
        return value

    def _get_handle(self, index: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(
            self._funcs.get_handle_by_index(index, ctypes.byref(handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        return handle

    def _get_field_uints(self, handle: ctypes.c_void_p, field_id: int) -> dict[int, int]:
        fields = (NvmlFieldValue * NVML_NVLINK_MAX_LINKS)()
        for scope_id in range(NVML_NVLINK_MAX_LINKS):
            fields[scope_id].fieldId = field_id
            fields[scope_id].scopeId = scope_id
        self._check(
            self._funcs.get_field_values(handle, NVML_NVLINK_MAX_LINKS, fields),
            "nvmlDeviceGetFieldValues",
        )
        values: dict[int, int] = {}
        for field in fields:
            if field.nvmlReturn == NVML_SUCCESS:
                values[field.scopeId] = field.value.uiVal
        return values

    def _collect_nvlinks(self, handle: ctypes.c_void_p) -> list[NvLinkSnapshot]:
        speeds = self._get_field_uints(handle, NVML_FI_DEV_NVLINK_GET_SPEED)
        states = self._get_field_uints(handle, NVML_FI_DEV_NVLINK_GET_STATE)
        versions = self._get_field_uints(handle, NVML_FI_DEV_NVLINK_GET_VERSION)
        link_ids = sorted(set(speeds) | set(states) | set(versions))
        links: list[NvLinkSnapshot] = []
        for link_id in link_ids:
            links.append(
                NvLinkSnapshot(
                    link_id=link_id,
                    active=states.get(link_id, 0) == 1,
                    speed_mb_s=speeds.get(link_id),
                    version=versions.get(link_id),
                )
            )
        return links

    def _get_running_processes(
        self,
        func: ctypes._CFuncPtr,
        handle: ctypes.c_void_p,
        process_type: str,
    ) -> dict[int, dict[str, object]]:
        count = ctypes.c_uint(64)
        entries = (NvmlProcessInfo * 64)()
        rc = func(handle, ctypes.byref(count), entries)
        if rc in IGNORED_RETURN_CODES:
            return {}
        self._check(rc, func.__name__)
        rows: dict[int, dict[str, object]] = {}
        for index in range(count.value):
            entry = entries[index]
            gpu_memory_mib = None
            if entry.usedGpuMemory not in (0xFFFFFFFFFFFFFFFF,):
                gpu_memory_mib = int(entry.usedGpuMemory / (1024**2))
            rows[int(entry.pid)] = {
                "gpu_memory_mib": gpu_memory_mib,
                "process_type": process_type,
            }
        return rows

    def collect(self) -> SystemSnapshot:
        count = ctypes.c_uint()
        self._check(self._funcs.get_count(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
        driver_version = self._get_string(self._funcs.system_get_driver_version, handle=None, length=80)
        nvlink_matrix = self._topology_sampler.sample(count.value)
        raw_gpus: list[dict[str, object]] = []
        process_memory: dict[tuple[int, int], dict[str, object]] = {}
        for index in range(count.value):
            handle = self._get_handle(index)
            utilization = self._get_utilization(handle)
            memory = self._get_memory(handle)
            uuid = self._get_string(self._funcs.get_uuid, handle, 96) or ""
            raw_gpus.append(
                {
                    "index": index,
                    "name": self._get_string(self._funcs.get_name, handle, 96) or f"GPU-{index}",
                    "uuid": uuid,
                    "gpu_utilization_pct": None if utilization is None else utilization.gpu,
                    "memory_utilization_pct": None if utilization is None else utilization.memory,
                    "memory_used_bytes": None if memory is None else memory.used,
                    "memory_total_bytes": None if memory is None else memory.total,
                    "temperature_c": self._get_uint(self._funcs.get_temperature, handle, NVML_TEMPERATURE_GPU),
                    "power_usage_w": self._get_uint(self._funcs.get_power_usage, handle, scale=1000.0),
                    "power_limit_w": self._get_uint(self._funcs.get_power_limit, handle, scale=1000.0),
                    "fan_speed_pct": self._get_uint(self._funcs.get_fan_speed, handle),
                    "sm_clock_mhz": self._get_uint(self._funcs.get_clock_info, handle, NVML_CLOCK_SM),
                    "mem_clock_mhz": self._get_uint(self._funcs.get_clock_info, handle, NVML_CLOCK_MEM),
                    "graphics_clock_mhz": self._get_uint(self._funcs.get_clock_info, handle, NVML_CLOCK_GRAPHICS),
                    "pcie_gen": self._get_uint(self._funcs.get_pcie_link_gen, handle),
                    "pcie_width": self._get_uint(self._funcs.get_pcie_link_width, handle),
                    "pcie_rx_kib_s": self._get_uint(self._funcs.get_pcie_throughput, handle, NVML_PCIE_UTIL_RX_BYTES),
                    "pcie_tx_kib_s": self._get_uint(self._funcs.get_pcie_throughput, handle, NVML_PCIE_UTIL_TX_BYTES),
                    "nvlinks": self._collect_nvlinks(handle),
                }
            )
            for pid, info in self._get_running_processes(self._funcs.get_compute_running_processes, handle, "C").items():
                process_memory[(index, pid)] = info
            for pid, info in self._get_running_processes(self._funcs.get_graphics_running_processes, handle, "G").items():
                existing = process_memory.get((index, pid), {})
                process_memory[(index, pid)] = {**existing, **info}

        active_links_by_gpu = {
            raw_gpu["index"]: [link.link_id for link in raw_gpu["nvlinks"] if link.active]
            for raw_gpu in raw_gpus
        }
        dcgm_samples = self._dcgmi_sampler.sample(sorted(active_links_by_gpu))
        gpm_samples = self._nvlink_sampler.sample(active_links_by_gpu)
        processes = self._process_sampler.sample(process_memory)

        gpus: list[GpuSnapshot] = []
        for raw_gpu in raw_gpus:
            gpu_index = int(raw_gpu["index"])
            dcgm_sample = dcgm_samples.get(gpu_index)
            gpm_sample = gpm_samples.get(gpu_index)
            source = "state-poll"
            if dcgm_sample is not None and gpm_sample is not None:
                source = "dcgm-total+gpm-link"
            elif dcgm_sample is not None:
                source = self._dcgmi_sampler.source
            elif gpm_sample is not None:
                source = self._nvlink_sampler.source
            nvlinks: list[NvLinkSnapshot] = []
            for link in raw_gpu["nvlinks"]:
                rx_bytes_per_s: float | None = None
                tx_bytes_per_s: float | None = None
                if gpm_sample is not None:
                    rx_bytes_per_s, tx_bytes_per_s = gpm_sample.link_throughput_bytes_per_s.get(
                        link.link_id,
                        (None, None),
                    )
                nvlinks.append(
                    NvLinkSnapshot(
                        link_id=link.link_id,
                        active=link.active,
                        speed_mb_s=link.speed_mb_s,
                        version=link.version,
                        rx_bytes_per_s=rx_bytes_per_s,
                        tx_bytes_per_s=tx_bytes_per_s,
                    )
                )
            gpus.append(
                GpuSnapshot(
                    index=gpu_index,
                    name=str(raw_gpu["name"]),
                    uuid=str(raw_gpu["uuid"]),
                    nvlink_metrics_source=source,
                    gpu_utilization_pct=raw_gpu["gpu_utilization_pct"],
                    memory_utilization_pct=raw_gpu["memory_utilization_pct"],
                    memory_used_bytes=raw_gpu["memory_used_bytes"],
                    memory_total_bytes=raw_gpu["memory_total_bytes"],
                    temperature_c=raw_gpu["temperature_c"],
                    power_usage_w=raw_gpu["power_usage_w"],
                    power_limit_w=raw_gpu["power_limit_w"],
                    fan_speed_pct=raw_gpu["fan_speed_pct"],
                    sm_clock_mhz=raw_gpu["sm_clock_mhz"],
                    mem_clock_mhz=raw_gpu["mem_clock_mhz"],
                    graphics_clock_mhz=raw_gpu["graphics_clock_mhz"],
                    pcie_gen=raw_gpu["pcie_gen"],
                    pcie_width=raw_gpu["pcie_width"],
                    pcie_rx_kib_s=raw_gpu["pcie_rx_kib_s"],
                    pcie_tx_kib_s=raw_gpu["pcie_tx_kib_s"],
                    nvlink_total_rx_bytes_per_s=None if dcgm_sample is None else dcgm_sample.total_rx_bytes_per_s,
                    nvlink_total_tx_bytes_per_s=None if dcgm_sample is None else dcgm_sample.total_tx_bytes_per_s,
                    nvlinks=nvlinks,
                )
            )
        return SystemSnapshot(
            hostname=socket.gethostname(),
            driver_version=driver_version,
            timestamp=time.time(),
            gpus=gpus,
            processes=processes,
            nvlink_matrix=nvlink_matrix,
        )
