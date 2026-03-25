from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime

from .models import GpuProcessSnapshot, GpuSnapshot, NvLinkSnapshot, SystemSnapshot

RESET = "\x1b[0m"
REVERSE = "\x1b[7m"
FG_GREEN = "\x1b[32m"
FG_YELLOW = "\x1b[33m"
FG_RED = "\x1b[31m"
FG_CYAN = "\x1b[36m"
FG_DIM = "\x1b[2m"
FG_BLUE = "\x1b[34m"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
RULE = "-"
SPARK_LEVELS = ".:-=+*#%@"
MAX_CONTENT_WIDTH = 85


@dataclass(frozen=True)
class RenderOptions:
    color: bool = True
    bars: bool = True
    links_mode: str = "summary"
    interval: float = 1.0
    events_limit: int = 6


@dataclass
class HistoryBuffer:
    gpu_util: dict[int, list[int | None]] = field(default_factory=dict)
    mem_util: dict[int, list[int | None]] = field(default_factory=dict)

    def update(self, snapshot: SystemSnapshot, points: int) -> None:
        points = max(8, points)
        for gpu in snapshot.gpus:
            self._push(self.gpu_util, gpu.index, gpu.gpu_utilization_pct, points)
            self._push(self.mem_util, gpu.index, memory_pct(gpu), points)

    def series(self, gpu_index: int, metric: str, points: int) -> list[int | None]:
        points = max(8, points)
        source = self.gpu_util if metric == "gpu" else self.mem_util
        values = source.get(gpu_index, [])
        if not values:
            return [None] * points
        if len(values) >= points:
            return values[-points:]
        return [values[0]] * (points - len(values)) + values

    @staticmethod
    def _push(store: dict[int, list[int | None]], gpu_index: int, value: int | None, points: int) -> None:
        bucket = store.setdefault(gpu_index, [])
        bucket.append(value)
        if len(bucket) > points:
            del bucket[: len(bucket) - points]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def visible_width(text: str) -> int:
    return len(strip_ansi(text))


def colorize(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{code}{text}{RESET}"


def reverse(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{REVERSE}{text}{RESET}"


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def pad_visible(text: str, width: int) -> str:
    current = visible_width(text)
    if current >= width:
        return text
    return text + (" " * (width - current))


def clamp_render_width(width: int) -> int:
    return max(1, min(width, MAX_CONTENT_WIDTH))


def compose_lr(left: str, right: str, width: int) -> str:
    if width <= 1:
        return truncate(left, width)
    left_width = visible_width(left)
    right_width = visible_width(right)
    if left_width + right_width + 1 <= width:
        return left + (" " * (width - left_width - right_width)) + right
    available_for_left = max(0, width - right_width - 1)
    return truncate(strip_ansi(left), available_for_left) + " " + strip_ansi(right)[: max(0, width - available_for_left - 1)]


def wrap_plain(text: str, width: int, indent: str = "  ") -> list[str]:
    if width <= 4:
        return [truncate(text, width)]
    return textwrap.wrap(
        text,
        width=width,
        initial_indent="",
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def section_heading(title: str, width: int, options: RenderOptions) -> str:
    clean_title = truncate(title, max(1, width))
    base = colorize(clean_title, FG_BLUE, options.color)
    if width <= visible_width(base) + 1:
        return pad_visible(base, width)
    return base + " " + (RULE * (width - visible_width(base) - 1))


def format_rate_mb_s(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1000.0:.3f} GB/s"


def format_throughput_mib_s(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{(value * (1024**2)) / 1_000_000_000:.3f} GB/s"


def format_gb_s(value: float | None, precision: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}G"


def format_kib_per_s(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f} KiB/s"


def format_number(value: int | float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def format_power_watts(value: float | None) -> str:
    if value is None:
        return "N/A"
    return str(int(round(value)))


def format_process_value(value: int | float | None, suffix: str = "", precision: int = 0) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"


def bandwidth_pct(current_gb_s: float | None, max_gb_s: float | None) -> int | None:
    if current_gb_s is None or not max_gb_s:
        return None
    return round(min(100.0, (current_gb_s / max_gb_s) * 100.0))


def bandwidth_suffix(current_gb_s: float | None, max_gb_s: float | None) -> str:
    if max_gb_s is None:
        return "N/A"
    if current_gb_s is None:
        return f"N/A/{max_gb_s:.1f}G"
    return f"{current_gb_s:.1f}/{max_gb_s:.1f}G"


def temperature_pct(temperature_c: int | None) -> int | None:
    if temperature_c is None:
        return None
    return round(min(100.0, max(0.0, float(temperature_c))))


def power_pct(power_usage_w: float | None, power_limit_w: float | None) -> int | None:
    if power_usage_w is None or not power_limit_w:
        return None
    return round(min(100.0, max(0.0, (power_usage_w / power_limit_w) * 100.0)))


def dual_bar_body_widths(
    width: int,
    left_label: str,
    left_suffix: str,
    right_label: str,
    right_suffix: str,
) -> tuple[int, int]:
    fixed = len(left_label) + len(right_label) + 1
    fixed += (2 + 1 + len(left_suffix)) + (2 + 1 + len(right_suffix))
    body_total = max(2, width - fixed)
    left_body = max(1, body_total // 2)
    right_body = max(1, body_total - left_body)
    return left_body, right_body


def memory_pct(gpu: GpuSnapshot) -> int | None:
    if gpu.memory_used_bytes is None or not gpu.memory_total_bytes:
        return None
    return round(gpu.memory_used_bytes * 100 / gpu.memory_total_bytes)


def memory_ratio(gpu: GpuSnapshot) -> str:
    if gpu.memory_used_bytes is None or gpu.memory_total_bytes is None:
        return "N/A"
    used = gpu.memory_used_bytes / (1024**3)
    total = gpu.memory_total_bytes / (1024**3)
    return f"{used:.3f}Gi/{total:.3f}Gi"


def summarize_sources(snapshot: SystemSnapshot) -> str:
    sources = sorted({gpu.nvlink_metrics_source for gpu in snapshot.gpus if gpu.nvlink_metrics_source})
    return ",".join(sources) if sources else "n/a"


def draw_nvtop_bar(value: int | None, body_width: int, options: RenderOptions, suffix: str) -> str:
    suffix = suffix.strip()
    body_width = max(1, body_width)
    if value is None:
        body = ("?" * body_width)
        return f"[{body} {suffix}]"
    fill = max(0, min(body_width, round(body_width * value / 100)))
    body = ("|" * fill) + (" " * (body_width - fill))
    if options.color and options.bars:
        color = FG_GREEN
        if value >= 80:
            color = FG_RED
        elif value >= 50:
            color = FG_YELLOW
        body = colorize(body, color, True)
    return f"[{body} {suffix}]"


def format_link_chunk(link: NvLinkSnapshot) -> str:
    chunk = (
        f"L{link.link_id} {'up' if link.active else 'down'} "
        f"{format_rate_mb_s(link.speed_mb_s)} "
        f"{'v' + str(link.version) if link.version is not None else 'v?'}"
    )
    if link.rx_bytes_per_s is not None or link.tx_bytes_per_s is not None:
        rx_gbps = None if link.rx_bytes_per_s is None else link.rx_bytes_per_s / (1024**2)
        tx_gbps = None if link.tx_bytes_per_s is None else link.tx_bytes_per_s / (1024**2)
        chunk += f" rx {format_throughput_mib_s(rx_gbps)} tx {format_throughput_mib_s(tx_gbps)}"
    return chunk


def render_system_overview(snapshot: SystemSnapshot, width: int, options: RenderOptions) -> list[str]:
    timestamp = datetime.fromtimestamp(snapshot.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    active_links = sum(gpu.active_nvlink_count for gpu in snapshot.gpus)
    left = f"1CatLinkStat  host {snapshot.hostname}"
    right = (
        f"driver {snapshot.driver_version or 'unknown'}  "
        f"gpus {len(snapshot.gpus)}  procs {len(snapshot.processes)}  time {timestamp}"
    )
    line1 = compose_lr(left, right, width)
    left2 = (
        f"NVLink live {summarize_sources(snapshot)}  "
        f"active links {active_links}  links {options.links_mode}"
    )
    right2 = f"refresh {options.interval:.1f}s"
    return [pad_visible(line1, width), pad_visible(compose_lr(left2, right2, width), width)]


def render_device_block(gpu: GpuSnapshot, width: int, options: RenderOptions) -> list[str]:
    header_left = f"Device {gpu.index} [{gpu.name}]"
    total_links = len(gpu.nvlinks)
    version = f"v{gpu.dominant_nvlink_version}" if gpu.dominant_nvlink_version is not None else "v?"

    line1 = pad_visible(header_left, width)
    line2 = pad_visible(
        "  ".join(
            [
                f"PCIe GEN {format_number(gpu.pcie_gen)}",
                f"WIDTH {format_number(gpu.pcie_width)}x",
                f"RX {format_kib_per_s(gpu.pcie_rx_kib_s)}",
                f"TX {format_kib_per_s(gpu.pcie_tx_kib_s)}",
            ]
        ),
        width,
    )
    line3 = pad_visible(
        "  ".join(
            [
                f"GPU {format_number(gpu.graphics_clock_mhz, 'MHz')}",
                f"MEM {format_number(gpu.mem_clock_mhz, 'MHz')}",
                f"TEMP {format_number(gpu.temperature_c, 'C')}",
                f"FAN {format_number(gpu.fan_speed_pct, '%')}",
                f"POW {format_power_watts(gpu.power_usage_w)} / {format_power_watts(gpu.power_limit_w)} W",
            ]
        ),
        width,
    )

    gpu_suffix = format_number(gpu.gpu_utilization_pct, '%')
    mem_suffix = memory_ratio(gpu)
    gpu_bar_width, mem_bar_width = dual_bar_body_widths(width, "GPU", gpu_suffix, "MEM", mem_suffix)
    line4 = pad_visible(
        " ".join(
            [
                f"GPU{draw_nvtop_bar(gpu.gpu_utilization_pct, gpu_bar_width, options, gpu_suffix)}",
                f"MEM{draw_nvtop_bar(memory_pct(gpu), mem_bar_width, options, mem_suffix)}",
            ]
        ),
        width,
    )

    temp_suffix = "N/A" if gpu.temperature_c is None else f"{gpu.temperature_c}C/100C"
    power_suffix = "N/A" if gpu.power_usage_w is None or gpu.power_limit_w is None else f"{format_power_watts(gpu.power_usage_w)}/{format_power_watts(gpu.power_limit_w)}W"
    temp_bar_width, power_bar_width = dual_bar_body_widths(width, "TEMP", temp_suffix, "POW", power_suffix)
    line5 = pad_visible(
        " ".join(
            [
                f"TEMP{draw_nvtop_bar(temperature_pct(gpu.temperature_c), temp_bar_width, options, temp_suffix)}",
                f"POW{draw_nvtop_bar(power_pct(gpu.power_usage_w, gpu.power_limit_w), power_bar_width, options, power_suffix)}",
            ]
        ),
        width,
    )

    nvrx_suffix = bandwidth_suffix(gpu.total_nvlink_rx_gb_s, gpu.total_nvlink_bandwidth_gb_s)
    nvtx_suffix = bandwidth_suffix(gpu.total_nvlink_tx_gb_s, gpu.total_nvlink_bandwidth_gb_s)
    nvrx_bar_width, nvtx_bar_width = dual_bar_body_widths(width, "NVRX", nvrx_suffix, "NVTX", nvtx_suffix)
    line6 = pad_visible(
        " ".join(
            [
                f"NVRX{draw_nvtop_bar(bandwidth_pct(gpu.total_nvlink_rx_gb_s, gpu.total_nvlink_bandwidth_gb_s), nvrx_bar_width, options, nvrx_suffix)}",
                f"NVTX{draw_nvtop_bar(bandwidth_pct(gpu.total_nvlink_tx_gb_s, gpu.total_nvlink_bandwidth_gb_s), nvtx_bar_width, options, nvtx_suffix)}",
            ]
        ),
        width,
    )

    summary = [
        f"NVL {gpu.active_nvlink_count}/{total_links}",
        version,
        f"per-link {format_rate_mb_s(gpu.per_link_nvlink_rate_mb_s)}",
        f"total {format_rate_mb_s(gpu.total_nvlink_rate_mb_s)}",
        f"src {gpu.nvlink_metrics_source}",
    ]
    line7 = pad_visible("  ".join(summary), width)
    return [line1, line2, line3, line4, line5, line6, line7]


def render_gpu(gpu: GpuSnapshot, width: int, options: RenderOptions) -> list[str]:
    return render_device_block(gpu, width, options)


def downsample_series(series: list[int | None], width: int) -> list[float | None]:
    if width <= 0:
        return []
    if not series:
        return [None] * width
    result: list[float | None] = []
    total = len(series)
    for index in range(width):
        start = round(index * total / width)
        end = round((index + 1) * total / width)
        chunk = [value for value in series[start:end] if value is not None]
        if chunk:
            result.append(sum(chunk) / len(chunk))
        else:
            result.append(None)
    return result


def render_sparkline(series: list[int | None], width: int, color: str, options: RenderOptions) -> str:
    values = downsample_series(series, width)
    chars: list[str] = []
    max_index = len(SPARK_LEVELS) - 1
    for value in values:
        if value is None:
            chars.append(" ")
            continue
        level = max(0, min(max_index, round((value / 100.0) * max_index)))
        chars.append(SPARK_LEVELS[level])
    return colorize("".join(chars), color, options.color)


def render_activity_section(
    snapshot: SystemSnapshot,
    width: int,
    available_height: int,
    options: RenderOptions,
    history: HistoryBuffer,
) -> list[str]:
    if available_height < 2 or not snapshot.gpus:
        return []

    spark_budget = max(8, min(28, (width - 28) // 2))
    window_seconds = int(round(max(1, spark_budget - 1) * options.interval))
    lines = [pad_visible(section_heading(f"Activity ({window_seconds}s recent)", width, options), width)]

    visible_gpu_count = max(0, available_height - 1)
    for gpu in snapshot.gpus[:visible_gpu_count]:
        gpu_line = render_sparkline(history.series(gpu.index, "gpu", spark_budget), spark_budget, FG_GREEN, options)
        mem_line = render_sparkline(history.series(gpu.index, "mem", spark_budget), spark_budget, FG_CYAN, options)
        row = (
            f"GPU{gpu.index} U {format_process_value(gpu.gpu_utilization_pct, '%'):>4} {gpu_line}  "
            f"M {format_process_value(memory_pct(gpu), '%'):>4} {mem_line}"
        )
        lines.append(pad_visible(row, width))

    hidden = len(snapshot.gpus) - visible_gpu_count
    if hidden > 0 and len(lines) < available_height:
        lines.append(pad_visible(f"... {hidden} more GPU activity rows", width))
    return lines[:available_height]


def matrix_cell(label: str, cell_width: int, options: RenderOptions, diagonal: bool = False) -> str:
    text = truncate(label, cell_width).center(cell_width)
    if not options.color:
        return text
    if diagonal:
        return colorize(text, FG_DIM, True)
    if label.startswith("NV"):
        return colorize(text, FG_CYAN, True)
    if label in {"PIX", "PXB", "PHB", "NODE", "SYS"}:
        return colorize(text, FG_YELLOW, True)
    if label == "?":
        return colorize(text, FG_RED, True)
    return text


def render_nvlink_matrix_section(
    snapshot: SystemSnapshot,
    width: int,
    available_height: int,
    options: RenderOptions,
) -> list[str]:
    if available_height < 4 or not snapshot.gpus:
        return []

    gpu_indices = [gpu.index for gpu in snapshot.gpus]
    if not gpu_indices:
        return []

    row_label_width = 5
    cell_width = max(4, min(8, (width - row_label_width - 2) // max(1, len(gpu_indices))))
    lines = [pad_visible(section_heading("NVLink Matrix", width, options), width)]

    header = " " * row_label_width + " " + " ".join(
        matrix_cell(f"G{index}", cell_width, options) for index in gpu_indices
    )
    lines.append(pad_visible(header, width))

    gpu_map = {gpu.index: gpu for gpu in snapshot.gpus}
    for gpu_index in gpu_indices:
        if len(lines) >= available_height:
            return lines[:available_height]
        row_cells = []
        for column_index in gpu_indices:
            label = snapshot.nvlink_matrix.get(gpu_index, {}).get(column_index, "X" if gpu_index == column_index else "?")
            row_cells.append(matrix_cell(label, cell_width, options, diagonal=gpu_index == column_index))
        row = f"G{gpu_index:<{row_label_width - 1}} " + " ".join(row_cells)
        lines.append(pad_visible(row, width))

    for gpu_index in gpu_indices:
        if len(lines) >= available_height:
            return lines[:available_height]
        gpu = gpu_map[gpu_index]
        nv_bar_width = 20 if width >= 140 else 16 if width >= 120 else 12 if width >= 100 else 10
        bar_row = (
            f"G{gpu_index} "
            f"RX{draw_nvtop_bar(bandwidth_pct(gpu.total_nvlink_rx_gb_s, gpu.total_nvlink_bandwidth_gb_s), nv_bar_width, options, bandwidth_suffix(gpu.total_nvlink_rx_gb_s, gpu.total_nvlink_bandwidth_gb_s))} "
            f"TX{draw_nvtop_bar(bandwidth_pct(gpu.total_nvlink_tx_gb_s, gpu.total_nvlink_bandwidth_gb_s), nv_bar_width, options, bandwidth_suffix(gpu.total_nvlink_tx_gb_s, gpu.total_nvlink_bandwidth_gb_s))}"
        )
        lines.append(pad_visible(bar_row, width))

    if len(lines) < available_height:
        legend = "Legend: NV# bonded NVLinks, PIX/PXB/PHB/SYS are PCIe-path topology classes"
        lines.append(pad_visible(truncate(legend, width), width))
    return lines[:available_height]


def render_nvlink_section(
    snapshot: SystemSnapshot,
    width: int,
    available_height: int,
    options: RenderOptions,
) -> list[str]:
    if available_height < 2 or not snapshot.gpus or options.links_mode != "expanded":
        return []

    lines = [pad_visible(section_heading("NVLink", width, options), width)]
    used = 1

    for gpu in snapshot.gpus:
        if used >= available_height:
            break
        total_links = len(gpu.nvlinks)
        summary = (
            f"GPU{gpu.index} active={gpu.active_nvlink_count}/{total_links} "
            f"ver={'v' + str(gpu.dominant_nvlink_version) if gpu.dominant_nvlink_version is not None else 'v?'} "
            f"per-link={format_rate_mb_s(gpu.per_link_nvlink_rate_mb_s)} "
            f"total={format_rate_mb_s(gpu.total_nvlink_rate_mb_s)} "
            f"src={gpu.nvlink_metrics_source}"
        )
        if gpu.has_live_nvlink_throughput:
            summary += (
                f" rx={format_throughput_mib_s(gpu.total_nvlink_rx_mib_s)}"
                f" tx={format_throughput_mib_s(gpu.total_nvlink_tx_mib_s)}"
            )
        lines.append(pad_visible(truncate(summary, width), width))
        used += 1

    for gpu in snapshot.gpus:
        if used >= available_height:
            break
        detail = "  " + " | ".join(format_link_chunk(link) for link in gpu.nvlinks)
        for line in wrap_plain(detail, width, indent="    "):
            if used >= available_height:
                break
            lines.append(pad_visible(line, width))
            used += 1

    if used < available_height and len(snapshot.gpus) > max(0, available_height - 1):
        lines.append(pad_visible(f"... {len(snapshot.gpus) - (available_height - 1)} more GPU NVLink summaries", width))
    return lines[:available_height]


def render_process_table(
    processes: list[GpuProcessSnapshot],
    width: int,
    available_height: int,
    options: RenderOptions,
) -> list[str]:
    if available_height < 3:
        return []

    pid_w = 7
    user_w = 10
    dev_w = 3
    type_w = 7
    sm_w = 5
    gpu_mem_w = 11
    cpu_w = 6
    host_mem_w = 10
    fixed = pid_w + user_w + dev_w + type_w + sm_w + gpu_mem_w + cpu_w + host_mem_w + 8
    cmd_w = max(12, width - fixed)

    title = pad_visible(section_heading(f"Processes ({len(processes)})", width, options), width)
    header = (
        f"{'PID':>{pid_w}} {'USER':<{user_w}} {'GPU':>{dev_w}} {'TYPE':<{type_w}} "
        f"{'SM':>{sm_w}} {'GPU MEM':>{gpu_mem_w}} {'CPU':>{cpu_w}} {'HOST MEM':>{host_mem_w}} {'Command':<{cmd_w}}"
    )
    lines = [title, reverse(pad_visible(header, width), options.color)]

    if not processes:
        lines.append(pad_visible("No GPU processes.", width))
        return lines[:available_height]

    row_limit = max(1, available_height - len(lines))
    visible_processes = processes[:row_limit]
    truncated = len(processes) - len(visible_processes)
    if truncated > 0 and row_limit > 1:
        visible_processes = processes[: row_limit - 1]
        truncated = len(processes) - len(visible_processes)

    for process in visible_processes:
        process_type = "Compute" if process.process_type == "C" else "Graphic" if process.process_type == "G" else (process.process_type or "-")
        command = truncate(process.command, cmd_w)
        row = (
            f"{process.pid:>{pid_w}} "
            f"{truncate(process.username or '-', user_w):<{user_w}} "
            f"{process.gpu_index:>{dev_w}} "
            f"{truncate(process_type, type_w):<{type_w}} "
            f"{format_process_value(process.gpu_sm_pct, '%'):>{sm_w}} "
            f"{format_process_value(process.gpu_memory_mib, 'MiB'):>{gpu_mem_w}} "
            f"{format_process_value(process.cpu_pct, '%', precision=0):>{cpu_w}} "
            f"{format_process_value(process.host_memory_mib, 'MiB', precision=0):>{host_mem_w}} "
            f"{command:<{cmd_w}}"
        )
        lines.append(pad_visible(row, width))

    if truncated > 0 and len(lines) < available_height:
        lines.append(pad_visible(f"... {truncated} more GPU processes", width))
    return lines[:available_height]


def render_screen(
    snapshot: SystemSnapshot,
    width: int,
    options: RenderOptions,
    events: list[str] | None = None,
) -> str:
    width = clamp_render_width(width)
    lines = render_system_overview(snapshot, width, options)
    lines.append("")
    if not snapshot.gpus:
        lines.append("No NVIDIA GPUs detected.")
    else:
        for gpu_index, gpu in enumerate(snapshot.gpus):
            lines.extend(render_gpu(gpu, width, options))
            if gpu_index != len(snapshot.gpus) - 1:
                lines.append("")
    nvlink_lines: list[str] = []
    if options.links_mode == "expanded":
        nvlink_lines = render_nvlink_section(snapshot, width, 1 + (len(snapshot.gpus) * 3), options)
    elif options.links_mode == "matrix":
        nvlink_lines = render_nvlink_matrix_section(snapshot, width, 3 + (len(snapshot.gpus) * 2), options)
    if nvlink_lines:
        lines.append("")
        lines.extend(nvlink_lines)
    process_lines = render_process_table(snapshot.processes, width, len(snapshot.processes) + 3, options)
    if process_lines:
        lines.append("")
        lines.extend(process_lines)
    if events is not None:
        lines.append("")
        lines.append("Recent NVLink Events")
        if events:
            for event in events[-options.events_limit :]:
                lines.append(f"  {event}")
        else:
            lines.append("  watching for NVLink link-count and rate changes")
    return "\n".join(lines)


def render_dashboard(
    snapshot: SystemSnapshot,
    width: int,
    height: int,
    options: RenderOptions,
    history: HistoryBuffer,
    events: list[str] | None = None,
) -> str:
    width = clamp_render_width(width)
    history_points = max(24, min(120, width))
    history.update(snapshot, history_points)

    event_line_count = 1 if options.events_limit > 0 else 0
    footer_lines = 1 + event_line_count
    content_limit = max(0, height - footer_lines)

    lines = render_system_overview(snapshot, width, options)
    lines.append("")
    for gpu_index, gpu in enumerate(snapshot.gpus):
        lines.extend(render_device_block(gpu, width, options))
        if gpu_index != len(snapshot.gpus) - 1:
            lines.append("")

    remaining = max(0, content_limit - len(lines))
    min_process_lines = 4 if snapshot.processes else 3

    if options.links_mode == "matrix":
        matrix_budget = min(max(5, 3 + (len(snapshot.gpus) * 2)), max(0, remaining - min_process_lines - 1))
        matrix_lines = render_nvlink_matrix_section(snapshot, width, matrix_budget, options)
        if matrix_lines and remaining >= len(matrix_lines) + min_process_lines + 1:
            lines.append("")
            lines.extend(matrix_lines)
            remaining = max(0, content_limit - len(lines))

        activity_budget = min(1 + len(snapshot.gpus), max(0, remaining - min_process_lines - 1))
        activity_lines = render_activity_section(snapshot, width, activity_budget, options, history)
        if activity_lines and remaining >= len(activity_lines) + min_process_lines + 1:
            lines.append("")
            lines.extend(activity_lines)
            remaining = max(0, content_limit - len(lines))
    else:
        activity_budget = min(1 + len(snapshot.gpus), max(0, remaining - min_process_lines - 1))
        activity_lines = render_activity_section(snapshot, width, activity_budget, options, history)
        if activity_lines and remaining >= len(activity_lines) + min_process_lines + 1:
            lines.append("")
            lines.extend(activity_lines)
            remaining = max(0, content_limit - len(lines))

        nvlink_budget = 0
        if options.links_mode == "expanded":
            nvlink_budget = min(1 + (len(snapshot.gpus) * 3), max(0, remaining - min_process_lines - 1))
        nvlink_lines = render_nvlink_section(snapshot, width, nvlink_budget, options)
        if nvlink_lines and remaining >= len(nvlink_lines) + min_process_lines + 1:
            lines.append("")
            lines.extend(nvlink_lines)
            remaining = max(0, content_limit - len(lines))

    process_budget = max(min_process_lines, remaining - 1)
    process_lines = render_process_table(snapshot.processes, width, process_budget, options)
    if process_lines:
        lines.append("")
        lines.extend(process_lines)

    if len(lines) > content_limit:
        lines = lines[:content_limit]
    while len(lines) < content_limit:
        lines.append("")

    if options.events_limit > 0:
        if events:
            event_text = f"Recent NVLink Event: {events[-1]}"
        else:
            event_text = "Recent NVLink Event: watching for link-count, state, rate, or source changes"
        lines.append(colorize(pad_visible(truncate(event_text, width), width), FG_DIM, options.color))

    status_left = (
        f"Ctrl-C Quit  refresh {options.interval:.1f}s  "
        f"links {options.links_mode}  nvlink {summarize_sources(snapshot)}"
    )
    live_state = "live RX/TX on" if any(gpu.has_live_nvlink_throughput for gpu in snapshot.gpus) else "live RX/TX unavailable"
    lines.append(reverse(pad_visible(compose_lr(status_left, live_state, width), width), options.color))
    return "\n".join(lines[:height])
