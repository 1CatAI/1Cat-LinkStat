from __future__ import annotations

import unittest

from cat_linkstat.models import GpuProcessSnapshot, GpuSnapshot, NvLinkSnapshot, SystemSnapshot
from cat_linkstat.render import HistoryBuffer, RenderOptions, format_rate_mb_s, render_dashboard, render_screen


class RenderTests(unittest.TestCase):
    def test_format_rate_mb_s(self) -> None:
        self.assertEqual(format_rate_mb_s(25781), "25.781 GB/s")

    def test_render_screen_contains_nvlink_summary(self) -> None:
        snapshot = SystemSnapshot(
            hostname="node-a",
            driver_version="570.153.02",
            timestamp=1742865600.0,
            gpus=[
                GpuSnapshot(
                    index=0,
                    name="Tesla V100-SXM2-16GB",
                    uuid="GPU-123",
                    nvlink_metrics_source="state-poll",
                    gpu_utilization_pct=12,
                    memory_utilization_pct=8,
                    memory_used_bytes=2 * 1024**3,
                    memory_total_bytes=16 * 1024**3,
                    temperature_c=35,
                    power_usage_w=42.0,
                    power_limit_w=300.0,
                    fan_speed_pct=None,
                    sm_clock_mhz=1230,
                    mem_clock_mhz=877,
                    graphics_clock_mhz=1230,
                    nvlink_total_rx_bytes_per_s=40_000_000_000.0,
                    nvlink_total_tx_bytes_per_s=41_000_000_000.0,
                    nvlinks=[
                        NvLinkSnapshot(link_id=0, active=True, speed_mb_s=25781, version=2),
                        NvLinkSnapshot(link_id=1, active=True, speed_mb_s=25781, version=2),
                        NvLinkSnapshot(link_id=2, active=False, speed_mb_s=None, version=None),
                    ],
                )
            ],
        )

        text = render_screen(snapshot, 120, RenderOptions(color=False, bars=True, links_mode="summary", interval=1.0))
        self.assertIn("1CatLinkStat", text)
        self.assertIn("NVLink live state-poll", text)
        self.assertIn("active links 2", text)
        self.assertIn("per-link 25.781 GB/s", text)
        self.assertIn("total 51.562 GB/s", text)
        self.assertIn("NVRX", text)
        self.assertIn("NVTX", text)
        self.assertIn("40.0/51.6G", text)
        self.assertIn("41.0/51.6G", text)

    def test_render_screen_caps_content_width_at_85(self) -> None:
        snapshot = SystemSnapshot(
            hostname="node-a",
            driver_version="570.153.02",
            timestamp=1742865600.0,
            gpus=[
                GpuSnapshot(
                    index=0,
                    name="Tesla V100-SXM2-16GB",
                    uuid="GPU-123",
                    nvlink_metrics_source="state-poll",
                    gpu_utilization_pct=12,
                    memory_utilization_pct=8,
                    memory_used_bytes=2 * 1024**3,
                    memory_total_bytes=16 * 1024**3,
                    temperature_c=35,
                    power_usage_w=42.0,
                    power_limit_w=300.0,
                    fan_speed_pct=None,
                    sm_clock_mhz=1230,
                    mem_clock_mhz=877,
                    graphics_clock_mhz=1230,
                    nvlink_total_rx_bytes_per_s=40_000_000_000.0,
                    nvlink_total_tx_bytes_per_s=41_000_000_000.0,
                    nvlinks=[NvLinkSnapshot(link_id=0, active=True, speed_mb_s=25781, version=2)],
                )
            ],
        )

        text_at_85 = render_screen(
            snapshot,
            85,
            RenderOptions(color=False, bars=True, links_mode="summary", interval=1.0),
        )
        text_at_120 = render_screen(
            snapshot,
            120,
            RenderOptions(color=False, bars=True, links_mode="summary", interval=1.0),
        )
        self.assertEqual(text_at_85, text_at_120)

    def test_render_dashboard_uses_honest_footer_and_processes(self) -> None:
        snapshot = SystemSnapshot(
            hostname="node-a",
            driver_version="570.153.02",
            timestamp=1742865600.0,
            gpus=[
                GpuSnapshot(
                    index=0,
                    name="Tesla V100-SXM2-16GB",
                    uuid="GPU-123",
                    nvlink_metrics_source="dcgm-total",
                    gpu_utilization_pct=87,
                    memory_utilization_pct=8,
                    memory_used_bytes=2 * 1024**3,
                    memory_total_bytes=16 * 1024**3,
                    temperature_c=35,
                    power_usage_w=42.0,
                    power_limit_w=300.0,
                    fan_speed_pct=None,
                    sm_clock_mhz=1230,
                    mem_clock_mhz=877,
                    graphics_clock_mhz=1230,
                    nvlink_total_rx_bytes_per_s=40_000_000_000.0,
                    nvlink_total_tx_bytes_per_s=41_000_000_000.0,
                    nvlinks=[
                        NvLinkSnapshot(link_id=0, active=True, speed_mb_s=25781, version=2),
                        NvLinkSnapshot(link_id=1, active=True, speed_mb_s=25781, version=2),
                    ],
                )
            ],
            processes=[
                GpuProcessSnapshot(
                    pid=1234,
                    gpu_index=0,
                    process_type="C",
                    username="onecatai",
                    gpu_memory_mib=512,
                    gpu_sm_pct=77,
                    cpu_pct=31.0,
                    host_memory_mib=256.0,
                    command="./peer_copy 256 600",
                )
            ],
        )
        history = HistoryBuffer()
        for _ in range(20):
            history.update(snapshot, 64)

        text = render_dashboard(
            snapshot,
            120,
            32,
            RenderOptions(color=False, bars=True, links_mode="expanded", interval=1.0, events_limit=0),
            history,
        )
        self.assertIn("1CatLinkStat", text)
        self.assertIn("Activity", text)
        self.assertIn("NVLink", text)
        self.assertIn("Processes (1)", text)
        self.assertIn("Ctrl-C Quit", text)
        self.assertNotIn("F10 Quit", text)
        self.assertNotIn("PCIe GEN N/A@", text)
        self.assertIn("L0 up 25.781 GB/s v2", text)
        self.assertIn("NVRX", text)
        self.assertIn("NVTX", text)

    def test_render_dashboard_matrix_mode_shows_topology_grid(self) -> None:
        snapshot = SystemSnapshot(
            hostname="node-a",
            driver_version="570.153.02",
            timestamp=1742865600.0,
            gpus=[
                GpuSnapshot(
                    index=0,
                    name="Tesla V100-SXM2-16GB",
                    uuid="GPU-123",
                    nvlink_metrics_source="dcgm-total",
                    gpu_utilization_pct=10,
                    memory_utilization_pct=8,
                    memory_used_bytes=2 * 1024**3,
                    memory_total_bytes=16 * 1024**3,
                    temperature_c=35,
                    power_usage_w=42.0,
                    power_limit_w=300.0,
                    fan_speed_pct=None,
                    sm_clock_mhz=1230,
                    mem_clock_mhz=877,
                    graphics_clock_mhz=1230,
                    nvlink_total_rx_bytes_per_s=40_000_000_000.0,
                    nvlink_total_tx_bytes_per_s=41_000_000_000.0,
                    nvlinks=[NvLinkSnapshot(link_id=0, active=True, speed_mb_s=25781, version=2)],
                ),
                GpuSnapshot(
                    index=1,
                    name="Tesla V100-SXM2-16GB",
                    uuid="GPU-456",
                    nvlink_metrics_source="dcgm-total",
                    gpu_utilization_pct=20,
                    memory_utilization_pct=8,
                    memory_used_bytes=2 * 1024**3,
                    memory_total_bytes=16 * 1024**3,
                    temperature_c=36,
                    power_usage_w=43.0,
                    power_limit_w=300.0,
                    fan_speed_pct=None,
                    sm_clock_mhz=1230,
                    mem_clock_mhz=877,
                    graphics_clock_mhz=1230,
                    nvlink_total_rx_bytes_per_s=39_000_000_000.0,
                    nvlink_total_tx_bytes_per_s=38_000_000_000.0,
                    nvlinks=[NvLinkSnapshot(link_id=0, active=True, speed_mb_s=25781, version=2)],
                ),
            ],
            nvlink_matrix={
                0: {0: "X", 1: "NV6"},
                1: {0: "NV6", 1: "X"},
            },
        )
        history = HistoryBuffer()
        for _ in range(8):
            history.update(snapshot, 32)

        text = render_dashboard(
            snapshot,
            120,
            28,
            RenderOptions(color=False, bars=True, links_mode="matrix", interval=1.0, events_limit=0),
            history,
        )
        self.assertIn("NVLink Matrix", text)
        self.assertIn("G0", text)
        self.assertIn("G1", text)
        self.assertIn("NV6", text)
        self.assertIn("RX[", text)
        self.assertIn("TX[", text)

    def test_render_screen_with_events(self) -> None:
        snapshot = SystemSnapshot(
            hostname="node-a",
            driver_version="570.153.02",
            timestamp=1742865600.0,
            gpus=[],
        )
        text = render_screen(
            snapshot,
            120,
            RenderOptions(color=False, bars=True, links_mode="summary", interval=1.0),
            events=["12:00:01 GPU 0 NVLink L1 down -> up"],
        )
        self.assertIn("Recent NVLink Events", text)
        self.assertIn("GPU 0 NVLink L1 down -> up", text)


if __name__ == "__main__":
    unittest.main()
