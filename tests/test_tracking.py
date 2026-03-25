from __future__ import annotations

import unittest

from cat_linkstat.models import GpuSnapshot, NvLinkSnapshot, SystemSnapshot
from cat_linkstat.tracking import NvLinkChangeTracker


def make_snapshot(active: bool, speed_mb_s: int | None, timestamp: float) -> SystemSnapshot:
    return SystemSnapshot(
        hostname="node-a",
        driver_version="570.153.02",
        timestamp=timestamp,
        gpus=[
            GpuSnapshot(
                index=0,
                name="Tesla V100-SXM2-16GB",
                uuid="GPU-123",
                nvlink_metrics_source="state-poll",
                gpu_utilization_pct=0,
                memory_utilization_pct=0,
                memory_used_bytes=0,
                memory_total_bytes=16 * 1024**3,
                temperature_c=35,
                power_usage_w=40.0,
                power_limit_w=300.0,
                fan_speed_pct=None,
                sm_clock_mhz=135,
                mem_clock_mhz=877,
                graphics_clock_mhz=135,
                nvlinks=[NvLinkSnapshot(link_id=0, active=active, speed_mb_s=speed_mb_s, version=2)],
            )
        ],
    )


class TrackingTests(unittest.TestCase):
    def test_tracker_reports_nvlink_state_change(self) -> None:
        tracker = NvLinkChangeTracker(max_events=4)
        tracker.update(make_snapshot(active=True, speed_mb_s=25781, timestamp=1.0))
        events = tracker.update(make_snapshot(active=False, speed_mb_s=25781, timestamp=2.0))
        rendered = [event.render() for event in events]
        self.assertTrue(any("active NVLink channels 1 -> 0" in event for event in rendered))
        self.assertTrue(any("NVLink L0 up -> down" in event for event in rendered))

    def test_tracker_reports_nvlink_rate_change(self) -> None:
        tracker = NvLinkChangeTracker(max_events=4)
        tracker.update(make_snapshot(active=True, speed_mb_s=25781, timestamp=1.0))
        events = tracker.update(make_snapshot(active=True, speed_mb_s=20000, timestamp=2.0))
        rendered = [event.render() for event in events]
        self.assertTrue(any("aggregate NVLink rate 25.781 GB/s -> 20.000 GB/s" in event for event in rendered))
        self.assertTrue(any("NVLink L0 rate 25.781 GB/s -> 20.000 GB/s" in event for event in rendered))
