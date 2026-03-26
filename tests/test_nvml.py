from __future__ import annotations

import unittest

from cat_linkstat.nvml import DcgmiNvLinkSampler, NvidiaTopologySampler


class NvmlTopologyTests(unittest.TestCase):
    def test_parse_dcgmi_dmon_uses_latest_sample(self) -> None:
        raw = (
            "#Entity   NVLTX                       NVLRX                       \n"
            "ID                                                                \n"
            "GPU 1     0                           0                           \n"
            "GPU 0     0                           0                           \n"
            "GPU 1     144419184248                144423560785               \n"
            "GPU 0     144429797240                144425422196               \n"
        )
        samples = DcgmiNvLinkSampler.parse_dmon_output(raw)
        self.assertEqual(samples[0].total_tx_bytes_per_s, 144429797240.0)
        self.assertEqual(samples[0].total_rx_bytes_per_s, 144425422196.0)
        self.assertEqual(samples[1].total_tx_bytes_per_s, 144419184248.0)
        self.assertEqual(samples[1].total_rx_bytes_per_s, 144423560785.0)

    def test_parse_topology_matrix(self) -> None:
        raw = (
            "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
            "GPU0\t X \tNV6\t0-15\t0\t\tN/A\n"
            "GPU1\tNV6\t X \t0-15\t0\t\tN/A\n"
        )
        matrix = NvidiaTopologySampler.parse_matrix(raw, gpu_count=2)
        self.assertEqual(matrix[0][0].strip(), "X")
        self.assertEqual(matrix[0][1], "NV6")
        self.assertEqual(matrix[1][0], "NV6")
        self.assertEqual(matrix[1][1].strip(), "X")


if __name__ == "__main__":
    unittest.main()
