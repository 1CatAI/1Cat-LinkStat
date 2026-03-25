from __future__ import annotations

import unittest

from cat_linkstat.nvml import NvidiaTopologySampler


class NvmlTopologyTests(unittest.TestCase):
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
