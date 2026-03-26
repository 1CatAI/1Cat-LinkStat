from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cat_linkstat import __version__
from cat_linkstat.bench import BENCH_BINARY_STEM, binary_path, build_command, default_cache_dir, needs_rebuild


class BenchTests(unittest.TestCase):
    def test_default_cache_dir_prefers_xdg(self) -> None:
        cache_dir = default_cache_dir({"XDG_CACHE_HOME": "/tmp/xdg-cache"}, home=Path("/home/tester"))
        self.assertEqual(cache_dir, Path("/tmp/xdg-cache") / "1catlinkstat")

    def test_binary_path_uses_versioned_name(self) -> None:
        path = binary_path(Path("/tmp/cache"))
        self.assertEqual(path, Path("/tmp/cache") / f"{BENCH_BINARY_STEM}-{__version__}")

    def test_needs_rebuild_when_binary_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "source.cu"
            source.write_text("// bench\n", encoding="utf-8")
            binary = Path(tempdir) / "bench"
            self.assertTrue(needs_rebuild(source, binary))

    def test_needs_rebuild_when_source_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "source.cu"
            binary = Path(tempdir) / "bench"
            source.write_text("// bench\n", encoding="utf-8")
            binary.write_text("bin\n", encoding="utf-8")
            os.utime(binary, (source.stat().st_mtime - 5, source.stat().st_mtime - 5))
            self.assertTrue(needs_rebuild(source, binary))

    def test_build_command_points_to_source_and_output(self) -> None:
        source = Path("/tmp/source.cu")
        binary = Path("/tmp/bench")
        command = build_command("nvcc", source, binary)
        self.assertEqual(command, ["nvcc", "-O3", "-std=c++17", str(source), "-o", str(binary)])


if __name__ == "__main__":
    unittest.main()
