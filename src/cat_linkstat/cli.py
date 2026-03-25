from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

from .models import SystemSnapshot
from .nvml import NvmlError, NvmlMonitor
from .render import HistoryBuffer, RenderOptions, render_dashboard, render_screen
from .tracking import NvLinkChangeTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="1CatLinkStat",
        description="Real-time GPU monitor with NVLink channel count and rate reporting.",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument("--json", action="store_true", help="Print snapshot as JSON. Requires --once.")
    parser.add_argument(
        "--links",
        choices=("summary", "expanded", "matrix"),
        default="summary",
        help="NVLink display mode.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument("--no-bars", action="store_true", help="Disable utilization bars.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between refreshes.")
    parser.add_argument("--events", type=int, default=0, help="Number of recent NVLink change events to show in watch mode.")
    return parser


def snapshot_to_json(snapshot: SystemSnapshot) -> str:
    return json.dumps(snapshot.to_dict(), indent=2)


def emit_screen(text: str, clear: bool) -> None:
    if clear:
        sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def run_once(monitor: NvmlMonitor, args: argparse.Namespace) -> int:
    snapshot = monitor.collect()
    if args.json:
        print(snapshot_to_json(snapshot))
        return 0
    width = shutil.get_terminal_size(fallback=(120, 40)).columns
    options = RenderOptions(
        color=not args.no_color,
        bars=not args.no_bars,
        links_mode=args.links,
        interval=args.interval,
        events_limit=max(0, args.events),
    )
    print(render_screen(snapshot, width, options))
    return 0


def watch(monitor: NvmlMonitor, args: argparse.Namespace) -> int:
    options = RenderOptions(
        color=not args.no_color,
        bars=not args.no_bars,
        links_mode=args.links,
        interval=args.interval,
        events_limit=max(0, args.events),
    )
    tracker = NvLinkChangeTracker(max_events=max(1, args.events)) if args.events > 0 else None
    history = HistoryBuffer()
    use_alt_screen = not args.no_clear
    if use_alt_screen:
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
    while True:
        loop_started = time.monotonic()
        snapshot = monitor.collect()
        terminal_size = shutil.get_terminal_size(fallback=(160, 44))
        width = terminal_size.columns
        height = terminal_size.lines
        events = None
        if tracker is not None:
            events = [event.render() for event in tracker.update(snapshot)]
        screen = render_dashboard(snapshot, width, height, options, history, events=events)
        emit_screen(screen, clear=not args.no_clear)
        elapsed = time.monotonic() - loop_started
        sleep_for = max(0.0, args.interval - elapsed)
        time.sleep(sleep_for)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be greater than 0.")
    if args.events < 0:
        parser.error("--events must be greater than or equal to 0.")
    if args.json and not args.once:
        parser.error("--json requires --once.")

    try:
        monitor = NvmlMonitor()
    except NvmlError as exc:
        print(f"1CatLinkStat: {exc}", file=sys.stderr)
        return 1

    try:
        if args.once:
            return run_once(monitor, args)
        return watch(monitor, args)
    except KeyboardInterrupt:
        return 0
    except NvmlError as exc:
        print(f"1CatLinkStat: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.once and not args.no_clear:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
        try:
            monitor.close()
        except NvmlError:
            pass
