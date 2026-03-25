#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"

mkdir -p "$DIST_DIR"
cd "$ROOT_DIR"

dpkg-buildpackage -us -uc -b

find "$ROOT_DIR/.." -maxdepth 1 -type f -name '1catlinkstat_*_all.deb' -exec cp -f {} "$DIST_DIR/" \;

echo "Built package(s):"
find "$DIST_DIR" -maxdepth 1 -type f -name '1catlinkstat_*_all.deb' -print | sort
