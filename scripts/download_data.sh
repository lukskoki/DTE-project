#!/usr/bin/env bash
# Downloads the SDHAR-HOME dataset (62 daily zips, ~900 MB) into data/raw/SDHAR-HOME.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw/SDHAR-HOME"
URL="https://github.com/raugom13/SDHAR-HOME-A-Sensor-Dataset-for-Human-Activity-Recognition-at-Home"
if [ -d "$DEST/.git" ]; then
  echo "Dataset already cloned at $DEST"
else
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 "$URL" "$DEST"
fi
echo "Done: $(ls "$DEST"/DAY_*.zip | wc -l) day zips in $DEST (the loader reads the zips directly)"
