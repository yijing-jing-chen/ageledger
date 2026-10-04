#!/usr/bin/env bash
# Dev only (not published): sync the published copies from their sources of truth, then wrap the
# fragment in a minimal document so it can be opened from `python3 -m http.server` in this folder.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
cp "$root/ageledger/levers/rules.yaml" "$here/rules.yaml"
mkdir -p "$here/sample"
cp "$root/data/synthetic/whoop_synthetic_180d.zip" "$here/sample/"
[ -f "$root/data/synthetic/calendar_synthetic.ics" ] && cp "$root/data/synthetic/calendar_synthetic.ics" "$here/sample/"
{
  printf '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>\n'
  cat "$here/index.html"
  printf '\n</body></html>\n'
} > "$here/_dev.html"
echo "wrote $here/_dev.html  (serve: cd $here && python3 -m http.server 8765)"
