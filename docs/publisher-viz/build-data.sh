#!/bin/sh
# Regenerate data.js from the canonical JSON sources in data/.
# Run from the repo root.
set -eu

cd "$(dirname "$0")/../.."
python3 - <<'PY'
from json import load, dumps
with open("data/publisher_imprints.json") as f:
    publishers = load(f)
with open("data/publisher_events.json") as f:
    events = load(f)
out = "window.PUBLISHERS = " + dumps(publishers, ensure_ascii=False, separators=(",", ":")) + ";\n"
out += "window.EVENTS = " + dumps(events, ensure_ascii=False, separators=(",", ":")) + ";\n"
with open("docs/publisher-viz/data.js", "w", encoding="utf-8") as f:
    f.write(out)
print(f"wrote {len(out):,} chars to docs/publisher-viz/data.js")
PY
