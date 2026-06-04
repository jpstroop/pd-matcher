# publisher-viz

A static, single-page visualization of mid-20th-century U.S. and U.K. book-publishing consolidation, derived from the hand-curated reference at `data/publisher_imprints.json` and the structured events at `data/publisher_events.json`.

## What it shows

A force-directed graph of publishing houses (~100 entities + ~40 external acquirers). Drag the **year slider** at the top to scrub from 1820 to 2025; nodes appear as houses are founded, edges appear as acquisitions/mergers happen, and node colors shift as houses go from independent → acquired → defunct. Click any node for full event history, imprints, and source citations.

## Running locally

The page is fully self-contained except for the d3 CDN and the bundled `data.js`. To view it, you have two options:

**1. Open `index.html` directly in a browser** — works in Safari and most browsers because `data.js` is a `<script>` (not `fetch`'d JSON).

**2. Run any static server from this directory**:

```sh
cd docs/publisher-viz
python3 -m http.server 8000
# then open http://localhost:8000/
```

## Updating the data

The `data.js` file is a bundled mirror of the two canonical JSON sources in `data/`. After editing `data/publisher_imprints.json` or `data/publisher_events.json`, regenerate it:

```sh
docs/publisher-viz/build-data.sh
```

## Deploying to GitHub Pages

Settings → Pages → "Build and deployment" → Source: **Deploy from a branch** → Branch: **`main`** → Folder: **`/docs`**. The page lands at `https://<user>.github.io/pd-matcher/publisher-viz/`.

## Files

- `index.html` — the whole viz (CSS + JS inline).
- `data.js` — bundled `data/publisher_imprints.json` + `data/publisher_events.json`, exposed on `window`.
- `build-data.sh` — regenerator for `data.js`.
- `README.md` — this file.

## Caveats

- Dates are pulled from the events JSON; for entities without a `founded` event (most external acquirers) the founding year is *inferred* from the earliest event in which they appear as an actor.
- Force-directed layout is non-deterministic — the network's shape will differ run-to-run. Use the zoom + drag controls if a layout settles awkwardly.
- The legend's "External acquirer" category covers parent conglomerates (Bertelsmann, RCA, News Corp, etc.) that aren't book publishers themselves; they're shown to make ownership flow legible.
