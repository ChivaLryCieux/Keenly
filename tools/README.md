# Keenly analysis & maintenance scripts (Python)

These scripts used to live in `out/` - a gitignored directory - which meant
they were never version controlled. They are tooling, not output, so they
live here now. Run them from the repo root; they need `Pillow` + `numpy`.

Scripts:

- `grid_stats.py`    - per-cell stats for a 3x3 showcase grid (lum/std/edge/sat
                       + 8-bin histogram). Usage: `python tools\grid_stats.py [png] [tile] [gap]`.
- `comp_stats.py`    - composition metric (8x8 block-mean std) for rendered
                       frames. Usage: `python tools\comp_stats.py img.png name [...]`.
- `validate_rd.py`   - Gray-Scott parameter validation: mirrors `src/turing.cuh`
                       (same hash, update, wrap) in numpy and prints B-field
                       stats for the Pearson-morphology parameter pairs.
- `fixdocs.py`       - one-shot docs restructure helper (art.md -> pipeline.md),
                       kept for the record; already applied, do not re-run.