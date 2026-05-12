# Repli-seq Completion Bounds

This repository contains a compact, notebook-first workflow for comparing
Repli-seq-derived replication timing profiles with Proposition 1 completion-time
and expected-time bounds.

The analysis is organized around two geometries:

- **Line-bound chromosome profiles**: non-periodic chromosome-scale timing
  profiles, simulated with `perQ=False` and compared with the full-line bound.
- **Torus-bound periodic intervals**: selected genomic windows treated as
  periodic domains, simulated with `perQ=True` and compared with the torus bound.

## Repository Layout

- `repliseq_completion_bounds.ipynb`: main analysis notebook.
- `repliseq_completion_bounds.py`: trimmed helper module used by the notebook.
- `requirements.txt`: Python runtime dependencies.
- `data/`: Repli-seq bigWig inputs used by the notebook.

Generated plots are written to `figures/` when figure saving is enabled in the
notebook.

## Installation

From the repository root, install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then open `repliseq_completion_bounds.ipynb` in Jupyter from the repository root
so the notebook can import the local helper module and find the `data/` files.

## Workflow

The notebook first defines shared utilities for:

- reading Repli-seq bigWig tracks;
- smoothing and refining timing curves;
- fitting initiation-rate profiles;
- running stochastic replication simulations;
- computing completion-time and expected-time bounds;
- plotting simulation and bound comparisons.

It then provides two runnable analysis sections:

- **Analysis A** builds line-profile configurations, runs one chromosome-scale
  non-periodic example by default, and includes an optional batch loop.
- **Analysis B** builds periodic-interval configurations, runs one torus-bound
  interval example by default, and includes an optional batch loop.

## Units

The physical fork speed is specified once in kb/min. The notebook converts it to
grid units using the current spatial resolution:

```python
fork_speed_grid = fork_speed_kb_min / dx_kb
```

The same grid speed is used for simulation and for the theoretical bounds.
Fitted initiation rates are treated as rates per grid site per minute; the kb
conversion is used only for plotting lengths.

## Outputs

For each dataset, the notebook can generate:

- fitted timing and initiation-rate profiles;
- replicated-fraction maps from simulation;
- theoretical completion-time bounds versus empirical simulation curves;
- expected-time summaries derived from the same survival bound;
- local initiation-mass and tightness diagnostics.

This repository is intentionally not packaged as an installable library. It is a
small, self-contained analysis workspace for the Repli-seq completion-bound
calculations.
