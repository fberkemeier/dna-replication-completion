# Theoretical Bounds for DNA Replication Timing

This repository provides a compact research workflow for studying DNA
replication timing and theoretical bounds on replication completion. Starting
from Repli-seq timing tracks, the code fits initiation-rate landscapes, runs
stochastic replication simulations, and compares the resulting timing statistics
with the analytical bounds developed by Alkhaled et al. (2026).

The emphasis is on connecting three views of the same replication process:
experimental timing profiles, computational simulations, and theoretical
completion estimates. The same machinery can be applied to
chromosome-scale profiles, periodic genomic intervals, or other one-dimensional
replication domains.

## Repository layout

- `replication_timing_bounds.ipynb`: main analysis notebook.
- `replication_src.py`: reusable timing extraction, simulation,
  completion-bound, plotting, and summary helpers used by the notebook.
- `requirements.txt`: Python runtime dependencies.
- `data/`: Repli-seq bigWig inputs used by the notebook.

Generated plots are written to `figures/` when figure saving is enabled in the
notebook.

## Installation

From the repository root, install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then open `replication_timing_bounds.ipynb` in Jupyter from the repository root
so the notebook can import the local tools module and find the `data/` files.

## Workflow

The notebook is organized as an executable analysis record. It imports reusable
utilities from `replication_src.py`, then applies them to example
domains.

1. **Timing-profile extraction**

   Repli-seq bigWig tracks are read on a selected chromosome or interval and
   converted into a one-dimensional replication timing curve. The notebook
   includes smoothing, rescaling, and optional grid refinement so that the same
   input data can be used at chromosome scale or in higher-resolution local
   interval analyses.

2. **Initiation-rate fitting**

   The observed timing curve is mapped to a fitted initiation-rate landscape using the methods presented in [Berkemeier et al. (2025)](https://www.nature.com/articles/s41467-025-59991-w).
   This fitted landscape is the common input for both the stochastic simulations
   and the theoretical calculations, keeping the comparison tied to the same
   inferred replication program.

3. **Stochastic replication simulation**

   The code simulates one-dimensional replication with stochastic origin firing
   and fork propagation. Simulations can be run on non-periodic domains
   (`perQ=False`) or periodic domains (`perQ=True`). The simulated ensembles
   produce replicated-fraction curves, replication timing profiles, local
   replication-time samples, and completion-time statistics.

4. **Theoretical and computational comparisons**

   The fitted initiation landscape is also used to compute the theoretical
   completion bounds from Alkhaled et al. (2026). The notebook compares these
   bounds with empirical simulation curves using an L-infinity-in-space
   criterion. Equivalently, completion is assessed in the L-infinity norm over
   spatial positions with tolerance `epsilon`: for each `epsilon`, the code
   reports the theoretical time bound and the corresponding empirical
   `(1 - epsilon)` quantile of the simulated replication times. This gives a
   uniform completion-time comparison rather than an averaged pointwise
   comparison.

5. **Expected local replication timing bounds**

   In addition to epsilon-dependent completion-time curves, the notebook
   integrates the same survival bound to obtain an upper bound on expected local
   replication timing. These expected-time summaries are compared with the
   empirical simulation estimate `max_x E[T(x)]`.

## Domains

The notebook contains two ready-to-run analysis sections:

- **Line domains**: chromosome-scale, non-periodic profiles compared with the
  full-line theoretical bound.
- **Torus domains**: selected intervals treated as periodic domains and compared
  with the torus theoretical bound.

New domains can be added by extending the dataset dictionaries in the notebook.
Each dataset specifies the cell line, chromosome, start and end coordinates,
input resolution, simulation periodicity, and the theoretical geometry used for
comparison.

## Usage

For a single dataset, edit or select an entry in `LINE_PROFILE_DATASETS` or
`PERIODIC_INTERVAL_DATASETS`, then run:

```python
result = run_single_dataset(
    cfg,
    fork_speed_kb_min=1.4,
    sim_number=10_000,
    refine_factor=1,
    smooth_window=25,
)
```

Use `refine_factor=1` for large chromosome-scale analyses, and a larger
refinement factor, such as `10`, for local interval analyses where a 1 kb grid is
desired from 10 kb input tracks.

To process several domains with the same settings, use `run_dataset_collection`.
The summary helpers then collect the comparison statistics:

```python
completion_summary_table(results)
expected_time_summary_table(results)
```

Plots are produced with `make_standard_plots(result)`. When figure saving is
enabled, PDF outputs are written to `figures/`.

## Units

The physical fork speed is specified once in kb/min. The notebook converts it
internally to grid units using the current spatial resolution:

```python
fork_speed_grid = fork_speed_kb_min / dx_kb
```

The same grid speed is used for simulations and theoretical bounds.
Fitted initiation rates are treated as rates per grid site per minute; the kb
conversion is used only for plotting lengths.

## Outputs

For each dataset, the notebook can generate:

- fitted timing and initiation-rate profiles;
- optional replicated-fraction maps from simulation;
- theoretical completion-time bounds across `epsilon` values versus empirical
  simulation curves;
- expected local replication timing bounds derived from the same survival estimate;
- local initiation-mass and tightness diagnostics.

This repository is intentionally not packaged as an installable library. It is a
small, self-contained analysis workspace for replication timing, simulation, and
theoretical completion-bound calculations.

## System requirements

This codebase was developed and tested on Python 3.13.9 under Windows 11. No installation procedure is required beyond installing standard Python 3 and the key dependencies. All scripts should remain compatible with standard Python 3 distributions on other operating systems.

## License

This project is openly distributed under the MIT License. This license allows unrestricted use, redistribution, and modification, provided that proper attribution to the original creators is maintained.

## Contact information

For further information, contributions, or queries, please contact:

- **Email**: [fp409@cam.ac.uk](mailto:fp409@cam.ac.uk)
- **GitHub**: [fberkemeier](https://github.com/fberkemeier)

Should any bugs arise or if you have any questions about usage, please raise a [GitHub issue](https://github.com/fberkemeier/replication-timing-bounds/issues).

## References

Alkhaled, A., Berkemeier, F., & Nik, K. Title to be determined. _arXiv_ (2026).
