# Repli-seq Completion Bounds

This repository is now flattened around one workflow for Repli-seq-based completion-bound analysis, with the ecDNA and whole-chromosome use cases collected in a single notebook.

## Layout

- `repliseq_completion_bounds.ipynb`: the main notebook
- `repliseq_completion_bounds.py`: the small helper module used by the notebook
- `data/`: the Repli-seq bigWig inputs

Generated figures are written to `figures/` when the notebook runs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run `repliseq_completion_bounds.ipynb` from the repository root. The notebook imports:

```python
from repliseq_completion_bounds import plotf, rescale, rfit, rsim
```

The repo no longer ships as a package because it is intended only for this specific analysis.
