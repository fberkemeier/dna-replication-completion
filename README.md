<p align="center">
  <img src="docs/assets/dnascape_logo.png" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is a computational framework for simulating, mapping, and analysing DNA replication kinetics at genome scale.
It provides a unified modelling environment to relate replication origin activity, fork propagation, and replication timing through explicit mechanistic assumptions.

DNAscape supports both data integration and hypothesis-driven modelling by enabling experimental replication datasets to be mapped onto a common kinetic representation, and by allowing systematic comparison of alternative mechanistic descriptions of replication dynamics.

# Key features

- Genome-scale simulation of DNA replication dynamics
- Explicit modelling of origin firing and bidirectional fork progression
- Mapping between replication timing, origin activity, and fork directionality
- Ensemble simulations capturing stochastic variability
- Designed for integration with experimental datasets (e.g. Repli-seq, OK-seq)

# Documentation

Full documentation, including installation instructions, tutorials, and API reference, is available in the [Documentation](https://dnascape.readthedocs.io/en/latest/).

The documentation includes:
- Installation and environment setup
- Worked examples and tutorial notebooks
- Conceptual description of the modelling framework
- Detailed module and function reference

# Installation (quick start)

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See the documentation for recommended workflows and examples.

# Citation

If you use DNAscape in your research, please cite our publication.

# Questions, bugs, and feature requests

If you encounter bugs or have questions regarding usage, you may open a GitHub issue:
https://github.com/fberkemeier/DNAscape/issues

For more detailed discussions or potential collaborations, please contact Francisco Berkemeier at fp409@cam.ac.uk.
