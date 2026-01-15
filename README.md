<p align="center">
  <img src="docs/assets/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is a computational framework for simulating, mapping, and analysing DNA replication kinetics at genome scale. It provides a unified modelling environment that links replication origin activity, fork propagation, and replication timing through explicit mechanistic assumptions.

DNAscape integrates experimental replication datasets such as Repli-seq, SNS-seq, OK-seq, and related assays into a common kinetic representation. This enables genome-wide reconstruction of DNA replication programmes and supports the inference of key replication features, including origin firing rates and efficiency, fork directionality, fork speed distributions, inter-origin distances, and the temporal progression of replication.

In addition to data integration, DNAscape supports hypothesis-driven modelling of DNA replication dynamics. Users can simulate alternative mechanistic scenarios, test assumptions about origin firing regulation or fork dynamics, and assess how changes in replication kinetics reshape replication timing and overall genome replication programmes. DNAscape is intended as a reusable research software framework for quantitative inference, mechanistic testing, and predictive modelling of DNA replication kinetics.

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
