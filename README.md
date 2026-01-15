<p align="center">
  <img src="docs/assets/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is a computational framework for simulating, mapping, and analysing DNA replication kinetics at genome scale. It provides a unified modelling environment that links replication origin activity, fork propagation, and replication timing through explicit mechanistic assumptions.

This framework integrates replication datasets such as Repli-seq, SNS-seq, OK-seq, and related assays into a mathematically grounded kinetic framework, enabling genome-wide reconstruction of replication programmes and inference of core replication features. Importantly, it supports hypothesis-driven modelling, allowing users to simulate alternative mechanistic scenarios, test assumptions about origin firing or fork behaviour, and quantify how changes in kinetics reshape genome-wide replication dynamics.

Full details of the DNAscape framework, together with worked examples and tutorials, are available in the [Documentation](https://dnascape.readthedocs.io/en/latest/).

## Key features

- Genome-wide simulation and analysis of DNA replication kinetics within a unified modelling framework.
- Explicit modelling of origin firing, bidirectional fork progression, fork speed, and inter-origin distances.
- Mapping between replication timing, origin activity, fork directionality, and other latent kinetic parameters.
- Stochastic ensemble simulations capturing variability in origin firing and fork dynamics.
- Support for modelling replication stress, including fork stalling and heterogeneous fork behaviour.
- Adjustable spatial resolution to accommodate different experimental scales and data granularities.
- Integration with replication datasets such as Repli-seq, SNS-seq, OK-seq, and related assays.
- Gene- and region-level comparison of replication features using genome annotations.
- Reproducible example workflows demonstrating data integration, inference, and simulation.

## Documentation

Full documentation, including installation instructions, tutorials, and API reference, is available in the [Documentation](https://dnascape.readthedocs.io/en/latest/).

The documentation includes:
- Installation and environment setup
- Worked examples and tutorial notebooks
- Conceptual description of the modelling framework
- Detailed module and function reference

## Installation (quick start)

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See the documentation for recommended workflows and examples.

## Citation

If you use DNAscape in your research, please cite our publication.

## Questions, bugs, and feature requests

If you encounter bugs or have questions regarding usage, you may open a GitHub issue:
https://github.com/fberkemeier/DNAscape/issues

For more detailed discussions or potential collaborations, please contact Francisco Berkemeier at fp409@cam.ac.uk.
