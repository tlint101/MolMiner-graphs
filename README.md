# MolMiner - Graphs
[![Molecular Miner - Graphs](https://img.shields.io/pypi/v/MolMiner-graphs.svg?label=MolMiner-Graphs&style=flat)]()
[![Python Versions](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://pypi.org/project/mlx-mol-cluster/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![MLX](https://img.shields.io/badge/MLX-0.31.0+-black?logo=apple&labelColor=gray)](https://github.com/ml-explore/mlx)

## Overview
This repo is a companion to the MolMiner project. It is focused on building deep learning molecular graph models built
using PyTorch or MLX.

## Install

Installation of MolMiner-graphs is done as follows:

```bash
pip install MolMiner-graphs
```

By default, this will only install the PyTorch backend. For users interested in MLX, this requires workstations using 
Apple Silicon and can be installed as follows:
```bash
pip install "MolMiner-graphs[mlx]"
```
This will install both PyTorch and MLX on the Apple Silicon workstation.

## Tutorials

Tutorials and examples of how to use MolMiner-Graphs can be seen at the [Tutorials](tutorials/) directory. 