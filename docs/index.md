# The Roofline Model

By [Arun Sharma](https://arunshar.com) and [Claude Code](https://claude.com/claude-code)

**This tutorial teaches the roofline model to readers who know basic Python, and every lab runs on an ordinary CPU.**

The labs use [NumPy](https://github.com/numpy/numpy), [pandas](https://github.com/pandas-dev/pandas),
[matplotlib](https://github.com/matplotlib/matplotlib) and
[threadpoolctl](https://github.com/joblib/threadpoolctl), and nothing else. The GPU examples use
NVIDIA's published peaks for the A100 and H100, and the last section of the lecture times real
kernels on the CPU that runs the notebook. The labs follow the original roofline paper
([Williams et al., CACM '09](https://doi.org/10.1145/1498765.1498785)).

Every lecture explains one idea and then tests it in a notebook you can run. The
**Open in Colab** and **Launch Binder** buttons above each lab page run that notebook in
your browser, so you do not have to install anything. Begin with
{doc}`lectures/00-getting-ready`.

Claude Code helped write these pages and labs, so please tell me about any mistake you find.

```{list-table}
:header-rows: 1
:widths: 6 40 54

* - #
  - Lecture
  - Topics
* - 0
  - {doc}`lectures/00-getting-ready`
  - vocabulary and background reading
* - 1
  - {doc}`lectures/01-the-roofline-model`
  - arithmetic intensity, the ridge point, memory-bound and compute-bound kernels, batching, a measured CPU roofline
```

```{toctree}
:hidden:
:caption: Getting Started

getting-started/index
getting-started/run-in-browser
getting-started/run-locally
```

```{toctree}
:hidden:
:caption: Lectures
:maxdepth: 1
:glob:

lectures/*
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/how-it-works
```
