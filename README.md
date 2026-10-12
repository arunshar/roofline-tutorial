# The Roofline Model

By [Arun Sharma](https://arunshar.com) and [Claude Code](https://claude.com/claude-code)

This tutorial teaches the roofline model to readers who know basic Python, and every lab runs on
an ordinary CPU. Lecture 1 counts the arithmetic intensity of four kernels, draws the roofline of
an NVIDIA A100, and then measures a roofline for one core of the CPU that runs the notebook.

## Run the lecture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arunshar/roofline-tutorial/blob/main/docs/lectures/01-the-roofline-model.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arunshar/roofline-tutorial/main?labpath=docs%2Flectures%2F01-the-roofline-model.ipynb)

To run lecture 1 on your own machine, clone the repository and install the lab packages with
these commands.

```bash
git clone https://github.com/arunshar/roofline-tutorial.git
cd roofline-tutorial
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lab.txt jupyterlab
jupyter lab docs/lectures/01-the-roofline-model.ipynb
```

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | the Sphinx site, with the executed lecture notebook in `docs/lectures/` |
| `labs/src/` | the lecture source in jupytext format, which you edit by hand |
| `labs/build_notebooks.py` | the script that executes `labs/src/*.py` into `docs/lectures/*.ipynb` |
| `labs/tutorial_helpers.py` | the quiz widget, which every lecture notebook imports and the `{quiz}` directive uses on Markdown pages |

## Build the site

```bash
pip install -r requirements-build.txt
python labs/build_notebooks.py
sphinx-build -W -b html docs docs/_build/html
```

The build re-times the CPU cells, so the measured numbers change on every run. The lecture quotes
the range that several runs produced.

## License

The code and the lecture text are released under the [MIT License](LICENSE).
