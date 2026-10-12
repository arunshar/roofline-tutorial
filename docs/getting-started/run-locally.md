# Run Locally

By [Arun Sharma](https://arunshar.com) and [Claude Code](https://claude.com/claude-code)

You need Linux, macOS, or Windows with WSL2, plus **Python 3.10 or newer**, **git**, and
**4 GB of RAM**.

## 1. Get the course

```bash
git clone https://github.com/arunshar/roofline-tutorial
cd roofline-tutorial
```

## 2. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lab.txt jupyterlab
```

## 3. Launch Jupyter

```bash
jupyter lab docs/lectures/
```

Open a lecture notebook and run its first cell.

## Where things are stored

| Path | Contents | Safe to delete? |
|---|---|---|
| `docs/lectures/*.ipynb` | the lecture notebooks, with their outputs | no, because they are the course |
| `labs/src/*.py` | the lecture sources the notebooks are built from | no |
| `.venv/` | the Python environment | yes, and step 2 recreates it |

## Troubleshooting

`urllib.error.HTTPError: HTTP Error 404: Not Found`
: The setup cell did not find `labs/tutorial_helpers.py` next to the notebook, so it tried
  to download the module from GitHub. A 404 means the repository is private or has not been
  pushed. Open the notebook from `docs/lectures/` inside your clone, or make the repository public.

`urllib.error.URLError`
: The download failed because the machine is offline. Work inside your clone, where the setup
  cell loads the module from `labs/` and needs no network.

The measured bandwidth is far above your memory's rated speed
: The test vectors fit in your processor's cache, so the timing measured the cache. Raise `n`
  in the measurement cell until the two vectors together are several times larger than the
  last-level cache.

`MemoryError` or a crashed kernel during the CPU measurement
: The measurement needs about 600 MB of free RAM. Close other programs, or lower `n` and the
  matrix size together.
