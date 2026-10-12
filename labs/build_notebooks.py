"""Run each lecture source and save the result as the notebook the site shows.

Usage, from the course root:
    python labs/build_notebooks.py            # every lecture in labs/src/
    python labs/build_notebooks.py 02 03      # lectures whose file names start with 02 or 03

Each labs/src/NN-slug.py file (jupytext percent format) is executed in a fresh kernel, and
the notebook is written with its outputs to docs/lectures/NN-slug.ipynb. Commit those
notebooks, because the docs build shows them as they are and never executes them.
"""

import argparse
import time
from pathlib import Path

import jupytext
from nbclient import NotebookClient

COURSE = Path(__file__).resolve().parents[1]
SOURCES = COURSE / "labs" / "src"
NOTEBOOKS = COURSE / "docs" / "lectures"


def execute(source: Path) -> Path:
    """Execute one lecture source and return the path of the notebook it wrote."""
    notebook = jupytext.read(source)
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=1800,
        record_timing=False,  # keeps per-cell timestamps out of the committed notebooks
        resources={"metadata": {"path": str(NOTEBOOKS)}},  # cells run from docs/lectures
    )
    client.execute()
    target = NOTEBOOKS / f"{source.stem}.ipynb"
    jupytext.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute lecture sources into notebooks.")
    parser.add_argument("prefixes", nargs="*", help="only build sources whose names start with these")
    args = parser.parse_args()

    sources = sorted(SOURCES.glob("*.py"))
    if args.prefixes:
        sources = [s for s in sources if s.name.startswith(tuple(args.prefixes))]
    if not sources:
        wanted = f" whose name starts with {' or '.join(args.prefixes)}" if args.prefixes else ""
        raise SystemExit(f"{SOURCES.relative_to(COURSE)} has no lecture source{wanted}.")

    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for source in sources:
        start = time.perf_counter()
        target = execute(source)
        print(f"{source.relative_to(COURSE)} -> {target.relative_to(COURSE)} "
              f"in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
