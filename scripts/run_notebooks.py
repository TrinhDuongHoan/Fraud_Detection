import argparse
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient


def execute_notebook(path: Path, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = AsyncKernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=timeout,
        startup_timeout=30,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.on_cell_start = lambda cell, cell_index: print(
        f"  Cell {cell_index + 1}/{len(notebook.cells)}",
        flush=True,
    )
    client.execute(cleanup_kc=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute notebooks and save their outputs")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = Path(tempfile.gettempdir()) / "fraud-notebooks"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(cache / "jupyter"))
    os.environ.setdefault("IPYTHONDIR", str(cache / "ipython"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    paths = (
        [args.notebook.resolve()] if args.notebook else sorted((root / "notebooks").glob("*.ipynb"))
    )
    if not paths:
        parser.error("No notebooks found")
    for path in paths:
        print(f"Executing {path.name}", flush=True)
        execute_notebook(path, args.timeout)
        print(f"Saved outputs: {path.name}", flush=True)


if __name__ == "__main__":
    main()
