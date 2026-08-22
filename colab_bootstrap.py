"""Bootstrap script for Google Colab — run BEFORE ``import project``.

Usage in Colab::

    exec(open("/content/ABPSEES/colab_bootstrap.py").read())
    ROOT = setup_colab_path()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the ABPSEES repository root on Colab or locally."""
    env_root = os.environ.get("ABPSEES_ROOT")
    if env_root:
        candidate = Path(env_root)
        if (candidate / "project" / "__init__.py").exists():
            return candidate.resolve()

    candidates = [
        Path("/content/ABPSEES"),
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]
    for candidate in candidates:
        if (candidate / "project" / "__init__.py").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the ABPSEES project package. "
        "Clone the repo to /content/ABPSEES and verify that "
        "/content/ABPSEES/project/__init__.py exists."
    )


def setup_colab_path(root: str | Path | None = None) -> Path:
    """Add the repository root to ``sys.path`` so ``import project`` works.

    Args:
        root: Optional explicit project root. Auto-detected if omitted.

    Returns:
        Resolved repository root path.
    """
    project_root = Path(root).resolve() if root else _find_project_root()
    root_str = str(project_root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    os.environ["ABPSEES_ROOT"] = root_str
    return project_root


if __name__ == "__main__":
    root = setup_colab_path()
    print(f"ABPSEES root: {root}")
    print(f"sys.path[0]: {sys.path[0]}")
    import project  # noqa: F401

    print("import project — OK")
