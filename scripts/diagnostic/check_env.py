"""Print the installed torch variant and CUDA availability.

Useful for diagnosing which lock file the current environment was built from.
Run via: ``uv run python scripts/diagnostic/check_env.py``
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    """Print interpreter, torch, and CUDA diagnostics; exit 0 always."""
    print(f"Python:       {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform:     {platform.platform()}")
    print(f"Machine:      {platform.machine()}")

    try:
        import torch
    except ImportError:
        print("torch:        not installed (ingest-only environment)")
        return 0

    print(f"torch:        {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA avail:   {cuda_available}")
    if cuda_available:
        print(f"CUDA build:   {torch.version.cuda}")
        print(f"GPUs:         {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
