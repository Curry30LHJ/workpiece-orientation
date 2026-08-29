#!/usr/bin/env python3
from pathlib import Path
import sys


# Keep this wrapper self-contained when launched by path from any working
# directory (as the portable release script does).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from release_tools.portable_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
