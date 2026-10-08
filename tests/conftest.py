"""server/ isn't an installed package (like scripts/), so put the repo root
on sys.path for tests that import it, e.g. `import server.app`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
