"""One coordinate system, shared by every encoding and query script."""
from pathlib import Path
import os

DIMENSIONS = 4096
SEED = 2026
ROOT = Path(__file__).resolve().parents[2]
# Override for an isolated test run. No database service is needed.
DB_PATH = Path(os.environ.get("GRAPH_HV_DB", str(ROOT / "data" / "lancedb")))
