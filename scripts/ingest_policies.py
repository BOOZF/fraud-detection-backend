"""(Re)build the copilot knowledge base from every PDF/MD/TXT file in docs/.

Uses the same service as the upload endpoint (POST /api/documents).
Run from this folder:  python scripts/ingest_policies.py  [--keep]   (--keep skips the schema reset)"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import connect  # noqa: E402
from service.services import documents  # noqa: E402

if __name__ == "__main__":
    connect()
    if "--keep" not in sys.argv:
        documents.reset_schema()
    for path in sorted((ROOT / "docs").iterdir()):
        if path.suffix.lower() not in documents.SUPPORTED:
            continue
        t0 = time.time()
        d = documents.ingest(path.name, path.read_bytes())
        print(f"{d['doc']}: {d['kind']}, pages={d['pages']}, chunks={d['chunks']} ({time.time() - t0:.1f}s)")
