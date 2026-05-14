"""Deprecated — use `build_mimic_parquet.py` with local MIMIC-IV CSV extracts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    print(
        "Use MIMIC-IV CSV extracts only (no synthetic generator in this repo).\n"
        "  python scripts/build_mimic_parquet.py\n"
        "or set mimic.data_root in config/config.yaml and run the same script.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
