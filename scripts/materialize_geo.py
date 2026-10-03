"""Download GEO metadata/raw archives without committing source data.

Examples
--------
python scripts/materialize_geo.py --accession GSE153480 --kind family_soft --cache data/raw
python scripts/materialize_geo.py --accession GSE153480 --kind raw_tar --cache data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cardilearn.geo import download_resource


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a public GEO resource into a local cache")
    parser.add_argument("--accession", required=True)
    parser.add_argument("--kind", choices=["family_soft", "raw_tar"], default="family_soft")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    resource = download_resource(
        args.accession,
        args.kind,
        Path(args.cache) / args.accession.upper(),
        overwrite=args.overwrite,
    )
    print(json.dumps(resource.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
