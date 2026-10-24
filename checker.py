#!/usr/bin/env python3
"""Weather-station data checker.

    python checker.py <pack_folder>

Reads <pack_folder>/data/*.csv, gps/*.nmea, stations.csv and maintenance_log.md and writes flags.csv
into the folder you run it from (use --out to change that).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find unreliable readings in a weather-station data pack.")
    ap.add_argument("pack", help="folder containing data/, gps/, stations.csv, maintenance_log.md")
    ap.add_argument("--out", default="flags.csv", help="output file (default: flags.csv in the current folder)")
    ap.add_argument("--iforest", action="store_true", help="also write iforest_candidates.csv (optional review aid; needs scikit-learn)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    result = run(args.pack, quiet=args.quiet, use_iforest=args.iforest)
    df = result["flags"]
    df.to_csv(args.out, index=False)
    if args.iforest and result["iforest"] is not None:
        result["iforest"].to_csv("iforest_candidates.csv", index=False)
    if not args.quiet:
        counts = df.action.value_counts().to_dict()
        print(f"wrote {len(df)} flags to {args.out}  {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
