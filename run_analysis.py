#!/usr/bin/env python3
"""
CLI entrypoint for BDD object detection dataset analysis.
Delegates to data_analysis.

Usage:
    python run_analysis.py [--bdd-root /path/to/BDD] [--plots] [--out-dir ./output]
    python -m data_analysis.run_analysis [--bdd-root /path/to/BDD] [--plots] [--out-dir ./output]
"""
from data_analysis.run_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
