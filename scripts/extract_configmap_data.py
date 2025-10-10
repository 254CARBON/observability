#!/usr/bin/env python3
"""
Extract the embedded file content from a ConfigMap manifest.

Usage:
    python extract_configmap_data.py <path> <data-key>

The script reads the YAML manifest (without external dependencies) and emits
the value stored under `data[<data-key>]`. It assumes the manifest stores the
data using the block scalar (`|`) style, which is standard for Prometheus
rule/config maps in this repository.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def extract_block(lines: list[str], key: str) -> list[str]:
    """Extract lines belonging to a `data:<key>: |` block."""
    block_prefix = f"  {key}: |"
    capturing = False
    extracted: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if capturing:
            # Stop when we hit another top-level data entry or the end.
            if line.startswith("  ") and not line.startswith("    "):
                break
            # Remove the 4-space indentation applied by the ConfigMap.
            if line.startswith("    "):
                extracted.append(line[4:])
            else:
                extracted.append(line.lstrip())
        elif line == block_prefix:
            capturing = True

    if not extracted:
        raise ValueError(f"Could not find data entry '{key}' in ConfigMap")

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the ConfigMap manifest")
    parser.add_argument("key", help="Key under the ConfigMap data section")
    args = parser.parse_args()

    content = Path(args.path).read_text(encoding="utf-8").splitlines()
    block = extract_block(content, args.key)
    sys.stdout.write("\n".join(block))
    if block and block[-1] != "":
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
