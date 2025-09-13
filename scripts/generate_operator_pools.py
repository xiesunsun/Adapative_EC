#!/usr/bin/env python3
"""
Generate config/operator_pools.json from the unified operator registry.

Usage:
  python scripts/generate_operator_pools.py [--out config/operator_pools.json]
"""
import json
import os
import sys
import argparse


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=os.path.join(project_root(), "config", "operator_pools.json"))
    args = parser.parse_args()

    # Ensure project root on path to import aos
    sys.path.append(project_root())
    from aos.operators import default_registry

    reg = default_registry()
    data = reg.export_prompt_json()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote operator pools to {args.out}")


if __name__ == "__main__":
    main()

