"""Command-line entry points for data inventory and inspection."""

import argparse
import json
import sys
from pathlib import Path
import zipfile

from .data import inspect_file, inventory, summarize, write_inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect climate data without modifying the originals.")
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="List files in a directory or ZIP archive.")
    inv.add_argument("--source", required=True, type=Path)
    inv.add_argument("--output", type=Path, default=Path("reports/data_inventory.csv"))
    inspect = commands.add_parser("inspect", help="Show variables, shapes, dates or table previews.")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--member", help="Exact ZIP member name, as listed by inventory.")
    inspect.add_argument("--output", type=Path, help="Also save the metadata as JSON.")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            rows = inventory(args.source)
            write_inventory(rows, args.output)
            result = {**summarize(rows), "inventory_file": str(args.output.resolve())}
        else:
            result = inspect_file(args.source, args.member)
        payload = json.dumps(result, indent=2, default=str, ensure_ascii=False)
        if args.command == "inspect" and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
