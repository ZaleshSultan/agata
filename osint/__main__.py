"""CLI independent from Agent and conversational memory."""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from osint import investigate
def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m osint")
    sub = parser.add_subparsers(dest="command", required=True); command = sub.add_parser("investigate")
    command.add_argument("target"); command.add_argument("--target-type", default="auto"); command.add_argument("--depth", choices=("quick", "standard", "deep"), default="standard"); command.add_argument("--image", type=Path)
    args = parser.parse_args(); image = args.image.read_bytes() if args.image else None
    print(json.dumps(asyncio.run(investigate(args.target, args.target_type, image, args.depth)), ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
