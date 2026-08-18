import argparse
from pathlib import Path

from comic_codex.datasets.manifest import build_manifest, write_manifest

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".xml", ".json", ".txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible dataset manifest")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extension", action="append", dest="extensions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extensions = frozenset(args.extensions or DEFAULT_EXTENSIONS)
    manifest = build_manifest(args.root, extensions)
    write_manifest(manifest, args.output)
    total_bytes = sum(asset.size_bytes for asset in manifest.assets)
    print(f"assets={len(manifest.assets)} bytes={total_bytes} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
