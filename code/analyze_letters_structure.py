from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FileInfo:
    rel_path: str
    suffix: str
    size_bytes: int
    npz_arrays: list[dict[str, Any]] | None = None


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def inspect_npz(npz_path: Path) -> list[dict[str, Any]]:
    arrays_info: list[dict[str, Any]] = []
    with np.load(npz_path, allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            arrays_info.append(
                {
                    "key": key,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size": int(arr.size),
                }
            )
    return arrays_info


def analyze_directory(root: Path, inspect_npz_files: bool = True) -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    files: list[FileInfo] = []
    ext_counter: Counter[str] = Counter()
    depth_counter: Counter[int] = Counter()
    dir_file_counts: defaultdict[str, int] = defaultdict(int)
    dir_size_counts: defaultdict[str, int] = defaultdict(int)

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(root)
        suffix = file_path.suffix.lower() or "<no_ext>"
        size_bytes = file_path.stat().st_size

        npz_arrays = None
        if inspect_npz_files and suffix == ".npz":
            try:
                npz_arrays = inspect_npz(file_path)
            except Exception as exc:  # pragma: no cover
                npz_arrays = [{"error": str(exc)}]

        info = FileInfo(
            rel_path=str(rel_path).replace("\\", "/"),
            suffix=suffix,
            size_bytes=size_bytes,
            npz_arrays=npz_arrays,
        )
        files.append(info)
        ext_counter[suffix] += 1
        depth_counter[len(rel_path.parts) - 1] += 1

        dir_key = str(rel_path.parent).replace("\\", "/")
        if dir_key == ".":
            dir_key = "<root>"
        dir_file_counts[dir_key] += 1
        dir_size_counts[dir_key] += size_bytes

    total_size = sum(f.size_bytes for f in files)
    result = {
        "root": str(root),
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_human": format_size(total_size),
        "extensions": dict(ext_counter),
        "depth_distribution": dict(depth_counter),
        "directories": {
            d: {
                "file_count": dir_file_counts[d],
                "size_bytes": dir_size_counts[d],
                "size_human": format_size(dir_size_counts[d]),
            }
            for d in sorted(dir_file_counts.keys())
        },
        "files": [asdict(f) for f in files],
    }
    return result


def print_report(report: dict[str, Any]) -> None:
    print("=== data/letters structure report ===")
    print(f"Root: {report['root']}")
    print(f"Total files: {report['total_files']}")
    print(f"Total size: {report['total_size_human']} ({report['total_size_bytes']} bytes)")
    print("")

    print("[Extensions]")
    if report["extensions"]:
        for ext, count in sorted(report["extensions"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {ext}: {count}")
    else:
        print("  (none)")
    print("")

    print("[Directories]")
    if report["directories"]:
        for dname, stats in report["directories"].items():
            print(f"  {dname}: {stats['file_count']} files, {stats['size_human']}")
    else:
        print("  (none)")
    print("")

    print("[Files]")
    if report["files"]:
        for file_info in report["files"]:
            print(f"  - {file_info['rel_path']} ({format_size(file_info['size_bytes'])})")
            if file_info.get("npz_arrays"):
                for arr_info in file_info["npz_arrays"]:
                    if "error" in arr_info:
                        print(f"      npz error: {arr_info['error']}")
                    else:
                        print(
                            "      "
                            f"{arr_info['key']}: shape={arr_info['shape']}, "
                            f"dtype={arr_info['dtype']}, size={arr_info['size']}"
                        )
    else:
        print("  (none)")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Analyze file structure under data/letters")
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "data" / "letters",
        help="Target directory to analyze (default: data/letters)",
    )
    parser.add_argument(
        "--no-inspect-npz",
        action="store_true",
        help="Disable npz internal array inspection",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save full report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_directory(root=args.root, inspect_npz_files=not args.no_inspect_npz)
    print_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
