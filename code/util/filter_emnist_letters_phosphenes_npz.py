from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from multiprocessing import get_context
from typing import Tuple

import numpy as np
from numpy.lib.format import (
    open_memmap,
    read_array_header_1_0,
    read_array_header_2_0,
    read_magic,
    write_array_header_1_0,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description=(
            "Filter emnist_letters_phosphenes.npz with a basecode-like semicircle/sector mask.\n"
            "Designed to run in limited RAM by streaming npz entries in chunks."
        )
    )
    p.add_argument(
        "--in-npz",
        type=Path,
        default=project_root / "data" / "letters" / "emnist_letters_phosphenes.npz",
        help="Input NPZ (default: data/letters/emnist_letters_phosphenes.npz)",
    )
    p.add_argument(
        "--out-npz",
        type=Path,
        default=project_root / "data" / "letters" / "emnist_letters_phosphenes_filtered.npz",
        help="Output NPZ path (default: data/letters/emnist_letters_phosphenes_filtered.npz)",
    )
    p.add_argument(
        "--angle1",
        type=float,
        default=-90.0,
        help="Sector start angle in degrees (default: -90, matches basecode)",
    )
    p.add_argument(
        "--angle2",
        type=float,
        default=90.0,
        help="Sector end angle in degrees (default: 90, matches basecode)",
    )
    p.add_argument(
        "--radius-low-frac",
        type=float,
        default=0.0,
        help="Inner radius as fraction of half-size (default: 0.0)",
    )
    p.add_argument(
        "--radius-high-frac",
        type=float,
        default=1.0,
        help="Outer radius as fraction of half-size (default: 1.0 -> full half-disk)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of samples processed per chunk (default: 256). Lower if RAM is tight.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of worker processes for masking (default: 20). Use 1 to disable multiprocessing.",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help="Write output NPZ with deflate compression (smaller but slower). Default: store uncompressed.",
    )
    p.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars (falls back to periodic prints).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra step logs and flush immediately (useful if terminal appears stuck).",
    )
    return p.parse_args()


_WSL_MNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


def _normalize_cross_os_path(p: Path) -> Path:
    """
    Normalize common Windows/WSL path forms so the script works regardless of
    whether it's launched from Windows Python or WSL Python.
    """
    s = os.fspath(p)
    is_win = sys.platform.startswith("win")
    is_posix = os.sep == "/"

    if is_win:
        m = _WSL_MNT_RE.match(s.replace("\\", "/"))
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
        return Path(s)

    if is_posix:
        # Convert "D:\\foo\\bar" or "D:/foo/bar" into "/mnt/d/foo/bar"
        if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
            drive = s[0].lower()
            rest = s[2:].lstrip("\\/").replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
        return Path(s)

    return Path(s)


def sector_mask(
    shape: Tuple[int, int],
    centre: Tuple[float, float],
    radius_low: float,
    radius_high: float,
    angle_range: Tuple[float, float],
) -> np.ndarray:
    """Basecode-equivalent sector mask (ported from basecode/lossfunc.py)."""
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)
    if tmax < tmin:
        tmax += 2 * np.pi

    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin
    theta %= 2 * np.pi

    circmask_low = r2 >= radius_low * radius_low
    circmask_high = r2 <= radius_high * radius_high
    circmask = circmask_low * circmask_high
    anglemask = theta <= (tmax - tmin)
    return (circmask * anglemask).astype(bool)


def _read_npy_header(f) -> tuple[tuple[int, ...], np.dtype, bool]:
    """Return (shape, dtype, fortran_order) for a .npy stream."""
    major, minor = read_magic(f)
    if (major, minor) == (1, 0):
        shape, fortran, dtype = read_array_header_1_0(f)
    else:
        shape, fortran, dtype = read_array_header_2_0(f)
    return tuple(shape), np.dtype(dtype), bool(fortran)


def _copy_npy_entry(zf_in: zipfile.ZipFile, name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf_in.open(name, "r") as src, out_path.open("wb") as dst:
        while True:
            buf = src.read(1024 * 1024)
            if not buf:
                break
            dst.write(buf)


def _extract_npz_entry_to_file(zf_in: zipfile.ZipFile, name: str, out_path: Path, verbose: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[step] extracting: {name} -> {out_path}", flush=True)
    with zf_in.open(name, "r") as src, out_path.open("wb") as dst:
        while True:
            buf = src.read(1024 * 1024)
            if not buf:
                break
            dst.write(buf)
    if verbose:
        try:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[step] extracted: {name} ({size_mb:.1f} MB)", flush=True)
        except Exception:
            print(f"[step] extracted: {name}", flush=True)


def _mask_worker(args_tuple) -> None:
    in_npy, out_npy, start, end, mask, shape, dtype_str, verbose = args_tuple
    dtype = np.dtype(dtype_str)
    n_dims = len(shape)
    h = shape[-2]
    w = shape[-1]
    mask_hw = mask.reshape(h, w).astype(bool)

    if n_dims == 4:
        mm_in = np.memmap(in_npy, mode="r", dtype=dtype, offset=0, shape=shape)
        mm_out = np.memmap(out_npy, mode="r+", dtype=dtype, offset=0, shape=shape)
        x = np.array(mm_in[start:end, :, :, :], copy=True)
        x[:, :, ~mask_hw] = 0.0
        mm_out[start:end, :, :, :] = x
    elif n_dims == 3:
        mm_in = np.memmap(in_npy, mode="r", dtype=dtype, offset=0, shape=shape)
        mm_out = np.memmap(out_npy, mode="r+", dtype=dtype, offset=0, shape=shape)
        x = np.array(mm_in[start:end, :, :], copy=True)
        x[:, ~mask_hw] = 0.0
        mm_out[start:end, :, :] = x
    else:
        raise ValueError(f"Unsupported ndim={n_dims} shape={shape}")

    if verbose:
        print(f"[step] masked samples {start}:{end}", flush=True)


def _filter_phos_entry_to_npy(
    zf_in: zipfile.ZipFile,
    name: str,
    out_path: Path,
    mask_hw: np.ndarray,
    chunk_size: int,
    show_progress: bool,
    verbose: bool,
    workers: int,
) -> None:
    """
    Stream a phosphenes .npy entry from npz, apply mask, and write to .npy file.
    Supports input shapes [N,1,H,W] or [N,H,W].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # To support multiprocessing, extract the npy entry to a file first, then memmap it.
    in_path = out_path.parent / f".{out_path.stem}__in.npy"
    _extract_npz_entry_to_file(zf_in, name, in_path, verbose=verbose)

    with in_path.open("rb") as f:
        shape, dtype, fortran = _read_npy_header(f)
        if verbose:
            print(f"[step] header: {name} shape={shape} dtype={dtype} fortran={fortran}", flush=True)
        if fortran:
            raise ValueError(f"Unsupported Fortran-order array for {name}")
        data_offset = f.tell()

    if len(shape) == 4:
        n, c, h, w = shape
        if c != 1:
            raise ValueError(f"Expected channel dim=1 for {name}, got shape={shape}")
    elif len(shape) == 3:
        n, h, w = shape
    else:
        raise ValueError(f"Unsupported shape for {name}: {shape}")

    if mask_hw.shape != (h, w):
        raise ValueError(f"Mask shape {mask_hw.shape} != map shape {(h, w)} for {name}")

    # Create output .npy (includes header) and then use memmap on raw data region.
    if verbose:
        print(f"[step] creating memmap output: {out_path} shape={shape} dtype={dtype}", flush=True)
    open_memmap(out_path, mode="w+", dtype=dtype, shape=shape).flush()

    mm_in = np.memmap(in_path, mode="r", dtype=dtype, offset=data_offset, shape=shape)
    mm_out = np.memmap(out_path, mode="r+", dtype=dtype, offset=data_offset, shape=shape)

    it = list(range(0, n, chunk_size))
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            it = list(tqdm(it, total=(n + chunk_size - 1) // chunk_size, desc=f"mask {name}", unit="chunk"))
        except Exception:
            show_progress = False

    last_print_t = time.time()
    t0 = time.time()
    if verbose:
        print(f"[step] begin masking: {name} n={n} chunk_size={chunk_size} workers={workers}", flush=True)

    workers_eff = max(1, int(workers))
    # Disable multiprocessing to avoid memmap corruption with concurrent writes
    if workers_eff > 1:
        print(f"[warning] Multiprocessing disabled (workers={workers_eff} -> 1) to avoid memmap corruption", flush=True)
        workers_eff = 1
    if workers_eff == 1:
        mask = mask_hw.astype(bool)
        for start in it:
            end = min(start + chunk_size, n)
            if len(shape) == 4:
                x = np.array(mm_in[start:end, :, :, :], copy=True)
                x[:, :, ~mask] = 0.0
                mm_out[start:end, :, :, :] = x
            else:
                x = np.array(mm_in[start:end, :, :], copy=True)
                x[:, ~mask] = 0.0
                mm_out[start:end, :, :] = x

            if not show_progress:
                now = time.time()
                if now - last_print_t >= 5.0:
                    done = end
                    rate = done / max(now - t0, 1e-8)
                    remain = (n - done) / max(rate, 1e-8)
                    print(f"[progress] {name}: {done}/{n} samples ({done/n:.1%}) | ETA ~ {remain/60:.1f} min")
                    last_print_t = now
    else:
        # Avoid passing huge arrays; pass mask as uint8.
        mask_u8 = mask_hw.astype(np.uint8, copy=False)
        tasks = []
        for start in it:
            end = min(start + chunk_size, n)
            tasks.append((str(in_path), str(out_path), start, end, mask_u8, tuple(shape), dtype.str, False))

        ctx = get_context("fork" if os.name != "nt" else "spawn")
        with ctx.Pool(processes=workers_eff) as pool:
            for _ in pool.imap_unordered(_mask_worker, tasks, chunksize=1):
                if not show_progress:
                    now = time.time()
                    # Progress estimation is approximate here.
                    if now - last_print_t >= 5.0:
                        # Count finished tasks via internal iterator state is hard; just emit heartbeat.
                        print(f"[progress] {name}: working... (workers={workers_eff})", flush=True)
                        last_print_t = now

    mm_out.flush()
    if verbose:
        print(f"[step] finished entry: {name}", flush=True)


def _pack_npys_to_npz(out_npz: Path, files: dict[str, Path], compress: bool, verbose: bool) -> None:
    """Pack individual .npy files into a single .npz archive using numpy's native format."""
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # Only include phosphenes keys (exclude labels for compatibility with train.py)
    phosphene_keys = [k for k in files.keys() if 'phosphenes' in k]

    arrays_dict = {}
    for key in phosphene_keys:
        npy_path = files[key]
        if verbose:
            try:
                size_mb = npy_path.stat().st_size / (1024 * 1024)
                print(f"[step] loading for packing: {key} ({size_mb:.1f} MB)", flush=True)
            except Exception:
                print(f"[step] loading for packing: {key}", flush=True)

        # Load with memmap directly to avoid np.load issues
        with open(npy_path, 'rb') as f:
            shape, dtype, fortran = _read_npy_header(f)
            data_offset = f.tell()
        array = np.memmap(npy_path, mode='r', dtype=dtype, offset=data_offset, shape=shape)
        # Force loading into memory and ensure float32 dtype
        arrays_dict[key] = np.array(array, copy=True, dtype=np.float32)

    # Save to NPZ using numpy's native format (preserves metadata and dtype)
    if verbose:
        print(f"[step] writing npz with numpy native format: {out_npz}", flush=True)

    if compress:
        np.savez_compressed(out_npz, **arrays_dict)
    else:
        np.savez(out_npz, **arrays_dict)

    if verbose:
        print(f"[step] npz write complete: {out_npz}", flush=True)


def main() -> None:
    args = parse_args()
    show_progress = not args.no_tqdm
    t_start = time.time()
    def log(msg: str) -> None:
        if args.verbose:
            print(msg, flush=True)

    # Normalize paths early so existence checks and downstream I/O behave.
    args.in_npz = _normalize_cross_os_path(args.in_npz)
    args.out_npz = _normalize_cross_os_path(args.out_npz)

    if not args.in_npz.exists():
        raise FileNotFoundError(f"Input NPZ not found: {args.in_npz}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if not (0.0 <= args.radius_low_frac <= args.radius_high_frac):
        raise ValueError("--radius-low-frac must be <= --radius-high-frac and both >= 0")

    log(f"[step] args.in_npz={args.in_npz}")
    log(f"[step] args.out_npz={args.out_npz}")
    log(
        f"[step] mask params: angle={args.angle1}..{args.angle2}, "
        f"r_frac={args.radius_low_frac}..{args.radius_high_frac}, chunk={args.chunk_size}, "
        f"compress={args.compress}, tqdm={show_progress}"
    )

    # Build mask for 256x256 maps (will be validated against file header)
    h = w = 256
    half = min(h, w) / 2.0
    r_low = float(args.radius_low_frac) * half
    r_high = float(args.radius_high_frac) * half
    centre = ((h - 1) / 2.0, (w - 1) / 2.0)
    mask = sector_mask(
        shape=(h, w),
        centre=centre,
        radius_low=r_low,
        radius_high=r_high,
        angle_range=(float(args.angle1), float(args.angle2)),
    )

    # Temporary npy outputs (created alongside out_npz)
    tmp_dir = args.out_npz.parent / f".{args.out_npz.stem}_tmp_npys"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log(f"[step] tmp_dir={tmp_dir}")

    needed = {
        "train_phosphenes": "train_phosphenes.npy",
        "test_phosphenes": "test_phosphenes.npy",
        "train_labels": "train_labels.npy",
        "test_labels": "test_labels.npy",
    }

    out_files: dict[str, Path] = {}
    with zipfile.ZipFile(args.in_npz, "r") as zf_in:
        names = set(zf_in.namelist())
        for key, entry in needed.items():
            if entry not in names:
                raise KeyError(f"Expected entry '{entry}' not found in input npz. Found: {sorted(names)}")

        # Filter large arrays with streaming
        for split in ["train", "test"]:
            in_entry = f"{split}_phosphenes.npy"
            out_path = tmp_dir / f"{split}_phosphenes.npy"
            print(f"[filter] streaming+masking: {in_entry} -> {out_path}", flush=True)
            _filter_phos_entry_to_npy(
                zf_in=zf_in,
                name=in_entry,
                out_path=out_path,
                mask_hw=mask,
                chunk_size=args.chunk_size,
                show_progress=show_progress,
                verbose=bool(args.verbose),
                workers=int(args.workers),
            )
            out_files[f"{split}_phosphenes"] = out_path
            log(f"[step] done masking {in_entry}")

        # Copy label arrays as-is (fast streaming copy)
        for split in ["train", "test"]:
            in_entry = f"{split}_labels.npy"
            out_path = tmp_dir / f"{split}_labels.npy"
            print(f"[filter] copying: {in_entry} -> {out_path}", flush=True)
            _copy_npy_entry(zf_in, in_entry, out_path)
            out_files[f"{split}_labels"] = out_path
            log(f"[step] done copying {in_entry}")

    # Pack into new npz
    print(f"[filter] writing npz: {args.out_npz} (compress={args.compress})", flush=True)
    if args.verbose:
        for key, p in out_files.items():
            try:
                size_mb = p.stat().st_size / (1024 * 1024)
                log(f"[step] pack input: {key} -> {p} ({size_mb:.1f} MB)")
            except Exception:
                log(f"[step] pack input: {key} -> {p} (size unknown)")
        log("[step] begin packing npz")
    _pack_npys_to_npz(args.out_npz, out_files, compress=args.compress, verbose=bool(args.verbose))
    log("[step] finished packing npz")

    kept_frac = float(mask.mean())
    elapsed = time.time() - t_start
    print(f"[filter] done.", flush=True)
    print(f"[filter] in:  {args.in_npz}", flush=True)
    print(f"[filter] out: {args.out_npz}", flush=True)
    print(
        f"[filter] mask kept fraction: {kept_frac:.3f} "
        f"(angle={args.angle1:.1f}..{args.angle2:.1f}, r_frac={args.radius_low_frac:.2f}..{args.radius_high_frac:.2f})"
    )
    print(f"[filter] elapsed: {elapsed/60:.1f} min", flush=True)
    print(f"[filter] tmp npy dir (can delete): {tmp_dir}", flush=True)


if __name__ == "__main__":
    main()

