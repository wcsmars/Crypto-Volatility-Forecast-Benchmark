#!/usr/bin/env python3
"""Download the public source dataset and verify the saved snapshot digest."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset"
MAX_DOWNLOAD_BYTES = 50_000_000
MAX_ARCHIVE_BYTES = 200_000_000


def load_manifest(path: Path) -> dict:
    """Read the source manifest and reject one that cannot drive verification."""
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    digest = manifest.get("archive_sha256")
    files = manifest.get("files")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{path}: 'archive_sha256' must be a 64-character hexadecimal SHA-256")
    if not isinstance(files, dict) or not files or any(
        not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name
        or Path(name).suffix.casefold() != ".csv"
        or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for name, value in files.items()
    ):
        raise ValueError(f"{path}: 'files' must map every CSV name to a SHA-256 digest")
    if len({name.casefold() for name in files}) != len(files):
        raise ValueError(f"{path}: 'files' contains colliding CSV filenames")
    return manifest


def extract_csvs(payload: bytes, destination: Path) -> int:
    """Validate ZIP members before writing; flatten CSVs without overwriting."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # The same suffix rule as the manifest and the loader: a bare ".csv" dotfile
        # has no suffix and is never staged, so every staged file is hash-checked.
        members = [m for m in archive.infolist()
                   if not m.is_dir() and Path(m.filename).suffix.casefold() == ".csv"]
        if not members:
            raise ValueError("Downloaded archive contains no CSV files")
        names = [Path(m.filename).name for m in members]
        if len({n.casefold() for n in names}) != len(names):
            raise ValueError("Archive contains colliding CSV filenames")
        if sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES:
            raise ValueError("Unexpected archive size (over 200 MB uncompressed)")
        destination.mkdir(parents=True, exist_ok=True)
        # Read all files (including CRC checks) before changing the destination.
        contents = {name: archive.read(member) for name, member in zip(names, members)}
        existing = {p.name.casefold() for p in destination.iterdir()}
        for name in contents:
            if name.casefold() in existing:
                raise FileExistsError(f"Refusing to overwrite {destination / name}; use a new directory")
        for name, data in contents.items():
            with (destination / name).open("xb") as target:
                target.write(data)
    return len(contents)


def publish(staging: Path, output: Path) -> None:
    """Move verified files into ``output``: one atomic rename when it is new, otherwise a pre-checked copy."""
    staged = sorted(p for p in staging.iterdir() if p.is_file())
    if not output.exists():
        os.replace(staging, output)
        # mkdtemp creates the staging directory owner-only; restore the umask
        # default so other accounts can read the published data directory.
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(output, 0o777 & ~mask)
        return
    if not output.is_dir():
        raise ValueError(f"{output} exists and is not a directory")
    # Case-insensitive comparison: macOS and Windows file systems would otherwise
    # fail part way through the copy and leave a half-populated directory.
    existing = {p.name.casefold() for p in output.iterdir()}
    clashes = [p.name for p in staged if p.name.casefold() in existing]
    if clashes:
        raise FileExistsError(f"Refusing to overwrite existing files in {output}: {', '.join(clashes)}")
    created = []
    try:
        for file in staged:
            destination = output / file.name
            with destination.open("xb") as target:
                created.append(destination)
                target.write(file.read_bytes())
    except BaseException:
        # A full disk, failed source read or Ctrl-C must not leave a partial
        # dataset which then prevents a retry. Preserve pre-existing output files.
        for destination in reversed(created):
            destination.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--zip", type=Path, help="Use an already downloaded Kaggle ZIP instead of the network")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "source_manifest.json")
    args = parser.parse_args(argv)
    if args.output.is_dir() and any(p.suffix.casefold() == ".csv" for p in args.output.iterdir()):
        parser.error("Output already contains CSVs; use existing data or specify a new --output")
    stage = "Reading the manifest"
    try:
        manifest = load_manifest(args.manifest)
        if args.zip:
            stage = f"Reading {args.zip}"
            with args.zip.open("rb") as source:
                payload = source.read(MAX_DOWNLOAD_BYTES + 1)
        else:
            stage = "Download"
            request = urllib.request.Request(URL, headers={"User-Agent": "crypto-volatility/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(MAX_DOWNLOAD_BYTES + 1)
        if len(payload) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Unexpected download size (over 50 MB)")
        stage = "Verification"
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest["archive_sha256"]:
            raise ValueError("Kaggle archive differs from the recorded snapshot; obtain the matching version. "
                             f"Expected {manifest['archive_sha256']}; received {digest}")
        # Stage and verify every file before publishing it to the requested path.
        stage = "Preparing the output directory"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".crypto-download-", dir=args.output.parent))
        try:
            stage = "Verification"
            count = extract_csvs(payload, staging)
            # Hash every staged file, so the published set is exactly the verified set.
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()}
            if hashes != manifest["files"]:
                raise ValueError("CSV hashes do not match source manifest")
            stage = "Publishing"
            publish(staging, args.output)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        print(f"Verified {count} source CSVs in {args.output}; SHA-256 {digest}")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as exc:
        hint = "" if args.zip else "\nYou can also download the ZIP from Kaggle and use --zip PATH."
        parser.exit(1, f"{stage} failed: {exc}{hint}\n")


if __name__ == "__main__":
    raise SystemExit(main())
