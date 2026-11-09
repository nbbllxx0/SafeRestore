"""Fetch selected members of the M2AD category zips with HTTP range requests.

Only the members needed for the WP6 study are fetched: images of the chosen
illuminations (default I07, I10) and the ground-truth ``*_mask.png`` files.
zipfile verifies each member's CRC-32 on read; members are written with
validated relative paths only.

Usage:
  python -I -B m2ad_fetch_subset.py --dest <m2ad_root> --dry-run Bird Car ...
  python -I -B m2ad_fetch_subset.py --dest <m2ad_root> Bird Car ...
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests

BASE = "https://huggingface.co/datasets/ChengYuQi99/M2AD/resolve/main/{name}"
CHUNK = 4 << 20  # read-ahead per request


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with a read-ahead buffer."""

    def __init__(self, url: str, size: int, session: requests.Session):
        self.url, self.size, self.session = url, size, session
        self.pos = 0
        self.buf_start, self.buf = 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                     timeout=120, allow_redirects=True)
                if r.status_code == 206:
                    return r.content
                raise IOError(f"HTTP {r.status_code} for range {start}-{end}")
            except (requests.RequestException, IOError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            fetch_end = min(max(end, self.pos + CHUNK), self.size) - 1
            self.buf_start, self.buf = self.pos, self._get(self.pos, fetch_end)
        out = self.buf[self.pos - self.buf_start:end - self.buf_start]
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def remote_size(session: requests.Session, url: str) -> int:
    r = session.get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
    if r.status_code != 206:
        raise IOError(f"range not supported: HTTP {r.status_code}")
    return int(r.headers["Content-Range"].split("/")[-1])


def wanted(name: str, cls: str, illums: tuple[str, ...]) -> bool:
    if name.endswith("/"):
        return False
    if re.fullmatch(rf"{cls}/GT/[^/]+/A\d{{3}}_mask\.png", name):
        return True
    return any(re.fullmatch(rf"{cls}/(Good|NG)/[^/]+/A\d{{3}}_I{il}\.png", name) for il in illums)


def safe_target(dest: Path, name: str) -> Path:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or ":" in name:
        raise ValueError(f"unsafe member path: {name!r}")
    return dest.joinpath(*p.parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--illums", default="07,10")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("categories", nargs="+")
    args = ap.parse_args()
    illums = tuple(args.illums.split(","))
    session = requests.Session()
    grand = 0
    for cls in args.categories:
        url = BASE.format(name=f"{cls}.zip")
        size = remote_size(session, url)
        zf = zipfile.ZipFile(RangeFile(url, size, session))
        members = [i for i in zf.infolist() if wanted(i.filename, cls, illums)]
        nbytes = sum(i.compress_size for i in members)
        grand += nbytes
        kinds = {"mask": sum(i.filename.endswith("_mask.png") for i in members)}
        for il in illums:
            kinds[f"I{il}"] = sum(i.filename.endswith(f"_I{il}.png") for i in members)
        print(f"{cls}: zip {size/1e9:.2f} GB, selected {len(members)} members, {nbytes/1e9:.2f} GB, {kinds}",
              flush=True)
        if args.dry_run:
            continue
        todo = [i for i in members
                if not (safe_target(args.dest, i.filename).exists()
                        and safe_target(args.dest, i.filename).stat().st_size == i.file_size)]
        lock = threading.Lock()
        local = threading.local()

        def fetch(info: zipfile.ZipInfo) -> str:
            if not hasattr(local, "zf"):
                s = requests.Session()
                local.zf = zipfile.ZipFile(RangeFile(url, size, s))
            target = safe_target(args.dest, info.filename)
            data = local.zf.read(info.filename)  # CRC-32 checked by zipfile
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)
            return info.filename

        done = 0
        with ThreadPoolExecutor(args.workers) as pool:
            for fut in as_completed([pool.submit(fetch, i) for i in todo]):
                fut.result()
                with lock:
                    done += 1
                    if done % 500 == 0:
                        print(f"  {cls}: {done}/{len(todo)}", flush=True)
        manifest = args.dest / f"{cls}_subset_manifest.json"
        manifest.write_text(json.dumps({
            "source": url, "zip_size": size, "illuminations": illums,
            "members": {i.filename: {"size": i.file_size, "crc32": f"{i.CRC:08x}"} for i in members},
        }, indent=1), encoding="utf-8")
        print(f"{cls}: done ({len(todo)} fetched, {len(members) - len(todo)} already present)", flush=True)
    print(f"TOTAL selected {grand/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
