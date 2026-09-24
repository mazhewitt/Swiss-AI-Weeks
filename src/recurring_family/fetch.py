"""Unpack the challenge zip into the raw data area, idempotently."""

import zipfile
import zlib
from pathlib import Path


def _crc32(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            crc = zlib.crc32(chunk, crc)
    return crc


def fetch_data(zip_path: Path, raw_dir: Path) -> list[str]:
    """Extract every member not already present byte-identically. Returns the names written."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            dest = raw_dir / name
            if dest.exists() and dest.stat().st_size == info.file_size and _crc32(dest) == info.CRC:
                continue
            tmp = dest.with_name(dest.name + ".part")
            with zf.open(info) as src, open(tmp, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)
            written.append(name)
    return written
