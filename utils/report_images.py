"""Validate report figure PNGs so stub/corrupt files are not treated as present."""

from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path

# Real matplotlib/SHAP exports are typically many KB; magic-only stubs are 8 bytes.
_MIN_PNG_BYTES = 64
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_IEND = b"IEND"


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(tag)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def is_valid_report_png(path: Path | str, *, min_bytes: int = _MIN_PNG_BYTES) -> bool:
    """True when path is a structurally complete PNG figure.

    Checks: file size, PNG magic, IHDR (positive dims + CRC), and trailing IEND.
    Optionally verifies pixel decode via Pillow when installed.
    Rejects missing files, tiny stubs, truncated headers, and header-only forgeries.
    """
    p = Path(path)
    try:
        if not p.is_file():
            return False
        size = p.stat().st_size
        if size < min_bytes:
            return False
        data = p.read_bytes()
    except OSError:
        return False
    if len(data) < 33 or not data.startswith(_PNG_MAGIC):
        return False
    # Bytes 8-11: chunk length; 12-15: type; 16-23: width/height (IHDR)
    if data[12:16] != b"IHDR":
        return False
    ihdr_len = struct.unpack(">I", data[8:12])[0]
    if ihdr_len != 13:
        return False
    ihdr_data = data[16:29]
    width, height = struct.unpack(">II", ihdr_data[:8])
    if width <= 0 or height <= 0:
        return False
    expected_crc = struct.unpack(">I", data[29:33])[0]
    actual_crc = binascii.crc32(b"IHDR")
    actual_crc = binascii.crc32(ihdr_data, actual_crc) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        return False
    # Require a complete file terminator (rejects truncated / header-padded stubs).
    if _IEND not in data[33:]:
        return False
    # Prefer full decode when Pillow is available (catches CRC/IDAT corruption).
    try:
        from PIL import Image  # type: ignore

        with Image.open(p) as im:
            im.verify()
        return True
    except ImportError:
        return True
    except Exception:
        return False


def remove_invalid_report_png(path: Path | str) -> bool:
    """Delete a corrupt/stub PNG so presence checks stay honest. Returns True if removed."""
    p = Path(path)
    if not p.is_file():
        return False
    if is_valid_report_png(p):
        return False
    try:
        p.unlink()
        return True
    except OSError:
        return False


def require_valid_report_png(path: Path | str, *, label: str = "report PNG") -> Path:
    """Raise ValueError when path is missing or not a real PNG figure."""
    p = Path(path)
    if is_valid_report_png(p):
        return p
    size = p.stat().st_size if p.is_file() else 0
    raise ValueError(
        f"{label} invalid or missing at {p} (bytes={size}). "
        "Re-run SHAP / train to regenerate a real figure."
    )


def minimal_png_bytes(width: int = 8, height: int = 8) -> bytes:
    """Small structurally valid PNG (IHDR + IDAT + IEND) for tests."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # 1x1+ RGB raw scanlines compressed; pad so file exceeds _MIN_PNG_BYTES.
    raw = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _ in range(height))
    idat = zlib.compress(raw, 9)
    body = _PNG_MAGIC + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    if len(body) < _MIN_PNG_BYTES:
        # Extra tEXt chunk as padding (still a valid PNG).
        pad_needed = _MIN_PNG_BYTES - len(body)
        text = b"Comment\x00" + (b"x" * max(1, pad_needed))
        body = (
            _PNG_MAGIC
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"tEXt", text)
            + _chunk(b"IDAT", idat)
            + _chunk(b"IEND", b"")
        )
    return body
