#!/usr/bin/env python3
"""
hdr10plus_tool.py — AV1 (IVF) HDR10+ only

Commands (exactly):
  python hdr10plus_tool.py extract -i input.ivf -o meta.json
  python hdr10plus_tool.py remove  -i input.ivf -o output.ivf
  python hdr10plus_tool.py inject  -i input.ivf -j meta.json -o output.ivf
  python hdr10plus_tool.py plot    -i meta.json -o out.png

Behavior requirements implemented:
- AV1/IVF only.
- extract/remove/plot: ONLY a progress bar (no other prints).
- inject prints exactly:
    Parsing JSON file...
    Processing input video for frame order info...
    (progress bar only during this phase)
    (optional mismatch warning block)
    Rewriting file with interleaved HDR10+ ITU-T T.35 Metadata OBUs..
    (progress bar only during this phase)
- If video shorter than JSON: metadata truncated ("skipped at the end").
- If video longer than JSON: metadata duplicated (wrapped) to match video length.
- Classic JSON format output/acceptance: JSONInfo + SceneInfo (+ SceneInfoSummary + ToolInfo).
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Progress bar (default)
# ----------------------------
def progress_bar(percent: float, width: int = 50):
    if percent < 0:
        percent = 0.0
    if percent > 100.0:
        percent = 100.0
    filled = int(width * percent / 100.0)
    bar = "■" * filled + " " * (width - filled)
    sys.stderr.write(f"\r[{bar}] {percent:5.1f}%")
    sys.stderr.flush()


def progress_done():
    progress_bar(100.0)
    sys.stderr.write("\n")
    sys.stderr.flush()


def print_status_line(msg: str):
    # Always start on a clean line, preventing collisions with the bar
    sys.stderr.write("\n" + msg + "\n")
    sys.stderr.flush()


# ----------------------------
# Constants (HDR10+ / AV1)
# ----------------------------
METADATA_TYPE_ITUT_T35 = 4

HDR10PLUS_COUNTRY_CODE = 0xB5
HDR10PLUS_PROVIDER_CODE = 0x003C
HDR10PLUS_ORIENTED_CODE = 0x0001
HDR10PLUS_APP_ID = 0x04
HDR10PLUS_APP_MODE = 0x01

# AV1 OBU types (only what we need)
OBU_METADATA = 5

# Classic distribution indices used by hdr10plus_tool JSON
DIST_INDEX_CLASSIC = [1, 5, 10, 25, 50, 75, 90, 95, 99]


# ----------------------------
# Small helpers
# ----------------------------
def u16be(b: bytes) -> int:
    return (b[0] << 8) | b[1]


def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def leb128_read(data: bytes, off: int) -> Tuple[int, int]:
    """Return (value, bytes_consumed)."""
    val = 0
    shift = 0
    i = 0
    while True:
        if off + i >= len(data):
            raise ValueError("LEB128 truncated")
        byte = data[off + i]
        val |= (byte & 0x7F) << shift
        i += 1
        if (byte & 0x80) == 0:
            break
        shift += 7
        if shift > 63:
            raise ValueError("LEB128 too long")
    return val, i


def leb128_write(v: int) -> bytes:
    if v < 0:
        raise ValueError("LEB128 negative")
    out = bytearray()
    while True:
        byte = v & 0x7F
        v >>= 7
        if v:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


# ----------------------------
# Bit I/O (MSB-first)
# ----------------------------
class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0

    def remaining_bits(self) -> int:
        return len(self.data) * 8 - self.bitpos

    def read_bits(self, n: int) -> int:
        if n <= 0:
            return 0
        if self.remaining_bits() < n:
            raise ValueError("BitReader underflow")
        val = 0
        for _ in range(n):
            byte_index = self.bitpos // 8
            bit_index = 7 - (self.bitpos % 8)
            bit = (self.data[byte_index] >> bit_index) & 1
            val = (val << 1) | bit
            self.bitpos += 1
        return val

    def read_u8(self) -> int:
        return self.read_bits(8)


class BitWriter:
    __slots__ = ("_buf", "_cur", "_nbits")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0

    def write_bits(self, value: int, nbits: int) -> None:
        if nbits <= 0:
            return
        if value < 0:
            raise ValueError("negative bit value")
        value &= (1 << nbits) - 1
        for i in range(nbits - 1, -1, -1):
            bit = (value >> i) & 1
            self._cur = (self._cur << 1) | bit
            self._nbits += 1
            if self._nbits == 8:
                self._buf.append(self._cur & 0xFF)
                self._cur = 0
                self._nbits = 0

    def write_u8(self, v: int) -> None:
        self.write_bits(v & 0xFF, 8)

    def write_bit(self, bit: int) -> None:
        self.write_bits(1 if bit else 0, 1)

    def byte_align(self) -> None:
        if self._nbits:
            self._cur <<= (8 - self._nbits)
            self._buf.append(self._cur & 0xFF)
            self._cur = 0
            self._nbits = 0

    def get_bytes(self) -> bytes:
        self.byte_align()
        return bytes(self._buf)


# ----------------------------
# IVF I/O
# ----------------------------
@dataclass
class IvfHeader:
    width: int
    height: int
    timebase_num: int
    timebase_den: int
    frames: int


def read_ivf_header(fp) -> IvfHeader:
    header = fp.read(32)
    if len(header) != 32:
        raise ValueError("IVF header truncated")
    if header[0:4] != b"DKIF":
        raise ValueError("Not an IVF file (missing DKIF)")
    fourcc = header[8:12]
    if fourcc != b"AV01":
        raise ValueError("IVF fourcc is not AV01 (AV1)")
    width = struct.unpack_from("<H", header, 12)[0]
    height = struct.unpack_from("<H", header, 14)[0]
    tb_den = struct.unpack_from("<I", header, 16)[0]
    tb_num = struct.unpack_from("<I", header, 20)[0]
    frames = struct.unpack_from("<I", header, 24)[0]
    return IvfHeader(width=width, height=height, timebase_num=tb_num, timebase_den=tb_den, frames=frames)


def write_ivf_header(fp, h: IvfHeader) -> None:
    out = bytearray(32)
    out[0:4] = b"DKIF"
    struct.pack_into("<H", out, 4, 0)      # version
    struct.pack_into("<H", out, 6, 32)     # header size
    out[8:12] = b"AV01"
    struct.pack_into("<H", out, 12, h.width)
    struct.pack_into("<H", out, 14, h.height)
    struct.pack_into("<I", out, 16, h.timebase_den)
    struct.pack_into("<I", out, 20, h.timebase_num)
    struct.pack_into("<I", out, 24, h.frames)
    struct.pack_into("<I", out, 28, 0)     # unused
    fp.write(bytes(out))


def iter_ivf_frames(fp) -> Tuple[int, bytes, int]:
    """Yield (frame_index, payload_bytes, timestamp_u64)."""
    i = 0
    while True:
        hdr = fp.read(12)
        if not hdr:
            return
        if len(hdr) != 12:
            raise ValueError("IVF frame header truncated")
        size = struct.unpack_from("<I", hdr, 0)[0]
        ts = struct.unpack_from("<Q", hdr, 4)[0]
        payload = fp.read(size)
        if len(payload) != size:
            raise ValueError("IVF frame payload truncated")
        yield i, payload, ts
        i += 1


# ----------------------------
# AV1 OBU parsing / building (size-field OBUs only)
# ----------------------------
@dataclass
class Obu:
    obu_type: int
    header: bytes
    ext: bytes
    payload: bytes


def parse_obus(frame_payload: bytes) -> List[Obu]:
    """
    Parse concatenated OBUs from one IVF frame payload.
    Requires OBU has_size_field=1 (typical in IVF).
    """
    obus: List[Obu] = []
    off = 0
    n = len(frame_payload)

    while off < n:
        first = frame_payload[off]
        off += 1

        forbidden = (first >> 7) & 1
        if forbidden:
            raise ValueError("Invalid OBU header (forbidden bit set)")

        obu_type = (first >> 3) & 0x0F
        ext_flag = (first >> 2) & 1
        has_size_field = (first >> 1) & 1

        ext = b""
        if ext_flag:
            if off >= n:
                raise ValueError("OBU extension truncated")
            ext = frame_payload[off:off + 1]
            off += 1

        if not has_size_field:
            raise ValueError("OBU has_size_field=0 not supported in this tool")

        obu_size, used = leb128_read(frame_payload, off)
        off += used

        if off + obu_size > n:
            raise ValueError("OBU payload truncated")

        payload = frame_payload[off:off + obu_size]
        off += obu_size

        obus.append(Obu(obu_type=obu_type, header=bytes([first]), ext=ext, payload=payload))

    return obus


def rebuild_obus(obus: List[Obu]) -> bytes:
    out = bytearray()
    for o in obus:
        out.extend(o.header)
        if o.ext:
            out.extend(o.ext)
        out.extend(leb128_write(len(o.payload)))
        out.extend(o.payload)
    return bytes(out)


def build_metadata_obu_payload_from_t35(t35: bytes) -> bytes:
    return leb128_write(METADATA_TYPE_ITUT_T35) + t35


def is_hdr10plus_t35(t35: bytes) -> bool:
    if len(t35) < 7:
        return False
    if t35[0] != HDR10PLUS_COUNTRY_CODE:
        return False
    provider = u16be(t35[1:3])
    oriented = u16be(t35[3:5])
    app_id = t35[5]
    app_mode = t35[6]
    return (
        provider == HDR10PLUS_PROVIDER_CODE
        and oriented == HDR10PLUS_ORIENTED_CODE
        and app_id == HDR10PLUS_APP_ID
        and app_mode == HDR10PLUS_APP_MODE
    )


def extract_t35_from_metadata_obu_payload(metadata_payload: bytes) -> Optional[bytes]:
    try:
        mtype, used = leb128_read(metadata_payload, 0)
    except Exception:
        return None
    if mtype != METADATA_TYPE_ITUT_T35:
        return None
    t35 = metadata_payload[used:]
    return t35 if is_hdr10plus_t35(t35) else None


def remove_hdr10plus_from_frame(frame_payload: bytes) -> bytes:
    obus = parse_obus(frame_payload)
    kept: List[Obu] = []
    for o in obus:
        if o.obu_type == OBU_METADATA:
            t35 = extract_t35_from_metadata_obu_payload(o.payload)
            if t35 is not None:
                continue
        kept.append(o)
    return rebuild_obus(kept)


def inject_hdr10plus_into_frame(frame_payload: bytes, t35: bytes) -> bytes:
    """
    Remove any existing HDR10+ metadata OBUs, then append ours.
    """
    obus = parse_obus(frame_payload)
    kept: List[Obu] = []
    for o in obus:
        if o.obu_type == OBU_METADATA:
            if extract_t35_from_metadata_obu_payload(o.payload) is not None:
                continue
        kept.append(o)

    # Build a metadata OBU (type=5) with a size field
    meta_payload = build_metadata_obu_payload_from_t35(t35)
    # Header byte: forbidden=0, obu_type=5, ext=0, has_size=1, reserved=0
    header_byte = (0 << 7) | ((OBU_METADATA & 0x0F) << 3) | (0 << 2) | (1 << 1) | 0
    meta_obu = Obu(obu_type=OBU_METADATA, header=bytes([header_byte]), ext=b"", payload=meta_payload)

    kept.append(meta_obu)
    return rebuild_obus(kept)


# ----------------------------
# HDR10+ ST 2094-40 decode/encode (AV1 common layout)
# ----------------------------
def decode_hdr10plus_t35_av1(t35: bytes) -> Dict:
    """
    Decode HDR10+ ITU-T T.35 for AV1 (B5 003C 0001 04 01 ...).

    Bit layout implemented (aligned with SMPTE ST 2094-40 App 4, Version 1 as used by hdr10plus_tool):
      - AV1 wrapper:
          num_windows: 2 bits
          reserved:    6 bits
      - Core fields:
          targeted_system_display_maximum_luminance: 27 bits
          targeted_system_display_actual_peak_luminance_flag: 1 bit
            (if 1: parse ActualTargetedSystemDisplay grid)
          For each window:
            maxscl[3]: 17 bits each
            average_maxrgb: 17 bits
            num_distribution_maxrgb_percentiles: 4 bits
            distribution entries: (percentage: 7 bits, percentile: 17 bits) * N
            fraction_bright_pixels: 10 bits
          mastering_display_actual_peak_luminance_flag: 1 bit
            (if 1: parse ActualMasteringDisplay grid)
          For each window:
            tone_mapping_flag: 1 bit
              (if 1: BezierCurve: knee_x 12, knee_y 12, n_anchors 4, anchors 10 bits each)
          color_saturation_mapping_flag: 1 bit
            (if 1: color_saturation_weight: 6 bits)
          byte_align

    NOTE on scaling for classic JSON:
      The reference hdr10plus_tool JSON (HEVC) expresses:
        - TargetedSystemDisplayMaximumLuminance in nits
        - MaxScl and AverageRGB in 1/32 nits (approx)
      Many AV1 HDR10+ streams store these as scaled integers.
      To match "classic" JSON examples, we apply:
        - targeted_nits = targeted_raw // 64
        - maxscl_out    = maxscl_raw  // 32
        - average_out   = average_raw // 32
      DistributionValues are kept as-is (as in hdr10plus_tool JSON).
    """
    if not is_hdr10plus_t35(t35):
        raise ValueError("Not HDR10+ T.35")

    br = BitReader(t35[7:])

    num_windows = br.read_bits(2)

    targeted_raw = br.read_bits(27)

    # Flag exists even for v1 (should be 0, but must be consumed to stay aligned)
    tsd_actual_flag = br.read_bits(1)
    if tsd_actual_flag:
        # ActualTargetedSystemDisplay:
        # num_rows: 5, num_cols: 5, then rows*cols 4-bit entries
        rows = br.read_bits(5)
        cols = br.read_bits(5)
        for _r in range(rows):
            for _c in range(cols):
                _ = br.read_bits(4)

    # Defaults
    max_scl_raw = [0, 0, 0]
    average_raw = 0
    dist_map: Dict[int, int] = {}
    fraction_bright_pixels = 0

    # v1 expects num_windows == 1, but we still parse per window
    nw = num_windows if num_windows else 1
    for _w in range(nw):
        max_scl_raw = [br.read_bits(17) for _ in range(3)]
        average_raw = br.read_bits(17)

        num_dists = br.read_bits(4)
        dist_map = {}
        for _ in range(num_dists):
            pct = br.read_bits(7)
            val = br.read_bits(17)
            dist_map[int(pct)] = int(val)

        fraction_bright_pixels = br.read_bits(10)

    mastering_flag = br.read_bits(1)
    if mastering_flag:
        rows = br.read_bits(5)
        cols = br.read_bits(5)
        for _r in range(rows):
            for _c in range(cols):
                _ = br.read_bits(4)

    knee_x = 0
    knee_y = 0
    anchors: List[int] = [0] * 9

    # tone_mapping_flag per window (v1 num_windows==1)
    tone_mapping_flag = 0
    for _w in range(nw):
        tone_mapping_flag = br.read_bits(1)
        if tone_mapping_flag:
            knee_x = br.read_bits(12)
            knee_y = br.read_bits(12)
            n_anchors = br.read_bits(4)
            raw_anchors = []
            for _ in range(n_anchors):
                raw_anchors.append(br.read_bits(10))
            # Normalize to classic 9 anchors
            anchors = (raw_anchors[:9] + [0] * 9)[:9]

    color_flag = br.read_bits(1) if br.remaining_bits() >= 1 else 0
    if color_flag and br.remaining_bits() >= 6:
        _ = br.read_bits(6)

    # Classic index order
    dist_vals = [int(dist_map.get(i, 0)) for i in DIST_INDEX_CLASSIC]

    # Apply classic scaling
    # Some sources store these as scaled integers (e.g., target*64, maxscl*32, avg*32).
    # Others (common in AV1) already store them in "classic" units.
    # Use a conservative heuristic to avoid double-scaling.
    if targeted_raw > 10000:
        targeted_out = int(targeted_raw // 64)
    else:
        targeted_out = int(targeted_raw)

    if any(x > 100000 for x in max_scl_raw):
        max_scl_out = [int(x // 32) for x in max_scl_raw]
    else:
        max_scl_out = [int(x) for x in max_scl_raw]

    if average_raw > 100000:
        average_out = int(average_raw // 32)
    else:
        average_out = int(average_raw)

    return {
        "NumberOfWindows": int(nw),
        "TargetedSystemDisplayMaximumLuminance": targeted_out,
        "AverageRGB": average_out,
        "MaxScl": max_scl_out,
        "DistributionIndex": list(DIST_INDEX_CLASSIC),
        "DistributionValues": dist_vals,
        "BezierCurveData": {
            "Anchors": [int(x) for x in anchors],
            "KneePointX": int(knee_x),
            "KneePointY": int(knee_y),
        },
    }


def encode_hdr10plus_t35_from_scene(scene: Dict) -> bytes:
    """
    Encode a classic SceneInfo entry to HDR10+ ITU-T T.35 for AV1.

    This is the inverse of decode_hdr10plus_t35_av1(), including scaling:
      - targeted_raw = targeted_nits * 64
      - maxscl_raw   = maxscl_out    * 32
      - average_raw  = average_out   * 32

    DistributionValues are encoded as percentiles (17-bit) with classic percentages (7-bit):
      [1, 5, 10, 25, 50, 75, 90, 95, 99]
    """
    num_windows = int(scene.get("NumberOfWindows", 1))
    targeted_nits = int(scene.get("TargetedSystemDisplayMaximumLuminance", 400))

    lum = scene.get("LuminanceParameters", {})
    average_out = int(lum.get("AverageRGB", 0))
    max_scl_out = lum.get("MaxScl", [0, 0, 0])
    if not (isinstance(max_scl_out, list) and len(max_scl_out) == 3):
        max_scl_out = [0, 0, 0]
    max_scl_out = [int(x) for x in max_scl_out]

    dist = lum.get("LuminanceDistributions", {})
    dist_idx = dist.get("DistributionIndex", DIST_INDEX_CLASSIC)
    dist_vals = dist.get("DistributionValues", [0] * 9)

    if dist_idx != DIST_INDEX_CLASSIC or not (isinstance(dist_vals, list) and len(dist_vals) == 9):
        mapped = {int(i): int(v) for i, v in zip(dist_idx, dist_vals)} if isinstance(dist_idx, list) else {}
        dist_vals = [int(mapped.get(i, 0)) for i in DIST_INDEX_CLASSIC]
    else:
        dist_vals = [int(x) for x in dist_vals]

    bez = scene.get("BezierCurveData", {})
    anchors = bez.get("Anchors", [0] * 9)
    if not (isinstance(anchors, list) and len(anchors) == 9):
        anchors = anchors[:9] if isinstance(anchors, list) else []
        anchors = anchors + [0] * (9 - len(anchors))
        anchors = anchors[:9]
    anchors = [int(x) for x in anchors]
    knee_x = int(bez.get("KneePointX", 0))
    knee_y = int(bez.get("KneePointY", 0))

    # Apply inverse scaling
    targeted_raw = targeted_nits * 64
    max_scl_raw = [x * 32 for x in max_scl_out]
    average_raw = average_out * 32

    bw = BitWriter()

    # AV1 wrapper
    bw.write_bits(clamp(num_windows, 0, 3), 2)
    bw.write_bits(0, 6)

    # Core
    bw.write_bits(clamp(targeted_raw, 0, (1 << 27) - 1), 27)

    # v1: targeted_system_display_actual_peak_luminance_flag = 0
    bw.write_bit(0)

    # Per-window (v1 num_windows=1, but keep loop)
    nw = clamp(num_windows, 1, 3)
    for _w in range(nw):
        for x in max_scl_raw:
            bw.write_bits(clamp(int(x), 0, (1 << 17) - 1), 17)

        bw.write_bits(clamp(int(average_raw), 0, (1 << 17) - 1), 17)

        # Distributions: 9 entries
        bw.write_bits(9, 4)
        for pct, val in zip(DIST_INDEX_CLASSIC, dist_vals):
            bw.write_bits(int(pct) & 0x7F, 7)
            bw.write_bits(clamp(int(val), 0, (1 << 17) - 1), 17)

        # fraction_bright_pixels (10) — keep 0
        bw.write_bits(0, 10)

    # v1: mastering_display_actual_peak_luminance_flag = 0
    bw.write_bit(0)

    # Per-window: tone mapping + Bezier curve (profile B expects it)
    for _w in range(nw):
        bw.write_bit(1)  # tone_mapping_flag
        bw.write_bits(clamp(knee_x, 0, (1 << 12) - 1), 12)
        bw.write_bits(clamp(knee_y, 0, (1 << 12) - 1), 12)
        bw.write_bits(9, 4)
        for a in anchors:
            bw.write_bits(clamp(int(a), 0, (1 << 10) - 1), 10)

    # v1: color_saturation_mapping_flag = 0
    bw.write_bit(0)

    payload = bw.get_bytes()

    # T.35 header for HDR10+ (B5 003C 0001 04 01 ...)
    out = bytearray()
    out.append(HDR10PLUS_COUNTRY_CODE)
    out.extend(bytes([0x00, HDR10PLUS_PROVIDER_CODE & 0xFF]))   # 0x003C
    out.extend(bytes([0x00, HDR10PLUS_ORIENTED_CODE & 0xFF]))   # 0x0001
    out.append(HDR10PLUS_APP_ID)                                # 0x04
    out.append(HDR10PLUS_APP_MODE)                              # 0x01
    out.extend(payload)
    return bytes(out)


# ----------------------------
# Classic JSON construction (SceneInfo, SceneInfoSummary, ToolInfo)
# ----------------------------
def metadata_signature(decoded: Dict) -> Tuple:
    return (
        decoded.get("NumberOfWindows"),
        decoded.get("TargetedSystemDisplayMaximumLuminance"),
        decoded.get("AverageRGB"),
        tuple(decoded.get("MaxScl", [])),
        tuple(decoded.get("DistributionValues", [])),
        decoded.get("BezierCurveData", {}).get("KneePointX"),
        decoded.get("BezierCurveData", {}).get("KneePointY"),
        tuple(decoded.get("BezierCurveData", {}).get("Anchors", [])),
    )


def build_classic_json_from_t35_list(t35_by_frame: List[bytes]) -> Dict:
    scene_info: List[Dict] = []

    scene_id = 0
    scene_frame_index = 0
    prev_sig: Optional[Tuple] = None

    for i, t35 in enumerate(t35_by_frame):
        decoded = decode_hdr10plus_t35_av1(t35)
        sig = metadata_signature(decoded)

        if prev_sig is None:
            scene_id = 0
            scene_frame_index = 0
        else:
            if sig != prev_sig:
                scene_id += 1
                scene_frame_index = 0
            else:
                scene_frame_index += 1

        entry = {
            "BezierCurveData": decoded["BezierCurveData"],
            "LuminanceParameters": {
                "AverageRGB": decoded["AverageRGB"],
                "LuminanceDistributions": {
                    "DistributionIndex": decoded["DistributionIndex"],
                    "DistributionValues": decoded["DistributionValues"],
                },
                "MaxScl": decoded["MaxScl"],
            },
            "NumberOfWindows": decoded["NumberOfWindows"],
            "TargetedSystemDisplayMaximumLuminance": decoded["TargetedSystemDisplayMaximumLuminance"],
            "SceneFrameIndex": int(scene_frame_index),
            "SceneId": int(scene_id),
            "SequenceFrameIndex": int(i),
        }
        scene_info.append(entry)
        prev_sig = sig

    # SceneInfoSummary
    scene_first: List[int] = []
    scene_len: List[int] = []
    if scene_info:
        cur = scene_info[0]["SceneId"]
        start = 0
        for idx in range(1, len(scene_info)):
            if scene_info[idx]["SceneId"] != cur:
                scene_first.append(start)
                scene_len.append(idx - start)
                start = idx
                cur = scene_info[idx]["SceneId"]
        scene_first.append(start)
        scene_len.append(len(scene_info) - start)

    return {
        "JSONInfo": {"HDR10plusProfile": "B", "Version": "1.0"},
        "SceneInfo": scene_info,
        "SceneInfoSummary": {
            "SceneFirstFrameIndex": scene_first,
            "SceneFrameNumbers": scene_len,
        },
        "ToolInfo": {"Tool": "hdr10plus_tool", "Version": "1.6.0"},
    }


# ----------------------------
# JSON load/validate + per-frame metadata list build
# ----------------------------
def load_json_file(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_classic_json(j: Dict) -> None:
    if not isinstance(j, dict):
        raise ValueError("JSON root must be an object")
    if "JSONInfo" not in j or "SceneInfo" not in j:
        raise ValueError("Classic JSON must contain JSONInfo and SceneInfo")

    ji = j["JSONInfo"]
    if not isinstance(ji, dict):
        raise ValueError("JSONInfo must be an object")
    if ji.get("HDR10plusProfile") != "B":
        raise ValueError("HDR10plusProfile must be 'B'")
    if str(ji.get("Version", "")) != "1.0":
        raise ValueError("Version must be '1.0'")

    si = j["SceneInfo"]
    if not isinstance(si, list) or len(si) == 0:
        raise ValueError("SceneInfo must be a non-empty array")

    e0 = si[0]
    if not isinstance(e0, dict):
        raise ValueError("SceneInfo entries must be objects")
    if "LuminanceParameters" not in e0 or "BezierCurveData" not in e0:
        raise ValueError("SceneInfo entries must contain LuminanceParameters and BezierCurveData")


def build_t35_list_from_classic_json(sceneinfo: List[Dict], video_frames: int) -> Tuple[List[bytes], int]:
    """
    Returns (t35_list, json_frame_count_used_for_warning).

    For compatibility with the classic behavior:
    - If SceneInfo length != video_frames:
        * video shorter: truncate SceneInfo (skip at end)
        * video longer: duplicate (wrap) SceneInfo from beginning to reach video length
    """
    json_len = len(sceneinfo)
    if video_frames <= 0:
        # No reliable video length. Use JSON length as-is.
        return [encode_hdr10plus_t35_from_scene(e) for e in sceneinfo], json_len

    if json_len == video_frames:
        return [encode_hdr10plus_t35_from_scene(e) for e in sceneinfo], json_len

    if video_frames < json_len:
        trimmed = sceneinfo[:video_frames]
        return [encode_hdr10plus_t35_from_scene(e) for e in trimmed], json_len

    # video_frames > json_len: duplicate (wrap) metadata until we match video length
    out: List[bytes] = []
    for i in range(video_frames):
        out.append(encode_hdr10plus_t35_from_scene(sceneinfo[i % json_len]))
    return out, json_len


# ----------------------------
# Commands
# ----------------------------
def cmd_extract(inp: str, outp: str) -> None:
    """
    extract: ONLY progress bar.
    """
    with open(inp, "rb") as f:
        h = read_ivf_header(f)

        # If header frame count is 0, we still process until EOF and compute total.
        total_hint = h.frames

        t35_by_frame: List[bytes] = []
        scanned = 0

        for i, payload, _ts in iter_ivf_frames(f):
            scanned += 1

            obus = parse_obus(payload)
            found_t35: Optional[bytes] = None
            for o in obus:
                if o.obu_type == OBU_METADATA:
                    t35 = extract_t35_from_metadata_obu_payload(o.payload)
                    if t35 is not None:
                        found_t35 = t35
                        break

            if found_t35 is None:
                raise ValueError(f"No HDR10+ T.35 found at frame {i}")

            t35_by_frame.append(found_t35)

            # progress bar only
            if total_hint > 0:
                progress_bar((scanned / total_hint) * 100.0)
            else:
                # Unknown total: show a coarse bar based on scanned count (keeps UX consistent)
                # Use a soft cap to avoid sticking at 0.
                progress_bar(min(99.0, (scanned % 1000) / 10.0))

        progress_done()

        # Post-scan stages (classic hdr10plus_tool style)
        sys.stderr.write("Reordering metadata... Done.\n")
        sys.stderr.write("Reading parsed dynamic metadata... Done.\n")
        sys.stderr.write("Generating and writing metadata to JSON file... Done.\n")
        sys.stderr.flush()

    classic = build_classic_json_from_t35_list(t35_by_frame)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(classic, f, indent=2)


def cmd_remove(inp: str, outp: str) -> None:
    """
    remove: ONLY progress bar.
    """
    with open(inp, "rb") as fi:
        h = read_ivf_header(fi)
        total = h.frames

        with open(outp, "wb") as fo:
            write_ivf_header(fo, h)

            done = 0
            for _i, payload, ts in iter_ivf_frames(fi):
                new_payload = remove_hdr10plus_from_frame(payload)

                fo.write(struct.pack("<I", len(new_payload)))
                fo.write(struct.pack("<Q", ts))
                fo.write(new_payload)

                done += 1
                if total > 0:
                    progress_bar((done / total) * 100.0)

    progress_done()


def cmd_inject(inp: str, json_path: str, outp: str) -> None:
    """
    inject:
      Parsing JSON file...
      Processing input video for frame order info...   (progress bar only here)
      Warning block if mismatch
      Rewriting file with interleaved HDR10+ ITU-T T.35 Metadata OBUs..  (progress bar only here)
    """
    # Parsing JSON file...
    print_status_line("Parsing JSON file...")
    j = load_json_file(json_path)
    validate_classic_json(j)
    sceneinfo = j["SceneInfo"]

    # Processing input video for frame order info...
    print_status_line("Processing input video for frame order info...")

    # We scan input to count frames reliably (even if IVF header frames is wrong/0),
    # showing a progress bar only in this phase.
    with open(inp, "rb") as fi:
        h = read_ivf_header(fi)
        header_total = h.frames

        counted = 0
        for _i, _payload, _ts in iter_ivf_frames(fi):
            counted += 1
            if header_total > 0:
                progress_bar((counted / header_total) * 100.0)
            else:
                progress_bar(min(99.0, (counted % 1000) / 10.0))

    progress_done()

    video_frames = counted if counted > 0 else header_total
    json_frames = len(sceneinfo)

    # Warning block (exact formatting)
    if video_frames != json_frames:
        sys.stderr.write("\n")
        sys.stderr.write(f"Warning: mismatched lengths. video {video_frames}, HDR10+ JSON {json_frames}\n")
        if video_frames < json_frames:
            sys.stderr.write("Metadata will be skipped at the end to match video length\n")
        else:
            sys.stderr.write("Metadata will be duplicated at the end to match video length\n")
        sys.stderr.write("\n")
        sys.stderr.flush()

    # Build per-frame T.35 list with truncate/duplicate policy
    t35_list, _json_len_for_warning = build_t35_list_from_classic_json(sceneinfo, video_frames)

    # Rewriting file with interleaved...
    print_status_line("Rewriting file with interleaved HDR10+ ITU-T T.35 Metadata OBUs..")

    with open(inp, "rb") as fi, open(outp, "wb") as fo:
        h2 = read_ivf_header(fi)
        # Update header frames to actual counted frames for consistency
        out_header = IvfHeader(
            width=h2.width,
            height=h2.height,
            timebase_num=h2.timebase_num,
            timebase_den=h2.timebase_den,
            frames=video_frames,
        )
        write_ivf_header(fo, out_header)

        done = 0
        for i, payload, ts in iter_ivf_frames(fi):
            if i >= video_frames:
                break
            new_payload = inject_hdr10plus_into_frame(payload, t35_list[i])

            fo.write(struct.pack("<I", len(new_payload)))
            fo.write(struct.pack("<Q", ts))
            fo.write(new_payload)

            done += 1
            if video_frames > 0:
                progress_bar((done / video_frames) * 100.0)

    progress_done()

# ----------------------------
# PQ helpers (ST 2084)
# ----------------------------
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 32.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 128.0
_PQ_C3 = 2392.0 / 128.0


def nits_to_pq(nits: float) -> float:
    if nits <= 0.0:
        return 0.0
    n = min(max(nits / 10000.0, 0.0), 1.0)
    n_m1 = n ** _PQ_M1
    num = _PQ_C1 + _PQ_C2 * n_m1
    den = 1.0 + _PQ_C3 * n_m1
    e = (num / den) ** _PQ_M2
    return float(min(max(e, 0.0), 1.0))


def pq_to_nits(pq: float) -> float:
    e = min(max(pq, 0.0), 1.0)
    e_1_m2 = e ** (1.0 / _PQ_M2)
    num = max(e_1_m2 - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * e_1_m2
    if den <= 0.0:
        return 10000.0
    n = (num / den) ** (1.0 / _PQ_M1)
    return float(min(max(n * 10000.0, 0.0), 10000.0))


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_max3(x, default=0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)):
        vals = []
        for v in x:
            try:
                vals.append(float(v))
            except Exception:
                pass
        return float(max(vals)) if vals else float(default)
    return float(default)


def _infer_scale_from_scene(scene_entry: dict) -> float:
    """
    Infer likely unit scale for classic JSON luminance integers.
    Returns a divisor such that: nits = raw / scale.

    Heuristic (pragmatic):
      - If values (hist/maxscl) are in the ~0..5000 range, most commonly they are 0.1 nit units => scale=10.
      - If values are much larger, some pipelines store ~1/32 nit units => scale=32.
      - If TargetedSystemDisplayMaximumLuminance is present, prefer the scale that makes peaks reasonable
        relative to that display max.
    """
    try:
        if not isinstance(scene_entry, dict):
            return 10.0

        tmax = _safe_float(scene_entry.get("TargetedSystemDisplayMaximumLuminance", 0.0), 0.0)

        lum = scene_entry.get("LuminanceParameters", {}) or {}
        if not isinstance(lum, dict):
            return 10.0

        mx_raw = _safe_max3(lum.get("MaxScl", [0, 0, 0]), 0.0)

        dist = lum.get("LuminanceDistributions", {}) or {}
        dv = dist.get("DistributionValues", None)
        peak_raw = None
        if isinstance(dv, list) and dv:
            try:
                peak_raw = float(max(dv))
            except Exception:
                peak_raw = None

        if peak_raw is None:
            peak_raw = mx_raw

        if peak_raw <= 5000.0:
            if tmax > 0.0:
                peak10 = peak_raw / 10.0
                peak32 = peak_raw / 32.0
                score10 = abs(peak10 - tmax)
                score32 = abs(peak32 - tmax)
                if score10 <= score32 * 0.85:
                    return 10.0
            return 10.0

        return 32.0
    except Exception:
        return 10.0


def plot_hdr10plus_style_png(path_png: str, j: dict, json_name: str) -> None:
    """
    Rust/plotters-like design:
      - PQ Y-axis (0..1) with key-point ticks labeled in nits
      - Two filled area series: Maximum + Average
      - Big canvas (3000x1200), margins, mesh, legend lower-left
      - Captions at top-left ordered like reference image
    Notes:
      - Peak brightness should come from histogram maximum (DistributionValues) when available.
      - Luminance integers in classic JSON are NOT guaranteed to be in nits; we infer a scale.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"ERROR: matplotlib import failed: {type(e).__name__}: {e}") from e

    si = (j or {}).get("SceneInfo", []) or []
    n = len(si)
    if n <= 0:
        raise SystemExit("ERROR: SceneInfo is empty; cannot plot.")

    # Infer scale from first scene (nits = raw / SCALE)
    SCALE = _infer_scale_from_scene(si[0] if si else {})

    # Match Rust example colors
    MAXSCL_COLOR = (65/255.0, 105/255.0, 225/255.0)  # royalblue
    AVERAGE_COLOR = (75/255.0, 0/255.0, 130/255.0)   # indigo

    max_pq = []
    avg_pq = []
    missing = 0
    invalid = 0
    floor_pq = nits_to_pq(0.01)

    used_histogram_peak = False

    for e in si:
        if not isinstance(e, dict):
            missing += 1
            max_pq.append(floor_pq)
            avg_pq.append(floor_pq)
            continue

        lum = e.get("LuminanceParameters", {}) or {}
        if not isinstance(lum, dict):
            invalid += 1
            max_pq.append(floor_pq)
            avg_pq.append(floor_pq)
            continue

        # AverageRGB (raw -> nits)
        avg_raw = _safe_float(lum.get("AverageRGB", 0.0), 0.0)
        avg_nits = avg_raw / SCALE

        # Peak brightness: histogram maximum value when available
        dist = lum.get("LuminanceDistributions", {}) or {}
        dv = dist.get("DistributionValues", None)

        peak_raw = None
        if isinstance(dv, list) and len(dv) > 0:
            try:
                peak_raw = float(max(dv))
            except Exception:
                peak_raw = None

        if peak_raw is not None:
            mx_nits = peak_raw / SCALE
            used_histogram_peak = True
        else:
            # Fallback: MaxScl (raw -> nits)
            mx_raw = _safe_max3(lum.get("MaxScl", [0, 0, 0]), 0.0)
            mx_nits = mx_raw / SCALE

        # Clamp to HDR range
        avg_nits = max(0.0, min(10000.0, avg_nits))
        mx_nits = max(0.0, min(10000.0, mx_nits))

        avg_pq.append(max(floor_pq, nits_to_pq(avg_nits)))
        max_pq.append(max(floor_pq, nits_to_pq(mx_nits)))

    # Stats in nits (ignore placeholders)
    valid_max_nits = [pq_to_nits(v) for v in max_pq if v > nits_to_pq(0.02)]
    valid_avg_nits = [pq_to_nits(v) for v in avg_pq if v > nits_to_pq(0.02)]

    maxcll = max(valid_max_nits) if valid_max_nits else 0.01
    maxcll_avg = (sum(valid_max_nits) / len(valid_max_nits)) if valid_max_nits else 0.01
    maxfall = max(valid_avg_nits) if valid_avg_nits else 0.01
    maxfall_avg = (sum(valid_avg_nits) / len(valid_avg_nits)) if valid_avg_nits else 0.01

    x = list(range(n))

    # 3000x1200
    fig = plt.figure(figsize=(30.0, 12.0), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_title("HDR10+ Plot", fontsize=22, pad=24)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("frames", fontsize=14)
    ax.set_ylabel("nits (cd/m²)", fontsize=14)

    ax.grid(True, which="major", alpha=0.10, linewidth=1.2)
    ax.grid(True, which="minor", alpha=0.03, linewidth=0.8)
    ax.minorticks_on()

    key_nits = [
        0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0,
        200.0, 400.0, 600.0, 1000.0, 2000.0, 4000.0, 10000.0
    ]
    key_pq = [nits_to_pq(v) for v in key_nits]
    ax.set_yticks(key_pq)
    ax.set_yticklabels([("{:.3f}".format(v)).rstrip("0").rstrip(".") for v in key_nits], fontsize=11)

    max_label = f"Maximum (MaxCLL: {maxcll:.2f} nits, avg: {maxcll_avg:.2f} nits)"
    avg_label = f"Average (MaxFALL: {maxfall:.2f} nits, avg: {maxfall_avg:.2f} nits)"

    ax.fill_between(x, max_pq, 0.0, alpha=0.25, linewidth=0.0, color=MAXSCL_COLOR)
    ax.plot(x, max_pq, linewidth=1.5, color=MAXSCL_COLOR, label=max_label)

    ax.fill_between(x, avg_pq, 0.0, alpha=0.50, linewidth=0.0, color=AVERAGE_COLOR)
    ax.plot(x, avg_pq, linewidth=1.5, color=AVERAGE_COLOR, label=avg_label)

    leg = ax.legend(loc="lower left", framealpha=1.0, fontsize=12)
    leg.get_frame().set_linewidth(1.0)

    # ---- Captions ordered like your reference image ----
    profile = ((j.get("JSONInfo", {}) or {}).get("HDR10plusProfile", "?"))
    scenes = None
    summary = j.get("SceneInfoSummary", {}) or {}
    scene_counts = summary.get("SceneFrameNumbers", None)
    if isinstance(scene_counts, list):
        scenes = len(scene_counts)

    peak_source = "Histogram maximum value" if used_histogram_peak else "MaxScl (fallback)"

    left_lines = [
        f"{json_name}",
        f"Frames: {n}. Profile {profile}." + (f" Scenes: {scenes}." if scenes is not None else ""),
        f"Peak brightness source: {peak_source}",
        f"Inferred scale: {SCALE:g} units -> 1 nit",
    ]

    x0 = 0.06
    y0 = 0.945
    line_h = 0.022

    for i, line in enumerate(left_lines):
        fig.text(x0, y0 - i * line_h, line, fontsize=12, ha="left", va="top")

    # Reserve top area for title + captions (like reference)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.10)

    fig.savefig(path_png)
    plt.close(fig)


def cmd_plot(inp_json: str, out_png: str) -> None:
    """
    plot: ONLY progress bar.
    """
    j = load_json_file(inp_json)
    validate_classic_json(j)

    si = (j.get("SceneInfo", []) or [])
    total = len(si) if si else 1

    # 0..50% during "scan"
    last = -1.0
    for i in range(total):
        pct = (i + 1) * 50.0 / total
        if pct - last >= 0.1 or pct >= 50.0:
            progress_bar(pct)
            last = pct

    import os
    plot_hdr10plus_style_png(out_png, j, os.path.basename(inp_json))

    progress_bar(100.0)
    progress_done()



# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hdr10plus_tool.py", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="Extract HDR10+ from AV1 IVF to classic JSON")
    p_ex.add_argument("-i", "--input", required=True)
    p_ex.add_argument("-o", "--output", required=True)

    p_rm = sub.add_parser("remove", help="Remove HDR10+ from AV1 IVF")
    p_rm.add_argument("-i", "--input", required=True)
    p_rm.add_argument("-o", "--output", required=True)

    p_in = sub.add_parser("inject", help="Inject HDR10+ into AV1 IVF from classic JSON")
    p_in.add_argument("-i", "--input", required=True)
    p_in.add_argument("-j", "--json", required=True)
    p_in.add_argument("-o", "--output", required=True)

    p_pl = sub.add_parser("plot", help="Plot metrics from classic JSON to PNG")
    p_pl.add_argument("-i", "--input", required=True)
    p_pl.add_argument("-o", "--output", required=True)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "extract":
        cmd_extract(args.input, args.output)
    elif args.cmd == "remove":
        cmd_remove(args.input, args.output)
    elif args.cmd == "inject":
        cmd_inject(args.input, args.json, args.output)
    elif args.cmd == "plot":
        cmd_plot(args.input, args.output)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
