"""
MP4 telemetry stream extraction — this module parses the GPMF telemetry track embedded in GoPro MP4 files.

For each MP4 file, parse the GPMF telemetry track and produce three output files.

1. `{flight_tag}_mp4_meta.json` — flight metadata (§2.5)
2. `{flight_tag}_mp4.csv` — GPS-rate trajectory CSV (§2.3)
3. `{flight_tag}_mp4_hf.csv` — high-frequency IMU / orientation CSV (§2.4)

#### 2.1 GPMF parsing and timestamping

The GoPro MP4 contains a `meta` track (sample description type `gpmd`) holding GPMF payloads. Each payload covers a time window determined by the MP4 track's `stts` (sample durations) and `stco` (chunk offsets). Within each payload, GPMF data is organized as nested `DEVC > STRM` containers. Each stream carries:

- **TSMP** — running total sample count since recording start.
- **SCAL** — scaling divisor(s) to convert raw integers to physical units.
- **STMP** — microsecond timestamp for sub-payload precision.

Timestamps for individual samples are derived by distributing them evenly across the payload's time window (from the MP4 track clock), refined by TSMP deltas and STMP where available.

#### 2.2 Relevant GPMF streams

The following GPMF streams are extracted. All other streams (DVID, ISOE, SHUT, WBAL, WRGB, YAVG, UNIF, AALP, WNDM, MWET, MSKP, LSKP, LRVO, LRVS — camera/video metadata) are ignored.

| FourCC | Description | ~Rate | Physical Units (after SCAL) |
|--------|-------------|-------|-----------------------------|
| GPS5 | lat, lon, alt (WGS 84), 2D ground speed, 3D speed | 18 Hz | deg, deg, m, m/s, m/s |
| GPSF | GPS fix status | 1 Hz | 0 = no lock, 2 = 2D, 3 = 3D |
| GPSP | GPS precision (DOP × 100) | 1 Hz | unitless; < 500 is good |
| ACCL | 3-axis accelerometer | 200 Hz | m/s² |
| GYRO | 3-axis gyroscope | 200 Hz | rad/s |
| CORI | Camera orientation quaternion (w, x, y, z) | frame rate | unitless (sensor-fused) |
| IORI | Image orientation quaternion (relative to camera body) | frame rate | unitless |
| GRAV | Gravity direction vector (x, y, z) | frame rate | unitless (normalized) |

#### 2.3 Output: GPS-rate trajectory CSV

Output path: `config.DIR.TELEMETRY / f"{flight_tag}_mp4.csv"`

One row per GPS5 sample (~18 Hz). Slow-rate streams (GPSF, GPSP) are forward-filled to the GPS sample rate. Accelerometer (ACCL) is downsampled to the GPS rate by taking the nearest sample.

| Column | Source | Units | Description |
|--------|--------|-------|-------------|
| `timestamp_s` | MP4 track + TSMP | s | seconds since recording start |
| `gps_lat_deg` | GPS5[0] / SCAL | deg | latitude WGS 84 |
| `gps_lon_deg` | GPS5[1] / SCAL | deg | longitude WGS 84 |
| `gps_alt_m` | GPS5[2] / SCAL | m | altitude WGS 84 |
| `gps_speed2d_ms` | GPS5[3] / SCAL | m/s | 2D ground speed |
| `gps_speed3d_ms` | GPS5[4] / SCAL | m/s | 3D speed |
| `gps_fix` | GPSF | — | 0 / 2 / 3 |
| `gps_dop` | GPSP / 100 | — | dilution of precision |
| `accl_x_ms2` | ACCL[0] / SCAL | m/s² | accelerometer X (nearest to GPS timestamp) |
| `accl_y_ms2` | ACCL[1] / SCAL | m/s² | accelerometer Y |
| `accl_z_ms2` | ACCL[2] / SCAL | m/s² | accelerometer Z |

This CSV is the primary input for pipeline phases 1–6 (trajectory reconstruction, cleaning, alignment, fusion, kinematics). The `timestamp_s` column contains seconds since recording start; to obtain absolute UTC times, add `gps_start_utc` from the metadata JSON (§2.5).

#### 2.4 Output: high-frequency IMU / orientation CSV

Output path: `config.DIR.TELEMETRY / f"{flight_tag}_mp4_hf.csv"`

One row per ACCL/GYRO sample (~200 Hz). Frame-rate streams (CORI, IORI, GRAV) are linearly interpolated to the IMU rate.

| Column | Source | Units | Description |
|--------|--------|-------|-------------|
| `timestamp_s` | MP4 track + TSMP | s | seconds since recording start |
| `accl_x_ms2` | ACCL[0] / SCAL | m/s² | accelerometer X |
| `accl_y_ms2` | ACCL[1] / SCAL | m/s² | accelerometer Y |
| `accl_z_ms2` | ACCL[2] / SCAL | m/s² | accelerometer Z |
| `gyro_x_rads` | GYRO[0] / SCAL | rad/s | gyroscope X |
| `gyro_y_rads` | GYRO[1] / SCAL | rad/s | gyroscope Y |
| `gyro_z_rads` | GYRO[2] / SCAL | rad/s | gyroscope Z |
| `cori_w` | CORI[0] / SCAL | — | camera orientation quaternion W |
| `cori_x` | CORI[1] / SCAL | — | camera orientation quaternion X |
| `cori_y` | CORI[2] / SCAL | — | camera orientation quaternion Y |
| `cori_z` | CORI[3] / SCAL | — | camera orientation quaternion Z |
| `iori_w` | IORI[0] / SCAL | — | image orientation quaternion W |
| `iori_x` | IORI[1] / SCAL | — | image orientation quaternion X |
| `iori_y` | IORI[2] / SCAL | — | image orientation quaternion Y |
| `iori_z` | IORI[3] / SCAL | — | image orientation quaternion Z |
| `grav_x` | GRAV[0] / SCAL | — | gravity vector X |
| `grav_y` | GRAV[1] / SCAL | — | gravity vector Y |
| `grav_z` | GRAV[2] / SCAL | — | gravity vector Z |

This CSV is the primary input for phase 7 (high-frequency jitter extraction). CORI is the preferred orientation source; GYRO is used as a fallback or for validation.

> **Note on IMU axis order:** GoPro IMU axis conventions vary by camera model (e.g. HERO5: Z,X,Y; HERO6+: Y,-X,Z for ACCL). The extraction code must detect the camera model from GPMF header metadata (DVNM) and remap axes to a consistent body-frame convention (X-Forward, Y-Right, Z-Down) before writing the CSV.

#### 2.5 Output: flight metadata JSON

Output path: `config.DIR.TELEMETRY / f"{flight_tag}_mp4_meta.json"`

Contains recording-level metadata extracted from three sources:
- MP4 container tags (via ``ffprobe``)
- ffprobe stream properties (video resolution, codec, frame rate, audio format)
- GoPro "Global Settings" GPMF payload stored in the ``moov > udta > GPMF`` atom

##### 2.5.1 Top-level fields

| Key | Source | Description |
|-----|--------|-------------|
| `device_name` | GPMF `DVNM` | Camera model, e.g. `"HERO9 Black"` |
| `gps_start_utc` | GPMF `GPSU` (first payload) | ISO 8601 UTC timestamp of the first GPS sample |
| `creation_time` | MP4 container tag | File creation timestamp as written by the camera |
| `firmware` | GPMF `FMWR` / MP4 tag | Camera firmware version, e.g. `"HD9.01.01.72.00"` |
| `timecode` | MP4 video stream tag | Wall-clock timecode at recording start, e.g. `"12:47:35:23"` |
| `location` | MP4 container tag | Approximate start location, e.g. `"+52.2700+020.9148/"` |
| `duration_s` | MP4 container | Total recording duration in seconds |
| `camera_serial` | GPMF `CASN` | Camera serial number |
| `lens_serial` | GPMF `LINF` | Lens / Max Lens Mod serial number |
| `metadata_version` | GPMF `VERS` | GPMF metadata format version, e.g. `"8.1.2"` |

##### 2.5.2 `video` sub-object

| Key | Source | Description |
|-----|--------|-------------|
| `resolution` | ffprobe | e.g. `"1920x1080"` |
| `frame_rate_hz` | ffprobe `r_frame_rate` | Frames per second, e.g. `59.94` |
| `codec` | ffprobe | e.g. `"h264"` |
| `profile` | ffprobe | e.g. `"High"` |
| `bit_rate_mbps` | ffprobe | Video stream bit rate in Mbps |
| `color_space` | ffprobe | e.g. `"bt709"` |
| `color_range` | ffprobe | e.g. `"pc"` (full range) or `"tv"` (limited) |

##### 2.5.3 `audio` sub-object

| Key | Source | Description |
|-----|--------|-------------|
| `codec` | ffprobe | e.g. `"aac"` |
| `sample_rate_hz` | ffprobe | e.g. `48000` |
| `channels` | ffprobe | e.g. `2` |
| `bit_rate_kbps` | ffprobe | Audio stream bit rate in kbps |

##### 2.5.4 `settings` sub-object — GoPro camera settings

Extracted from the GPMF "Global Settings" payload stored in the MP4 ``moov > udta > GPMF`` atom.
These are the camera settings that were active at the time of recording.

| Key | GPMF FourCC | Example | Description |
|-----|-------------|---------|-------------|
| `protune` | `PRTN` | `true` | Protune mode enabled |
| `white_balance` | `PTWB` | `"AUTO"` | Protune white balance setting |
| `sharpness` | `PTSH` | `"HIGH"` | Protune sharpness |
| `color_mode` | `PTCL` | `"GOPRO"` | Protune color profile |
| `exposure_type` | `EXPT` | `"AUTO"` | Exposure control mode |
| `auto_iso_max` | `PIMX` | `1600` | Maximum auto ISO |
| `auto_iso_min` | `PIMN` | `100` | Minimum auto ISO |
| `ev_compensation` | `PTEV` | `"0.0"` | Exposure compensation |
| `field_of_view` | `VFOV` | `"W"` | FOV setting: W=Wide, N=Narrow, L=Linear, etc. |
| `digital_zoom_on` | `DZOM` | `true` | Digital zoom enabled |
| `digital_zoom` | `DZST` | `255` | Digital zoom level (0–255) |
| `electronic_stabilization_on` | `EISE` | `true` | EIS (HyperSmooth) enabled |
| `electronic_stabilization` | `EISA` | `"HS High"` | EIS algorithm/level (e.g. `"HS High"`, `"HS Boost"`) |
| `lens_projection` | `PRJT` | `"GPRO"` | Lens projection: `"GPRO"` = standard, `"MLNS"` = Max Lens Mod |
| `spot_meter` | `SMTR` | `false` | Spot metering enabled |
| `auto_rotation` | `OREN` | `"U"` | Auto rotation: `"U"` = Up, `"A"` = Auto |
| `audio_setting` | `AUDO` | `"AUTO"` | Audio processing mode |
| `auto_protune` | `AUPT` | `false` | Auto Protune enabled |
| `highlight_tag_rate` | `RATE` | `"2_1SEC"` | HiLight tag rate |
| `sensor_readout_time_ms` | `SROT` | `11.689` | Sensor rolling-shutter readout time in ms |
| `diagonal_fov_deg` | `ZFOV` | `64.805` | Actual diagonal field of view in degrees |
| `camera_mode` | `CMOD` | `12` | Internal camera mode ID |
| `media_type` | `MTYP` | `0` | Media type (0 = video) |
| `vlte` | `VLTE` | `false` | Video low-light mode |
| `hilight_level` | `HLVL` | `false` | HiLight level |
| `broadcast` | `BROD` | `""` | Broadcast/live streaming destination |

> **Max Lens Mod detection:** The ``lens_projection`` field indicates the lens
> type. Standard GoPro lens = ``"GPRO"``; Max Lens Mod = ``"MLNS"``.
> Additionally, ``lens_serial`` will contain the MLM serial when attached.

> **Stabilization detection:** ``electronic_stabilization_on`` is ``true`` when
> HyperSmooth is active. ``electronic_stabilization`` gives the exact level
> (e.g. ``"HS High"``, ``"HS Boost"``, ``"Standard"``).

The `gps_start_utc` field is the authoritative absolute time reference. To convert any `timestamp_s` value from the CSV files to UTC: `utc = gps_start_utc + timedelta(seconds=timestamp_s - first_gps_timestamp_s)`, where `first_gps_timestamp_s` is the earliest `timestamp_s` in the GPS-rate CSV.



"""

from __future__ import annotations

import json
import logging
import re
import struct
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rich.console
import rich.logging

from flight_composer.config import config
from flight_composer.processing.flight_data import (
    FlightData,
    MP4Telemetry,
    WGS84Coordinate,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    log_format = r"\[[bold]%(name)s[/bold]] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=log_format,
        datefmt="[%X]",
        handlers=[
            rich.logging.RichHandler(
                console=rich.console.Console(color_system="auto"),
                show_level=True,
                show_path=False,
                enable_link_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                markup=True,
            )
        ],
    )


# ---------------------------------------------------------------------------
# GPMF binary type → Python struct format character
# ---------------------------------------------------------------------------
_TYPE_FMT: dict[str, str] = {
    "b": "b",
    "B": "B",
    "s": "h",
    "S": "H",
    "l": "i",
    "L": "I",
    "f": "f",
    "d": "d",
    "J": "Q",
}

# FourCC codes we care about inside each STRM container
_RELEVANT_DATA = {
    "GPS5",
    "GPSF",
    "GPSP",
    "ACCL",
    "GYRO",
    "CORI",
    "IORI",
    "GRAV",
}

# Metadata KLV keys found alongside data inside a STRM
_META_KEYS = {
    "TSMP",
    "STMP",
    "SCAL",
    "ORIN",
    "STNM",
    "SIUN",
    "DVNM",
    "DVID",
    "GPSU",
    "GPSA",
    "UNIT",
    "TYPE",
}


# ---------------------------------------------------------------------------
# Data-classes for parsed payload data
# ---------------------------------------------------------------------------


@dataclass
class StreamSamples:
    """Parsed samples from one STRM container within a single GPMF payload."""

    fourcc: str  # e.g. "ACCL", "GPS5"
    raw_values: np.ndarray  # shape (N, C) — N samples, C components
    scal: np.ndarray  # shape (C,) or (1,) scaling divisors
    stmp_us: int  # STMP microsecond timestamp of first sample
    tsmp: int  # TSMP running total count
    orin: str | None  # e.g. "ZXY" — axis order label
    # Per-payload metadata that lives in the same STRM as certain data
    gpsf: int | None = None  # GPS fix (only in GPS STRM)
    gpsp: int | None = None  # GPS precision ×100 (only in GPS STRM)


@dataclass
class ParsedPayload:
    """All relevant streams from one GPMF payload (≈1 s of data)."""

    pts_time: float  # seconds — MP4 track PTS
    duration: float  # seconds — MP4 track packet duration
    device_name: str  # e.g. "HERO9 Black"
    gpsu: str | None = None  # GPSU UTC timestamp string, e.g. "250508104822.180"
    streams: dict[str, StreamSamples] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low-level GPMF KLV parsing
# ---------------------------------------------------------------------------


def _parse_klv(
    data: bytes,
) -> list[tuple[str, tuple[str, int, int] | None, bytes | list]]:
    """Recursively parse GPMF KLV items.

    Returns a list of ``(fourcc, (type_char, sample_size, repeat) | None, payload)``
    where *payload* is ``bytes`` for leaf items or a nested ``list`` for containers.
    """
    pos = 0
    results: list[tuple] = []
    length = len(data)
    while pos + 8 <= length:
        key = data[pos : pos + 4].decode("latin1", errors="replace")
        type_char = chr(data[pos + 4])
        sample_size = data[pos + 5]
        repeat = struct.unpack(">H", data[pos + 6 : pos + 8])[0]
        data_len = sample_size * repeat
        total = 8 + data_len
        padding = (4 - (total % 4)) % 4
        payload = data[pos + 8 : pos + 8 + data_len]
        if type_char == "\x00":
            results.append((key, None, _parse_klv(payload)))
        else:
            results.append((key, (type_char, sample_size, repeat), payload))
        pos += total + padding
    return results


def _unpack_values(
    type_char: str, sample_size: int, repeat: int, data: bytes
) -> np.ndarray | None:
    """Unpack a leaf KLV item into an ndarray of shape ``(repeat, elems_per_sample)``."""
    fmt_char = _TYPE_FMT.get(type_char)
    if fmt_char is None:
        return None
    elem_size = struct.calcsize(">" + fmt_char)
    if elem_size == 0:
        return None
    elems_per_sample = sample_size // elem_size
    total_elems = elems_per_sample * repeat
    nbytes = total_elems * elem_size
    if nbytes > len(data) or total_elems == 0:
        return None
    flat = struct.unpack(">" + fmt_char * total_elems, data[:nbytes])
    arr = np.array(flat, dtype=np.float64)
    if elems_per_sample > 1:
        arr = arr.reshape(repeat, elems_per_sample)
    else:
        arr = arr.reshape(repeat, 1)
    return arr


def _decode_string(data: bytes, sample_size: int, repeat: int) -> str:
    return (
        data[: sample_size * repeat].decode("latin1", errors="replace").rstrip("\x00")
    )


# ---------------------------------------------------------------------------
# Payload-level parsing
# ---------------------------------------------------------------------------


def _parse_single_payload(
    raw: bytes, pts_time: float, duration: float
) -> ParsedPayload:
    """Parse one GPMF payload (one MP4 packet) into a :class:`ParsedPayload`."""
    tree = _parse_klv(raw)
    device_name = ""
    payload_gpsu: str | None = None
    streams: dict[str, StreamSamples] = {}

    for key, info, children in tree:
        if info is not None:
            continue  # only process containers (DEVC)

        # children is a list — iterate DEVC contents
        assert isinstance(children, list)
        for ckey, cinfo, cdata in children:
            if ckey == "DVNM" and cinfo is not None and isinstance(cdata, bytes):
                tc, ss, rp = cinfo
                device_name = _decode_string(cdata, ss, rp)

        # Iterate STRM containers
        for ckey, cinfo, cchildren in children:
            if cinfo is not None or not isinstance(cchildren, list):
                continue  # skip non-container items at DEVC level

            # cchildren is a list of items inside this STRM
            tsmp: int = 0
            stmp_us: int = 0
            scal: np.ndarray | None = None
            orin: str | None = None
            gpsf: int | None = None
            gpsp: int | None = None
            gpsu: str | None = None
            data_fourcc: str | None = None
            data_arr: np.ndarray | None = None

            for skey, sinfo, sdata in cchildren:
                if sinfo is None:
                    continue  # skip nested containers (e.g. UNIT)
                tc, ss, rp = sinfo
                if skey == "TSMP":
                    vals = _unpack_values(tc, ss, rp, sdata)
                    if vals is not None:
                        tsmp = int(vals[0, 0])
                elif skey == "STMP":
                    vals = _unpack_values(tc, ss, rp, sdata)
                    if vals is not None:
                        stmp_us = int(vals[0, 0])
                elif skey == "SCAL":
                    scal = _unpack_values(tc, ss, rp, sdata)
                    if scal is not None:
                        scal = scal.flatten()
                elif skey == "ORIN" and tc == "c":
                    orin = _decode_string(sdata, ss, rp)
                elif skey == "GPSF":
                    vals = _unpack_values(tc, ss, rp, sdata)
                    if vals is not None and vals.size > 0:
                        gpsf = int(vals.flat[0])
                elif skey == "GPSP":
                    vals = _unpack_values(tc, ss, rp, sdata)
                    if vals is not None and vals.size > 0:
                        gpsp = int(vals.flat[0])
                elif skey == "GPSU" and tc == "U":
                    gpsu = _decode_string(sdata, ss, rp)
                elif skey in _RELEVANT_DATA:
                    arr = _unpack_values(tc, ss, rp, sdata)
                    if arr is not None:
                        data_fourcc = skey
                        data_arr = arr

            if data_fourcc is not None and data_arr is not None:
                if scal is None:
                    scal = np.ones(data_arr.shape[1], dtype=np.float64)
                assert scal is not None  # for type checker
                streams[data_fourcc] = StreamSamples(
                    fourcc=data_fourcc,
                    raw_values=data_arr,
                    scal=scal,
                    stmp_us=stmp_us,
                    tsmp=tsmp,
                    orin=orin,
                    gpsf=gpsf,
                    gpsp=gpsp,
                )
            # GPSU lives in the GPS STRM — propagate to payload level
            if gpsu is not None:
                payload_gpsu = gpsu

    return ParsedPayload(
        pts_time=pts_time,
        duration=duration,
        device_name=device_name,
        gpsu=payload_gpsu,
        streams=streams,
    )


# ---------------------------------------------------------------------------
# GPSU parsing
# ---------------------------------------------------------------------------

_GPSU_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.(\d+)$")


def _parse_gpsu(gpsu: str) -> datetime | None:
    """Parse a GPMF ``GPSU`` string into a UTC :class:`datetime`.

    Format: ``YYMMDDHHMMSS.sss``  e.g. ``"250508104822.180"``
    """
    m = _GPSU_RE.match(gpsu)
    if m is None:
        return None
    yy, mo, dd, hh, mi, ss, frac = m.groups()
    year = 2000 + int(yy)
    microsecond = int(frac.ljust(6, "0")[:6])
    return datetime(
        year,
        int(mo),
        int(dd),
        int(hh),
        int(mi),
        int(ss),
        microsecond,
        tzinfo=timezone.utc,
    )


# ---------------------------------------------------------------------------
# MP4 container metadata
# ---------------------------------------------------------------------------


def _extract_mp4_metadata(mp4_path: Path) -> dict[str, Any]:
    """Extract container-level and per-stream metadata via ``ffprobe``.

    Returns a dict with keys: ``creation_time``, ``firmware``, ``location``,
    ``timecode``, ``duration_s``, ``video`` (sub-dict), ``audio`` (sub-dict).
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    probe = json.loads(result.stdout)
    fmt_tags = probe.get("format", {}).get("tags", {})
    duration = probe.get("format", {}).get("duration")

    # Find timecode and video/audio stream info
    timecode: str | None = None
    video_info: dict[str, Any] = {}
    audio_info: dict[str, Any] = {}

    for s in probe.get("streams", []):
        tc = s.get("tags", {}).get("timecode")
        if tc and timecode is None:
            timecode = tc

        if s.get("codec_type") == "video" and not video_info:
            # Parse frame rate from r_frame_rate fraction (e.g. "60000/1001")
            fr_str = s.get("r_frame_rate", "")
            frame_rate: float | None = None
            if "/" in fr_str:
                num, den = fr_str.split("/", 1)
                try:
                    frame_rate = round(int(num) / int(den), 4)
                except (ValueError, ZeroDivisionError):
                    pass

            bit_rate_raw = s.get("bit_rate")
            bit_rate_mbps: float | None = None
            if bit_rate_raw is not None:
                try:
                    bit_rate_mbps = round(int(bit_rate_raw) / 1_000_000, 2)
                except ValueError:
                    pass

            video_info = {
                "resolution": f"{s.get('width')}x{s.get('height')}"
                if s.get("width")
                else None,
                "frame_rate_hz": frame_rate,
                "codec": s.get("codec_name"),
                "profile": s.get("profile"),
                "bit_rate_mbps": bit_rate_mbps,
                "color_space": s.get("color_space"),
                "color_range": s.get("color_range"),
            }

        if s.get("codec_type") == "audio" and not audio_info:
            a_bit_rate_raw = s.get("bit_rate")
            a_bit_rate_kbps: float | None = None
            if a_bit_rate_raw is not None:
                try:
                    a_bit_rate_kbps = round(int(a_bit_rate_raw) / 1_000, 1)
                except ValueError:
                    pass

            audio_info = {
                "codec": s.get("codec_name"),
                "sample_rate_hz": int(s["sample_rate"])
                if s.get("sample_rate")
                else None,
                "channels": s.get("channels"),
                "bit_rate_kbps": a_bit_rate_kbps,
            }

    return {
        "creation_time": fmt_tags.get("creation_time"),
        "firmware": fmt_tags.get("firmware"),
        "location": fmt_tags.get("location"),
        "timecode": timecode,
        "duration_s": float(duration) if duration else None,
        "video": video_info or None,
        "audio": audio_info or None,
    }


# ---------------------------------------------------------------------------
# GoPro settings from moov > udta > GPMF atom
# ---------------------------------------------------------------------------

# Map of GPMF FourCC → (output_key, value_type)
# value_type: "str", "bool_yn", "int", "float"
_GOPRO_SETTINGS_MAP: dict[str, tuple[str, str]] = {
    "CASN": ("camera_serial", "str"),
    "LINF": ("lens_serial", "str"),
    "VERS": ("metadata_version", "version"),
    "PRTN": ("protune", "bool_yn"),
    "PTWB": ("white_balance", "str"),
    "PTSH": ("sharpness", "str"),
    "PTCL": ("color_mode", "str"),
    "EXPT": ("exposure_type", "str"),
    "PIMX": ("auto_iso_max", "int"),
    "PIMN": ("auto_iso_min", "int"),
    "PTEV": ("ev_compensation", "str"),
    "VFOV": ("field_of_view", "str"),
    "DZOM": ("digital_zoom_on", "bool_yn"),
    "DZST": ("digital_zoom", "int"),
    "SMTR": ("spot_meter", "bool_yn"),
    "EISE": ("electronic_stabilization_on", "bool_yn"),
    "EISA": ("electronic_stabilization", "str"),
    "PRJT": ("lens_projection", "fourcc"),
    "OREN": ("auto_rotation", "str"),
    "AUDO": ("audio_setting", "str"),
    "AUPT": ("auto_protune", "bool_yn"),
    "RATE": ("highlight_tag_rate", "str"),
    "SROT": ("sensor_readout_time_ms", "float"),
    "ZFOV": ("diagonal_fov_deg", "float"),
    "CMOD": ("camera_mode", "int"),
    "MTYP": ("media_type", "int"),
    "VLTE": ("vlte", "bool_yn"),
    "HLVL": ("hilight_level", "bool_yn"),
    "BROD": ("broadcast", "str"),
}


def _find_mp4_atom(f, parent_end: int, target: bytes) -> tuple[int, int] | None:
    """Scan sibling atoms within *parent_end* for *target* FourCC.

    Returns ``(data_offset, data_size)`` or ``None`` if not found.
    """
    while f.tell() + 8 <= parent_end:
        pos = f.tell()
        header = f.read(8)
        if len(header) < 8:
            break
        size = struct.unpack(">I", header[:4])[0]
        fourcc = header[4:8]
        if size == 0:
            break
        if size == 1:  # 64-bit extended size
            ext = f.read(8)
            if len(ext) < 8:
                break
            size = struct.unpack(">Q", ext)[0]
            data_start = pos + 16
        else:
            data_start = pos + 8
        atom_end = pos + size
        if fourcc == target:
            return data_start, atom_end - data_start
        f.seek(atom_end)
    return None


def _extract_gopro_settings(mp4_path: Path) -> dict[str, Any]:
    """Read GoPro camera settings from the ``moov > udta > GPMF`` atom.

    The GoPro firmware writes a GPMF "Global Settings" payload into the MP4
    ``udta`` box.  This contains camera serial, lens info, Protune settings,
    stabilisation mode, FOV, digital zoom, lens projection, etc.

    Returns a flat dict with human-readable keys (see ``_GOPRO_SETTINGS_MAP``).
    Returns an empty dict if the atom is not found or cannot be parsed.
    """
    try:
        with open(mp4_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)

            # Find moov atom
            moov = _find_mp4_atom(f, file_size, b"moov")
            if moov is None:
                logger.debug("No moov atom found in %s", mp4_path)
                return {}
            moov_offset, moov_size = moov

            # Find udta inside moov
            f.seek(moov_offset)
            udta = _find_mp4_atom(f, moov_offset + moov_size, b"udta")
            if udta is None:
                logger.debug("No udta atom found in %s", mp4_path)
                return {}
            udta_offset, udta_size = udta

            # Find GPMF inside udta
            f.seek(udta_offset)
            gpmf = _find_mp4_atom(f, udta_offset + udta_size, b"GPMF")
            if gpmf is None:
                logger.debug("No GPMF atom in udta for %s", mp4_path)
                return {}
            gpmf_offset, gpmf_size = gpmf

            f.seek(gpmf_offset)
            gpmf_data = f.read(gpmf_size)
    except OSError as exc:
        logger.warning("Failed to read GoPro settings from %s: %s", mp4_path, exc)
        return {}

    # Parse the GPMF payload
    tree = _parse_klv(gpmf_data)

    # Walk DEVC > children looking for the "Global Settings" device
    settings: dict[str, Any] = {}
    for key, info, children in tree:
        if info is not None:
            continue  # only process containers (DEVC)
        assert isinstance(children, list)

        for ckey, cinfo, cdata in children:
            if cinfo is None:
                continue
            tc, ss, rp = cinfo

            if ckey in _GOPRO_SETTINGS_MAP:
                out_key, val_type = _GOPRO_SETTINGS_MAP[ckey]
                try:
                    if val_type == "str":
                        settings[out_key] = _decode_string(cdata, ss, rp).strip()
                    elif val_type == "bool_yn":
                        s = _decode_string(cdata, ss, rp).strip()
                        settings[out_key] = s.upper() == "Y"
                    elif val_type == "int":
                        vals = _unpack_values(tc, ss, rp, cdata)
                        if vals is not None and vals.size > 0:
                            settings[out_key] = int(vals.flat[0])
                    elif val_type == "float":
                        vals = _unpack_values(tc, ss, rp, cdata)
                        if vals is not None and vals.size > 0:
                            settings[out_key] = round(float(vals.flat[0]), 6)
                    elif val_type == "fourcc":
                        # FourCC stored as 4-byte type 'F'
                        settings[out_key] = (
                            cdata[:4].decode("ascii", errors="replace").strip("\x00")
                        )
                    elif val_type == "version":
                        # VERS is stored as B (unsigned bytes), e.g. [8, 1, 2]
                        vals = _unpack_values(tc, ss, rp, cdata)
                        if vals is not None:
                            parts = [str(int(v)) for v in vals.flat]
                            settings[out_key] = ".".join(parts)
                except Exception as exc:
                    logger.debug("Failed to decode GPMF setting %s: %s", ckey, exc)

    if settings:
        logger.debug("Extracted %d GoPro settings from udta GPMF atom", len(settings))
    return settings


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# ---------------------------------------------------------------------------


def _find_gpmf_stream_index(mp4_path: Path) -> int:
    """Return the stream index of the ``gpmd`` track (GoPro Metadata)."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    for s in info.get("streams", []):
        if s.get("codec_tag_string") == "gpmd":
            return int(s["index"])
    raise RuntimeError(f"No gpmd stream found in {mp4_path}")


def _extract_gpmf_packets(mp4_path: Path) -> list[ParsedPayload]:
    """Extract all GPMF payloads from *mp4_path* with per-packet timing.

    Uses ``ffprobe`` for packet timing and ``ffmpeg`` for the raw binary data,
    then splits the data by packet sizes reported by ffprobe.
    """
    stream_idx = _find_gpmf_stream_index(mp4_path)
    logger.debug("Found gpmd metadata on stream index %d", stream_idx)

    # 1. Get packet timing via ffprobe
    logger.debug("Running ffprobe for packet timing …")
    t0 = time.monotonic()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_packets",
            "-select_streams",
            str(stream_idx),
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    packets_meta = json.loads(result.stdout)["packets"]
    logger.debug(
        "ffprobe returned %d packet descriptors (%.2f s)",
        len(packets_meta),
        time.monotonic() - t0,
    )

    # 2. Extract raw binary GPMF stream
    logger.debug("Running ffmpeg to extract raw GPMF binary …")
    t0 = time.monotonic()
    result2 = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "quiet",
            "-i",
            str(mp4_path),
            "-map",
            f"0:{stream_idx}",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    raw = result2.stdout
    logger.debug(
        "ffmpeg extracted %.1f KiB of GPMF data (%.2f s)",
        len(raw) / 1024,
        time.monotonic() - t0,
    )

    total_expected = sum(int(p["size"]) for p in packets_meta)
    if total_expected != len(raw):
        logger.warning(
            "Raw GPMF size mismatch: expected %d, got %d bytes",
            total_expected,
            len(raw),
        )

    # 3. Split raw data by packet sizes and parse each payload
    logger.debug("Parsing %d GPMF payloads …", len(packets_meta))
    t0 = time.monotonic()
    payloads: list[ParsedPayload] = []
    offset = 0
    for pmeta in packets_meta:
        psize = int(pmeta["size"])
        pts_time = float(pmeta["pts_time"])
        duration = float(pmeta["duration_time"])
        chunk = raw[offset : offset + psize]
        offset += psize
        payloads.append(_parse_single_payload(chunk, pts_time, duration))
    parse_elapsed = time.monotonic() - t0

    # --- Collect per-stream sample counts across all payloads ---
    stream_counts: dict[str, int] = {}
    for p in payloads:
        for fourcc, ss in p.streams.items():
            stream_counts[fourcc] = (
                stream_counts.get(fourcc, 0) + ss.raw_values.shape[0]
            )
    total_duration = payloads[-1].pts_time + payloads[-1].duration if payloads else 0.0

    logger.info(
        "Parsed %d GPMF payloads (%.1f s, device: %s) in %.2f s",
        len(payloads),
        total_duration,
        payloads[0].device_name if payloads else "?",
        parse_elapsed,
    )
    for fourcc in sorted(stream_counts):
        n = stream_counts[fourcc]
        rate = n / total_duration if total_duration > 0 else 0
        logger.info("  %-5s  %7d samples  (~%.0f Hz)", fourcc, n, rate)

    return payloads


# ---------------------------------------------------------------------------
# Axis remapping from ORIN
# ---------------------------------------------------------------------------


def _remap_axes(values: np.ndarray, orin: str | None) -> np.ndarray:
    """Remap raw IMU columns from the order given by *orin* to canonical XYZ.

    ``orin`` is a 3-character string like ``"ZXY"`` where each character
    indicates which logical axis the corresponding raw column represents.
    Upper-case → positive, lower-case → negative.

    Returns an ``(N, 3)`` array in ``(X, Y, Z)`` order.
    """
    if orin is None or len(orin) < 3 or values.shape[1] != 3:
        return values

    target = "XYZ"
    out = np.empty_like(values)
    for dst_idx, axis_char in enumerate(target):
        # Find which raw column maps to this axis
        for src_idx, o in enumerate(orin):
            if o.upper() == axis_char:
                sign = -1.0 if o.islower() else 1.0
                out[:, dst_idx] = values[:, src_idx] * sign
                break
        else:
            # Axis not found in ORIN — keep as-is
            out[:, dst_idx] = values[:, dst_idx]
    return out


# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------


def _make_timestamps(stream: StreamSamples, payload_duration: float) -> np.ndarray:
    """Generate per-sample timestamps (seconds since recording start).

    Uses STMP (µs) for the first-sample time and distributes *N* samples
    evenly across the payload duration.
    """
    n = stream.raw_values.shape[0]
    start_s = stream.stmp_us / 1_000_000.0
    if n == 1:
        return np.array([start_s])
    # Samples span from start_s to start_s + (n-1)/n * payload_duration
    step = payload_duration / n
    return start_s + np.arange(n) * step


# ---------------------------------------------------------------------------
# Apply SCAL — raw values → physical units
# ---------------------------------------------------------------------------


def _apply_scal(values: np.ndarray, scal: np.ndarray) -> np.ndarray:
    """Divide *values* by *scal*, broadcasting along the component axis."""
    if scal.shape[0] == 1:
        return values / scal[0]
    # scal has one entry per component column
    return values / scal[np.newaxis, :]


# ---------------------------------------------------------------------------
# Build GPS-rate trajectory DataFrame  (§2.3)
# ---------------------------------------------------------------------------


def _build_gps_rate_df(payloads: Sequence[ParsedPayload]) -> pd.DataFrame:
    """Build the ~18 Hz GPS-rate trajectory DataFrame.

    Slow-rate streams (GPSF, GPSP) are forward-filled to the GPS sample
    rate.  ACCL is downsampled to GPS rate via nearest-timestamp.
    """
    logger.debug("Building GPS-rate trajectory DataFrame …")
    rows_ts: list[np.ndarray] = []
    rows_gps: list[np.ndarray] = []
    rows_fix: list[float] = []
    rows_dop: list[float] = []

    # Collect per-payload ACCL data for nearest-sample lookup
    accl_ts_all: list[np.ndarray] = []
    accl_vals_all: list[np.ndarray] = []

    for p in payloads:
        gps = p.streams.get("GPS5")
        if gps is None:
            continue

        ts = _make_timestamps(gps, p.duration)
        scaled = _apply_scal(gps.raw_values, gps.scal)
        # GPS5 columns after SCAL: lat, lon, alt, speed2d, speed3d
        n = ts.shape[0]

        rows_ts.append(ts)
        rows_gps.append(scaled)

        gpsf_val = float(gps.gpsf) if gps.gpsf is not None else np.nan
        gpsp_val = float(gps.gpsp) / 100.0 if gps.gpsp is not None else np.nan
        rows_fix.extend([gpsf_val] * n)
        rows_dop.extend([gpsp_val] * n)

        # Collect ACCL for this payload
        accl = p.streams.get("ACCL")
        if accl is not None:
            accl_ts = _make_timestamps(accl, p.duration)
            accl_scaled = _apply_scal(accl.raw_values, accl.scal)
            accl_remapped = _remap_axes(accl_scaled, accl.orin)
            accl_ts_all.append(accl_ts)
            accl_vals_all.append(accl_remapped)

    if not rows_ts:
        return pd.DataFrame()

    all_ts = np.concatenate(rows_ts)
    all_gps = np.concatenate(rows_gps, axis=0)

    logger.debug(
        "GPS5: %d samples over %.1f s (%.1f Hz)",
        len(all_ts),
        all_ts[-1] - all_ts[0] if len(all_ts) > 1 else 0,
        len(all_ts) / (all_ts[-1] - all_ts[0])
        if len(all_ts) > 1 and all_ts[-1] > all_ts[0]
        else 0,
    )

    df = pd.DataFrame(
        {
            "timestamp_s": all_ts,
            "gps_lat_deg": all_gps[:, 0],
            "gps_lon_deg": all_gps[:, 1],
            "gps_alt_m": all_gps[:, 2],
            "gps_speed2d_ms": all_gps[:, 3],
            "gps_speed3d_ms": all_gps[:, 4],
            "gps_fix": rows_fix,
            "gps_dop": rows_dop,
        }
    )

    # Nearest-sample ACCL lookup
    if accl_ts_all:
        accl_ts_cat = np.concatenate(accl_ts_all)
        accl_vals_cat = np.concatenate(accl_vals_all, axis=0)
        # For each GPS timestamp, find nearest ACCL sample index
        idxs = np.searchsorted(accl_ts_cat, all_ts, side="left")
        idxs = np.clip(idxs, 0, len(accl_ts_cat) - 1)
        # Also check the previous index for closer match
        idxs_prev = np.clip(idxs - 1, 0, len(accl_ts_cat) - 1)
        d_right = np.abs(accl_ts_cat[idxs] - all_ts)
        d_left = np.abs(accl_ts_cat[idxs_prev] - all_ts)
        use_prev = d_left < d_right
        nearest = np.where(use_prev, idxs_prev, idxs)
        df["accl_x_ms2"] = accl_vals_cat[nearest, 0]
        df["accl_y_ms2"] = accl_vals_cat[nearest, 1]
        df["accl_z_ms2"] = accl_vals_cat[nearest, 2]
    else:
        df["accl_x_ms2"] = np.nan
        df["accl_y_ms2"] = np.nan
        df["accl_z_ms2"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Build high-frequency IMU / orientation DataFrame  (§2.4)
# ---------------------------------------------------------------------------


def _interp_to_target(
    target_ts: np.ndarray,
    src_ts: np.ndarray,
    src_vals: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate ``src_vals`` (N, C) at ``target_ts`` timestamps."""
    ncols = src_vals.shape[1]
    out = np.empty((target_ts.shape[0], ncols), dtype=np.float64)
    for c in range(ncols):
        out[:, c] = np.interp(target_ts, src_ts, src_vals[:, c])
    return out


def _build_hf_imu_df(payloads: Sequence[ParsedPayload]) -> pd.DataFrame:
    """Build the ~200 Hz high-frequency IMU / orientation DataFrame.

    ACCL / GYRO define the time base.  CORI, IORI, GRAV (≈60 Hz) are
    linearly interpolated to the IMU rate.
    """
    logger.debug("Building high-frequency IMU/orientation DataFrame …")
    # Collect all streams across payloads
    accl_ts_list: list[np.ndarray] = []
    accl_vals_list: list[np.ndarray] = []
    gyro_ts_list: list[np.ndarray] = []
    gyro_vals_list: list[np.ndarray] = []
    cori_ts_list: list[np.ndarray] = []
    cori_vals_list: list[np.ndarray] = []
    iori_ts_list: list[np.ndarray] = []
    iori_vals_list: list[np.ndarray] = []
    grav_ts_list: list[np.ndarray] = []
    grav_vals_list: list[np.ndarray] = []

    for p in payloads:
        accl = p.streams.get("ACCL")
        if accl is not None:
            ts = _make_timestamps(accl, p.duration)
            scaled = _apply_scal(accl.raw_values, accl.scal)
            remapped = _remap_axes(scaled, accl.orin)
            accl_ts_list.append(ts)
            accl_vals_list.append(remapped)

        gyro = p.streams.get("GYRO")
        if gyro is not None:
            ts = _make_timestamps(gyro, p.duration)
            scaled = _apply_scal(gyro.raw_values, gyro.scal)
            remapped = _remap_axes(scaled, gyro.orin)
            gyro_ts_list.append(ts)
            gyro_vals_list.append(remapped)

        cori = p.streams.get("CORI")
        if cori is not None:
            ts = _make_timestamps(cori, p.duration)
            scaled = _apply_scal(cori.raw_values, cori.scal)
            cori_ts_list.append(ts)
            cori_vals_list.append(scaled)

        iori = p.streams.get("IORI")
        if iori is not None:
            ts = _make_timestamps(iori, p.duration)
            scaled = _apply_scal(iori.raw_values, iori.scal)
            iori_ts_list.append(ts)
            iori_vals_list.append(scaled)

        grav = p.streams.get("GRAV")
        if grav is not None:
            ts = _make_timestamps(grav, p.duration)
            scaled = _apply_scal(grav.raw_values, grav.scal)
            grav_ts_list.append(ts)
            grav_vals_list.append(scaled)

    if not accl_ts_list:
        return pd.DataFrame()

    accl_ts = np.concatenate(accl_ts_list)
    accl_vals = np.concatenate(accl_vals_list, axis=0)
    hf_span = accl_ts[-1] - accl_ts[0] if len(accl_ts) > 1 else 0.0
    hf_rate = len(accl_ts) / hf_span if hf_span > 0 else 0.0

    logger.debug(
        "ACCL: %d samples over %.1f s (%.1f Hz)",
        len(accl_ts),
        hf_span,
        hf_rate,
    )

    # GYRO — same rate as ACCL; align by nearest if counts differ
    if gyro_ts_list:
        gyro_ts = np.concatenate(gyro_ts_list)
        gyro_vals = np.concatenate(gyro_vals_list, axis=0)
        if gyro_ts.shape[0] == accl_ts.shape[0]:
            gyro_aligned = gyro_vals
            logger.debug("GYRO: %d samples — aligned 1:1 with ACCL", len(gyro_ts))
        else:
            gyro_aligned = _interp_to_target(accl_ts, gyro_ts, gyro_vals)
            logger.debug(
                "GYRO: %d samples — interpolated to %d ACCL timestamps",
                len(gyro_ts),
                len(accl_ts),
            )
    else:
        gyro_aligned = np.full((accl_ts.shape[0], 3), np.nan)
        logger.debug("GYRO: no data — filled with NaN")

    # CORI (4 components — w, x, y, z)
    if cori_ts_list:
        cori_ts = np.concatenate(cori_ts_list)
        cori_vals = np.concatenate(cori_vals_list, axis=0)
        cori_aligned = _interp_to_target(accl_ts, cori_ts, cori_vals)
        logger.debug(
            "CORI: %d samples (~%.0f Hz) — interpolated to IMU rate",
            len(cori_ts),
            len(cori_ts) / hf_span if hf_span > 0 else 0,
        )
    else:
        cori_aligned = np.full((accl_ts.shape[0], 4), np.nan)
        logger.debug("CORI: no data — filled with NaN")

    # IORI (4 components)
    if iori_ts_list:
        iori_ts = np.concatenate(iori_ts_list)
        iori_vals = np.concatenate(iori_vals_list, axis=0)
        iori_aligned = _interp_to_target(accl_ts, iori_ts, iori_vals)
        logger.debug(
            "IORI: %d samples (~%.0f Hz) — interpolated to IMU rate",
            len(iori_ts),
            len(iori_ts) / hf_span if hf_span > 0 else 0,
        )
    else:
        iori_aligned = np.full((accl_ts.shape[0], 4), np.nan)
        logger.debug("IORI: no data — filled with NaN")

    # GRAV (3 components)
    if grav_ts_list:
        grav_ts = np.concatenate(grav_ts_list)
        grav_vals = np.concatenate(grav_vals_list, axis=0)
        grav_aligned = _interp_to_target(accl_ts, grav_ts, grav_vals)
        logger.debug(
            "GRAV: %d samples (~%.0f Hz) — interpolated to IMU rate",
            len(grav_ts),
            len(grav_ts) / hf_span if hf_span > 0 else 0,
        )
    else:
        grav_aligned = np.full((accl_ts.shape[0], 3), np.nan)
        logger.debug("GRAV: no data — filled with NaN")

    df = pd.DataFrame(
        {
            "timestamp_s": accl_ts,
            "accl_x_ms2": accl_vals[:, 0],
            "accl_y_ms2": accl_vals[:, 1],
            "accl_z_ms2": accl_vals[:, 2],
            "gyro_x_rads": gyro_aligned[:, 0],
            "gyro_y_rads": gyro_aligned[:, 1],
            "gyro_z_rads": gyro_aligned[:, 2],
            "cori_w": cori_aligned[:, 0],
            "cori_x": cori_aligned[:, 1],
            "cori_y": cori_aligned[:, 2],
            "cori_z": cori_aligned[:, 3],
            "iori_w": iori_aligned[:, 0],
            "iori_x": iori_aligned[:, 1],
            "iori_y": iori_aligned[:, 2],
            "iori_z": iori_aligned[:, 3],
            "grav_x": grav_aligned[:, 0],
            "grav_y": grav_aligned[:, 1],
            "grav_z": grav_aligned[:, 2],
        }
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_mp4_telemetry(flight_data: FlightData) -> FlightData:
    """Extract consolidated telemetry CSVs from a GoPro MP4 file.

    Parameters
    ----------
    mp4_path:
        Path to the source ``.MP4`` file.
    flight_tag:
        Flight tag string (e.g. ``"07_niskie_ladowanie"``), used to
        construct output filenames.

    Returns
    -------
    FlightData
        A copy of *flight_data* with ``mp4_telemetry.trajectory``,
        ``mp4_telemetry.metadata``, ``mp4_telemetry.hf_imu``, and
        ``mp4_telemetry.origin`` populated.  ``origin`` is set to the
        WGS 84 coordinates (lat, lon, GPS altitude) of the first
        recorded GPS5 sample.
    """
    if flight_data.mp4_telemetry is None:
        raise ValueError(
            "FlightData object does not have a valid mp4_telemetry field. Use find_data_sources first."
        )

    if flight_data.mp4_telemetry.mp4_path is None:
        raise ValueError(
            "FlightData object does not have a valid mp4_telemetry.mp4_path field. Use find_data_sources first."
        )

    mp4_path = flight_data.mp4_telemetry.mp4_path

    if flight_data.flight_tag is None:
        raise ValueError(
            "FlightData object does not have a valid flight_tag field. Use find_data_sources first."
        )

    flight_tag = flight_data.flight_tag

    mp4_size_mb = mp4_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Extracting GPMF telemetry from %s (%.1f MiB)",
        mp4_path.name,
        mp4_size_mb,
    )
    t_total = time.monotonic()

    payloads = _extract_gpmf_packets(mp4_path)
    if not payloads:
        logger.warning("No GPMF payloads found in %s", mp4_path)
        return flight_data.model_copy()

    output_dir: Path = config.DIR.TELEMETRY
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_telemetry: MP4Telemetry = flight_data.mp4_telemetry.model_copy()

    # --- Flight metadata JSON ---
    mp4_meta = _extract_mp4_metadata(mp4_path)

    # Find first GPSU across payloads
    first_gpsu_str: str | None = None
    for p in payloads:
        if p.gpsu is not None:
            first_gpsu_str = p.gpsu
            break
    gps_start_utc: datetime | None = None
    if first_gpsu_str is not None:
        gps_start_utc = _parse_gpsu(first_gpsu_str)

    # --- GoPro camera settings from udta GPMF atom ---
    gopro_settings = _extract_gopro_settings(mp4_path)

    # Top-level identifiers from settings (promoted out of settings sub-dict)
    camera_serial = gopro_settings.pop("camera_serial", None)
    lens_serial = gopro_settings.pop("lens_serial", None)
    metadata_version = gopro_settings.pop("metadata_version", None)

    # Prefer firmware from GPMF Global Settings; fall back to ffprobe tag
    firmware = gopro_settings.pop("firmware", None) or mp4_meta.get("firmware")

    meta: dict[str, Any] = {
        "device_name": payloads[0].device_name if payloads else None,
        "gps_start_utc": gps_start_utc.isoformat() if gps_start_utc else None,
        "creation_time": mp4_meta.get("creation_time"),
        "firmware": firmware,
        "timecode": mp4_meta.get("timecode"),
        "location": mp4_meta.get("location"),
        "duration_s": mp4_meta.get("duration_s"),
        "camera_serial": camera_serial,
        "lens_serial": lens_serial,
        "metadata_version": metadata_version,
        "video": mp4_meta.get("video"),
        "audio": mp4_meta.get("audio"),
        "settings": gopro_settings if gopro_settings else None,
    }

    meta_json = output_dir / f"{flight_tag}_mp4_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("Wrote metadata:     %s", meta_json.name)
    if gps_start_utc is not None:
        logger.info("  GPS start (UTC):  %s", gps_start_utc.isoformat())
    if gopro_settings:
        eis = gopro_settings.get("electronic_stabilization", "off")
        eis_on = gopro_settings.get("electronic_stabilization_on", False)
        proj = gopro_settings.get("lens_projection", "?")
        fov = gopro_settings.get("field_of_view", "?")
        logger.info(
            "  Settings:  EIS=%s (%s)  |  lens=%s  |  FOV=%s",
            eis,
            "on" if eis_on else "off",
            proj,
            fov,
        )

    mp4_telemetry.metadata = meta_json
    extracted_files = 1

    # --- GPS-rate trajectory CSV ---
    gps_df = _build_gps_rate_df(payloads)
    if not gps_df.empty:
        gps_csv = output_dir / f"{flight_tag}_mp4.csv"
        gps_df.to_csv(gps_csv, index=False)
        csv_size_kb = gps_csv.stat().st_size / 1024
        ts = gps_df["timestamp_s"]
        span = ts.iloc[-1] - ts.iloc[0]
        logger.info(
            "Wrote GPS-rate CSV: %s  (%d rows, %.1f s, %.0f KiB)",
            gps_csv.name,
            len(gps_df),
            span,
            csv_size_kb,
        )
        # GPS quality summary
        fix_counts = gps_df["gps_fix"].value_counts().to_dict()
        logger.info(
            "  GPS fix: %s  |  DOP: %.2f – %.2f  |  lat: %.4f – %.4f  |  lon: %.4f – %.4f",
            ", ".join(f"{int(k)}={int(v)}" for k, v in sorted(fix_counts.items())),
            gps_df["gps_dop"].min(),
            gps_df["gps_dop"].max(),
            gps_df["gps_lat_deg"].min(),
            gps_df["gps_lat_deg"].max(),
            gps_df["gps_lon_deg"].min(),
            gps_df["gps_lon_deg"].max(),
        )
        logger.info(
            "  alt: %.0f – %.0f m  |  speed2d: %.1f – %.1f m/s",
            gps_df["gps_alt_m"].min(),
            gps_df["gps_alt_m"].max(),
            gps_df["gps_speed2d_ms"].min(),
            gps_df["gps_speed2d_ms"].max(),
        )
        mp4_telemetry.trajectory = gps_csv
        extracted_files += 1

        # Origin — WGS 84 coordinates of the first recorded GPS fix
        first_row = gps_df.iloc[0]
        mp4_telemetry.origin = WGS84Coordinate(
            lat=float(first_row["gps_lat_deg"]),
            lon=float(first_row["gps_lon_deg"]),
            alt=float(first_row["gps_alt_m"]),
        )
        logger.info(
            "  Origin (WGS 84):  lat=%.6f  lon=%.6f  alt=%.0f m",
            mp4_telemetry.origin.lat,
            mp4_telemetry.origin.lon,
            mp4_telemetry.origin.alt,
        )
    else:
        logger.warning("No GPS5 data found — skipping GPS-rate CSV")

    # --- High-frequency IMU / orientation CSV ---
    hf_df = _build_hf_imu_df(payloads)
    if not hf_df.empty:
        hf_csv = output_dir / f"{flight_tag}_mp4_hf.csv"
        hf_df.to_csv(hf_csv, index=False)
        csv_size_mb = hf_csv.stat().st_size / (1024 * 1024)
        ts = hf_df["timestamp_s"]
        span = ts.iloc[-1] - ts.iloc[0]
        logger.info(
            "Wrote HF IMU CSV:   %s  (%d rows, %.1f s, %.1f MiB)",
            hf_csv.name,
            len(hf_df),
            span,
            csv_size_mb,
        )
        mp4_telemetry.hf_imu = hf_csv
        extracted_files += 1
    else:
        logger.warning("No ACCL data found — skipping HF IMU CSV")

    elapsed = time.monotonic() - t_total
    logger.info("Extraction complete — %d files in %.1f s", extracted_files, elapsed)

    return flight_data.model_copy(
        update={
            "mp4_telemetry": mp4_telemetry,
        }
    )
