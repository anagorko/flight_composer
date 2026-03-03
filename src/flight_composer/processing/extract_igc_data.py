"""
IGC telemetry stream extraction — parse an IGC file into an ``IGCTelemetry`` object.

Signature: ``extract_igc_data(igc_path: Path) -> IGCTelemetry``

Takes the path to an ``.igc`` file and returns a fully populated
``IGCTelemetry`` object.

The flight tag is derived from ``igc_path.stem`` (e.g.
``20250508_igc.igc`` → ``"20250508_igc"``).  The trajectory DataFrame is
saved to ``config.DIR.TELEMETRY / f"{flight_tag}.parquet"``, and that path
is stored in ``data_path``.

#### 3.1 IGC format overview

The IGC (International Gliding Commission) file format is a plain-text standard for GNSS flight recorders. Each line starts with a single uppercase letter identifying the record type. The records relevant to extraction are:

| Record | Description | Cardinality |
|--------|-------------|-------------|
| **A** | Flight recorder manufacturer and serial | single |
| **H** | Header — date, pilot, glider, hardware, etc. | single (multi-line) |
| **I** | Defines extension fields appended to each B record | single |
| **J** | Defines extension fields in K records | single |
| **B** | Fix — UTC time, lat, lon, pressure alt, GNSS alt, validity, extensions | multiple (~1 Hz) |
| **K** | Less-frequent extension data (e.g. wind from SeeYou) | multiple |
| **E** | Events (altimeter setting changes, pilot events) | multiple |
| **L** | Comments / logbook (includes SeeYou phase annotations) | multiple |
| **G** | Security signature | single (last) |

Our IGC files are produced by **SeeYou Navigator** running on a smartphone. Key characteristics:

- **No pressure altitude sensor** — `HFPRSPRESSALTSENSOR:NIL`. The pressure altitude field in B records is always `00000`.
- **GPS altitude reference** — `HFALGALTGPS:ELL` means altitude is above the WGS 84 ellipsoid (not the geoid).
- **Fix rate** — 1 Hz (one B record per second).
- **K records** — contain wind direction (`WDI`) and wind speed (`WSP`) from SeeYou's internal estimation. Currently observed as sentinel values (`999` for speed); not extracted.
- **L records** — contain SeeYou phase annotations (`onGround`, `takingOff`, `flying`, `landing`, etc.) and post-flight statistics. Not extracted in this step.

#### 3.2 Parsing

The parser is implemented without external libraries (the IGC text format is simple enough to parse directly, consistent with the GPMF parser in step 2). Parsing proceeds in a single pass over the file lines:

1. **H records** → extract metadata key-value pairs. The H record subtypes are identified by their three-letter code (TLC) after the source byte (`F`, `O`, or `P`). We handle both the legacy format (`HFDTEDDMMYY`) and the newer format with a long name and colon (`HFDTEDATE:DDMMYY`).
2. **I record** → parse extension definitions to know byte positions of any additional fields in B records. Store as a list of `(start_byte, end_byte, TLC)` tuples (1-based byte positions).
3. **B records** → parse each fix into structured data. The B record format (first 35 bytes) is fixed:

   ```
   B HHMMSS DDMMmmmN/S DDDMMmmmE/W V PPPPP GGGGG [extensions...]
   ```

   - Time: `HHMMSS` (UTC). Combined with the date from `HFDTE` to form an absolute timestamp.
   - Latitude: `DDMMmmmN/S` → decimal degrees (positive = North).
   - Longitude: `DDDMMmmmE/W` → decimal degrees (positive = East).
   - Fix validity: `A` (3D fix, valid) or `V` (2D / no GPS).
   - Pressure altitude: `PPPPP` (meters above ISA 1013.25 hPa sea-level datum). Can be negative (prefix `-`).
   - GNSS altitude: `GGGGG` (meters above WGS 84 ellipsoid or geoid, per `HFALG`).

4. **Midnight rollover** — if B-record times decrease (e.g. `235959` → `000001`), the date is advanced by one day.

#### 3.3 Output: 1 Hz trajectory DataFrame (Parquet)

The parsed B-record fixes are assembled into a pandas DataFrame and saved as a
Parquet file at ``config.DIR.TELEMETRY / f"{flight_tag}.parquet"``. The path is
stored in ``igc_data.data_path``.

One row per B record (~1 Hz). Column naming follows the MP4 conventions where applicable so that downstream steps can consume both formats uniformly.

| Column | Source | Units | Description |
|--------|--------|-------|-------------|
| `timestamp_s` | B-record time | s | Seconds since the first B-record fix. Float. |
| `gps_lat_deg` | B-record lat | deg | Latitude WGS 84, decimal degrees (positive N) |
| `gps_lon_deg` | B-record lon | deg | Longitude WGS 84, decimal degrees (positive E) |
| `alt_baro_m` | B-record PPPPP | m | Pressure altitude (ISA 1013.25 hPa). 0 when sensor not available. |
| `alt_gps_m` | B-record GGGGG | m | GNSS altitude (ellipsoid or geoid per `HFALG`) |
| `fix_validity` | B-record V flag | — | `A` (3D valid) or `V` (2D / no GPS) |

The `timestamp_s` column contains seconds since the first B-record; to obtain absolute UTC times, use ``metadata.gps_start_utc`` from the ``IGCMetadata`` object (§3.5).

> **Note on altitude columns:** The MP4 telemetry has a single `gps_alt_m` column (WGS 84 GPS altitude from the GoPro). The IGC format natively provides two altitude references — barometric and GNSS — so both are preserved. When the pressure sensor is absent (`HFPRSPRESSALTSENSOR:NIL`), `alt_baro_m` will be 0 for every row; downstream steps should check the ``metadata.pressure_sensor`` field before using this column.

#### 3.4 Data not extracted

The following IGC data is **not** extracted in this step:

- **K records** (wind direction/speed) — SeeYou Navigator fills these with sentinel/placeholder values. If future IGC sources provide reliable K-record data, extraction can be added.
- **L records** (SeeYou phase annotations, post-flight statistics) — app-specific; not needed for trajectory reconstruction.
- **E records** (events) — altimeter setting changes and pilot events; not needed for trajectory reconstruction.
- **C records** (task declarations) — competition task waypoints; not relevant to visualization.
- **I-record extensions** beyond the base 35 bytes — e.g. FXA (fix accuracy), SIU (satellites in use), ENL (engine noise). These could be extracted in the future if present and useful; for now the base B-record fields are sufficient.

#### 3.5 Output: IGCMetadata object

Recording-level metadata extracted from H records and derived from B records is
stored directly in the ``igc_data.metadata`` field as an :class:`IGCMetadata`
pydantic model — no external JSON file is written.

| Field | H-record TLC | Example | Description |
|-------|-------------|---------|-------------|
| ``date`` | ``DTE`` | ``date(2025, 5, 8)`` | UTC date of the flight |
| ``gps_start_utc`` | derived | ``datetime(2025, 5, 8, 10, 48, 23)`` | UTC timestamp of the first B-record fix |
| ``duration_s`` | derived | ``392.0`` | Seconds from first to last B-record fix |
| ``num_fixes`` | derived | ``393`` | Total number of B records |
| ``timezone_offset_h`` | ``TZN`` | ``2.0`` | Hours from UTC to local time |
| ``pilot`` | ``PLT`` | ``"Andrzej Nagorko"`` | Pilot in charge |
| ``copilot`` | ``CM2`` | ``None`` | Second crew member (``None`` if ``NIL``) |
| ``glider_type`` | ``GTY`` | ``"SZD 51-1 Junior"`` | Glider model |
| ``glider_id`` | ``GID`` | ``"SP-3303"`` | Glider registration |
| ``competition_id`` | ``CID`` | ``"LK"`` | Competition / fin ID |
| ``competition_class`` | ``CCL`` | ``"NKN"`` | Competition class |
| ``fr_type`` | ``FTY`` | ``"Naviter,SeeYou Navigator"`` | Flight recorder manufacturer and model |
| ``fr_id`` | A record | ``"XNA8C5..."`` | Flight recorder manufacturer code and serial |
| ``firmware`` | ``RFW`` | ``"3.4.0+2662"`` | Firmware / app version |
| ``hardware`` | ``RHW`` | ``"Samsung SM-A356B,Android 14"`` | Hardware description |
| ``gps_receiver`` | ``GPS`` | ``"Internal sensors,Internal,0,0"`` | GPS receiver info |
| ``pressure_sensor`` | ``PRS`` | ``"NIL"`` | Pressure altitude sensor (``"NIL"`` = not available) |
| ``gps_datum`` | ``DTM`` | ``"WGS84"`` | Geodetic datum |
| ``altitude_gps_ref`` | ``ALG`` | ``"ELL"`` | GPS altitude reference: ``ELL`` (ellipsoid) or ``GEO`` (geoid) |
| ``altitude_pressure_ref`` | ``ALP`` | ``"NIL"`` | Pressure altitude reference (or ``"NIL"``) |

#### 3.6 Implementation notes

- **No external dependencies.** The parser uses only the Python standard library and pandas (already a project dependency). The IGC format is a simple line-oriented text protocol — unlike GPMF binary parsing, no struct unpacking or atom traversal is needed.
- **Function signature:** ``extract_igc_data(igc_path: Path) -> IGCTelemetry``. The function takes a path to an ``.igc`` file, parses it, builds an ``IGCMetadata`` object in memory, saves the trajectory DataFrame as Parquet to ``config.DIR.TELEMETRY``, and returns a complete ``IGCTelemetry`` with ``igc_path``, ``metadata``, and ``data_path`` populated. The flight tag used for the output filename is derived from ``igc_path.stem``.
- **Logging** follows the step 2 pattern: log file sizes, row counts, time span, coordinate ranges, and fix validity statistics.

"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from flight_composer.config import config
from flight_composer.flight_data import (
    IGCMetadata,
    IGCTelemetry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# H-record three-letter code → metadata JSON key
# ---------------------------------------------------------------------------
_H_TLC_MAP: dict[str, str] = {
    "PLT": "pilot",
    "CM2": "copilot",
    "GTY": "glider_type",
    "GID": "glider_id",
    "CID": "competition_id",
    "CCL": "competition_class",
    "FTY": "fr_type",
    "RFW": "firmware",
    "RHW": "hardware",
    "GPS": "gps_receiver",
    "PRS": "pressure_sensor",
    "DTM": "gps_datum",
    "ALG": "altitude_gps_ref",
    "ALP": "altitude_pressure_ref",
    "TZN": "timezone_offset_h",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_igc_latitude(raw: str) -> float:
    """Convert IGC latitude ``DDMMmmmN`` (8 chars) to decimal degrees.

    The format is ``DDMMmmmH`` where DD = degrees, MM = whole minutes,
    mmm = fractional minutes (thousandths), and H = hemisphere (N/S).
    """
    dd = int(raw[0:2])
    mm_mmm = int(raw[2:7])  # MMmmm as an integer
    hemisphere = raw[7]
    decimal_deg = dd + mm_mmm / 60_000.0
    if hemisphere == "S":
        decimal_deg = -decimal_deg
    return decimal_deg


def _parse_igc_longitude(raw: str) -> float:
    """Convert IGC longitude ``DDDMMmmmE`` (9 chars) to decimal degrees.

    The format is ``DDDMMmmmH`` where DDD = degrees, MM = whole minutes,
    mmm = fractional minutes (thousandths), and H = hemisphere (E/W).
    """
    ddd = int(raw[0:3])
    mm_mmm = int(raw[3:8])  # MMmmm as an integer
    hemisphere = raw[8]
    decimal_deg = ddd + mm_mmm / 60_000.0
    if hemisphere == "W":
        decimal_deg = -decimal_deg
    return decimal_deg


def _parse_hfdte(value: str) -> date:
    """Parse the DTE value ``DDMMYY`` (with optional ``,NN`` flight
    number suffix).

       The IGC spec uses a two-digit year; we assume 2000+ for YY < 80.
    """
    # Strip optional flight-number suffix (e.g. "080525,01" → "080525")
    ddmmyy = value.split(",")[0].strip()
    dd = int(ddmmyy[0:2])
    mm = int(ddmmyy[2:4])
    yy = int(ddmmyy[4:6])
    year = 2000 + yy if yy < 80 else 1900 + yy
    return date(year, mm, dd)


def _parse_h_record(line: str) -> tuple[str, str]:
    """Return ``(TLC, value)`` from an H record line.

    Handles both legacy (``HFDTEDDMMYY``) and newer
    (``HFDTEDATE:DDMMYY,01``) formats.

    The *source byte* at position 1 (``F``, ``O``, or ``P``) is skipped.
    The TLC occupies positions 2–4 (inclusive).
    """
    # line[0] = 'H', line[1] = source, line[2:5] = TLC
    tlc = line[2:5]
    rest = line[5:]

    # Newer format: LONG_NAME:VALUE
    colon_pos = rest.find(":")
    if colon_pos != -1:
        value = rest[colon_pos + 1 :]
    else:
        # Legacy format: value starts immediately after TLC
        value = rest

    return tlc, value.strip()


def _nil_to_none(value: str) -> str | None:
    """Return *None* if value is ``NIL`` (case-insensitive), else the value."""
    return None if value.upper() == "NIL" else value


def _parse_igc_file(
    igc_path: Path,
) -> tuple[
    str | None, dict[str, str], list[tuple[int, int, str]], list[dict[str, Any]]
]:
    """Single-pass parse of an IGC file.

    Returns
    -------
    fr_id : str | None
        Flight-recorder ID from the A record.
    h_records : dict[str, str]
        TLC → raw value from all H records.
    i_extensions : list[tuple[int, int, str]]
        I-record extension definitions as ``(start_byte, end_byte, TLC)``
        with 1-based byte positions.
    b_fixes : list[dict[str, Any]]
        Parsed B-record fixes, each containing ``time_s``, ``lat``, ``lon``,
        ``alt_baro``, ``alt_gps``, ``validity``, and ``datetime``.
    """
    text = igc_path.read_text(encoding="latin-1")
    lines = text.splitlines()

    fr_id: str | None = None
    h_records: dict[str, str] = {}
    i_extensions: list[tuple[int, int, str]] = []
    b_fixes: list[dict[str, Any]] = []

    flight_date: date | None = None
    prev_time_s: int | None = None  # previous B-record time-of-day in seconds
    day_offset_days: int = 0  # midnight rollover counter

    for line in lines:
        if not line:
            continue
        rec_type = line[0]

        # --- A record: flight recorder ID ---
        if rec_type == "A":
            fr_id = line[1:].strip()

        # --- H record: header metadata ---
        elif rec_type == "H":
            tlc, value = _parse_h_record(line)
            h_records[tlc] = value

            # Parse date eagerly so B-record parsing can use it
            if tlc == "DTE":
                flight_date = _parse_hfdte(value)

        # --- I record: B-record extension definitions ---
        elif rec_type == "I":
            # Format: I NN SS EE TLC SS EE TLC ...
            # NN = number of extensions (2 digits)
            num_ext = int(line[1:3])
            for i in range(num_ext):
                offset = 3 + i * 7
                start = int(line[offset : offset + 2])
                end = int(line[offset + 2 : offset + 4])
                tlc = line[offset + 4 : offset + 7]
                i_extensions.append((start, end, tlc))

        # --- B record: fix ---
        elif rec_type == "B" and len(line) >= 35:
            hh = int(line[1:3])
            mm = int(line[3:5])
            ss = int(line[5:7])
            time_of_day_s = hh * 3600 + mm * 60 + ss

            # Midnight rollover detection
            if prev_time_s is not None and time_of_day_s < prev_time_s - 3600:
                # Large backward jump → crossed midnight
                day_offset_days += 1
            prev_time_s = time_of_day_s

            total_seconds = day_offset_days * 86400 + time_of_day_s

            lat = _parse_igc_latitude(line[7:15])
            lon = _parse_igc_longitude(line[15:24])
            validity = line[24]
            alt_baro = int(line[25:30])
            alt_gps = int(line[30:35])

            # Build absolute datetime if flight_date is known
            fix_dt: datetime | None = None
            if flight_date is not None:
                fix_dt = datetime(
                    flight_date.year,
                    flight_date.month,
                    flight_date.day,
                    tzinfo=timezone.utc,
                ) + timedelta(days=day_offset_days, seconds=time_of_day_s)

            b_fixes.append(
                {
                    "total_seconds": total_seconds,
                    "lat": lat,
                    "lon": lon,
                    "alt_baro": alt_baro,
                    "alt_gps": alt_gps,
                    "validity": validity,
                    "datetime": fix_dt,
                }
            )

    return fr_id, h_records, i_extensions, b_fixes


def _build_trajectory_df(b_fixes: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the 1 Hz trajectory DataFrame from parsed B-record fixes."""
    if not b_fixes:
        return pd.DataFrame()

    first_ts = b_fixes[0]["total_seconds"]

    rows = []
    for fix in b_fixes:
        rows.append(
            {
                "timestamp_s": float(fix["total_seconds"] - first_ts),
                "gps_lat_deg": fix["lat"],
                "gps_lon_deg": fix["lon"],
                "alt_baro_m": fix["alt_baro"],
                "alt_gps_m": fix["alt_gps"],
                "fix_validity": fix["validity"],
            }
        )

    return pd.DataFrame(rows)


def _build_metadata(
    fr_id: str | None,
    h_records: dict[str, str],
    b_fixes: list[dict[str, Any]],
) -> IGCMetadata:
    """Build an :class:`IGCMetadata` from parsed IGC data."""

    # --- Date ---
    dte_raw = h_records.get("DTE")
    flight_date: date | None = None
    if dte_raw is not None:
        flight_date = _parse_hfdte(dte_raw)

    # --- Derived from B records ---
    gps_start_utc: datetime | None = None
    duration_s: float | None = None
    num_fixes = len(b_fixes)

    if b_fixes:
        first_fix = b_fixes[0]
        last_fix = b_fixes[-1]
        duration_s = float(last_fix["total_seconds"] - first_fix["total_seconds"])

        if first_fix["datetime"] is not None:
            gps_start_utc = first_fix["datetime"]

    # --- H-record fields ---
    def _h(tlc: str) -> str | None:
        """Return the raw H-record value, or *None* if the TLC is absent."""
        return h_records.get(tlc)

    def _h_nil(tlc: str) -> str | None:
        """Return *None* when the value is ``NIL``; used for person fields."""
        val = h_records.get(tlc)
        if val is None:
            return None
        return _nil_to_none(val)

    # Timezone needs numeric conversion
    tzn_raw = h_records.get("TZN")
    timezone_offset_h: float | None = None
    if tzn_raw is not None and tzn_raw.upper() != "NIL":
        try:
            timezone_offset_h = float(tzn_raw)
        except ValueError:
            timezone_offset_h = None

    return IGCMetadata(
        date=flight_date,
        gps_start_utc=gps_start_utc,
        duration_s=duration_s,
        num_fixes=num_fixes,
        timezone_offset_h=timezone_offset_h,
        pilot=_h_nil("PLT"),
        copilot=_h_nil("CM2"),
        glider_type=_h("GTY"),
        glider_id=_h("GID"),
        competition_id=_h("CID"),
        competition_class=_h("CCL"),
        fr_type=_h("FTY"),
        fr_id=fr_id,
        firmware=_h("RFW"),
        hardware=_h("RHW"),
        gps_receiver=_h("GPS"),
        pressure_sensor=_h("PRS"),
        gps_datum=_h("DTM"),
        altitude_gps_ref=_h("ALG"),
        altitude_pressure_ref=_h("ALP"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_igc_data(igc_path: Path) -> IGCTelemetry:
    """Parse an IGC file and return a fully populated :class:`IGCTelemetry`.

    Parameters
    ----------
    igc_path:
        Path to the source ``.igc`` file.

    Returns
    -------
    IGCTelemetry
        A new object with ``igc_path``, ``metadata``, and ``data_path``
        populated.  The trajectory DataFrame is saved as Parquet to
        ``config.DIR.TELEMETRY / f"{flight_tag}.parquet"`` where
        *flight_tag* is derived from ``igc_path.stem``.
    """
    flight_tag = igc_path.stem

    igc_size_kb = igc_path.stat().st_size / 1024
    logger.info(
        "Extracting IGC telemetry from %s (%.1f KiB)",
        igc_path.name,
        igc_size_kb,
    )
    t_total = time.monotonic()

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------
    fr_id, h_records, i_extensions, b_fixes = _parse_igc_file(igc_path)

    if not b_fixes:
        logger.warning("No B-record fixes found in %s", igc_path.name)
        return IGCTelemetry(igc_path=igc_path)

    logger.info(
        "Parsed %d B-record fixes (I-extensions: %d, H-records: %d)",
        len(b_fixes),
        len(i_extensions),
        len(h_records),
    )

    output_dir: Path = config.DIR.TELEMETRY
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Flight metadata → IGCMetadata object
    # ------------------------------------------------------------------
    metadata = _build_metadata(fr_id, h_records, b_fixes)

    if metadata.gps_start_utc is not None:
        logger.info("  GPS start (UTC):  %s", metadata.gps_start_utc.isoformat())
    if metadata.pilot is not None:
        logger.info("  Pilot:            %s", metadata.pilot)
    if metadata.glider_type is not None:
        logger.info(
            "  Glider:           %s (%s)",
            metadata.glider_type,
            metadata.glider_id or "?",
        )
    if metadata.fr_type is not None:
        logger.info("  Recorder:         %s", metadata.fr_type)
    if metadata.pressure_sensor is not None:
        logger.info("  Pressure sensor:  %s", metadata.pressure_sensor)
    else:
        logger.info("  Pressure sensor:  NIL (not available)")

    # ------------------------------------------------------------------
    # 1 Hz trajectory DataFrame → Parquet
    # ------------------------------------------------------------------
    traj_df = _build_trajectory_df(b_fixes)
    data_path: Path | None = None

    if not traj_df.empty:
        data_path = output_dir / f"{flight_tag}.parquet"
        traj_df.to_parquet(data_path, index=False)
        parquet_size_kb = data_path.stat().st_size / 1024
        ts = traj_df["timestamp_s"]
        span = ts.iloc[-1] - ts.iloc[0]
        logger.info(
            "Wrote Parquet:      %s  (%d rows, %.1f s, %.0f KiB)",
            data_path.name,
            len(traj_df),
            span,
            parquet_size_kb,
        )

        # Fix validity summary
        fix_counts = traj_df["fix_validity"].value_counts().to_dict()
        logger.info(
            "  Fix validity: %s  |  lat: %.4f – %.4f  |  lon: %.4f – %.4f",
            ", ".join(f"{k}={int(v)}" for k, v in sorted(fix_counts.items())),
            traj_df["gps_lat_deg"].min(),
            traj_df["gps_lat_deg"].max(),
            traj_df["gps_lon_deg"].min(),
            traj_df["gps_lon_deg"].max(),
        )
        logger.info(
            "  alt_baro: %.0f – %.0f m  |  alt_gps: %.0f – %.0f m",
            traj_df["alt_baro_m"].min(),
            traj_df["alt_baro_m"].max(),
            traj_df["alt_gps_m"].min(),
            traj_df["alt_gps_m"].max(),
        )
    else:
        logger.warning("No B-record data found — skipping trajectory Parquet")

    elapsed = time.monotonic() - t_total
    logger.info("Extraction complete in %.2f s", elapsed)

    return IGCTelemetry(
        igc_path=igc_path,
        metadata=metadata,
        data_path=data_path,
    )
