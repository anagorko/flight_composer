"""
FlightTrack holds discrete series of flight data points before smoothing interpolation is applied.

FlightTrack imports data from GoProTelemetry or IGCTelemetry object. Some preliminary data processing is done by the class.

1. GoPro data sanitization: sometimes GoPro hangs and records invalid timestamps, e.g.

>>> d = extract_gopro_data(pathlib.Path("/srv/samba/share/GoProFlights/09_termika_maj_2025.MP4"))
>>> print(d.data["timestamp_s"][15880:15960].diff().to_string())

2. IGC data sanitization: sometimes IGC has 0 time delta between records. Also, data recorded with my phone often loses altitude reading and suddenly drops to the ground, e.g.

>>> from flight_composer.processing.extract_igc_data import extract_igc_data
>>> d = extract_igc_data(pathlib.Path("IGCData/07_niskie_ladowanie.igc"))
>>> d.data[:][133:142]
     timestamp_s  gps_lat_deg  gps_lon_deg  alt_baro_m  alt_gps_m fix_validity
133        134.0    52.279633    20.908900           0        382            A
134        135.0    52.279700    20.909250           0        140            A
135        136.0    52.279817    20.909550           0        140            A
136        137.0    52.279950    20.909800           0        140            A
137        138.0    52.280100    20.909967           0        140            A
138        139.0    52.280283    20.910067           0        140            A
139        140.0    52.280483    20.910083           0        140            A
140        141.0    52.280683    20.910017           0        140            A
141        142.0    52.280867    20.909867           0        372            A

3. Detect flight phases and add them to the dataframe (column "phase").

4. Trim ground phases: remove most of PRE_FLIGHT and POST_FLIGHT, keeping only 3 seconds
   at the boundary with the active flight to avoid long ground recordings distorting
   spline fitting.

5. Fix altitude drift and level ground phases to Z = 0.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import pyproj

from flight_composer.flight_data import (
    Airfield,
    GliderSpecs,
    GoProTelemetry,
    IGCTelemetry,
)
from flight_composer.processing.assign_flight_phases import (
    FlightPhase,
    assign_flight_phases,
    assign_flight_phases_hmm,
)
from flight_composer.processing.fix_altitude_drift import fix_altitude_drift
from flight_composer.processing.raw_data_sanitization import (
    sanitize_gopro_telemetry,
    sanitize_igc_telemetry,
)

# ---------------------------------------------------------------------------
# FlightTrack dataframe schema
# ---------------------------------------------------------------------------

flight_track_schema = pa.DataFrameSchema(
    columns={
        "timestamp_s": pa.Column(
            float,
            checks=pa.Check(
                lambda s: s.is_monotonic_increasing,
                name="is_monotonic",
                error="timestamp_s must be monotonically increasing. Did sanitization fail?",
            ),
            nullable=False,
        ),
        "x_m": pa.Column(float, nullable=False),
        "y_m": pa.Column(float, nullable=False),
        "z_m": pa.Column(float, nullable=False),
        "weight": pa.Column(float, nullable=False),
        "phase": pa.Column(
            str,
            checks=pa.Check.isin([e.value for e in FlightPhase]),
            required=False,
            nullable=False,
        ),
        "phase_hmm": pa.Column(
            str,
            checks=pa.Check.isin([e.value for e in FlightPhase]),
            required=False,
            nullable=True,
        ),
    },
    # Set strict=False so we don't throw errors if 'gps_speed2d_ms' or 'weight' are also present
    strict=False,
    # Coerce ints to floats automatically if needed
    coerce=True,
)

# ---------------------------------------------------------------------------
# FlightTrack
# ---------------------------------------------------------------------------


class FlightTrack:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Private constructor. Use the factory methods from_igc or from_gopro.
        The dataframe is expected to have standard columns:
        ['timestamp_s', 'x_m', 'y_m', 'z_m', 'weight']
        """
        self._df = dataframe
        self.glider_model: GliderSpecs = GliderSpecs()

    @staticmethod
    def trim_flight_phases(df: pd.DataFrame, keep_seconds: float = 3.0) -> pd.DataFrame:
        """Remove most of the PRE_FLIGHT and POST_FLIGHT phases, keeping only
        *keep_seconds* worth of data at the boundary with the active flight.

        For PRE_FLIGHT the last *keep_seconds* are kept (transition into
        flight).  For POST_FLIGHT the first *keep_seconds* are kept
        (transition out of flight).

        Uses ``phase_hmm`` if present, otherwise falls back to ``phase``.
        If neither column exists the dataframe is returned unchanged.
        """
        logger = logging.getLogger("flight_track.trim_flight_phases")

        if "phase_hmm" in df.columns:
            phase_col = "phase_hmm"
        elif "phase" in df.columns:
            phase_col = "phase"
        else:
            logger.info("No phase column found – skipping ground phase trimming")
            return df

        logger.info(
            f"Trimming ground phases using '{phase_col}' column (keep_seconds={keep_seconds})"
        )

        pre_value = FlightPhase.PRE_FLIGHT.value
        post_value = FlightPhase.POST_FLIGHT.value

        n_before = len(df)
        mask_keep = pd.Series(True, index=df.index)

        # --- trim PRE_FLIGHT: keep only the last `keep_seconds` ----------
        pre_mask = df[phase_col] == pre_value
        if pre_mask.any():
            n_pre = int(pre_mask.sum())
            pre_timestamps = df.loc[pre_mask, "timestamp_s"]
            pre_duration = pre_timestamps.iloc[-1] - pre_timestamps.iloc[0]
            pre_end = pre_timestamps.iloc[-1]
            trim_before = pre_end - keep_seconds
            mask_keep = mask_keep & ~(pre_mask & (df["timestamp_s"] < trim_before))
            n_pre_kept = int((mask_keep & pre_mask).sum())
            n_pre_removed = n_pre - n_pre_kept
            logger.info(
                f"PRE_FLIGHT: {n_pre} samples spanning {pre_duration:.1f}s – "
                f"removing {n_pre_removed}, keeping {n_pre_kept} (last {keep_seconds}s)"
            )
        else:
            logger.info("PRE_FLIGHT phase not present – nothing to trim")

        # --- trim POST_FLIGHT: keep only the first `keep_seconds` --------
        post_mask = df[phase_col] == post_value
        if post_mask.any():
            n_post = int(post_mask.sum())
            post_timestamps = df.loc[post_mask, "timestamp_s"]
            post_duration = post_timestamps.iloc[-1] - post_timestamps.iloc[0]
            post_start = post_timestamps.iloc[0]
            trim_after = post_start + keep_seconds
            mask_keep = mask_keep & ~(post_mask & (df["timestamp_s"] > trim_after))
            n_post_kept = int((mask_keep & post_mask).sum())
            n_post_removed = n_post - n_post_kept
            logger.info(
                f"POST_FLIGHT: {n_post} samples spanning {post_duration:.1f}s – "
                f"removing {n_post_removed}, keeping {n_post_kept} (first {keep_seconds}s)"
            )
        else:
            logger.info("POST_FLIGHT phase not present – nothing to trim")

        result = df.loc[mask_keep].reset_index(drop=True)
        n_after = len(result)
        logger.info(
            f"Trimming complete: {n_before} → {n_after} samples ({n_before - n_after} removed)"
        )
        return result

    @classmethod
    def from_igc(
        cls, igc: IGCTelemetry, airfield: Airfield = Airfield()
    ) -> FlightTrack:
        if igc.data is None:
            raise ValueError("IGC data is None")

        origin = airfield.origin
        proj_aeqd = pyproj.Proj(
            proj="aeqd", lat_0=origin.lat, lon_0=origin.lon, datum="WGS84", units="m"
        )

        df = igc.data.copy()
        df["x_m"], df["y_m"] = proj_aeqd(
            df["gps_lon_deg"].values, df["gps_lat_deg"].values
        )
        # IGC usually has barometric altitude as primary, but gps_alt is safer for 3D trajectory
        df["z_m"] = df["alt_gps_m"] - origin.alt
        df["weight"] = np.where(df["fix_validity"] == "A", 1.0, 0.01)

        df = sanitize_igc_telemetry(df)
        df = flight_track_schema.validate(df).copy()
        df = assign_flight_phases(df)
        df = assign_flight_phases_hmm(df, airfield)
        df = cls.trim_flight_phases(df)
        df = fix_altitude_drift(df, airfield)

        return cls(df)

    @classmethod
    def from_gopro(
        cls, gopro: GoProTelemetry, airfield: Airfield = Airfield()
    ) -> FlightTrack:
        if gopro.data is None:
            raise ValueError("GoPro data is None")

        origin = airfield.origin

        df = gopro.data.copy()

        proj_aeqd = pyproj.Proj(
            proj="aeqd", lat_0=origin.lat, lon_0=origin.lon, datum="WGS84", units="m"
        )

        df["x_m"], df["y_m"] = proj_aeqd(
            df["gps_lon_deg"].values, df["gps_lat_deg"].values
        )
        df["z_m"] = df["gps_alt_m"] - origin.alt
        safe_dop = df["gps_dop"].replace(0.0, 0.1)
        df["weight"] = 1.0 / safe_dop

        df = sanitize_gopro_telemetry(df)
        df = flight_track_schema.validate(df).copy()
        df = assign_flight_phases(df)
        df = assign_flight_phases_hmm(df, airfield)
        df = cls.trim_flight_phases(df)
        df = fix_altitude_drift(df, airfield)

        return cls(df)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Access the raw, cleaned pandas dataframe."""
        return self._df

    @property
    def t(self) -> npt.NDArray[np.float64]:
        return self._df["timestamp_s"].to_numpy()

    @property
    def points_2d(self) -> npt.NDArray[np.float64]:
        return self._df[["x_m", "y_m"]].to_numpy()

    @property
    def points_3d(self) -> npt.NDArray[np.float64]:
        return self._df[["x_m", "y_m", "z_m"]].to_numpy()

    @property
    def points_z(self) -> npt.NDArray[np.float64]:
        return self._df[["z_m"]].to_numpy()

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        return self._df["weight"].to_numpy()

    @property
    def phases(self) -> npt.NDArray[np.str_]:
        return self._df["phase_hmm"].to_numpy()
