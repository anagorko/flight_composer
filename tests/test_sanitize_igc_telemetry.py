"""Tests for sanitize_igc_telemetry – especially the altitude-drop interpolation pass."""

import numpy as np
import pandas as pd
import pytest

from flight_composer.flight_track import sanitize_igc_telemetry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_igc_df(
    timestamp_s: list[float],
    alt_gps_m: list[float],
    alt_baro_m: list[float] | None = None,
    z_m: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal IGC-style DataFrame for testing."""
    n = len(timestamp_s)
    df = pd.DataFrame(
        {
            "timestamp_s": timestamp_s,
            "gps_lat_deg": np.linspace(52.27, 52.28, n),
            "gps_lon_deg": np.linspace(20.90, 20.91, n),
            "alt_baro_m": alt_baro_m if alt_baro_m is not None else alt_gps_m,
            "alt_gps_m": alt_gps_m,
            "fix_validity": ["A"] * n,
        }
    )
    if z_m is not None:
        df["z_m"] = z_m
    return df


# ---------------------------------------------------------------------------
# PASS 1 – monotonicity (existing behaviour)
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_strictly_increasing_timestamps_unchanged(self):
        df = _make_igc_df(
            timestamp_s=[1.0, 2.0, 3.0, 4.0],
            alt_gps_m=[100.0, 110.0, 120.0, 130.0],
        )
        result = sanitize_igc_telemetry(df)
        assert len(result) == 4

    def test_duplicate_timestamp_dropped(self):
        df = _make_igc_df(
            timestamp_s=[1.0, 2.0, 2.0, 3.0],
            alt_gps_m=[100.0, 110.0, 115.0, 120.0],
        )
        result = sanitize_igc_telemetry(df)
        assert len(result) == 3
        assert list(result["timestamp_s"]) == [1.0, 2.0, 3.0]

    def test_backward_timestamp_dropped(self):
        df = _make_igc_df(
            timestamp_s=[1.0, 5.0, 3.0, 6.0],
            alt_gps_m=[100.0, 110.0, 105.0, 120.0],
        )
        result = sanitize_igc_telemetry(df)
        assert list(result["timestamp_s"]) == [1.0, 5.0, 6.0]


# ---------------------------------------------------------------------------
# PASS 2 – altitude drop interpolation
# ---------------------------------------------------------------------------


class TestAltitudeDropInterpolation:
    """Reproduce the scenario from the module docstring."""

    def test_single_drop_segment_is_interpolated(self):
        """A single altitude drop to ground level should be linearly filled."""
        # Mimics the docstring example (rows 133–141):
        #   382  →  140 140 140 140 140 140 140  →  372
        alt = [382.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 372.0]
        ts = list(range(134, 134 + len(alt)))
        ts_f = [float(t) for t in ts]

        df = _make_igc_df(timestamp_s=ts_f, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)

        # End-points must remain unchanged
        assert result.at[0, "alt_gps_m"] == pytest.approx(382.0)
        assert result.at[len(result) - 1, "alt_gps_m"] == pytest.approx(372.0)

        # Interior points should be linearly interpolated between 382 and 372
        for i in range(1, len(result) - 1):
            frac = (result.at[i, "timestamp_s"] - ts_f[0]) / (ts_f[-1] - ts_f[0])
            expected = 382.0 + frac * (372.0 - 382.0)
            assert result.at[i, "alt_gps_m"] == pytest.approx(expected, abs=0.01), (
                f"index {i}: expected {expected}, got {result.at[i, 'alt_gps_m']}"
            )

    def test_z_m_column_also_interpolated(self):
        """If z_m is present it must be interpolated in the same segment."""
        origin_alt = 100.0
        alt = [382.0, 140.0, 140.0, 372.0]
        z = [a - origin_alt for a in alt]
        ts = [1.0, 2.0, 3.0, 4.0]

        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt, z_m=z)
        result = sanitize_igc_telemetry(df)

        # z_m at anchors
        assert result.at[0, "z_m"] == pytest.approx(282.0)
        assert result.at[3, "z_m"] == pytest.approx(272.0)

        # Interior z_m should be linearly interpolated
        for i in [1, 2]:
            frac = (result.at[i, "timestamp_s"] - 1.0) / (4.0 - 1.0)
            expected_z = 282.0 + frac * (272.0 - 282.0)
            assert result.at[i, "z_m"] == pytest.approx(expected_z, abs=0.01)

    def test_alt_baro_m_also_interpolated(self):
        """alt_baro_m (often zeroed out during drops) should be interpolated too."""
        alt_gps = [400.0, 140.0, 140.0, 390.0]
        alt_baro = [300.0, 0.0, 0.0, 290.0]
        ts = [10.0, 11.0, 12.0, 13.0]

        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt_gps, alt_baro_m=alt_baro)
        result = sanitize_igc_telemetry(df)

        # Baro at interior points should be linearly interpolated
        for i in [1, 2]:
            frac = (result.at[i, "timestamp_s"] - 10.0) / 3.0
            expected_baro = 300.0 + frac * (290.0 - 300.0)
            assert result.at[i, "alt_baro_m"] == pytest.approx(expected_baro, abs=0.01)

    def test_no_drop_leaves_data_unchanged(self):
        """Normal altitude variations within threshold should not trigger interpolation."""
        alt = [300.0, 305.0, 310.0, 308.0, 315.0]
        ts = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)

        np.testing.assert_array_almost_equal(result["alt_gps_m"].values, alt)

    def test_drop_at_start_is_skipped(self):
        """A drop at the very first record cannot be interpolated – it must be skipped."""
        alt = [140.0, 140.0, 380.0]
        ts = [1.0, 2.0, 3.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        # The first diff is 0 (no change), but the big jump is at index 2 which is
        # a recovery without a preceding drop. This should leave the data untouched.
        result = sanitize_igc_telemetry(df)
        np.testing.assert_array_almost_equal(result["alt_gps_m"].values, alt)

    def test_drop_without_recovery_is_skipped(self):
        """If altitude drops and never recovers, the segment is not modified."""
        alt = [400.0, 140.0, 140.0, 140.0]
        ts = [1.0, 2.0, 3.0, 4.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)
        # Only the anchor (index 0) should be unchanged; bad values stay because
        # there's no recovery point to interpolate toward.
        assert result.at[0, "alt_gps_m"] == pytest.approx(400.0)
        assert result.at[1, "alt_gps_m"] == pytest.approx(140.0)

    def test_two_separate_drop_segments(self):
        """Two distinct drop–recovery events should both be independently fixed."""
        # Segment 1: indices 1–2 are bad (drop from 400 → 100, recovery at idx 3 to 390)
        # Segment 2: indices 5–6 are bad (drop from 395 → 100, recovery at idx 7 to 385)
        alt = [400.0, 100.0, 100.0, 390.0, 395.0, 100.0, 100.0, 385.0]
        ts = [float(i) for i in range(len(alt))]

        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)

        # Anchors unchanged
        assert result.at[0, "alt_gps_m"] == pytest.approx(400.0)
        assert result.at[3, "alt_gps_m"] == pytest.approx(390.0)
        assert result.at[4, "alt_gps_m"] == pytest.approx(395.0)
        assert result.at[7, "alt_gps_m"] == pytest.approx(385.0)

        # First segment interpolated
        for i in [1, 2]:
            frac = (ts[i] - ts[0]) / (ts[3] - ts[0])
            expected = 400.0 + frac * (390.0 - 400.0)
            assert result.at[i, "alt_gps_m"] == pytest.approx(expected, abs=0.01)

        # Second segment interpolated
        for i in [5, 6]:
            frac = (ts[i] - ts[4]) / (ts[7] - ts[4])
            expected = 395.0 + frac * (385.0 - 395.0)
            assert result.at[i, "alt_gps_m"] == pytest.approx(expected, abs=0.01)

    def test_small_altitude_change_not_treated_as_drop(self):
        """Altitude changes ≤ 50 m should never trigger the drop logic."""
        alt = [300.0, 260.0, 250.0, 260.0, 300.0]
        ts = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)
        np.testing.assert_array_almost_equal(result["alt_gps_m"].values, alt)

    def test_exact_threshold_not_triggered(self):
        """A drop of exactly 50 m should NOT trigger interpolation (threshold is strict >)."""
        alt = [300.0, 250.0, 250.0, 300.0]
        ts = [1.0, 2.0, 3.0, 4.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)
        np.testing.assert_array_almost_equal(result["alt_gps_m"].values, alt)

    def test_row_count_preserved(self):
        """Altitude interpolation should never add or remove rows."""
        alt = [382.0, 140.0, 140.0, 140.0, 372.0]
        ts = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)
        assert len(result) == len(df)

    def test_monotonicity_and_altitude_fix_combined(self):
        """Both passes should work together: first drop duplicates, then fix altitude."""
        # Timestamps 1, 2, 2 (dup), 3, 4, 5  with altitude drop in remaining data
        alt = [380.0, 100.0, 999.0, 100.0, 100.0, 370.0]
        ts = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)

        # After pass 1: duplicate at t=2.0 removed → [1,2,3,4,5] with alt [380,100,100,100,370]
        # After pass 2: indices 1-3 interpolated between 380 and 370
        assert len(result) == 5
        assert result.at[0, "alt_gps_m"] == pytest.approx(380.0)
        assert result.at[4, "alt_gps_m"] == pytest.approx(370.0)

        # Interior values should be linearly interpolated
        for i in [1, 2, 3]:
            frac = (result.at[i, "timestamp_s"] - 1.0) / 4.0
            expected = 380.0 + frac * (370.0 - 380.0)
            assert result.at[i, "alt_gps_m"] == pytest.approx(expected, abs=0.01)

    def test_single_bad_point(self):
        """A single bad point (drop + immediate recovery) should be interpolated."""
        alt = [400.0, 140.0, 390.0]
        ts = [1.0, 2.0, 3.0]
        df = _make_igc_df(timestamp_s=ts, alt_gps_m=alt)
        result = sanitize_igc_telemetry(df)

        assert result.at[0, "alt_gps_m"] == pytest.approx(400.0)
        assert result.at[2, "alt_gps_m"] == pytest.approx(390.0)
        # Middle point interpolated
        expected = 400.0 + 0.5 * (390.0 - 400.0)
        assert result.at[1, "alt_gps_m"] == pytest.approx(expected, abs=0.01)
