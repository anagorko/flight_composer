from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from flight_composer.flight_track import FlightTrack
from flight_composer.kinematic_spline import KinematicSpline
from flight_composer.processing.assign_flight_phases import FlightPhase

# ---------------------------------------------------------------------------
# FlightTrajectory dataframe schema
# ---------------------------------------------------------------------------

flight_trajectory_schema = pa.DataFrameSchema(
    columns={
        "timestamp_s": pa.Column(
            float,
            checks=pa.Check(
                lambda s: s.is_monotonic_increasing,
                name="is_monotonic",
                error="timestamp_s must be monotonically increasing.",
            ),
            nullable=False,
        ),
        "x_m": pa.Column(float, nullable=False),
        "y_m": pa.Column(float, nullable=False),
        "z_m": pa.Column(float, nullable=False),
        "vx_ms": pa.Column(float, nullable=False),
        "vy_ms": pa.Column(float, nullable=False),
        "vz_ms": pa.Column(float, nullable=False),
        "v_mag_ms": pa.Column(float, nullable=False),  # Total speed
        "ax_ms2": pa.Column(float, nullable=False),
        "ay_ms2": pa.Column(float, nullable=False),
        "az_ms2": pa.Column(float, nullable=False),
        "load_g": pa.Column(float, nullable=False),  # G-force magnitude
        "yaw_rad": pa.Column(float, nullable=False),
        "pitch_rad": pa.Column(float, nullable=False),
        "roll_rad": pa.Column(float, nullable=False),
        "phase": pa.Column(
            str,
            checks=pa.Check.isin([e.value for e in FlightPhase]),
            required=True,
            nullable=False,
        ),
    },
    strict=False,
    coerce=True,
)


class FlightTrajectory:
    """
    A continuous 3D representation of a flight, derived from discrete telemetry.
    Provides smoothed position, velocity, acceleration, and orientation (yaw/pitch/roll)
    at any arbitrary timestamp.
    """

    def __init__(
        self, xy_spline: KinematicSpline, z_spline: KinematicSpline, track: FlightTrack
    ):
        """
        Private constructor. Use the `from_track` factory method.
        """
        self._xy_spline = xy_spline
        self._z_spline = z_spline
        self._track = track
        self.glider_model = track.glider_model

    @classmethod
    def from_track(
        cls,
        track: FlightTrack,
        degree_xy: int = 5,
        degree_z: int = 3,
    ) -> FlightTrajectory:
        """
        Creates a continuous trajectory by fitting independent splines to the
        horizontal and vertical axes of a FlightTrack.
        """

        xy_smoothing = len(track.t) * 0.5
        xy_spline = KinematicSpline(degree=degree_xy, smoothing=xy_smoothing)
        xy_spline.fit(track.t, track.points_2d, track.weights)

        # Z data is usually noisier (barometer/GPS altitude), so it often needs more smoothing
        z_smoothing = len(track.t) * 1.0
        z_spline = KinematicSpline(degree=degree_z, smoothing=z_smoothing)
        z_spline.fit(track.t, track.points_z, weights=track.weights)

        return cls(xy_spline, z_spline, track)

    @property
    def t_begin(self) -> float:
        """The starting timestamp of the valid trajectory."""
        return self._xy_spline.t_begin

    @property
    def t_end(self) -> float:
        """The ending timestamp of the valid trajectory."""
        return self._xy_spline.t_end

    def position(self, t: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pos_xy = self._xy_spline(t, derivative=0)
        pos_z = self._z_spline(t, derivative=0)
        return (
            np.hstack((pos_xy, pos_z))
            if isinstance(t, np.ndarray)
            else np.append(pos_xy, pos_z)
        )

    def velocity(self, t: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        vel_xy = self._xy_spline(t, derivative=1)
        vel_z = self._z_spline(t, derivative=1)
        return (
            np.hstack((vel_xy, vel_z))
            if isinstance(t, np.ndarray)
            else np.append(vel_xy, vel_z)
        )

    def acceleration(
        self, t: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        acc_xy = self._xy_spline(t, derivative=2)
        acc_z = self._z_spline(t, derivative=2)
        return (
            np.hstack((acc_xy, acc_z))
            if isinstance(t, np.ndarray)
            else np.append(acc_xy, acc_z)
        )

    def orientation(
        self, t: float | npt.NDArray[np.float64], smoothing_window_s: float = 1.5
    ) -> npt.NDArray[np.float64]:
        """
        Computes the Euler angles [yaw, pitch, roll] in radians at time(s) t.
        Derived from the velocity and acceleration vectors assuming coordinated flight.
        """
        vel = self.velocity(t)
        acc = self.acceleration(t)

        is_scalar = vel.ndim == 1
        if is_scalar:
            vel = vel.reshape(1, 3)
            acc = acc.reshape(1, 3)

        vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
        ax, ay = acc[:, 0], acc[:, 1]

        v_horiz2 = vx**2 + vy**2
        v_horiz = np.sqrt(v_horiz2)
        v_mag2 = v_horiz2 + vz**2

        eps = 1e-6
        v_horiz_safe = np.maximum(v_horiz, eps)
        v_mag2_safe = np.maximum(v_mag2, eps)

        # --- 1. YAW (Heading) ---
        yaw = np.arctan2(vy, vx)

        # --- 2. ROLL (Bank Angle) ---
        g = 9.81
        a_c = (vy * ax - vx * ay) / v_horiz_safe
        roll = np.arctan2(a_c, g)

        # --- 3. PITCH ---
        gamma = np.arctan2(vz, v_horiz)

        pos = self.position(t)
        if is_scalar:
            pos = pos.reshape(1, 3)
        z_m = pos[:, 2]

        rho = 1.225 * (1 - 2.25577e-5 * z_m) ** 4.256
        m = self.glider_model.mass_kg
        S = self.glider_model.wing_area_m2
        AR = self.glider_model.aspect_ratio
        e = 0.85
        alpha_0 = -0.035

        a_w = (2 * np.pi) / (1 + (2 * np.pi) / (np.pi * e * AR))

        cos_roll_safe = np.maximum(np.cos(roll), 0.1)
        CL = (2 * m * g) / (rho * v_mag2_safe * S * cos_roll_safe)
        CL = np.clip(CL, -0.5, 1.5)

        alpha = (CL / a_w) + alpha_0
        pitch = gamma + alpha

        # --- Filter the final Euler angles ---
        if not is_scalar and smoothing_window_s > 0 and len(t) > 1:
            dt = t[1] - t[0]
            if dt > 0:
                sigma = smoothing_window_s / dt

                # Unwrap the angles to prevent 360-degree spins when crossing boundaries
                yaw = np.unwrap(yaw)
                roll = np.unwrap(roll)
                pitch = np.unwrap(pitch)

                # Smooth the visual angles directly
                yaw = gaussian_filter1d(yaw, sigma=sigma)
                roll = gaussian_filter1d(roll, sigma=sigma)
                pitch = gaussian_filter1d(pitch, sigma=sigma)

                # Re-wrap yaw to [-pi, pi] to keep the exported CSV clean
                yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        result = np.column_stack((yaw, pitch, roll))

        if is_scalar:
            return result[0]
        return result

    def trajectory_df(self, fps: float = 60.0) -> pd.DataFrame:
        """
        Generates a high-frequency DataFrame sampled at the specified FPS,
        containing position, velocity, acceleration, and phase data.
        """
        # 1. Create high-frequency time array
        step = 1.0 / fps
        t_high_freq = np.arange(self.t_begin, self.t_end, step)

        # 2. Evaluate kinematics
        pos = self.position(t_high_freq)
        vel = self.velocity(t_high_freq)
        acc = self.acceleration(t_high_freq)
        orient = self.orientation(t_high_freq)

        v_mag = np.linalg.norm(vel, axis=1)

        # Calculate felt G-load (kinematic acceleration + reaction to gravity)
        # Assuming Z is up. Standard gravity g = 9.81 m/s^2
        gravity_vector = np.array([0.0, 0.0, 9.81])
        acc_felt = acc + gravity_vector
        load_g = np.linalg.norm(acc_felt, axis=1) / 9.81

        # 3. Interpolate discrete phases using nearest neighbor
        # Scipy interp1d doesn't handle strings directly, so we interpolate indices
        original_t = self._track.t
        original_phases = self._track.phases

        min_idx = 0.0
        max_idx = float(len(original_t) - 1)

        nearest_idx_interpolator = interp1d(
            original_t,
            np.arange(len(original_t)),
            kind="nearest",
            bounds_error=False,
            fill_value=(min_idx, max_idx),  # Clamps out-of-bounds to the edge indices
        )
        nearest_indices = nearest_idx_interpolator(t_high_freq).astype(int)
        interpolated_phases = original_phases[nearest_indices]

        # 4. Construct the DataFrame
        df = pd.DataFrame(
            {
                "timestamp_s": t_high_freq,
                "x_m": pos[:, 0],
                "y_m": pos[:, 1],
                "z_m": pos[:, 2],
                "vx_ms": vel[:, 0],
                "vy_ms": vel[:, 1],
                "vz_ms": vel[:, 2],
                "v_mag_ms": v_mag,
                "ax_ms2": acc[:, 0],
                "ay_ms2": acc[:, 1],
                "az_ms2": acc[:, 2],
                "load_g": load_g,
                "yaw_rad": orient[:, 0],
                "pitch_rad": orient[:, 1],
                "roll_rad": orient[:, 2],
                "phase": interpolated_phases,
            }
        )

        # 5. Validate against our schema
        return flight_trajectory_schema.validate(df)
