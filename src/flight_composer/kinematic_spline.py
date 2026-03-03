"""
Provides continuous, mathematically smooth representations of discrete flight telemetry.

This module relies on time-parameterized B-splines to convert discrete, noisy
GPS/sensor fixes into analytically continuous paths. By using degree 3 or 5 splines,
it guarantees continuous first and second derivatives, allowing accurate extraction
of ground speed, acceleration, and kinematic orientations for downstream 3D visualization.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev, splprep


class KinematicSpline:
    def __init__(
        self,
        degree: int = 5,
        smoothing: float = 0.0,
    ):
        self.degree = degree
        self.smoothing = smoothing

        # Internal state to hold the FITPACK spline representation (knots, coefficients, degree)
        self._tck = None
        self._t_extents = None

    def fit(
        self,
        t: npt.NDArray[np.float64],
        points: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64] | None = None,
    ) -> KinematicSpline:
        """
        Fits the B-spline to the discrete data points.

        Args:
            t: 1D array of strictly increasing timestamps (shape: N,).
            points: 2D array of spatial coordinates (shape: N x D).
            weights: Optional 1D array of weights (shape: N,). Larger = closer fit.

        Returns:
            self, for method chaining (e.g. spline = KinematicSpline(degree=5, smoothing=10.0).fit(t, pts, w)).
        """
        # Guard against the SciPy Fortran crash: t must be strictly increasing.
        if not np.all(np.diff(t) > 0):
            raise ValueError(
                "Time array 't' must be strictly monotonically increasing. "
                "Ensure duplicate timestamps are dropped before fitting."
            )

        # splprep expects a sequence of 1D arrays (one per spatial dimension).
        # Transposing 'points' from (N, D) to (D, N) unpacks perfectly into this expected format.
        tck, _ = splprep(x=points.T, u=t, w=weights, k=self.degree, s=self.smoothing)

        self._tck = tck
        self._t_extents = (float(t[0]), float(t[-1]))

        return self

    @property
    def t_begin(self) -> float:
        """The starting timestamp of the fitted spline."""
        if self._t_extents is None:
            raise RuntimeError("Spline has not been fitted yet.")
        return self._t_extents[0]

    @property
    def t_end(self) -> float:
        """The ending timestamp of the fitted spline."""
        if self._t_extents is None:
            raise RuntimeError("Spline has not been fitted yet.")
        return self._t_extents[1]

    def __call__(
        self, t: float | npt.NDArray[np.float64], derivative: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the spline or its derivatives at the given time(s).

        Args:
            t: A single timestamp or a 1D array of timestamps to evaluate.
            derivative: 0 for position, 1 for velocity, 2 for acceleration.

        Returns:
            A numpy array of evaluated points.
            If t is a scalar, returns shape (D,).
            If t is an array, returns shape (len(t), D).
        """
        if self._tck is None:
            raise RuntimeError("Spline has not been fitted yet. Call .fit() first.")

        # splev evaluates the spline and returns a list of 1D arrays (one for each dimension).
        evaluated = splev(t, self._tck, der=derivative)

        # Transpose back so the output shape matches the (N, D) or (D,) input convention.
        return np.array(evaluated).T
