# Data Processing Pipeline

This document describes the data flow and filename conventions when processing flight data for visualization.

## General Information

The source code is contained in the `flight_composer` package in the `src/flight_composer` directory.

The entry point to the data processing pipeline is the `generate_actor_data.py` script in `src/flight_composer/scripts/`. It accepts a `--flight_uid` argument (default `"07"`) and processes a single flight per invocation.

### Configuration

The global `config` singleton (`FlightComposerConfig`) is built by `load_config()` in `src/flight_composer/config.py`. It loads all configuration from YAML files under `config/`:

* **`config.DIR`** (`FlightComposerDirs`) — directory paths, loaded from `config/global/settings.yaml`. Relative paths are automatically resolved against `PROJECT_ROOT`. See the `FlightComposerDirs` model and `settings.yaml` for the full set of fields.
* **`config.airfields`** — a `dict[str, Airfield]` registry loaded from `config/global/airfields.yaml`.
* **`config.gliders`** — a `dict[str, GliderSpecs]` registry loaded from `config/global/gliders.yaml`.
* **`config.sequences`** — a `dict[str, SequenceConfig]` loaded from all `config/sequences/*.yaml` files (used by the overlay pipeline, see [OVERLAY](OVERLAY.md)).

The `Airfield` and `GliderSpecs` Pydantic models are defined in `src/flight_composer/flight_data.py`.

Key directories used by this pipeline: `config.DIR.MP4`, `config.DIR.IGC`, `config.DIR.TELEMETRY`, `config.DIR.ACTOR_DATA`.

### Flight UIDs

We use logbook flight numbers as flight identifiers. Each identifier is a two- or three-digit number, e.g. `"07"` is flight 7 from the logbook.

For some visualizations, we use synthetic flight data (e.g. for video discussing emergency procedures). These flights are assigned UIDs in a form `"Axx"`, e.g. `"A01"`.

## Flight files

There are two types of source files:

- **MP4 files** recorded with GoPro, containing GPMF telemetry data. Found at `config.DIR.MP4 / "{UID}_*.(mp4|MP4)"`.
- **IGC files** recorded with SeeYou Navigator on a smartphone. Found at `config.DIR.IGC / "{UID}_*.igc"`.

## Computation Pipeline

The pipeline processes a **single `flight_uid`** per invocation. It uses **one data source** — GoPro is preferred when available, IGC is used as a fallback.

### 1. Source File Discovery

**Module:** `processing/flight_data_sources.py` — `find_data_sources(flight_uid)`

Glob for matching files in `config.DIR.MP4` and `config.DIR.IGC`. Returns a `FlightDataSources` pydantic model containing:

| Field | Type | Description |
|-------|------|-------------|
| `flight_tag` | `str` | Stem of the source filename, e.g. `"07_niskie_ladowanie"` |
| `gopro_path` | `Path \| None` | Path to the MP4 file, if found |
| `igc_path` | `Path \| None` | Path to the IGC file, if found |

Constraints enforced:
- At most one MP4 and one IGC file may exist per UID (raises `ValueError` otherwise).
- At least one source file must exist.
- If both MP4 and IGC files exist, their stems (flight tags) must match.

The `flight_tag` is derived from the MP4 stem if available, otherwise from the IGC stem.

### 2. Telemetry Extraction

The pipeline calls **one** of the two extraction functions based on which data source is available (GoPro preferred).

#### 2a. GoPro Telemetry Extraction

**Module:** `processing/extract_gopro_data.py` — `extract_gopro_data(mp4_path) -> GoProTelemetry | None`

Parses the GPMF telemetry track embedded in the GoPro MP4 file and produces two Parquet files in `config.DIR.TELEMETRY`:

##### GPS-rate trajectory DataFrame

**File:** `{flight_tag}_mp4.parquet`

One row per GPS5 sample (~18 Hz). Slow-rate streams (GPSF, GPSP) are forward-filled. ACCL is downsampled to GPS rate via nearest-timestamp lookup.

| Column | Units | Description |
|--------|-------|-------------|
| `timestamp_s` | s | Seconds since recording start |
| `gps_lat_deg` | deg | Latitude WGS 84 |
| `gps_lon_deg` | deg | Longitude WGS 84 |
| `gps_alt_m` | m | Altitude WGS 84 |
| `gps_speed2d_ms` | m/s | 2D ground speed |
| `gps_speed3d_ms` | m/s | 3D speed |
| `gps_fix` | — | 0 / 2 / 3 (no lock / 2D / 3D) |
| `gps_dop` | — | Dilution of precision (GPSP / 100) |
| `accl_x_ms2` | m/s² | Accelerometer X (nearest to GPS timestamp) |
| `accl_y_ms2` | m/s² | Accelerometer Y |
| `accl_z_ms2` | m/s² | Accelerometer Z |

##### High-frequency IMU / orientation DataFrame

**File:** `{flight_tag}_mp4_hf.parquet`

One row per ACCL/GYRO sample (~200 Hz). Frame-rate streams (CORI, IORI, GRAV at ~60 Hz) are linearly interpolated to the IMU rate.

| Column | Units | Description |
|--------|-------|-------------|
| `timestamp_s` | s | Seconds since recording start |
| `accl_{x,y,z}_ms2` | m/s² | 3-axis accelerometer |
| `gyro_{x,y,z}_rads` | rad/s | 3-axis gyroscope |
| `cori_{w,x,y,z}` | — | Camera orientation quaternion (sensor-fused) |
| `iori_{w,x,y,z}` | — | Image orientation quaternion |
| `grav_{x,y,z}` | — | Gravity direction vector |

> **Note:** The HF IMU data is extracted and persisted but is **not currently consumed** by downstream pipeline stages. It is available for future high-frequency jitter extraction.

##### GoProMetadata (in-memory)

Recording-level metadata is stored directly on the `GoProTelemetry.metadata` field as a `GoProMetadata` pydantic model — **no external JSON file is written**. Key fields include `device_name`, `gps_start_utc`, `creation_time`, `duration_s`, plus sub-objects for `video` (`GoProVideoConfig`), `audio` (`GoProAudioConfig`), and `settings` (`GoProSettings` — FOV, stabilization, lens projection).

#### 2b. IGC Telemetry Extraction

**Module:** `processing/extract_igc_data.py` — `extract_igc_data(igc_path) -> IGCTelemetry`

Parses the plain-text IGC file (B-records, H-records, I-record extensions) and produces one Parquet file in `config.DIR.TELEMETRY`:

##### 1 Hz trajectory DataFrame

**File:** `{flight_tag}.parquet`

One row per B-record (~1 Hz).

| Column | Units | Description |
|--------|-------|-------------|
| `timestamp_s` | s | Seconds since first B-record fix |
| `gps_lat_deg` | deg | Latitude WGS 84 |
| `gps_lon_deg` | deg | Longitude WGS 84 |
| `alt_baro_m` | m | Pressure altitude (0 when sensor unavailable) |
| `alt_gps_m` | m | GNSS altitude (ellipsoid or geoid per `HFALG`) |
| `fix_validity` | — | `A` (3D valid) or `V` (2D / no GPS) |

##### IGCMetadata (in-memory)

Recording-level metadata is stored directly on the `IGCTelemetry.metadata` field as an `IGCMetadata` pydantic model — **no external JSON file is written**. Key fields include `date`, `gps_start_utc`, `duration_s`, `num_fixes`, `pilot`, `glider_type`, `glider_id`, `fr_type`, `pressure_sensor`, and various hardware/datum fields.

### 3. FlightTrack Construction

**Module:** `flight_track.py` — `FlightTrack.from_gopro()` or `FlightTrack.from_igc()`

Constructs a `FlightTrack` from the extracted telemetry. This step performs several transformations on the raw data, producing a validated DataFrame with local coordinates, data quality weights, flight phase labels, and corrected altitude.

#### 3.1 Projection to Local Coordinate System

WGS 84 coordinates are projected to a Local Tangent Plane (East-North-Up) using **Azimuthal Equidistant Projection** via `pyproj`. The projection origin is the airfield reference point (default: EPBC Warsaw Babice, lat 52.2689, lon 20.9107, alt 106.1 m).

| Output Column | Source | Description |
|---------------|--------|-------------|
| `x_m` | `gps_lon_deg` → East | Easting in metres from origin |
| `y_m` | `gps_lat_deg` → North | Northing in metres from origin |
| `z_m` | `gps_alt_m - origin.alt` (GoPro) or `alt_gps_m - origin.alt` (IGC) | Height above airfield |

#### 3.2 Weight Assignment

Each data point receives a quality weight used by the downstream spline fitting:

- **GoPro:** `weight = 1.0 / DOP` (inverse dilution of precision; DOP clamped to ≥ 0.1).
- **IGC:** `weight = 1.0` for valid fixes (`A`), `weight = 0.01` for invalid fixes (`V`).

#### 3.3 Raw Data Sanitization

**Module:** `processing/raw_data_sanitization.py`

##### IGC sanitization (`sanitize_igc_telemetry`)

1. **Monotonicity pass:** Drop rows where `timestamp_s` does not strictly increase.
2. **Altitude glitch interpolation:** Detect physically impossible vertical velocity jumps (> 30 m/s). Two cases:
   - **Dropout gaps** (massive drop followed by recovery): Linearly interpolate altitude columns across the gap, set weight to 0.01.
   - **Missed climbs** (massive jump up from delayed GPS fix): Reconstruct a linear climb ramp at 10 m/s backward from the jump point, set weight to 0.01.

##### GoPro sanitization (`sanitize_gopro_telemetry`)

1. **Monotonicity pass:** Drop rows where `timestamp_s` does not strictly increase.
2. **Kinematic validation:** Compare implied speed (from position deltas) against hardware-reported `gps_speed2d_ms`. Keep points where the speed ratio is within [0.5, 2.0] and sample interval > 0.03 s. Points below a ground speed threshold (7 m/s) are always kept.

#### 3.4 Schema Validation

The DataFrame is validated against `flight_track_schema` (pandera), ensuring:
- `timestamp_s` is monotonically increasing (float).
- `x_m`, `y_m`, `z_m`, `weight` are non-nullable floats.
- `phase` values (when present) are valid `FlightPhase` enum members.

#### 3.5 Flight Phase Assignment

**Module:** `processing/assign_flight_phases.py`

Two independent phase classifiers are run sequentially, each adding a column to the DataFrame.

##### Phase Definitions

The `FlightPhase` enum defines the following states:

| Enum Member | String Value | Description |
|-------------|-------------|-------------|
| `PRE_FLIGHT` | `"PRE_FLIGHT"` | Stationary or slow movement before launch |
| `LAUNCH_ROLL` | `"LAUNCH_ROLL"` | Accelerating on the ground |
| `LAUNCH_CLIMB` | `"LAUNCH_CLIMB"` | Steep climb connected to the launch (winch/tow) |
| `CRUISING` | `"CRUISING"` | Airborne, relatively straight flight |
| `THERMALLING` | `"THERMALLING"` | Airborne, continuous circling |
| `LANDING_ROLL` | `"ROLLING_LANDING"` | Touchdown and deceleration |
| `POST_FLIGHT` | `"PUSHING_GROUND"` | Stationary or walking pace after landing |

##### Classifier 1: Debounced State Machine → `phase` column

`assign_flight_phases(df)` uses a bidirectional, debounced state machine. It computes kinematic features (v_2d, v_z, omega turn rate, z_agl) from 3-second central finite differences, smoothed with a 2-second rolling median. Transitions require sustained evidence (debouncing) to reject GPS noise:

| From | To | Trigger | Debounce |
|------|----|---------|----------|
| PRE_FLIGHT | LAUNCH_ROLL | v_2d > 3.0 m/s | 1 s |
| LAUNCH_ROLL | LAUNCH_CLIMB | v_2d > 15.0 AND v_z > 1.0 AND z_agl > 5.0 | 2 s |
| LAUNCH_CLIMB | CRUISING | v_z < 1.0 AND z_agl > 50.0 | 3 s |
| CRUISING | THERMALLING | \|omega\| > 6.0 deg/s | 5 s |
| THERMALLING | CRUISING | \|omega\| < 4.0 deg/s | 5 s |
| (Any airborne) | LANDING_ROLL | z_agl < 10.0 AND v_2d < 30.0 | 4 s |
| LANDING_ROLL | POST_FLIGHT | v_2d < 2.0 m/s | 5 s |

Error recovery: if in LANDING_ROLL or POST_FLIGHT but z_agl > 30.0 m and v_2d > 15.0 m/s, immediately revert to CRUISING.

##### Classifier 2: Hidden Markov Model → `phase_hmm` column

`assign_flight_phases_hmm(df, airfield)` uses a Hidden Markov Model with Viterbi decoding to find the globally optimal phase sequence. It uses the same kinematic observables (v_2d, v_z, omega, z_agl) but evaluates the entire flight globally rather than sequentially. Ground level is derived from the airfield origin altitude rather than from the first 10 seconds of data.

The HMM result in `phase_hmm` is used by downstream stages (altitude drift correction, trajectory phase interpolation).

#### 3.6 Ground Phase Trimming

**Method:** `FlightTrack.trim_flight_phases(df, keep_seconds=3.0)`

Removes the bulk of `PRE_FLIGHT` and `POST_FLIGHT` phases from the DataFrame, retaining only a short buffer at the boundary with the active flight. This prevents long ground recordings from distorting spline fitting and trajectory statistics.

- **PRE_FLIGHT:** Only the last `keep_seconds` (default 3 s) are kept — i.e. the rows closest to launch.
- **POST_FLIGHT:** Only the first `keep_seconds` (default 3 s) are kept — i.e. the rows immediately after landing.

If a phase is absent (e.g. the recording started mid-flight), no trimming is applied for that phase.

Uses the `phase_hmm` column if present, otherwise falls back to `phase`. If neither column exists the DataFrame is returned unchanged. The index is reset after trimming.

The retained 3-second ground segments are sufficient for downstream altitude drift correction anchoring.

#### 3.7 Altitude Drift Correction

**Module:** `processing/fix_altitude_drift.py` — `fix_altitude_drift(df, airfield)`

Corrects vertical drift in `z_m` using the `PRE_FLIGHT` and `POST_FLIGHT` ground phases (from `phase_hmm`). The goal is to anchor ground-level altitude to 0.0 m (the airfield reference).

**Correction modes:**

1. **Two ground phases** (both `PRE_FLIGHT` and `POST_FLIGHT` present): Compute the minimum `z_m` in each phase. Linearly interpolate the correction between the two anchor points (at each phase's median timestamp). The correction is held constant beyond the anchors.

2. **Single ground phase** (only one present): Subtract the minimum `z_m` of that phase as a constant offset from the entire `z_m` column.

3. **No ground phases:** Return data unmodified.

Outlier detection warns when: minimum z_m is far from the phase mean (> 3σ), absolute offset exceeds ±30 m, or phase noise is high (std > 5 m).

#### 3.8 FlightTrack Output

The resulting `FlightTrack` object holds a validated DataFrame with columns including `timestamp_s`, `x_m`, `y_m`, `z_m`, `weight`, `phase`, `phase_hmm`, plus any source-specific columns retained from the telemetry.

### 4. FlightTrajectory Construction (Spline Fitting)

**Module:** `flight_trajectory.py` — `FlightTrajectory.from_track(track)`

**Module:** `kinematic_spline.py` — `KinematicSpline`

Converts the discrete `FlightTrack` into a continuous, analytically differentiable trajectory by fitting B-splines independently to the horizontal and vertical axes.

#### 4.1 Spline Configuration

| Axis | Degree | Smoothing Parameter | Description |
|------|--------|-------------------|-------------|
| XY (horizontal) | 5 (quintic) | `N × 0.5` | C⁴ continuity; smooth through jerk |
| Z (vertical) | 3 (cubic) | `N × 1.0` | C² continuity; heavier smoothing to suppress GPS altitude bounce |

Where `N` is the number of data points. The heavier Z smoothing (2× the XY factor) counteracts the inherent noise in GPS-derived altitude.

#### 4.2 Implementation

`KinematicSpline` wraps `scipy.interpolate.splprep` / `splev` (FITPACK). It stores the fitted spline representation (knots, coefficients, degree) and provides evaluation at arbitrary timestamps with analytical derivatives up to degree order.

- `fit(t, points, weights)` — fits the spline. Requires strictly monotonic timestamps.
- `__call__(t, derivative=0)` — evaluates position (0), velocity (1st), or acceleration (2nd derivative).

#### 4.3 Continuous Evaluation

The `FlightTrajectory` object provides methods to evaluate at any time `t`:

- `position(t)` → `[x, y, z]` in metres (ENU)
- `velocity(t)` → `[vx, vy, vz]` in m/s (analytical 1st derivative)
- `acceleration(t)` → `[ax, ay, az]` in m/s² (analytical 2nd derivative)
- `orientation(t)` → `[yaw, pitch, roll]` in radians (derived from kinematics, see §5)

### 5. Orientation Computation

**Module:** `flight_trajectory.py` — `FlightTrajectory.orientation(t)`

Euler angles are derived purely from the velocity and acceleration vectors, assuming coordinated flight. No wind estimation or air mass conversion is performed — orientation is computed from ground-track kinematics.

#### 5.1 Yaw (Heading)

Aligned with the ground velocity direction:

`yaw = arctan2(vy, vx)`

#### 5.2 Roll (Bank Angle)

Derived from centripetal acceleration in the horizontal plane, assuming a coordinated turn:

`a_c = (vy · ax - vx · ay) / |v_horiz|`

`roll = arctan2(a_c, g)` where `g = 9.81 m/s²`

In straight flight, `a_c ≈ 0` and roll is approximately zero.

#### 5.3 Pitch

Computed as the sum of flight path angle and estimated angle of attack:

1. **Flight path angle:** `γ = arctan2(vz, |v_horiz|)`
2. **Angle of attack** from the required lift coefficient:
   - Air density `ρ` from the standard atmosphere model: `ρ = 1.225 × (1 - 2.25577×10⁻⁵ × z)^4.256`
   - Required `C_L = (2 × m × g) / (ρ × V² × S × cos(roll))`, clamped to [-0.5, 1.5]
   - 3D lift curve slope: `a_w = 2π / (1 + 2π / (π × e × AR))`
   - `α = C_L / a_w + α₀` where `α₀ ≈ -0.035 rad` (zero-lift AoA)
3. **Pitch** = `γ + α`

Glider parameters default to the SZD-51 Junior (`GliderSpecs`): mass 340 kg, wing area 12.51 m², wingspan 15.0 m, Oswald efficiency 0.85.

### 6. Trajectory DataFrame Assembly

**Module:** `flight_trajectory.py` — `FlightTrajectory.trajectory_df(fps=60.0)`

The continuous trajectory is sampled at 60 Hz to produce the final export DataFrame.

#### 6.1 Sampling

A uniform time array is created from `t_begin` to `t_end` at `1/fps` intervals. Position, velocity, acceleration, and orientation are evaluated at each sample via the spline.

#### 6.2 Load Factor

The felt G-load is computed from kinematic acceleration plus gravity reaction:

`acc_felt = acc + [0, 0, 9.81]`

`load_g = |acc_felt| / 9.81`

#### 6.3 Phase Interpolation

Flight phases from the `FlightTrack` (`phase_hmm` column) are interpolated to the high-frequency time array using nearest-neighbor interpolation, clamped at the edges.

#### 6.4 Schema Validation

The output DataFrame is validated against `flight_trajectory_schema` (pandera).

#### 6.5 Output Columns

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `timestamp_s` | float | s | Monotonically increasing timestamp |
| `x_m` | float | m | Position East (ENU) |
| `y_m` | float | m | Position North (ENU) |
| `z_m` | float | m | Position Up (ENU) |
| `vx_ms` | float | m/s | Velocity East |
| `vy_ms` | float | m/s | Velocity North |
| `vz_ms` | float | m/s | Velocity Up |
| `v_mag_ms` | float | m/s | Total speed magnitude |
| `ax_ms2` | float | m/s² | Acceleration East |
| `ay_ms2` | float | m/s² | Acceleration North |
| `az_ms2` | float | m/s² | Acceleration Up |
| `load_g` | float | g | Felt G-load magnitude |
| `yaw_rad` | float | rad | Heading (from velocity) |
| `pitch_rad` | float | rad | Pitch (flight path + AoA) |
| `roll_rad` | float | rad | Bank angle (from centripetal accel) |
| `phase` | str | — | Flight phase label (from `FlightPhase` enum) |

### 7. Export

**Module:** `scripts/generate_actor_data.py`

The trajectory DataFrame is saved in two formats to `config.DIR.ACTOR_DATA`:

| File | Format | Description |
|------|--------|-------------|
| `{flight_tag}_trajectory_df.parquet` | Parquet (pyarrow) | Primary output, compact binary |
| `{flight_tag}_trajectory_df.csv` | CSV (with index) | Human-readable companion |

## Pipeline Summary

```
generate_actor_data.py --flight_uid=UID
│
├─ 1. find_data_sources(UID)
│     → FlightDataSources { flight_tag, gopro_path?, igc_path? }
│
├─ 2. extract_gopro_data(path)          OR  extract_igc_data(path)
│     → GoProTelemetry                      → IGCTelemetry
│       ├ {tag}_mp4.parquet (GPS ~18Hz)       ├ {tag}.parquet (1Hz)
│       ├ {tag}_mp4_hf.parquet (IMU ~200Hz)   └ IGCMetadata (in-memory)
│       └ GoProMetadata (in-memory)
│
├─ 3. FlightTrack.from_gopro(telemetry) OR  FlightTrack.from_igc(telemetry)
│     ├ 3.1 WGS84 → ENU projection (aeqd, origin=EPBC)
│     ├ 3.2 Weight assignment (1/DOP or fix validity)
│     ├ 3.3 Raw data sanitization (monotonicity + glitch repair)
│     ├ 3.4 Schema validation (pandera)
│     ├ 3.5 Phase assignment (debounced state machine → phase)
│     ├ 3.5 Phase assignment (HMM/Viterbi → phase_hmm)
│     ├ 3.6 Ground phase trimming (keep 3s of PRE_FLIGHT / POST_FLIGHT)
│     └ 3.7 Altitude drift correction (anchor to 0m at ground phases)
│
├─ 4. FlightTrajectory.from_track(track)
│     ├ XY spline: quintic (k=5), smoothing = N×0.5
│     └ Z  spline: cubic  (k=3), smoothing = N×1.0
│
├─ 5. trajectory.trajectory_df(fps=60)
│     ├ Evaluate position, velocity, acceleration from splines
│     ├ Compute orientation (yaw, pitch, roll) from kinematics
│     ├ Compute load factor from felt acceleration
│     ├ Interpolate phases (nearest-neighbor)
│     └ Validate against flight_trajectory_schema
│
└─ 7. Export
      ├ {tag}_trajectory_df.parquet
      └ {tag}_trajectory_df.csv
```

## Features Not Yet Implemented

The following capabilities are planned but not currently part of the pipeline:

- **Multi-source temporal alignment:** When both GoPro and IGC data exist for the same flight, aligning and fusing the two data streams. Currently only one source is used (GoPro preferred).
- **Wind & atmosphere estimation:** Circle drift algorithm for wind extraction, air mass velocity computation, and netto vario from the glider polar curve.
- **High-frequency jitter extraction:** Using GoPro IMU/CORI data to add turbulence micro-motion on top of the smooth kinematic trajectory. The HF data is extracted (§2a) but not yet consumed.
- **Orientation damping:** SLERP low-pass filtering of orientation quaternions to simulate roll inertia.
- **State-based orientation overrides:** Forcing bank angle to zero during ground roll and launch phases.
- **Physics bounds clamping:** Clamping orientation to survivable limits (bank ±90°, pitch ±45°).
- **Total energy tracking:** Computing specific energy `E_tot = g·h + ½·V_TAS²` for physics validation.
- **Spline coefficient export:** Outputting the continuous spline representation rather than sampled frames, allowing consumers to evaluate at arbitrary frame rates.