# Actor Input Data

This document describes the data that Unreal Engine C++ actors consume.
It is produced by the flight data processing pipeline and saved to disk as
Parquet / CSV files.

## File Location

Output files are written to the `config.DIR.ACTOR_DATA` directory
(default: `<PROJECT_ROOT>/ProcessedData/ActorData/`).

### Naming Convention

Each flight produces two files:

| File | Format | Description |
|------|--------|-------------|
| `{flight_tag}_trajectory_df.parquet` | Parquet (pyarrow) | Primary binary format |
| `{flight_tag}_trajectory_df.csv` | CSV (with index) | Human-readable companion |

`flight_tag` is the stem of the source filename, e.g. `"07_niskie_ladowanie"`.

## Coordinate System

All positions, velocities, and accelerations use a **local East-North-Up (ENU)**
frame centred on the airfield reference point (default: EPBC Warsaw Babice).

- **X** → East
- **Y** → North
- **Z** → Up

Angles use the **right-hand rule** and are in **radians**.

## Trajectory DataFrame Columns

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `timestamp_s` | float | s | Monotonically increasing timestamp |
| `x_m` | float | m | Position East |
| `y_m` | float | m | Position North |
| `z_m` | float | m | Position Up (height above airfield) |
| `vx_ms` | float | m/s | Velocity East |
| `vy_ms` | float | m/s | Velocity North |
| `vz_ms` | float | m/s | Velocity Up |
| `v_mag_ms` | float | m/s | Total speed magnitude |
| `ax_ms2` | float | m/s² | Acceleration East |
| `ay_ms2` | float | m/s² | Acceleration North |
| `az_ms2` | float | m/s² | Acceleration Up |
| `load_g` | float | g | Felt G-load magnitude |
| `yaw_rad` | float | rad | Heading (derived from velocity vector) |
| `pitch_rad` | float | rad | Pitch angle (flight-path angle + angle of attack) |
| `roll_rad` | float | rad | Bank angle (derived from centripetal acceleration) |
| `phase` | str | — | Flight phase label (see below) |

## Flight Phase Labels

The `phase` column contains one of the following `FlightPhase` enum values:

| Value | Meaning |
|-------|---------|
| `PRE_FLIGHT` | Before take-off |
| `LAUNCH_ROLL` | Ground roll during launch |
| `LAUNCH_CLIMB` | Climbing on tow / winch |
| `CRUISING` | Straight flight / glide |
| `THERMALLING` | Circling in a thermal |
| `ROLLING_LANDING` | Ground roll after touchdown |
| `PUSHING_GROUND` | Post-flight ground handling |

## Glider Path Actor

`AGliderPathActor` (`GliderPathActor.h / .cpp`) renders the glider's flight
path as a procedural **ribbon mesh** — a wing-envelope strip that trails behind
the glider during animation.

### Data Flow

1. **Load** — The actor reads `{FlightTag}_trajectory_df.csv` from the
   configured `ActorDataDirectory`.
2. **Downsample & smooth** — Raw telemetry (typically 18 Hz) is reduced to a
   configurable interval (`DownsampleInterval`) with rolling-window averaging
   (`SmoothingWindowFrames`).
3. **Coordinate conversion** — ENU metres are converted to Unreal world
   coordinates (East → +X, North → −Y, Up → +Z) scaled by
   `MetresToUnrealScale` (default 100, i.e. centimetres).
4. **Mesh generation** — The entire flight path is built **once** as a single
   `ProceduralMeshComponent` section. Each ribbon cross-section is oriented by
   the velocity vector and banked by the roll angle.

### UV Encoding

The ribbon's UVs carry the information the material needs to compute
per-pixel visibility:

| Channel | Meaning |
|---------|---------|
| **U** | Absolute timestamp (`Point.Time`) of the ribbon vertex |
| **V** | Wing position: `0` = left wingtip, `1` = right wingtip |

### Material Interface

The actor creates a `UMaterialInstanceDynamic` from the editor-assigned
`RibbonMaterial` and pushes four scalar parameters every tick:

| Parameter | Type | Description |
|-----------|------|-------------|
| `CurrentAnimationTime` | float | Current playback time (seconds) |
| `FadeStartDelay` | float | Seconds a segment stays at full opacity |
| `FadeDuration` | float | Seconds to fade from full to minimum opacity |
| `MinimumOpacity` | float | Floor opacity so old path remains faintly visible |

The material shader uses `TimeDelta = CurrentAnimationTime − U` together with
the fade parameters to:

* Hide future path (`TimeDelta < 0`) at full transparency.
* Fade the ribbon in behind the glider over a configurable ramp.
* Gradually dim older segments down to `MinimumOpacity`.
* Use V (wing position) to make the centre more transparent than the tips,
  giving a smoke-trail / contrail look.

### Key Properties

| Property | Default | Purpose |
|----------|---------|---------|
| `FlightTag` | — | Selects which flight CSV to load |
| `Wingspan` | 1800 cm | Width of the ribbon |
| `DownsampleInterval` | 1.0 s | Time step between ribbon vertices |
| `SmoothingWindowFrames` | 30 | ± frames for rolling average |
| `CurrentAnimationTime` | 0.0 | Sequencer-driven playback time |
| `RibbonMaterial` | — | Base material (must expose the scalar parameters above) |

Because the mesh is built once and only a material parameter is updated per
frame, the actor is lightweight at runtime and produces perfectly smooth
animation regardless of the downsampling interval.

## Glider Playback Actor

`AGliderPlaybackActor` (`GliderPlaybackActor.h / .cpp`) animates a 3D glider
mesh along the recorded flight trajectory. While the Path Actor renders the
trail ribbon, this actor drives the glider model itself.

### Component Hierarchy

| Component | Type | Role |
|-----------|------|------|
| `SceneRoot` | `USceneComponent` | Root; defines the actor's world placement |
| `KinematicAnchor` | `USceneComponent` | Child of root; driven by interpolated flight data |
| `GliderMesh` | `UStaticMeshComponent` | Child of anchor; the visible 3D glider model (allows manual offset) |

The two-level attachment (root → anchor → mesh) separates manual world
placement from kinematic animation, letting the artist offset or rotate the
mesh without affecting the flight data transform.

### Data Flow

1. **Load** — The actor reads `{FlightTag}_trajectory_df.csv` from
   `ActorDataDirectory` (default `../../ProcessedData/ActorData`). CSV parsing
   uses the same header-map approach as the Path Actor.
2. **Coordinate conversion** — Each row is converted from ENU metres to Unreal
   world coordinates (East → +X, North → −Y, Up → +Z) scaled by
   `MetresToUnrealScale` (default 100, i.e. centimetres).
3. **Orientation pre-calculation** — For each data point, a world-space
   quaternion is built at load time:
   - The **forward** vector is derived from the velocity columns
     (`vx_ms`, `-vy_ms`, `vz_ms`).
   - A **right** vector is computed from the cross product of world-up and
     forward.
   - The right vector is rotated around the forward axis by the negated
     `roll_rad` to apply bank angle.
   - A rotation matrix is constructed from the forward and right axes and
     stored as `FQuat WorldRotation` on the `FGliderDataPoint`.
4. **Playback** — Each tick (or when `CurrentAnimationTime` changes), the actor
   performs a **binary search** over the sorted data points to find the two
   surrounding frames, then applies:
   - **Lerp** for position.
   - **Slerp** for rotation (avoids gimbal lock and twisting).
   The interpolated transform is applied to the `KinematicAnchor`.

### Sequencer / MRQ Integration

`CurrentAnimationTime` is marked `Interp`, making it directly keyable in
Unreal Sequencer timelines. The `BlueprintSetter` (`SetCurrentAnimationTime`)
ensures that if Movie Render Queue starts a new render session and the flight
data has been garbage-collected, a reload is triggered automatically before the
first frame is evaluated.

### Editor Integration

- **`OnConstruction`** — Automatically loads data and poses the glider when the
  actor is placed or the level is opened.
- **`PostEditChangeProperty`** — Reloads flight data when `FlightTag`,
  `ActorDataDirectory`, or `MetresToUnrealScale` are changed in the Details
  panel. Changing `CurrentAnimationTime` immediately scrubs the glider to the
  new pose, enabling interactive preview without entering Play mode.
- **`ShouldTickIfViewportsOnly`** returns `true`, so the actor ticks in the
  editor viewport.

### Key Properties

| Property | Category | Default | Purpose |
|----------|----------|---------|---------|
| `FlightTag` | Glider Playback | — | Selects which flight CSV to load |
| `ActorDataDirectory` | Glider Playback | `../../ProcessedData/ActorData` | Relative path from the Unreal project directory |
| `CurrentAnimationTime` | Glider Playback | 0.0 | Sequencer-driven playback time (seconds); `Interp` property |
| `MetresToUnrealScale` | Glider Playback | 100.0 | Conversion factor (metres → Unreal units) |
| `SequenceDataDirectory` | Flight Data | `Saved/FlightComposer/SequenceData` | Output directory for the exported overlay CSV |
| `bExportOverlayData` | Flight Data | `false` | Enable CSV overlay export during MRQ renders |

### Consumed CSV Columns

The actor reads all telemetry columns from the trajectory CSV. The four
positional columns (`timestamp_s`, `x_m`, `y_m`, `z_m`) are required; all
others are optional and default to zero (or `1.0` for `load_g`) if absent.

| Column | Usage |
|--------|-------|
| `timestamp_s` | Playback timeline key |
| `x_m`, `y_m`, `z_m` | Position (converted to Unreal coordinates) |
| `vx_ms`, `vy_ms`, `vz_ms` | Velocity (used to derive forward direction) |
| `v_mag_ms` | Total speed magnitude (stored & interpolated for export) |
| `ax_ms2`, `ay_ms2`, `az_ms2` | Acceleration components (stored & interpolated for export) |
| `load_g` | Felt G-load (stored & interpolated for export) |
| `yaw_rad` | Heading (stored & interpolated for export) |
| `pitch_rad` | Pitch angle (stored & interpolated for export) |
| `roll_rad` | Bank angle (applied as roll around forward axis; also interpolated for export) |
| `phase` | Flight phase label (stored for export; nearest-neighbour, not interpolated) |

Yaw and pitch are implicitly derived from the velocity vector for the 3D
orientation quaternion, but the raw `yaw_rad` / `pitch_rad` values are
preserved in `FGliderDataPoint` and written to the overlay export CSV.

### Interpolation

During playback the actor binary-searches for the two surrounding data points
and interpolates:

- **Position** — `FMath::Lerp` between the two world positions.
- **Rotation** — `FQuat::Slerp` between the two pre-computed quaternions.
- **All float telemetry** — `FMath::Lerp` for speed, accelerations, G-load,
  yaw, pitch, and roll.
- **Phase** — Nearest-neighbour (the value from the earlier data point).

### CSV Overlay Export (MRQ Integration)

When `bExportOverlayData` is enabled, the actor writes a frame-accurate CSV
during a **Movie Render Queue (MRQ)** render. This CSV is intended for a
Python post-production script that composites 2D data overlays onto the
rendered image sequence.

#### How It Works

1. The actor ticks in `TG_PostUpdateWork`, guaranteeing that Sequencer has
   already evaluated `CurrentAnimationTime` for the current frame.
2. On each tick during play, it searches for the active `UMoviePipeline`
   instance in the current world using `TObjectIterator`.
3. It queries `UMoviePipelineBlueprintLibrary::GetOverallOutputFrames()` to
   obtain the **MRQ output frame index**. MRQ holds this index constant across
   TAA temporal sub-samples, providing automatic deduplication — exactly one
   CSV row per output image.
4. The effective output FPS is read from MRQ's pipeline configuration:
   - If the user set a custom frame rate in `UMoviePipelineOutputSetting`, that
     value is used.
   - Otherwise the sequence's display rate is read via
     `UMovieSceneSequence::GetMovieScene()->GetDisplayRate()`.
   - A hardcoded 24 FPS fallback is used if neither is available.
5. `video_time_s` is computed as `MRQFrameIndex / EffectiveFPS`.

#### Stale-Frame Guard

Sequencer's first evaluation after `BeginPlay` carries the stale editor
playhead position (from wherever the user left the scrub head). The actor
records `GFrameCounter` at `BeginPlay` time and refuses to export on that
engine frame, ensuring the first CSV row contains the correct MRQ start time.

#### Export File

The CSV is written to
`<ProjectDir>/<SequenceDataDirectory>/<SequenceName>.csv`
(default: `Saved/FlightComposer/SequenceData/`).

The file name is derived dynamically from the active Movie Render Queue job:
`BuildExportFilePath()` iterates over live `UMoviePipeline` instances, finds
the one belonging to the current world, and reads
`UMoviePipelineExecutorJob::JobName` (which MRQ sets to the Level Sequence
asset name). If no active MRQ pipeline or job is found, the actor falls back
to the `FlightTag` property.

#### Export CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `frame_id` | int (zero-padded) | MRQ output frame index (`0000`, `0001`, …) |
| `timestamp_s` | float | Interpolated flight-data timestamp |
| `video_time_s` | float | Video timeline time (`frame_id / EffectiveFPS`) |
| `x_m` | float | Interpolated position East |
| `y_m` | float | Interpolated position North |
| `z_m` | float | Interpolated position Up |
| `vx_ms` | float | Interpolated velocity East |
| `vy_ms` | float | Interpolated velocity North |
| `vz_ms` | float | Interpolated velocity Up |
| `v_mag_ms` | float | Interpolated total speed |
| `ax_ms2` | float | Interpolated acceleration East |
| `ay_ms2` | float | Interpolated acceleration North |
| `az_ms2` | float | Interpolated acceleration Up |
| `load_g` | float | Interpolated G-load |
| `yaw_rad` | float | Interpolated heading |
| `pitch_rad` | float | Interpolated pitch |
| `roll_rad` | float | Interpolated roll / bank angle |
| `phase` | string | Flight phase (nearest-neighbour from earlier sample) |

#### Module Dependencies

The export feature requires the following Unreal modules (added to
`EPBC.Build.cs`):

- `MovieRenderPipelineCore` — `UMoviePipeline`, `UMoviePipelineBlueprintLibrary`, pipeline config classes.
- `MovieScene` — `UMovieSceneSequence`, `UMovieScene` (for reading the sequence display rate).
- `LevelSequence` — Transitive link dependency.
