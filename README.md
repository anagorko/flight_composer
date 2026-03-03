# Flight Composer

Flight trajectory interpolation and animation system for gliders in Unreal Engine.

## Installation

```bash
pixi install
```

### Fonts

Download `IBM Plex Mono` font and unzip it into `Fonts/` directory (in particular it should contain `IBMPlexMono-Regular.ttf`)

### Tools

Download `Gyroflow-linux64.AppImage` into `Tools/` directory.

## TODO

- create trajectory animation actor
- add mapbox tilesets

# Some setup to remember

- Cesium World Terrain > Bing Maps Aerial > Maximum Screen Space Error > 0.5: high-res tiles are shown even from high altitude
- Project Settings > Near Clip Plane > 1.0: for flying through clouds; AI suggestion, I didn't see a difference
         WARNING: when it somehow got reset to 0.0 rendering completely broke
- Directional Light > Per Pixel Atmosphere Transmittance > True
- Directional Light > Cast Cloud Shadows > True
- Cloud Material > Cloud_GlobalCoverage > 0.0
- Sky Atmosphere > Aerial Perspective Start Depth > 0.001 (or large value like 2.0)
- Sky Atmosphere > Planet Top at Component Transform; move it 100 meters down the z-axis; otherwise ground level is much darker and rendering flickers

# Resources

* EPBC coordinates: 52.26889595260726, 20.910683925025026
* Lighting and Clouds: https://www.youtube.com/watch?v=TDFDqpBynK8
* https://cesium.com/learn/unreal/unreal-edit-materials/ (tileset materials in cesium)

# Basic setup in Unreal Engine

## General

-  Edit -> Editor Preferences > Source Code Editor (sadly no Zed yet)

## Standard Lighting

* Create empty level
* Add Directional Light
* Add SkyAtmosphere
  - set "Transform Mode" to "Planet Top at Component Transform"
  - set Location Z to -1000000
* Add SkyLight
* Add ExponentialHeightFog
* Add VolumetricCloud
  - Set Layer Bottom Altitude to 12.0
  - Copy m_SimpleVolumetricCloud_Inst as m_MySimpleVolumetricCloud_Inst
  - Set cloud material to m_MySimpleVolumetricCloud_Inst
  - Edit it in material editor:
      - set Layout_CloudType Stratocumulus 1.5, Altostratus 0, Cirrostratus 0, Nimbostratus 1.5 
      - set Layout_CloudGlobalScale to 200

## Ultra Dynamic Sky

## Camera

* Add CineCamera
  - Current Camera Settings > Filmback > Sensor Width: 36mm
  - Current Camera Settings > Filmback > Sensor Height: 20.25mm
  - Focus Method: Disable
  - Shutter Speed: 1/60
  - Set Min EV100 = Max EV100 = 3.0

# C++ Actors

After adding new files to the project run

/srv/Unreal/Engine/Build/BatchFiles/Linux/GenerateProjectFiles.sh -project="/home/andrzej/reps/flight_composer/Unreal/EPBC/EPBC.uproject" -game

After modifying files run

make EPBCEditor 

in project root.

## UGeoJsonLoaderComponent

- Added necessary module names in Source/EPBC/EPBC.Build.cs 
- Created GeoJsonLoaderComponent.h and .cpp in Source/EPBC/

# Workflow

1. Join GoPro .MP4 files into a single file with Gyroflow.
2. Extract telemetry data from joined .MP4 file like:

python extract-gopro-data-feeds.py /srv/samba/share/GoPro\ Processed/GH010367_joined.MP4  > flight_data.csv

# Virtual environment setup

conda create -n flight_composer --yes
conda env update -f environment.yml --prune
conda activate flight_composer
python -m pip install -v -e .

# Older

gopro-dashboard.py --layout-xml overlay-simple.xml --use-gpx-only --profile quicktime --overlay-size 1980x1080 --gpx 26.06_agl.gpx 26.06.overlay-simple.mov
ffmpeg -i GH010367_joined_stabilized.mp4 -c:v prores_ks -profile:v 3 -qscale:v 9 -acodec pcm_s16le lipiec_stabilized.mov

# Cesium for Unreal

https://github.com/CesiumGS/cesium-unreal/releases
