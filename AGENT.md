This repository contains tool chain to process recorded glider flight data and animate it in Unreal Engine.

The repository contains flight_composer python package defined in pyproject.toml.
Package source code resides in src/flight_composer directory and its scripts in src/flight_composer/scripts directory.

To run any python code, use

  "pixi run python ..."

The repository contains C++ Unreal Engine project in Unreal/EPBC directory with a set of C++ classes in Unreal/EPBC/Source/EPBC.
To compile Unreal Engine C++ classes, run "make EPBCEditor" in Unreal/EPBC directory.
