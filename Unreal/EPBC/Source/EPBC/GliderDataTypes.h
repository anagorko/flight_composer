#pragma once

#include "CoreMinimal.h"
#include "GliderDataTypes.generated.h"

// Structure to hold a single flight data point from the trajectory CSV.
// All positions are in local ENU (East-North-Up) metres, angles in radians.
USTRUCT(BlueprintType)
struct EPBC_API FGliderDataPoint
{
    GENERATED_BODY()

    // --- Timestamp ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Time = 0.0f; // timestamp_s

    // --- Position (ENU, metres) ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float X_m = 0.0f; // East

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Y_m = 0.0f; // North

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Z_m = 0.0f; // Up

    // --- Velocity (m/s) ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Vx = 0.0f; // East

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Vy = 0.0f; // North

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Vz = 0.0f; // Up

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float VMag = 0.0f; // Total speed

    // --- Acceleration (m/s²) ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Ax = 0.0f; // East

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Ay = 0.0f; // North

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Az = 0.0f; // Up

    // --- G-load ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float LoadG = 1.0f;

    // --- Orientation (radians) ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float YawRad = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float PitchRad = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float RollRad = 0.0f;

    // --- Flight phase label ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Phase;

    // --- Cached Unreal world position (set during coordinate conversion) ---
    FVector WorldPosition = FVector::ZeroVector;

    // --- Cached rotation ---
    FQuat WorldRotation = FQuat::Identity;

    // Default constructor
    FGliderDataPoint() {}
};
