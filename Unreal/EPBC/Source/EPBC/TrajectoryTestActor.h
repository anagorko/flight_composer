#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"

#include "TrajectoryTestActor.generated.h"

// Structure to hold a single trajectory point
USTRUCT(BlueprintType)
struct EPBC_API FTrajectoryPoint
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Timestamp = 0.0f;

    // For ENU coordinate system (legacy)
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float X = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Y = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Z = 0.0f;

    // For WGS84 coordinate system (Cesium integration)
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    double Latitude = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    double Longitude = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    double AltitudeMSL = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Source = TEXT("UNKNOWN");

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Quality = 1.0f;

    // Default constructor
    FTrajectoryPoint()
    {
        Timestamp = 0.0f;
        X = 0.0f;
        Y = 0.0f;
        Z = 0.0f;
        Latitude = 0.0;
        Longitude = 0.0;
        AltitudeMSL = 0.0;
        Source = TEXT("UNKNOWN");
        Quality = 1.0f;
    }

    // Constructor with ENU parameters
    FTrajectoryPoint(float InTimestamp, float InX, float InY, float InZ, const FString& InSource, float InQuality)
        : Timestamp(InTimestamp), X(InX), Y(InY), Z(InZ), Source(InSource), Quality(InQuality)
    {
        Latitude = 0.0;
        Longitude = 0.0;
        AltitudeMSL = 0.0;
    }

    // Constructor with WGS84 parameters
    FTrajectoryPoint(float InTimestamp, double InLat, double InLon, double InAlt, const FString& InSource, float InQuality)
        : Timestamp(InTimestamp), Latitude(InLat), Longitude(InLon), AltitudeMSL(InAlt), Source(InSource), Quality(InQuality)
    {
        X = 0.0f;
        Y = 0.0f;
        Z = 0.0f;
    }
};

UCLASS()
class EPBC_API ATrajectoryTestActor : public AActor
{
    GENERATED_BODY()

public:
    ATrajectoryTestActor();

    // Called when properties are changed in the editor
    virtual void OnConstruction(const FTransform& Transform) override;

    // Called when properties are changed in the editor
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;

    // Load flight data from pre-existing JSON file and create spheres
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "Trajectory Test")
    void LoadTrajectoryData();

    // Clear all spheres
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "Trajectory Test")
    void ClearSpheres();

    // Refresh trajectory visualization
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "Trajectory Test")
    void RefreshTrajectory();

protected:
    // Flight number to load (e.g., "07")
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flight Data")
    FString FlightNumber = TEXT("07");

    // Scale factor for coordinates (to convert from meters to Unreal units)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization", meta = (ClampMin = "0.1", ClampMax = "100.0"))
    float CoordinateScale = 0.01f;

    // Scale factor for sphere size
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization", meta = (ClampMin = "0.1", ClampMax = "10.0"))
    float SphereScale = 1.0f;

    // Sphere mesh to use for trajectory points
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization")
    TSoftObjectPtr<UStaticMesh> SphereMesh;

    // Material for IGC data points
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization")
    TSoftObjectPtr<UMaterialInterface> IGCMaterial;

    // Material for GPX data points
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization")
    TSoftObjectPtr<UMaterialInterface> GPXMaterial;

    // Material for GeoJSON data points
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization")
    TSoftObjectPtr<UMaterialInterface> GeoJSONMaterial;

    // Material for unknown source data points
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization")
    TSoftObjectPtr<UMaterialInterface> UnknownMaterial;

    // Maximum number of spheres to create (to avoid performance issues)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization", meta = (ClampMin = "10", ClampMax = "10000"))
    int32 MaxSpheres = 1000;

    // Skip factor - only show every Nth point
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization", meta = (ClampMin = "1", ClampMax = "100"))
    int32 SkipFactor = 1;

    // Whether to use Cesium coordinate transformation (for WGS84 data)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Coordinates")
    bool bUseCesiumCoordinates = true;

    // JSON file suffix to use (_trajectory.json for ENU, _wgs84.json for WGS84)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flight Data")
    FString JSONFileSuffix = TEXT("_wgs84.json");

    // Loaded trajectory data points
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Flight Data")
    TArray<FTrajectoryPoint> TrajectoryData;

    // Created sphere components
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Visualization")
    TArray<UStaticMeshComponent*> SphereComponents;

    // Statistics about loaded data
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Flight Data")
    int32 TotalPoints = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Flight Data")
    float FlightDuration = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Flight Data")
    FString DataStatus = TEXT("No data loaded - run Python export first");

private:
    // Helper functions
    void CreateSphereAtPoint(const FTrajectoryPoint& Point, int32 Index);
    UMaterialInterface* GetMaterialForSource(const FString& Source);
    FVector ConvertToUnrealCoordinates(float X, float Y, float Z);
    FVector ConvertWGS84ToUnrealCoordinates(double Latitude, double Longitude, double AltitudeMSL);
    bool LoadTrajectoryFromJSON(const FString& JSONPath);
    bool LoadWGS84TrajectoryFromJSON(const FString& JSONPath);
    void UpdateStatistics();
    void LogTrajectoryBounds();

    // Internal state
    bool bDataLoaded = false;
    FString LastLoadedFlightNumber;
};