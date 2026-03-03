#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h"
#include "GliderDataTypes.h"
#include "GliderPathActor.generated.h"


UCLASS(meta = (PrioritizeCategories = "Glider Path"))
class EPBC_API AGliderPathActor : public AActor
{
    GENERATED_BODY()

public:
    AGliderPathActor();

    // Called when properties are changed in the editor
    virtual void OnConstruction(const FTransform& Transform) override;

    // Called when properties are changed in the editor
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;

    // Called every frame
    virtual void Tick(float DeltaTime) override;

    // Tells Unreal to run the Tick function even when not in Play mode
    virtual bool ShouldTickIfViewportsOnly() const override;

    // Load flight data from file
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "Glider Path")
    void LoadFlightData();

    // Update the path to show progress up to the specified time
    UFUNCTION(BlueprintCallable, Category = "Glider Path")
    void UpdatePathAtTime(float AnimationTime);

    // Clear all mesh data
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "Glider Path")
    void ClearPath();

    // Rebuild the path mesh (useful for updating after property changes in editor)
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "Glider Path")
    void RebuildPath();

protected:
    // The component that handles the mesh generation
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Glider Path")
    UProceduralMeshComponent* ProceduralMesh;

    // ---- Flight data source ----

    // Flight identifier / tag (e.g. "07_niskie_ladowanie").
    // The actor derives the CSV path: {ActorDataDirectory}/{FlightTag}_trajectory_df.csv
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path")
    FString FlightTag;

    // Base directory that contains the trajectory CSV files.
    // Relative to the Unreal project root (NOT Content).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path")
    FString ActorDataDirectory = TEXT("../../ProcessedData/ActorData");

    // ---- Ribbon parameters ----

    // The wingspan (width of the ribbon) in Unreal units (cm)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "1.0"))
    float Wingspan = 1800.0f;

    // ---- Downsampling & smoothing ----

    // Interval in seconds between downsampled ribbon points (e.g. 1.0 = one point per second).
    // Set to 0 to disable downsampling and use the raw 60 Hz data.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "0.0"))
    float DownsampleInterval = 1.0f;

    // Half-width of the smoothing window in raw frames (±N frames around each sample centre).
    // At 60 Hz, 30 means ±0.5 s.  Only used when DownsampleInterval > 0.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "0"))
    int32 SmoothingWindowFrames = 30;

    // ---- Ribbon Material ----

    // Base material to use for the ribbon. The material should have a
    // "CurrentAnimationTime" scalar parameter to control visibility.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ribbon Material")
    UMaterialInterface* RibbonMaterial;

    // Dynamic material instance created at runtime to pass CurrentAnimationTime
    UPROPERTY(Transient)
    UMaterialInstanceDynamic* RibbonMID;

    // ---- Animation ----

    // Current animation time (can be driven by sequencer)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, BlueprintSetter = SetCurrentAnimationTime, Interp, Category = "Glider Path", meta = (ClampMin = "0.0"))
    float CurrentAnimationTime = 0.0f;

    UFUNCTION(BlueprintSetter, CallInEditor)
    void SetCurrentAnimationTime(float NewTime);

    // How long segments remain at full opacity before starting to fade
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "0.0"))
    float FadeStartDelay = 360.0f;

    // How long it takes for segments to fade from full to minimum opacity
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "0.1"))
    float FadeDuration = 240.0f;

    // Minimum opacity for faded segments (they won't disappear completely)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float MinimumOpacity = 0.5f;

    // Whether to automatically update the path during play
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path")
    bool bAutoUpdatePath = true;

    // ---- Coordinate conversion ----

    // Scale factor applied when converting from data metres to Unreal units.
    // Unreal uses centimetres, so the default is 100.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "0.01"))
    float MetresToUnrealScale = 100.0f;

    // ---- Runtime data ----

    // Loaded flight data points
    UPROPERTY(Transient)
    TArray<FGliderDataPoint> FlightData;

    // Track which mesh sections are currently active
    TArray<int32> ActiveMeshSections;
    int32 NextMeshSectionID;

    // Cache the last time we updated the path to avoid redundant updates
    float LastUpdateTime;

    // ---- Internal helpers ----

    // Downsample the raw FlightData to ~1 Hz with rolling-window smoothing.
    // Called automatically at the end of LoadFlightData().
    void DownsampleAndSmooth();

    void GenerateRibbonSegments();
    FVector CalculateRibbonRight(const FVector& Direction, float RollRad);
    void ConvertCoordinatesToWorld();
    void CreateContinuousRibbonMesh(const TArray<FGliderDataPoint>& Points);

    // CSV parsing helpers
    TArray<FString> ParseCSVLine(const FString& Line);
    TMap<FString, int32> BuildHeaderMap(const FString& HeaderLine);

    // Build the full filesystem path from FlightTag + ActorDataDirectory
    FString BuildFlightDataPath() const;

    // Debug logging helper
    void LogFlightDataBounds();

private:
    // Internal state tracking
    int32 LastGeneratedPointIndex;
    bool bDataLoaded;
    bool bCoordinatesConverted;
    bool bMeshBuilt;
};