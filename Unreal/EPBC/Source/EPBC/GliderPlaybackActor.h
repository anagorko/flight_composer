#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GliderDataTypes.h"
#include "GliderPlaybackActor.generated.h"

UCLASS(meta = (PrioritizeCategories = "Glider Playback"))
class EPBC_API AGliderPlaybackActor : public AActor
{
    GENERATED_BODY()

public:
    AGliderPlaybackActor();

    virtual void Tick(float DeltaTime) override;
    virtual bool ShouldTickIfViewportsOnly() const override;
    virtual void OnConstruction(const FTransform& Transform) override;
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

#if WITH_EDITOR
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

    // Load flight data from the CSV
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "Glider Playback")
    void LoadFlightData();

protected:
    // ---- Components ----

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    USceneComponent* SceneRoot;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    USceneComponent* KinematicAnchor;

    // The actual 3D model of the glider
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    UStaticMeshComponent* GliderMesh;

    // ---- Playback Settings ----

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Playback", meta = (DisplayPriority = "1"))
    FString FlightTag;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Playback", meta = (DisplayPriority = "2"))
    FString ActorDataDirectory = TEXT("../../ProcessedData/ActorData");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, BlueprintSetter = SetCurrentAnimationTime, Interp, Category = "Glider Playback", meta = (ClampMin = "0.0"))
    float CurrentAnimationTime = 0.0f;

    UFUNCTION(BlueprintSetter, CallInEditor)
    void SetCurrentAnimationTime(float NewTime);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Playback", meta = (DisplayPriority = "4", ClampMin = "0.01"))
    float MetresToUnrealScale = 100.0f;

    // ---- Flight Data Export ----

    /** Relative directory under the project's Saved folder for exported CSV data. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flight Data")
    FString SequenceDataDirectory = TEXT("Saved/FlightComposer/SequenceData");

    /** When true, a frame-accurate CSV of interpolated overlay data is written during play. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flight Data")
    bool bExportOverlayData = false;

    // ---- Runtime Data ----

    UPROPERTY(Transient)
    TArray<FGliderDataPoint> FlightData;

    // ---- Internal Helpers ----

    // Finds the two surrounding frames and interpolates the glider's transform
    void UpdateGliderTransform(float Time);

    // Identical parsing setup to your Path actor
    FString BuildFlightDataPath() const;
    TArray<FString> ParseCSVLine(const FString& Line);
    TMap<FString, int32> BuildHeaderMap(const FString& HeaderLine);

private:
    bool bDataLoaded;

    // ---- CSV Export State ----

    /** The last interpolated data point computed by UpdateGliderTransform(). */
    FGliderDataPoint LastInterpPoint;

    /** Whether LastInterpPoint contains valid data from the current frame. */
    bool bInterpPointReady = false;

    /** GFrameCounter captured at BeginPlay time.
     *  Sequencer's first evaluation on this frame carries the stale editor playhead,
     *  so we refuse to export until the frame counter has advanced past this value. */
    uint64 BeginPlayFrameCounter = 0;

    /** The MRQ output frame index that was last written to the export CSV.
     *  MRQ holds this constant across TAA sub-samples, giving us perfect dedup. */
    int32 LastExportedVideoFrame = -1;

    /** Monotonically increasing row counter (only used if we ever need an internal count). */
    int32 ExportFrameId = 0;

    /** Full path to the currently open export CSV file. */
    FString ExportFilePath;

    /** Whether the CSV header has already been written for this play session. */
    bool bExportHeaderWritten = false;

    /** Build the full export file path based on project dir, SequenceDataDirectory, and FlightTag. */
    FString BuildExportFilePath() const;

    /** Write the CSV header row (called once per play session). */
    void WriteExportHeader();

    /** Write one data row for the current interpolated state.
     *  @param IP           The interpolated flight data point.
     *  @param VideoFrame   The MRQ output frame index (used as frame_id).
     *  @param VideoTimeS   The video timeline time in seconds. */
    void WriteExportRow(const FGliderDataPoint& IP, int32 VideoFrame, float VideoTimeS);
};