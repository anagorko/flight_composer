#include "GliderPlaybackActor.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Math/Quat.h"
#include "Math/RotationMatrix.h"
#include "UObject/UObjectIterator.h"
#include "MoviePipeline.h"
#include "MoviePipelineBlueprintLibrary.h"
#include "MoviePipelinePrimaryConfig.h"
#include "MoviePipelineOutputSetting.h"
#include "MoviePipelineQueue.h"
#include "MovieSceneSequence.h"
#include "MovieScene.h"


AGliderPlaybackActor::AGliderPlaybackActor()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostUpdateWork;

    // 1. Root Component (Handles your manual world placement)
    SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
    RootComponent = SceneRoot;

    // 2. Kinematic Anchor (Follows the math)
    KinematicAnchor = CreateDefaultSubobject<USceneComponent>(TEXT("KinematicAnchor"));
    KinematicAnchor->SetupAttachment(RootComponent);

    // 3. Glider Mesh (Follows the Anchor, allows manual offset)
    GliderMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("GliderMesh"));
    GliderMesh->SetupAttachment(KinematicAnchor);
    GliderMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);

    bDataLoaded = false;
}

void AGliderPlaybackActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);

    if (!FlightTag.IsEmpty() && !bDataLoaded)
    {
        LoadFlightData();
    }

    if (bDataLoaded)
    {
        UpdateGliderTransform(CurrentAnimationTime);
    }
}

void AGliderPlaybackActor::BeginPlay()
{
    Super::BeginPlay();

    // Reset export state at the start of every play / render session
    ExportFrameId = 0;
    LastExportedVideoFrame = -1;
    bExportHeaderWritten = false;
    bInterpPointReady = false;
    ExportFilePath.Empty();

    // Capture the engine frame on which BeginPlay fires.
    // Sequencer's first evaluation on this same frame carries the stale editor
    // playhead value, so we must refuse to export until GFrameCounter advances
    // past this point.
    BeginPlayFrameCounter = GFrameCounter;

    UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: === BeginPlay === Actor=%s  bExportOverlayData=%s  bDataLoaded=%s  FlightData=%d  HasBegunPlay=%s  BeginPlayFrame=%llu  ProjectDir=%s"),
        *FlightTag,
        *GetPathName(),
        bExportOverlayData ? TEXT("TRUE") : TEXT("FALSE"),
        bDataLoaded ? TEXT("TRUE") : TEXT("FALSE"),
        FlightData.Num(),
        (GetWorld() && GetWorld()->HasBegunPlay()) ? TEXT("TRUE") : TEXT("FALSE"),
        BeginPlayFrameCounter,
        *FPaths::ProjectDir());

    if (bExportOverlayData)
    {
        FString PreviewPath = BuildExportFilePath();
        UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: Export ENABLED – target path: %s"), *FlightTag, *PreviewPath);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("======================================================================"));
        UE_LOG(LogTemp, Error, TEXT("GliderExport [%s]: bExportOverlayData is FALSE – CSV export is DISABLED!"), *FlightTag);
        UE_LOG(LogTemp, Error, TEXT("GliderExport [%s]: To enable, check 'Export Overlay Data' in the Details panel under 'Flight Data' and SAVE THE LEVEL before rendering."), *FlightTag);
        UE_LOG(LogTemp, Error, TEXT("GliderExport [%s]: Actor: %s"), *FlightTag, *GetPathName());
        UE_LOG(LogTemp, Error, TEXT("======================================================================"));
    }
}

void AGliderPlaybackActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: === EndPlay === reason=%d  bExportHeaderWritten=%s  ExportFrameId=%d  LastExportedVideoFrame=%d"),
        *FlightTag, (int32)EndPlayReason, bExportHeaderWritten ? TEXT("TRUE") : TEXT("FALSE"), ExportFrameId, LastExportedVideoFrame);

    if (bExportHeaderWritten)
    {
        UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: Export finished – %d frames written to %s"),
            *FlightTag, ExportFrameId, *ExportFilePath);
    }
    else if (bExportOverlayData)
    {
        UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: Export was ENABLED but header was never written – no frames exported!"), *FlightTag);
    }

    Super::EndPlay(EndPlayReason);
}

void AGliderPlaybackActor::SetCurrentAnimationTime(float NewTime)
{
    CurrentAnimationTime = NewTime;

    // Log occasionally (every ~60 calls) to avoid spam but still show we're being driven
    static uint64 SetTimeCallCount = 0;
    SetTimeCallCount++;
    if (SetTimeCallCount <= 3 || SetTimeCallCount % 60 == 0)
    {
        UE_LOG(LogTemp, Log, TEXT("GliderExport [%s]: SetCurrentAnimationTime(%.4f) call #%llu  bDataLoaded=%s  HasBegunPlay=%s  GFrameCounter=%llu"),
            *FlightTag, NewTime, SetTimeCallCount,
            bDataLoaded ? TEXT("TRUE") : TEXT("FALSE"),
            (GetWorld() && GetWorld()->HasBegunPlay()) ? TEXT("TRUE") : TEXT("FALSE"),
            GFrameCounter);
    }

    // THE FIX: If MRQ just started a new session and RAM is empty, force a load!
    if (!bDataLoaded || FlightData.Num() == 0)
    {
        LoadFlightData();
    }

    // Now it is guaranteed to be loaded during the render
    if (bDataLoaded)
    {
        UpdateGliderTransform(CurrentAnimationTime);
    }
}

#if WITH_EDITOR
void AGliderPlaybackActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);

    if (PropertyChangedEvent.Property)
    {
        FName PropName = PropertyChangedEvent.Property->GetFName();

        if (PropName == GET_MEMBER_NAME_CHECKED(AGliderPlaybackActor, FlightTag) ||
            PropName == GET_MEMBER_NAME_CHECKED(AGliderPlaybackActor, ActorDataDirectory) ||
            PropName == GET_MEMBER_NAME_CHECKED(AGliderPlaybackActor, MetresToUnrealScale))
        {
            LoadFlightData();
        }

        // Scrub the timeline directly in the editor!
        if (PropName == GET_MEMBER_NAME_CHECKED(AGliderPlaybackActor, CurrentAnimationTime))
        {
            UpdateGliderTransform(CurrentAnimationTime);
        }
    }
}
#endif

void AGliderPlaybackActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // If you are driving this via Blueprints or Sequencer during play
    if (bDataLoaded)
    {
        UpdateGliderTransform(CurrentAnimationTime);
    }

    // ---- CSV Export via MRQ Pipeline query ----
    if (bExportOverlayData && bInterpPointReady && GetWorld() && GetWorld()->HasBegunPlay())
    {
        // Skip the initialization frame where Sequencer evaluates the stale editor playhead time
        if (GFrameCounter <= BeginPlayFrameCounter)
        {
            return;
        }

        UMoviePipeline* ActivePipeline = nullptr;

        // Find the currently active MRQ Pipeline in our world
        for (TObjectIterator<UMoviePipeline> It; It; ++It)
        {
            if (It->GetWorld() == GetWorld())
            {
                ActivePipeline = *It;
                break;
            }
        }

        if (ActivePipeline)
        {
            int32 CurrentMRQFrame = 0;
            int32 TotalFrames = 0;

            // Query MRQ directly for the exact output frame index.
            // MRQ holds CurrentMRQFrame constant during TAA sub-samples,
            // giving us perfect deduplication for free.
            UMoviePipelineBlueprintLibrary::GetOverallOutputFrames(ActivePipeline, CurrentMRQFrame, TotalFrames);

            if (CurrentMRQFrame != LastExportedVideoFrame)
            {
                if (!bExportHeaderWritten)
                {
                    UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: First export frame! MRQFrame=%d  TotalFrames=%d  Writing header now."),
                        *FlightTag, CurrentMRQFrame, TotalFrames);
                    WriteExportHeader();
                }

                // Read the effective output FPS directly from MRQ's pipeline config.
                // If the user set a custom frame rate in the Output Setting, use that.
                // Otherwise fall back to the sequence's display rate via UMovieScene.
                float EffectiveFPS = 24.0f; // safe default fallback
                if (UMoviePipelinePrimaryConfig* PrimaryConfig = ActivePipeline->GetPipelinePrimaryConfig())
                {
                    UMoviePipelineOutputSetting* OutputSettings = PrimaryConfig->FindSetting<UMoviePipelineOutputSetting>();
                    if (OutputSettings && OutputSettings->bUseCustomFrameRate)
                    {
                        EffectiveFPS = static_cast<float>(OutputSettings->OutputFrameRate.AsDecimal());
                    }
                    else
                    {
                        // No custom frame rate – read the sequence's display rate via UMovieSceneSequence.
                        // Cast to UMovieSceneSequence (base class in the MovieScene module) to avoid
                        // linking against the LevelSequence module.
                        UMoviePipelineExecutorJob* CurrentJob = ActivePipeline->GetCurrentJob();
                        if (CurrentJob)
                        {
                            UMovieSceneSequence* Sequence = Cast<UMovieSceneSequence>(CurrentJob->Sequence.TryLoad());
                            if (Sequence)
                            {
                                UMovieScene* MovieScene = Sequence->GetMovieScene();
                                if (MovieScene)
                                {
                                    EffectiveFPS = static_cast<float>(MovieScene->GetDisplayRate().AsDecimal());
                                }
                            }
                        }
                    }
                }

                float VideoTimeS = static_cast<float>(CurrentMRQFrame) / EffectiveFPS;

                WriteExportRow(LastInterpPoint, CurrentMRQFrame, VideoTimeS);
                LastExportedVideoFrame = CurrentMRQFrame;

                if (ExportFrameId <= 5 || ExportFrameId % 100 == 0)
                {
                    UE_LOG(LogTemp, Log, TEXT("GliderExport [%s]: Wrote frame_id=%04d  MRQFrame=%d/%d  video_t=%.4f  flight_t=%.4f  EffectiveFPS=%.2f"),
                        *FlightTag, CurrentMRQFrame, CurrentMRQFrame, TotalFrames, VideoTimeS, LastInterpPoint.Time, EffectiveFPS);
                }
            }
        }
    }
}

bool AGliderPlaybackActor::ShouldTickIfViewportsOnly() const
{
    return true;
}

// ---------------------------------------------------------------------------
// Core Interpolation Logic
// ---------------------------------------------------------------------------

void AGliderPlaybackActor::UpdateGliderTransform(float Time)
{
    if (FlightData.Num() == 0) return;

    // Helper lambda: populate LastInterpPoint from a single data point (clamped edge case)
    auto CachePointAndApply = [this](const FGliderDataPoint& Pt)
    {
        LastInterpPoint = Pt;
        bInterpPointReady = true;
        KinematicAnchor->SetRelativeLocationAndRotation(Pt.WorldPosition, Pt.WorldRotation);
    };

    // Handle bounds (Clamp to start)
    if (Time <= FlightData[0].Time)
    {
        CachePointAndApply(FlightData[0]);
        return;
    }

    // Handle bounds (Clamp to end)
    int32 LastIdx = FlightData.Num() - 1;
    if (Time >= FlightData[LastIdx].Time)
    {
        CachePointAndApply(FlightData[LastIdx]);
        return;
    }

    // Binary Search to instantly find the two surrounding frames
    int32 Low = 0;
    int32 High = LastIdx;
    while (Low <= High)
    {
        int32 Mid = Low + (High - Low) / 2;
        if (FlightData[Mid].Time < Time) Low = Mid + 1;
        else High = Mid - 1;
    }

    // Now 'High' is the frame just before Time, 'Low' is the frame just after Time
    const FGliderDataPoint& PointA = FlightData[High];
    const FGliderDataPoint& PointB = FlightData[Low];

    // Calculate how far we are between Frame A and Frame B (0.0 to 1.0)
    float Alpha = (Time - PointA.Time) / FMath::Max(PointB.Time - PointA.Time, 0.0001f);

    // Lerp Position
    FVector InterpPos = FMath::Lerp(PointA.WorldPosition, PointB.WorldPosition, Alpha);

    // Slerp Rotation (Spherical Linear Interpolation prevents gimbal lock/twisting)
    FQuat InterpRot = FQuat::Slerp(PointA.WorldRotation, PointB.WorldRotation, Alpha);

    // Apply to the kinematic anchor!
    KinematicAnchor->SetRelativeLocationAndRotation(InterpPos, InterpRot);

    // ---- Build the interpolated data point for export ----
    FGliderDataPoint& IP = LastInterpPoint;
    IP.Time   = FMath::Lerp(PointA.Time,   PointB.Time,   Alpha);
    IP.X_m    = FMath::Lerp(PointA.X_m,    PointB.X_m,    Alpha);
    IP.Y_m    = FMath::Lerp(PointA.Y_m,    PointB.Y_m,    Alpha);
    IP.Z_m    = FMath::Lerp(PointA.Z_m,    PointB.Z_m,    Alpha);
    IP.Vx     = FMath::Lerp(PointA.Vx,     PointB.Vx,     Alpha);
    IP.Vy     = FMath::Lerp(PointA.Vy,     PointB.Vy,     Alpha);
    IP.Vz     = FMath::Lerp(PointA.Vz,     PointB.Vz,     Alpha);
    IP.VMag   = FMath::Lerp(PointA.VMag,   PointB.VMag,   Alpha);
    IP.Ax     = FMath::Lerp(PointA.Ax,     PointB.Ax,     Alpha);
    IP.Ay     = FMath::Lerp(PointA.Ay,     PointB.Ay,     Alpha);
    IP.Az     = FMath::Lerp(PointA.Az,     PointB.Az,     Alpha);
    IP.LoadG  = FMath::Lerp(PointA.LoadG,  PointB.LoadG,  Alpha);
    IP.YawRad = FMath::Lerp(PointA.YawRad, PointB.YawRad, Alpha);
    IP.PitchRad = FMath::Lerp(PointA.PitchRad, PointB.PitchRad, Alpha);
    IP.RollRad  = FMath::Lerp(PointA.RollRad,  PointB.RollRad,  Alpha);

    // Phase: nearest-neighbour (take from PointA – the earlier sample)
    IP.Phase = PointA.Phase;

    IP.WorldPosition = InterpPos;
    IP.WorldRotation = InterpRot;

    bInterpPointReady = true;
}

// ---------------------------------------------------------------------------
// Data Loading & Pre-calculation
// ---------------------------------------------------------------------------

void AGliderPlaybackActor::LoadFlightData()
{
    if (FlightTag.IsEmpty()) return;

    FString FullPath = BuildFlightDataPath();
    FString FileContent;
    if (!FFileHelper::LoadFileToString(FileContent, *FullPath)) return;

    FlightData.Empty();
    bDataLoaded = false;

    TArray<FString> Lines;
    FileContent.ParseIntoArrayLines(Lines);
    if (Lines.Num() < 2) return;

    TMap<FString, int32> ColMap = BuildHeaderMap(Lines[0]);

    // Grab indices – required columns
    const int32* pTime  = ColMap.Find(TEXT("timestamp_s"));
    const int32* pX     = ColMap.Find(TEXT("x_m"));
    const int32* pY     = ColMap.Find(TEXT("y_m"));
    const int32* pZ     = ColMap.Find(TEXT("z_m"));

    if (!pTime || !pX || !pY || !pZ) return;

    // Optional columns
    const int32* pVx    = ColMap.Find(TEXT("vx_ms"));
    const int32* pVy    = ColMap.Find(TEXT("vy_ms"));
    const int32* pVz    = ColMap.Find(TEXT("vz_ms"));
    const int32* pVMag  = ColMap.Find(TEXT("v_mag_ms"));
    const int32* pAx    = ColMap.Find(TEXT("ax_ms2"));
    const int32* pAy    = ColMap.Find(TEXT("ay_ms2"));
    const int32* pAz    = ColMap.Find(TEXT("az_ms2"));
    const int32* pLoadG = ColMap.Find(TEXT("load_g"));
    const int32* pYaw   = ColMap.Find(TEXT("yaw_rad"));
    const int32* pPitch = ColMap.Find(TEXT("pitch_rad"));
    const int32* pRoll  = ColMap.Find(TEXT("roll_rad"));
    const int32* pPhase = ColMap.Find(TEXT("phase"));

    const float Scale = MetresToUnrealScale;

    for (int32 i = 1; i < Lines.Num(); i++)
    {
        if (Lines[i].TrimStartAndEnd().IsEmpty()) continue;
        TArray<FString> Fields = ParseCSVLine(Lines[i]);
        if (Fields.Num() <= FMath::Max3(*pX, *pY, *pZ)) continue;

        FGliderDataPoint Pt;

        auto ReadFloat = [&Fields](const int32* ColIdx, float Default) -> float {
            if (ColIdx && *ColIdx < Fields.Num()) return FCString::Atof(*Fields[*ColIdx]);
            return Default;
        };

        auto ReadString = [&Fields](const int32* ColIdx, const FString& Default) -> FString {
            if (ColIdx && *ColIdx < Fields.Num()) return Fields[*ColIdx];
            return Default;
        };

        Pt.Time     = ReadFloat(pTime, 0.0f);
        Pt.X_m      = ReadFloat(pX, 0.0f);
        Pt.Y_m      = ReadFloat(pY, 0.0f);
        Pt.Z_m      = ReadFloat(pZ, 0.0f);
        Pt.Vx       = ReadFloat(pVx, 0.0f);
        Pt.Vy       = ReadFloat(pVy, 0.0f);
        Pt.Vz       = ReadFloat(pVz, 0.0f);
        Pt.VMag     = ReadFloat(pVMag, 0.0f);
        Pt.Ax       = ReadFloat(pAx, 0.0f);
        Pt.Ay       = ReadFloat(pAy, 0.0f);
        Pt.Az       = ReadFloat(pAz, 0.0f);
        Pt.LoadG    = ReadFloat(pLoadG, 1.0f);
        Pt.YawRad   = ReadFloat(pYaw, 0.0f);
        Pt.PitchRad = ReadFloat(pPitch, 0.0f);
        Pt.RollRad  = ReadFloat(pRoll, 0.0f);
        Pt.Phase    = ReadString(pPhase, TEXT(""));

        // 1. Convert Position (ENU -> Unreal)
        Pt.WorldPosition = FVector(Pt.X_m * Scale, -Pt.Y_m * Scale, Pt.Z_m * Scale);

        // 2. Pre-calculate Orientation Quaternion to match the ribbon perfectly
        FVector Forward(Pt.Vx, -Pt.Vy, Pt.Vz);
        if (Forward.IsNearlyZero()) Forward = FVector::ForwardVector;
        Forward.Normalize();

        FVector BaseUp = FVector::UpVector;
        FVector BaseRight = FVector::CrossProduct(BaseUp, Forward).GetSafeNormal();
        if (BaseRight.SizeSquared() < 0.01f) BaseRight = FVector::RightVector;

        // Apply Roll
        float RollDegrees = FMath::RadiansToDegrees(-Pt.RollRad);
        FVector Right = BaseRight.RotateAngleAxis(RollDegrees, Forward);

        // Build Rotation Matrix from Forward and Right axes, extract Quat
        Pt.WorldRotation = FRotationMatrix::MakeFromXY(Forward, Right).ToQuat();

        FlightData.Add(Pt);
    }

    bDataLoaded = FlightData.Num() > 0;

    // DEBUG: Log discrete velocity between consecutive points
    if (bDataLoaded && FlightData.Num() > 1)
    {
        UE_LOG(LogTemp, Log, TEXT("=== GliderPlayback [%s] Discrete-Velocity Dump (%d points) ==="),
            *FlightTag, FlightData.Num());

        for (int32 k = 1; k < FlightData.Num(); ++k)
        {
            float Dist = FVector::Dist(FlightData[k - 1].WorldPosition, FlightData[k].WorldPosition);
            float Dt = FlightData[k].Time - FlightData[k - 1].Time;
            float SpeedMs = (Dt > 0.0001f) ? (Dist / (Dt * MetresToUnrealScale)) : 0.0f;

            UE_LOG(LogTemp, Log, TEXT("  [%d->%d] t=%.3f->%.3f  dt=%.4f s  dist=%.5f m  speed=%.3f m/s"),
                k - 1, k,
                FlightData[k - 1].Time, FlightData[k].Time,
                Dt,
                Dist / MetresToUnrealScale,
                SpeedMs);
        }

        UE_LOG(LogTemp, Log, TEXT("=== End Discrete-Velocity Dump ==="));
    }
}

// ---------------------------------------------------------------------------
// CSV Export Helpers
// ---------------------------------------------------------------------------

FString AGliderPlaybackActor::BuildExportFilePath() const
{
    // Fallback to the FlightTag if MRQ is somehow not active
    FString OutputName = FlightTag; 

    // Find the currently active MRQ Pipeline
    for (TObjectIterator<UMoviePipeline> It; It; ++It)
    {
        if (It->GetWorld() == GetWorld())
        {
            if (UMoviePipelineExecutorJob* Job = It->GetCurrentJob())
            {
                if (!Job->JobName.IsEmpty())
                {
                    OutputName = Job->JobName;
                }
            }
            break;
        }
    }

    // Grab the Actor's ID/Label from the Outliner
    FString ActorName = GetActorNameOrLabel();

    // Place the file under <ProjectDir>/<SequenceDataDirectory>/<SequenceName>_<ActorName>.csv
    FString Dir = FPaths::Combine(FPaths::ProjectDir(), SequenceDataDirectory);
    FString FileName = FString::Printf(TEXT("%s_%s.csv"), *OutputName, *ActorName);
    return FPaths::ConvertRelativePathToFull(FPaths::Combine(Dir, FileName));
}

void AGliderPlaybackActor::WriteExportHeader()
{
    ExportFilePath = BuildExportFilePath();

    // Ensure the directory exists
    FString Dir = FPaths::GetPath(ExportFilePath);
    UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: Creating directory tree: %s"), *FlightTag, *Dir);

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    const bool bDirCreated = PlatformFile.CreateDirectoryTree(*Dir);
    UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: CreateDirectoryTree returned %s  DirectoryExists=%s"),
        *FlightTag,
        bDirCreated ? TEXT("TRUE") : TEXT("FALSE"),
        PlatformFile.DirectoryExists(*Dir) ? TEXT("TRUE") : TEXT("FALSE"));

    const FString Header = TEXT("frame_id,timestamp_s,video_time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,v_mag_ms,ax_ms2,ay_ms2,az_ms2,load_g,yaw_rad,pitch_rad,roll_rad,phase\n");

    // Overwrite any previous file from an earlier render
    const bool bHeaderWriteOk = FFileHelper::SaveStringToFile(Header, *ExportFilePath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);

    UE_LOG(LogTemp, Warning, TEXT("GliderExport [%s]: Header write to '%s' %s"),
        *FlightTag, *ExportFilePath, bHeaderWriteOk ? TEXT("SUCCEEDED") : TEXT("FAILED"));

    if (!bHeaderWriteOk)
    {
        UE_LOG(LogTemp, Error, TEXT("GliderExport [%s]: FAILED to write header! Check path permissions: %s"), *FlightTag, *ExportFilePath);
    }

    bExportHeaderWritten = bHeaderWriteOk;
}

void AGliderPlaybackActor::WriteExportRow(const FGliderDataPoint& IP, int32 VideoFrame, float VideoTimeS)
{
    // Use the MRQ output frame index as frame_id, zero-padded to 4 digits
    FString FrameIdStr = FString::Printf(TEXT("%04d"), VideoFrame);

    FString Row = FString::Printf(
        TEXT("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%s\n"),
        *FrameIdStr,
        IP.Time,
        VideoTimeS,
        IP.X_m,
        IP.Y_m,
        IP.Z_m,
        IP.Vx,
        IP.Vy,
        IP.Vz,
        IP.VMag,
        IP.Ax,
        IP.Ay,
        IP.Az,
        IP.LoadG,
        IP.YawRad,
        IP.PitchRad,
        IP.RollRad,
        *IP.Phase
    );

    const bool bRowWriteOk = FFileHelper::SaveStringToFile(Row, *ExportFilePath,
        FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM,
        &IFileManager::Get(),
        EFileWrite::FILEWRITE_Append);

    if (!bRowWriteOk)
    {
        UE_LOG(LogTemp, Error, TEXT("GliderExport [%s]: FAILED to append row frame_id=%s to %s"), *FlightTag, *FrameIdStr, *ExportFilePath);
    }

    ExportFrameId++;
}

// ---------------------------------------------------------------------------
// Helpers (Identical to AGliderPathActor)
// ---------------------------------------------------------------------------

FString AGliderPlaybackActor::BuildFlightDataPath() const
{
    FString FileName = FString::Printf(TEXT("%s_trajectory_df.csv"), *FlightTag);
    return FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectDir(), ActorDataDirectory, FileName));
}

TMap<FString, int32> AGliderPlaybackActor::BuildHeaderMap(const FString& HeaderLine)
{
    TMap<FString, int32> Map;
    TArray<FString> Headers = ParseCSVLine(HeaderLine);
    for (int32 i = 0; i < Headers.Num(); i++)
    {
        FString H = Headers[i].TrimStartAndEnd().ToLower();
        if (!H.IsEmpty()) Map.Add(H, i);
    }
    return Map;
}

TArray<FString> AGliderPlaybackActor::ParseCSVLine(const FString& Line)
{
    TArray<FString> Fields;
    FString CurrentField;
    bool bInQuotes = false;
    for (int32 i = 0; i < Line.Len(); i++)
    {
        TCHAR Char = Line[i];
        if (Char == '"') bInQuotes = !bInQuotes;
        else if (Char == ',' && !bInQuotes)
        {
            Fields.Add(CurrentField.TrimStartAndEnd());
            CurrentField.Empty();
        }
        else CurrentField.AppendChar(Char);
    }
    Fields.Add(CurrentField.TrimStartAndEnd());
    return Fields;
}