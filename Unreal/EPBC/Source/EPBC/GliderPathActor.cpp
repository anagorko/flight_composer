#include "GliderPathActor.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "KismetProceduralMeshLibrary.h"
#include "Engine/World.h"
#include "Materials/MaterialInstanceDynamic.h"


AGliderPathActor::AGliderPathActor()
{
    PrimaryActorTick.bCanEverTick = true;

    // 1. Create a lightweight root component (so you have a gizmo to move around)
    USceneComponent* SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = SceneRoot;

    // 2. Create the procedural mesh and attach it to the root
    ProceduralMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("ProceduralMesh"));
    ProceduralMesh->SetupAttachment(RootComponent);

    ProceduralMesh->bUseAsyncCooking = true;

    // This stops the editor from freezing to calculate the highlight outline
    ProceduralMesh->bSelectable = false;

    // Optional but recommended: disable collision if it's purely visual
    ProceduralMesh->bUseComplexAsSimpleCollision = false;
    ProceduralMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);

    // Initialize variables
    NextMeshSectionID = 0;
    LastUpdateTime = -1.0f;
    LastGeneratedPointIndex = 0;
    bDataLoaded = false;
    bCoordinatesConverted = false;
    bMeshBuilt = false;
    RibbonMaterial = nullptr;
    RibbonMID = nullptr;
}

void AGliderPathActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);

    // Auto-load data in editor if FlightTag is set
    if (!FlightTag.IsEmpty() && !bDataLoaded)
    {
        LoadFlightData();
    }

    // Force path update after loading data or when properties change in editor
    if (bDataLoaded && bCoordinatesConverted)
    {
        UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Forcing path update on construction"));
        LastUpdateTime = -1.0f;
        UpdatePathAtTime(CurrentAnimationTime);
    }
}

void AGliderPathActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);

    if (PropertyChangedEvent.Property)
    {
        FName PropertyName = PropertyChangedEvent.Property->GetFName();

        // Reload data when FlightTag, directory, or downsampling parameters change
        if (PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, FlightTag) ||
            PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, ActorDataDirectory) ||
            PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, DownsampleInterval) ||
            PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, SmoothingWindowFrames))
        {
            bDataLoaded = false;
            bCoordinatesConverted = false;
            bMeshBuilt = false;
            ClearPath();
            if (!FlightTag.IsEmpty())
            {
                LoadFlightData();
                if (bDataLoaded && bCoordinatesConverted)
                {
                    RebuildPath();
                }
            }
        }

        // Geometry changes (wingspan, scale) require a mesh rebuild
        if (PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, Wingspan) ||
            PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, MetresToUnrealScale))
        {
            if (bDataLoaded)
            {
                if (PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, MetresToUnrealScale))
                {
                    ConvertCoordinatesToWorld();
                }
                if (bCoordinatesConverted)
                {
                    bMeshBuilt = false;
                    RebuildPath();
                }
            }
        }

        // Material change: recreate the dynamic material instance
        if (PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, RibbonMaterial))
        {
            RibbonMID = nullptr; // Force re-creation on next assignment
            if (bMeshBuilt && RibbonMaterial)
            {
                RibbonMID = UMaterialInstanceDynamic::Create(RibbonMaterial, this);
                ProceduralMesh->SetMaterial(0, RibbonMID);
                // Push current parameters to the new MID
                RibbonMID->SetScalarParameterValue(FName("CurrentAnimationTime"), CurrentAnimationTime);
                RibbonMID->SetScalarParameterValue(FName("FadeStartDelay"), FadeStartDelay);
                RibbonMID->SetScalarParameterValue(FName("FadeDuration"), FadeDuration);
                RibbonMID->SetScalarParameterValue(FName("MinimumOpacity"), MinimumOpacity);
            }
        }

        // Animation time: just update the material scalar — no mesh rebuild needed
        if (PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, CurrentAnimationTime))
        {
            if (bDataLoaded && bCoordinatesConverted)
            {
                UpdatePathAtTime(CurrentAnimationTime);
            }
        }

        // Fade parameters: forward to material — no mesh rebuild needed
        if (PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, FadeStartDelay) ||
            PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, FadeDuration) ||
            PropertyName == GET_MEMBER_NAME_CHECKED(AGliderPathActor, MinimumOpacity))
        {
            if (RibbonMID)
            {
                RibbonMID->SetScalarParameterValue(FName("FadeStartDelay"), FadeStartDelay);
                RibbonMID->SetScalarParameterValue(FName("FadeDuration"), FadeDuration);
                RibbonMID->SetScalarParameterValue(FName("MinimumOpacity"), MinimumOpacity);
            }
        }
    }
}

void AGliderPathActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Auto-update path if enabled and data is loaded
    if (bAutoUpdatePath && bDataLoaded && bCoordinatesConverted)
    {
        UpdatePathAtTime(CurrentAnimationTime);
    }
}

bool AGliderPathActor::ShouldTickIfViewportsOnly() const
{
    return true;
}

// ---------------------------------------------------------------------------
// File path construction
// ---------------------------------------------------------------------------

FString AGliderPathActor::BuildFlightDataPath() const
{
    // Naming convention: {FlightTag}_trajectory_df.csv
    FString FileName = FString::Printf(TEXT("%s_trajectory_df.csv"), *FlightTag);

    // ActorDataDirectory is relative to the Unreal project root
    FString BasePath = FPaths::Combine(FPaths::ProjectDir(), ActorDataDirectory);
    FString FullPath = FPaths::Combine(BasePath, FileName);
    FPaths::NormalizeFilename(FullPath);
    FullPath = FPaths::ConvertRelativePathToFull(FullPath);
    return FullPath;
}

// ---------------------------------------------------------------------------
// CSV header map
// ---------------------------------------------------------------------------

TMap<FString, int32> AGliderPathActor::BuildHeaderMap(const FString& HeaderLine)
{
    TMap<FString, int32> Map;
    TArray<FString> Headers = ParseCSVLine(HeaderLine);
    for (int32 i = 0; i < Headers.Num(); i++)
    {
        FString H = Headers[i].TrimStartAndEnd().ToLower();
        if (!H.IsEmpty())
        {
            Map.Add(H, i);
        }
    }
    return Map;
}

// ---------------------------------------------------------------------------
// Load flight data
// ---------------------------------------------------------------------------

void AGliderPathActor::LoadFlightData()
{
    if (FlightTag.IsEmpty())
    {
        UE_LOG(LogTemp, Warning, TEXT("GliderPathActor: No FlightTag specified"));
        return;
    }

    FString FullPath = BuildFlightDataPath();

    if (!FPaths::FileExists(FullPath))
    {
        UE_LOG(LogTemp, Error, TEXT("GliderPathActor: Flight data file does not exist: %s"), *FullPath);
        return;
    }

    FString FileContent;
    if (!FFileHelper::LoadFileToString(FileContent, *FullPath))
    {
        UE_LOG(LogTemp, Error, TEXT("GliderPathActor: Failed to read flight data file: %s"), *FullPath);
        return;
    }

    // Clear existing data
    FlightData.Empty();
    bDataLoaded = false;
    bCoordinatesConverted = false;

    // Split into lines
    TArray<FString> Lines;
    FileContent.ParseIntoArrayLines(Lines);

    if (Lines.Num() < 2) // header + at least one data row
    {
        UE_LOG(LogTemp, Error, TEXT("GliderPathActor: File has insufficient data lines"));
        return;
    }

    // ---- Build column-name → index map from the header row ----
    TMap<FString, int32> ColMap = BuildHeaderMap(Lines[0]);

    // Locate required columns
    const int32* pTime  = ColMap.Find(TEXT("timestamp_s"));
    const int32* pX     = ColMap.Find(TEXT("x_m"));
    const int32* pY     = ColMap.Find(TEXT("y_m"));
    const int32* pZ     = ColMap.Find(TEXT("z_m"));
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

    // Validate that the essential columns exist
    if (!pTime || !pX || !pY || !pZ)
    {
        UE_LOG(LogTemp, Error, TEXT("GliderPathActor: CSV is missing required columns (timestamp_s, x_m, y_m, z_m)"));
        return;
    }

    // ---- Parse data rows ----
    for (int32 i = 1; i < Lines.Num(); i++)
    {
        if (Lines[i].TrimStartAndEnd().IsEmpty())
        {
            continue;
        }

        TArray<FString> Fields = ParseCSVLine(Lines[i]);

        // Ensure we have enough fields for at least the position columns
        int32 MinRequired = FMath::Max3(*pTime, FMath::Max3(*pX, *pY, *pZ), 0) + 1;
        if (Fields.Num() < MinRequired)
        {
            UE_LOG(LogTemp, Warning, TEXT("GliderPathActor: Skipping malformed line %d (got %d fields, need %d)"),
                   i, Fields.Num(), MinRequired);
            continue;
        }

        FGliderDataPoint Pt;

        // Helper lambda — safely read a float from Fields at the given column index
        auto ReadFloat = [&Fields](const int32* ColIdx, float Default) -> float
        {
            if (ColIdx && *ColIdx < Fields.Num())
            {
                return FCString::Atof(*Fields[*ColIdx]);
            }
            return Default;
        };

        Pt.Time    = ReadFloat(pTime,  0.0f);
        Pt.X_m     = ReadFloat(pX,     0.0f);
        Pt.Y_m     = ReadFloat(pY,     0.0f);
        Pt.Z_m     = ReadFloat(pZ,     0.0f);
        Pt.Vx      = ReadFloat(pVx,    0.0f);
        Pt.Vy      = ReadFloat(pVy,    0.0f);
        Pt.Vz      = ReadFloat(pVz,    0.0f);
        Pt.VMag    = ReadFloat(pVMag,  0.0f);
        Pt.Ax      = ReadFloat(pAx,    0.0f);
        Pt.Ay      = ReadFloat(pAy,    0.0f);
        Pt.Az      = ReadFloat(pAz,    0.0f);
        Pt.LoadG   = ReadFloat(pLoadG, 1.0f);
        Pt.YawRad  = ReadFloat(pYaw,   0.0f);
        Pt.PitchRad= ReadFloat(pPitch, 0.0f);
        Pt.RollRad = ReadFloat(pRoll,  0.0f);

        if (pPhase && *pPhase < Fields.Num())
        {
            Pt.Phase = Fields[*pPhase].TrimStartAndEnd();
        }

        FlightData.Add(Pt);
    }

    if (FlightData.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("GliderPathActor: No valid data rows parsed from %s"), *FullPath);
        return;
    }

    bDataLoaded = true;

    UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Loaded %d raw flight data points from %s"),
           FlightData.Num(), *FlightTag);

    // Downsample & smooth the raw 60 Hz data before building the ribbon
    DownsampleAndSmooth();

    // Convert ENU metres to Unreal world positions
    ConvertCoordinatesToWorld();

    // Auto-set animation time to show the full path (Only in Editor, NOT during Render/Play)
    if (FlightData.Num() > 0)
    {
        // Check if we are safely in the editor viewport
        if (GetWorld() && !GetWorld()->IsGameWorld())
        {
            CurrentAnimationTime = FlightData.Last().Time;
            UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Auto-set animation time to %f"), CurrentAnimationTime);
        }
        else
        {
            UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Render/Play mode detected. Leaving CurrentAnimationTime at %f"), CurrentAnimationTime);
        }
    }
}

// ---------------------------------------------------------------------------
// Downsample & smooth — reduce 60 Hz raw data to ~1 Hz with rolling average
// ---------------------------------------------------------------------------

void AGliderPathActor::DownsampleAndSmooth()
{
    if (!bDataLoaded || FlightData.Num() < 2)
    {
        return;
    }

    // Skip if disabled
    if (DownsampleInterval <= 0.0f)
    {
        UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Downsampling disabled (interval=0)"));
        return;
    }

    const int32 RawCount = FlightData.Num();
    const float StartTime = FlightData[0].Time;
    const float EndTime   = FlightData.Last().Time;

    // Pre-compute a quick lookup: for a given time, find the closest raw-frame index.
    // Because the data is sorted by time we can walk a cursor forward.

    TArray<FGliderDataPoint> Downsampled;
    int32 Cursor = 0; // walks forward through FlightData

    for (float T = StartTime; T <= EndTime + SMALL_NUMBER; T += DownsampleInterval)
    {
        // Advance cursor to the frame closest to T
        while (Cursor < RawCount - 1 &&
               FMath::Abs(FlightData[Cursor + 1].Time - T) <= FMath::Abs(FlightData[Cursor].Time - T))
        {
            ++Cursor;
        }

        // Window bounds in raw-frame indices
        const int32 WinLo = FMath::Max(0, Cursor - SmoothingWindowFrames);
        const int32 WinHi = FMath::Min(RawCount - 1, Cursor + SmoothingWindowFrames);
        const int32 WinCount = WinHi - WinLo + 1;

        // Accumulators (linear quantities)
        double SumTime = 0.0, SumX = 0.0, SumY = 0.0, SumZ = 0.0;
        double SumVx = 0.0, SumVy = 0.0, SumVz = 0.0, SumVMag = 0.0;
        double SumAx = 0.0, SumAy = 0.0, SumAz = 0.0, SumLoadG = 0.0;

        // Circular-mean accumulators for angles (sin / cos)
        double SinYaw = 0.0, CosYaw = 0.0;
        double SinPitch = 0.0, CosPitch = 0.0;
        double SinRoll = 0.0, CosRoll = 0.0;

        for (int32 J = WinLo; J <= WinHi; ++J)
        {
            const FGliderDataPoint& P = FlightData[J];

            SumTime += P.Time;
            SumX    += P.X_m;
            SumY    += P.Y_m;
            SumZ    += P.Z_m;
            SumVx   += P.Vx;
            SumVy   += P.Vy;
            SumVz   += P.Vz;
            SumVMag += P.VMag;
            SumAx   += P.Ax;
            SumAy   += P.Ay;
            SumAz   += P.Az;
            SumLoadG+= P.LoadG;

            SinYaw   += FMath::Sin(P.YawRad);
            CosYaw   += FMath::Cos(P.YawRad);
            SinPitch += FMath::Sin(P.PitchRad);
            CosPitch += FMath::Cos(P.PitchRad);
            SinRoll  += FMath::Sin(P.RollRad);
            CosRoll  += FMath::Cos(P.RollRad);
        }

        const double Inv = 1.0 / WinCount;

        FGliderDataPoint Avg;
        Avg.Time     = static_cast<float>(SumTime * Inv);
        Avg.X_m      = static_cast<float>(SumX * Inv);
        Avg.Y_m      = static_cast<float>(SumY * Inv);
        Avg.Z_m      = static_cast<float>(SumZ * Inv);
        Avg.Vx       = static_cast<float>(SumVx * Inv);
        Avg.Vy       = static_cast<float>(SumVy * Inv);
        Avg.Vz       = static_cast<float>(SumVz * Inv);
        Avg.VMag     = static_cast<float>(SumVMag * Inv);
        Avg.Ax       = static_cast<float>(SumAx * Inv);
        Avg.Ay       = static_cast<float>(SumAy * Inv);
        Avg.Az       = static_cast<float>(SumAz * Inv);
        Avg.LoadG    = static_cast<float>(SumLoadG * Inv);

        // Circular means for angles
        Avg.YawRad   = FMath::Atan2(static_cast<float>(SinYaw),   static_cast<float>(CosYaw));
        Avg.PitchRad = FMath::Atan2(static_cast<float>(SinPitch), static_cast<float>(CosPitch));
        Avg.RollRad  = FMath::Atan2(static_cast<float>(SinRoll),  static_cast<float>(CosRoll));

        // Phase label from the centre frame
        Avg.Phase = FlightData[Cursor].Phase;

        Downsampled.Add(Avg);
    }

    UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Downsampled %d → %d points (interval=%.2fs, window=±%d frames)"),
           RawCount, Downsampled.Num(), DownsampleInterval, SmoothingWindowFrames);

    // Replace the raw data with the downsampled series
    FlightData = MoveTemp(Downsampled);
}

// ---------------------------------------------------------------------------
// Coordinate conversion — ENU metres to Unreal world position
// ---------------------------------------------------------------------------

void AGliderPathActor::ConvertCoordinatesToWorld()
{
    if (!bDataLoaded || FlightData.Num() == 0) return;

    const float Scale = MetresToUnrealScale;

    for (FGliderDataPoint& Point : FlightData)
    {
        // Convert Position
        Point.WorldPosition = FVector(
             Point.X_m * Scale,  // East  -> UE +X
            -Point.Y_m * Scale,  // North -> UE -Y
             Point.Z_m * Scale   // Up    -> UE +Z
        );

        // Convert Velocity so it matches the path direction!
        Point.Vy = -Point.Vy;

        // (Optional: If you ever use acceleration in C++, negate Point.Ay here too)
        Point.Ay = -Point.Ay;
    }

    bCoordinatesConverted = true;
    UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Converted %d points to Unreal coordinates"), FlightData.Num());
    LogFlightDataBounds();
}

// ---------------------------------------------------------------------------
// Path update
// ---------------------------------------------------------------------------

void AGliderPathActor::UpdatePathAtTime(float AnimationTime)
{
    if (!bDataLoaded || !bCoordinatesConverted || FlightData.Num() < 2)
    {
        return;
    }

    // Build the mesh once (the entire flight path)
    if (!bMeshBuilt)
    {
        GenerateRibbonSegments();
    }

    // Avoid redundant material updates
    if (FMath::IsNearlyEqual(AnimationTime, LastUpdateTime, 0.001f))
    {
        return;
    }

    LastUpdateTime = AnimationTime;
    CurrentAnimationTime = AnimationTime;

    // Push current time to the material — the shader handles all visibility
    if (RibbonMID)
    {
        RibbonMID->SetScalarParameterValue(FName("CurrentAnimationTime"), CurrentAnimationTime);
    }
}

void AGliderPathActor::GenerateRibbonSegments()
{
    if (FlightData.Num() < 2)
    {
        UE_LOG(LogTemp, Warning, TEXT("GliderPathActor: Not enough flight data points (%d)"), FlightData.Num());
        return;
    }

    // Wipe any existing mesh
    ProceduralMesh->ClearAllMeshSections();
    ActiveMeshSections.Empty();
    NextMeshSectionID = 0;

    // Build the entire flight path at once
    CreateContinuousRibbonMesh(FlightData);
    ActiveMeshSections.Add(NextMeshSectionID);
    NextMeshSectionID++;
    bMeshBuilt = true;

    // Create / assign the dynamic material instance
    if (RibbonMaterial && !RibbonMID)
    {
        RibbonMID = UMaterialInstanceDynamic::Create(RibbonMaterial, this);
    }
    if (RibbonMID)
    {
        ProceduralMesh->SetMaterial(0, RibbonMID);
        // Push all relevant parameters to the material
        RibbonMID->SetScalarParameterValue(FName("CurrentAnimationTime"), CurrentAnimationTime);
        RibbonMID->SetScalarParameterValue(FName("FadeStartDelay"), FadeStartDelay);
        RibbonMID->SetScalarParameterValue(FName("FadeDuration"), FadeDuration);
        RibbonMID->SetScalarParameterValue(FName("MinimumOpacity"), MinimumOpacity);
    }
}

// ---------------------------------------------------------------------------
// Ribbon mesh generation
// ---------------------------------------------------------------------------

void AGliderPathActor::CreateContinuousRibbonMesh(const TArray<FGliderDataPoint>& Points)
{
    if (Points.Num() < 2)
    {
        return;
    }

    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UVs;
    TArray<FProcMeshTangent> Tangents;

    const float HalfWidth = Wingspan * 0.5f;

    // Generate vertices for each point
    for (int32 i = 0; i < Points.Num(); i++)
    {
        const FGliderDataPoint& Point = Points[i];

        // --- Direction from velocity vector (preferred) or position difference (fallback) ---
        FVector VelocityDir(Point.Vx, Point.Vy, Point.Vz);
        FVector Direction;

        if (VelocityDir.SizeSquared() > 0.01f)
        {
            // Use velocity vector — already in ENU, convert to Unreal axes (same mapping as position)
            Direction = FVector(VelocityDir.X, VelocityDir.Y, VelocityDir.Z).GetSafeNormal();
        }
        else
        {
            // Fallback: derive direction from neighbouring positions
            if (i == 0 && Points.Num() > 1)
            {
                Direction = (Points[1].WorldPosition - Point.WorldPosition).GetSafeNormal();
            }
            else if (i == Points.Num() - 1)
            {
                Direction = (Point.WorldPosition - Points[i - 1].WorldPosition).GetSafeNormal();
            }
            else
            {
                FVector Forward  = (Points[i + 1].WorldPosition - Point.WorldPosition).GetSafeNormal();
                FVector Backward = (Point.WorldPosition - Points[i - 1].WorldPosition).GetSafeNormal();
                Direction = (Forward + Backward).GetSafeNormal();
            }
        }

        if (Direction.IsNearlyZero())
        {
            Direction = FVector::ForwardVector;
        }

        // Calculate right vector considering bank (roll) angle — already in radians
        FVector Right = CalculateRibbonRight(Direction, Point.RollRad);

        // Create ribbon vertices
        FVector LeftVertex  = Point.WorldPosition - (Right * HalfWidth);
        FVector RightVertex = Point.WorldPosition + (Right * HalfWidth);

        Vertices.Add(LeftVertex);
        Vertices.Add(RightVertex);

        // Normal perpendicular to ribbon surface
        FVector Normal = FVector::CrossProduct(Direction, Right).GetSafeNormal();
        Normals.Add(Normal);
        Normals.Add(Normal);

        // Tangents
        Tangents.Add(FProcMeshTangent(Direction, false));
        Tangents.Add(FProcMeshTangent(Direction, false));

        // UVs: U = Absolute Time, V = Wing position (0=left, 1=right)
        UVs.Add(FVector2D(Point.Time, 0.0f)); // Left wingtip
        UVs.Add(FVector2D(Point.Time, 1.0f)); // Right wingtip
    }

    // Generate triangles
    for (int32 i = 0; i < Points.Num() - 1; i++)
    {
        int32 BottomLeft  = i * 2;
        int32 BottomRight = i * 2 + 1;
        int32 TopLeft     = (i + 1) * 2;
        int32 TopRight    = (i + 1) * 2 + 1;

        // First triangle
        Triangles.Add(BottomLeft);
        Triangles.Add(TopLeft);
        Triangles.Add(BottomRight);

        // Second triangle
        Triangles.Add(BottomRight);
        Triangles.Add(TopLeft);
        Triangles.Add(TopRight);
    }

    if (Vertices.Num() > 0)
    {
        // Debug bounds
        FVector MinVertex = Vertices[0];
        FVector MaxVertex = Vertices[0];
        for (const FVector& Vertex : Vertices)
        {
            MinVertex = FVector::Min(MinVertex, Vertex);
            MaxVertex = FVector::Max(MaxVertex, Vertex);
        }
        UE_LOG(LogTemp, Verbose, TEXT("GliderPathActor: Ribbon mesh — %d verts, %d tris, bounds [%s] → [%s]"),
               Vertices.Num(), Triangles.Num() / 3, *MinVertex.ToString(), *MaxVertex.ToString());
    }

    // Create the mesh section (no vertex colors — using UV-based time encoding)
    ProceduralMesh->CreateMeshSection(
        0,
        Vertices,
        Triangles,
        Normals,
        UVs,
        TArray<FColor>(),
        Tangents,
        true
    );
}

// ---------------------------------------------------------------------------
// Ribbon orientation helpers
// ---------------------------------------------------------------------------

FVector AGliderPathActor::CalculateRibbonRight(const FVector& Direction, float RollRad)
{
    // 1. Get the base right vector (Up cross Forward = Right in Unreal)
    FVector BaseUp = FVector::UpVector;
    FVector BaseRight = FVector::CrossProduct(BaseUp, Direction).GetSafeNormal();

    // Gimbal lock prevention (if flying perfectly vertical)
    if (BaseRight.SizeSquared() < 0.01f)
    {
        BaseRight = FVector::RightVector;
    }

    // 2. Convert radians to degrees
    // We negate the angle because UE positive roll rotates the right wing UP.
    // We want positive roll to push the right wing DOWN.
    float RollDegrees = FMath::RadiansToDegrees(-RollRad);

    // 3. Rotate the base right vector around the forward direction by the roll angle
    FVector Right = BaseRight.RotateAngleAxis(RollDegrees, Direction);

    return Right.GetSafeNormal();
}

// ---------------------------------------------------------------------------
// Clear / Rebuild
// ---------------------------------------------------------------------------

void AGliderPathActor::ClearPath()
{
    if (ProceduralMesh)
    {
        ProceduralMesh->ClearAllMeshSections();
    }

    ActiveMeshSections.Empty();
    NextMeshSectionID = 0;
    LastUpdateTime = -1.0f;
    LastGeneratedPointIndex = 0;
    bMeshBuilt = false;
    RibbonMID = nullptr;

    UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Path cleared"));
}

void AGliderPathActor::RebuildPath()
{
    if (bDataLoaded && bCoordinatesConverted)
    {
        bMeshBuilt = false;
        RibbonMID = nullptr;
        LastUpdateTime = -1.0f;
        UpdatePathAtTime(CurrentAnimationTime);
        UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Path rebuilt"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("GliderPathActor: Cannot rebuild path — data not loaded or converted"));
    }
}

// ---------------------------------------------------------------------------
// CSV parsing helper
// ---------------------------------------------------------------------------

TArray<FString> AGliderPathActor::ParseCSVLine(const FString& Line)
{
    TArray<FString> Fields;
    FString CurrentField;
    bool bInQuotes = false;

    for (int32 i = 0; i < Line.Len(); i++)
    {
        TCHAR Char = Line[i];

        if (Char == '"')
        {
            bInQuotes = !bInQuotes;
        }
        else if (Char == ',' && !bInQuotes)
        {
            Fields.Add(CurrentField.TrimStartAndEnd());
            CurrentField.Empty();
        }
        else
        {
            CurrentField.AppendChar(Char);
        }
    }

    // Add the last field
    Fields.Add(CurrentField.TrimStartAndEnd());

    return Fields;
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

void AGliderPathActor::LogFlightDataBounds()
{
    if (FlightData.Num() == 0)
    {
        return;
    }

    FVector MinPos = FlightData[0].WorldPosition;
    FVector MaxPos = FlightData[0].WorldPosition;

    for (const FGliderDataPoint& Point : FlightData)
    {
        MinPos = FVector::Min(MinPos, Point.WorldPosition);
        MaxPos = FVector::Max(MaxPos, Point.WorldPosition);
    }

    FVector Size   = MaxPos - MinPos;
    FVector Center = (MinPos + MaxPos) * 0.5f;

    UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Path bounds — Min: %s, Max: %s"),
           *MinPos.ToString(), *MaxPos.ToString());
    UE_LOG(LogTemp, Log, TEXT("GliderPathActor: Path size: %s, Center: %s"),
           *Size.ToString(), *Center.ToString());
}

void AGliderPathActor::SetCurrentAnimationTime(float NewTime)
{
    CurrentAnimationTime = NewTime;

    if (!bDataLoaded || FlightData.Num() == 0)
    {
        LoadFlightData();
    }

    if (bDataLoaded)
    {
        UpdatePathAtTime(CurrentAnimationTime);
    }
}
