#include "TrajectoryTestActor.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInterface.h"
#include "UObject/ConstructorHelpers.h"
#include "Misc/FileHelper.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "CesiumGeoreference.h"
#include "CesiumGlobeAnchorComponent.h"

ATrajectoryTestActor::ATrajectoryTestActor()
{
    PrimaryActorTick.bCanEverTick = false;

    // Set default sphere mesh to engine's basic sphere
    static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMeshAsset(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    if (SphereMeshAsset.Succeeded())
    {
        SphereMesh = SphereMeshAsset.Object;
    }

    // Set default materials to engine's basic materials
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> RedMaterial(TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> BlueMaterial(TEXT("/Engine/EngineMaterials/DefaultMaterial.DefaultMaterial"));
    
    if (RedMaterial.Succeeded())
    {
        IGCMaterial = RedMaterial.Object;
        UnknownMaterial = RedMaterial.Object;
    }
    
    if (BlueMaterial.Succeeded())
    {
        GPXMaterial = BlueMaterial.Object;
        GeoJSONMaterial = BlueMaterial.Object;
    }

    // Initialize root component
    RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
}

void ATrajectoryTestActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    
    // Don't auto-load in construction to avoid performance issues
    // User should explicitly call LoadTrajectoryData()
}

void ATrajectoryTestActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);
    
    if (PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(ATrajectoryTestActor, FlightNumber))
    {
        // Clear data when flight number changes
        bDataLoaded = false;
        DataStatus = TEXT("Flight number changed - run Python export and click Load Trajectory Data");
    }
    
    if (PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(ATrajectoryTestActor, CoordinateScale) ||
        PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(ATrajectoryTestActor, SphereScale) ||
        PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(ATrajectoryTestActor, SkipFactor) ||
        PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(ATrajectoryTestActor, MaxSpheres))
    {
        // Refresh visualization when visual parameters change
        if (bDataLoaded)
        {
            RefreshTrajectory();
        }
    }
}

void ATrajectoryTestActor::LoadTrajectoryData()
{
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Loading trajectory data for flight %s"), *FlightNumber);
    
    // Clear existing data
    ClearSpheres();
    TrajectoryData.Empty();
    bDataLoaded = false;
    
    if (FlightNumber.IsEmpty())
    {
        DataStatus = TEXT("Error: Flight number is empty");
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Flight number is empty"));
        return;
    }
    
    // Load the pre-existing JSON data from flight_composer root directory
    FString ProjectDir = FPaths::ProjectDir();
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Project directory: %s"), *ProjectDir);
    
    // Navigate from Unreal/EPBC back to flight_composer root
    FString FlightComposerRoot = FPaths::Combine(ProjectDir, TEXT("../../"));
    FString ResolvedRoot = FPaths::ConvertRelativePathToFull(FlightComposerRoot);
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Flight composer root: %s"), *ResolvedRoot);
    
    FString JSONPath = FPaths::Combine(ResolvedRoot, TEXT("ProcessedData"), FString::Printf(TEXT("flight_%s%s"), *FlightNumber, *JSONFileSuffix));
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Looking for JSON at: %s"), *JSONPath);
    
    bool LoadSuccess = false;
    if (bUseCesiumCoordinates && JSONFileSuffix.Contains(TEXT("wgs84")))
    {
        LoadSuccess = LoadWGS84TrajectoryFromJSON(JSONPath);
    }
    else
    {
        LoadSuccess = LoadTrajectoryFromJSON(JSONPath);
    }
    
    if (!LoadSuccess)
    {
        DataStatus = FString::Printf(TEXT("Error: JSON file not found - run Python export first"));
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Failed to load JSON from %s"), *JSONPath);
        UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Run 'python test_trajectory_export.py %s' first to generate the JSON file"), *FlightNumber);
        return;
    }
    
    // Create spheres for visualization
    RefreshTrajectory();
    
    // Update status
    bDataLoaded = true;
    LastLoadedFlightNumber = FlightNumber;
    UpdateStatistics();
    
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Successfully loaded %d trajectory points"), TrajectoryData.Num());
}

void ATrajectoryTestActor::ClearSpheres()
{
    // Destroy all existing sphere components
    for (UStaticMeshComponent* Sphere : SphereComponents)
    {
        if (IsValid(Sphere))
        {
            Sphere->DestroyComponent();
        }
    }
    SphereComponents.Empty();
    
    UE_LOG(LogTemp, Log, TEXT("TrajectoryTestActor: Cleared all spheres"));
}

void ATrajectoryTestActor::RefreshTrajectory()
{
    if (!bDataLoaded || TrajectoryData.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: No data to refresh"));
        return;
    }
    
    // Clear existing spheres
    ClearSpheres();
    
    // Create new spheres based on current settings
    int32 PointsToShow = FMath::Min(TrajectoryData.Num() / SkipFactor, MaxSpheres);
    int32 ActualSkip = FMath::Max(1, TrajectoryData.Num() / MaxSpheres);
    
    UE_LOG(LogTemp, Log, TEXT("TrajectoryTestActor: Creating %d spheres (skip factor: %d)"), PointsToShow, ActualSkip);
    
    for (int32 i = 0; i < TrajectoryData.Num(); i += ActualSkip)
    {
        if (SphereComponents.Num() >= MaxSpheres)
        {
            break;
        }
        
        CreateSphereAtPoint(TrajectoryData[i], i);
    }
    
    LogTrajectoryBounds();
}

void ATrajectoryTestActor::CreateSphereAtPoint(const FTrajectoryPoint& Point, int32 Index)
{
    if (!SphereMesh.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: SphereMesh is not valid"));
        return;
    }
    
    // Create sphere component
    FString ComponentName = FString::Printf(TEXT("Sphere_%d"), Index);
    UStaticMeshComponent* SphereComponent = NewObject<UStaticMeshComponent>(this, *ComponentName);
    
    if (!SphereComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Failed to create sphere component"));
        return;
    }
    
    // Register component with world
    SphereComponent->RegisterComponent();
    
    // Set mesh and material
    SphereComponent->SetStaticMesh(SphereMesh.LoadSynchronous());
    UMaterialInterface* Material = GetMaterialForSource(Point.Source);
    if (Material)
    {
        SphereComponent->SetMaterial(0, Material);
    }
    
    // Set position and scale
    FVector WorldPosition;
    if (bUseCesiumCoordinates && Point.Latitude != 0.0 && Point.Longitude != 0.0)
    {
        WorldPosition = ConvertWGS84ToUnrealCoordinates(Point.Latitude, Point.Longitude, Point.AltitudeMSL);
    }
    else
    {
        WorldPosition = ConvertToUnrealCoordinates(Point.X, Point.Y, Point.Z);
    }
    SphereComponent->SetWorldLocation(WorldPosition);
    SphereComponent->SetWorldScale3D(FVector(SphereScale));
    
    // Attach to root
    SphereComponent->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepWorldTransform);
    
    // Add to our list
    SphereComponents.Add(SphereComponent);
}

UMaterialInterface* ATrajectoryTestActor::GetMaterialForSource(const FString& Source)
{
    if (Source == TEXT("IGC"))
    {
        return IGCMaterial.LoadSynchronous();
    }
    else if (Source == TEXT("GPX"))
    {
        return GPXMaterial.LoadSynchronous();
    }
    else if (Source == TEXT("GEOJSON"))
    {
        return GeoJSONMaterial.LoadSynchronous();
    }
    else
    {
        return UnknownMaterial.LoadSynchronous();
    }
}

FVector ATrajectoryTestActor::ConvertToUnrealCoordinates(float X, float Y, float Z)
{
    // Convert from ENU (East-North-Up) to Unreal coordinates (X-Forward, Y-Right, Z-Up)
    // Scale coordinates to Unreal units
    return FVector(
        Y * CoordinateScale * 100.0f,  // North -> X (Forward)
        X * CoordinateScale * 100.0f,  // East -> Y (Right)
        Z * CoordinateScale * 100.0f   // Up -> Z (Up)
    );
}

FVector ATrajectoryTestActor::ConvertWGS84ToUnrealCoordinates(double Latitude, double Longitude, double AltitudeMSL)
{
    // Use Cesium to transform WGS84 coordinates to Unreal coordinates
    ACesiumGeoreference* Georeference = ACesiumGeoreference::GetDefaultGeoreference(this);
    
    if (!Georeference)
    {
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Could not resolve Cesium Georeference - falling back to scaled coordinates"));
        
        // Fallback: use a simple approximation (not recommended for production)
        // This is just to prevent complete failure
        return FVector(
            (Latitude - 52.27) * 111320.0 * CoordinateScale * 100.0f,   // Rough lat to meters conversion
            (Longitude - 20.91) * 111320.0 * cos(FMath::DegreesToRadians(52.27)) * CoordinateScale * 100.0f,   // Rough lon to meters
            AltitudeMSL * CoordinateScale * 100.0f
        );
    }
    
    // Use Cesium's coordinate transformation
    FVector UnrealCoords = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(
        FVector(Longitude, Latitude, AltitudeMSL)
    );
    
    return UnrealCoords;
}

bool ATrajectoryTestActor::LoadTrajectoryFromJSON(const FString& JSONPath)
{
    FString JSONString;
    if (!FFileHelper::LoadFileToString(JSONString, *JSONPath))
    {
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Failed to load JSON file: %s"), *JSONPath);
        return false;
    }
    
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JSONString);
    
    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Failed to parse JSON"));
        return false;
    }
    
    // Clear existing data
    TrajectoryData.Empty();
    
    // Parse trajectory points
    const TArray<TSharedPtr<FJsonValue>>* TrajectoryArray;
    if (JsonObject->TryGetArrayField(TEXT("trajectory_points"), TrajectoryArray))
    {
        for (const TSharedPtr<FJsonValue>& Value : *TrajectoryArray)
        {
            const TSharedPtr<FJsonObject>& PointObj = Value->AsObject();
            if (PointObj.IsValid())
            {
                FTrajectoryPoint Point;
                Point.Timestamp = PointObj->GetNumberField(TEXT("timestamp"));
                Point.X = PointObj->GetNumberField(TEXT("x"));
                Point.Y = PointObj->GetNumberField(TEXT("y"));
                Point.Z = PointObj->GetNumberField(TEXT("z"));
                Point.Source = PointObj->GetStringField(TEXT("source"));
                Point.Quality = PointObj->GetNumberField(TEXT("quality"));
                
                TrajectoryData.Add(Point);
            }
        }
    }
    
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Loaded %d trajectory points from JSON"), TrajectoryData.Num());
    return TrajectoryData.Num() > 0;
}

bool ATrajectoryTestActor::LoadWGS84TrajectoryFromJSON(const FString& JSONPath)
{
    FString JSONString;
    if (!FFileHelper::LoadFileToString(JSONString, *JSONPath))
    {
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Failed to load WGS84 JSON file: %s"), *JSONPath);
        return false;
    }
    
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JSONString);
    
    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("TrajectoryTestActor: Failed to parse WGS84 JSON"));
        return false;
    }
    
    // Clear existing data
    TrajectoryData.Empty();
    
    // Parse trajectory points with WGS84 coordinates
    const TArray<TSharedPtr<FJsonValue>>* TrajectoryArray;
    if (JsonObject->TryGetArrayField(TEXT("trajectory_points"), TrajectoryArray))
    {
        for (const TSharedPtr<FJsonValue>& Value : *TrajectoryArray)
        {
            const TSharedPtr<FJsonObject>& PointObj = Value->AsObject();
            if (PointObj.IsValid())
            {
                FTrajectoryPoint Point;
                Point.Timestamp = PointObj->GetNumberField(TEXT("timestamp"));
                Point.Latitude = PointObj->GetNumberField(TEXT("latitude"));
                Point.Longitude = PointObj->GetNumberField(TEXT("longitude"));
                Point.AltitudeMSL = PointObj->GetNumberField(TEXT("altitude_msl"));
                Point.Source = PointObj->GetStringField(TEXT("source"));
                Point.Quality = PointObj->GetNumberField(TEXT("quality"));
                
                // Clear ENU coordinates since we're using WGS84
                Point.X = 0.0f;
                Point.Y = 0.0f;
                Point.Z = 0.0f;
                
                TrajectoryData.Add(Point);
            }
        }
    }
    
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Loaded %d WGS84 trajectory points from JSON"), TrajectoryData.Num());
    return TrajectoryData.Num() > 0;
}

void ATrajectoryTestActor::UpdateStatistics()
{
    TotalPoints = TrajectoryData.Num();
    
    if (TotalPoints > 0)
    {
        FlightDuration = TrajectoryData.Last().Timestamp - TrajectoryData[0].Timestamp;
        DataStatus = FString::Printf(TEXT("Loaded %d points, %.1f seconds"), TotalPoints, FlightDuration);
    }
    else
    {
        FlightDuration = 0.0f;
        DataStatus = TEXT("No trajectory points loaded");
    }
}

void ATrajectoryTestActor::LogTrajectoryBounds()
{
    if (TrajectoryData.Num() == 0)
    {
        return;
    }
    
    float MinX = TrajectoryData[0].X, MaxX = TrajectoryData[0].X;
    float MinY = TrajectoryData[0].Y, MaxY = TrajectoryData[0].Y;
    float MinZ = TrajectoryData[0].Z, MaxZ = TrajectoryData[0].Z;
    
    for (const FTrajectoryPoint& Point : TrajectoryData)
    {
        MinX = FMath::Min(MinX, Point.X);
        MaxX = FMath::Max(MaxX, Point.X);
        MinY = FMath::Min(MinY, Point.Y);
        MaxY = FMath::Max(MaxY, Point.Y);
        MinZ = FMath::Min(MinZ, Point.Z);
        MaxZ = FMath::Max(MaxZ, Point.Z);
    }
    
    UE_LOG(LogTemp, Warning, TEXT("TrajectoryTestActor: Trajectory bounds:"));
    
    if (bUseCesiumCoordinates && TrajectoryData.Num() > 0 && TrajectoryData[0].Latitude != 0.0)
    {
        // Show WGS84 bounds
        double MinLat = TrajectoryData[0].Latitude, MaxLat = TrajectoryData[0].Latitude;
        double MinLon = TrajectoryData[0].Longitude, MaxLon = TrajectoryData[0].Longitude;
        double MinAlt = TrajectoryData[0].AltitudeMSL, MaxAlt = TrajectoryData[0].AltitudeMSL;
        
        for (const FTrajectoryPoint& Point : TrajectoryData)
        {
            MinLat = FMath::Min(MinLat, Point.Latitude);
            MaxLat = FMath::Max(MaxLat, Point.Latitude);
            MinLon = FMath::Min(MinLon, Point.Longitude);
            MaxLon = FMath::Max(MaxLon, Point.Longitude);
            MinAlt = FMath::Min(MinAlt, Point.AltitudeMSL);
            MaxAlt = FMath::Max(MaxAlt, Point.AltitudeMSL);
        }
        
        UE_LOG(LogTemp, Warning, TEXT("  Latitude: %.6f° to %.6f°"), MinLat, MaxLat);
        UE_LOG(LogTemp, Warning, TEXT("  Longitude: %.6f° to %.6f°"), MinLon, MaxLon);
        UE_LOG(LogTemp, Warning, TEXT("  Altitude MSL: %.1f to %.1f meters"), MinAlt, MaxAlt);
    }
    else
    {
        // Show ENU bounds
        UE_LOG(LogTemp, Warning, TEXT("  X (East): %.1f to %.1f meters"), MinX, MaxX);
        UE_LOG(LogTemp, Warning, TEXT("  Y (North): %.1f to %.1f meters"), MinY, MaxY);
        UE_LOG(LogTemp, Warning, TEXT("  Z (Up): %.1f to %.1f meters"), MinZ, MaxZ);
    }
    
    UE_LOG(LogTemp, Warning, TEXT("  Created %d sphere components"), SphereComponents.Num());
}