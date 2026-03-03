#include "GeoJsonLoaderComponent.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Components/SplineComponent.h"
#include "CesiumGeoreference.h"
#include "CesiumGlobeAnchorComponent.h"
#include "CesiumGeospatial/Ellipsoid.h"
#include "CesiumGeospatial/Cartographic.h"

UGeoJsonLoaderComponent::UGeoJsonLoaderComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    FileName = "example.geojson"; // Default
}

void UGeoJsonLoaderComponent::BeginPlay()
{
    Super::BeginPlay();
    // Optional: Auto-load on play
    // LoadPolygonData();
}

void UGeoJsonLoaderComponent::LoadPolygonData()
{
    AActor* Owner = GetOwner();
    if (!Owner) return;

    ACesiumCartographicPolygon* CesiumPoly = Cast<ACesiumCartographicPolygon>(Owner);
    if (!CesiumPoly)
    {
        UE_LOG(LogTemp, Error, TEXT("Owner is not a CesiumCartographicPolygon"));
        return;
    }

    UCesiumGlobeAnchorComponent* GlobeAnchor = Owner->FindComponentByClass<UCesiumGlobeAnchorComponent>();
    USplineComponent* Spline = Owner->FindComponentByClass<USplineComponent>();

    if (!GlobeAnchor || !Spline)
    {
        UE_LOG(LogTemp, Error, TEXT("Missing GlobeAnchor or Spline component!"));
        return;
    }

    ACesiumGeoreference* Georeference = GlobeAnchor->ResolveGeoreference();
    if (!Georeference)
    {
        UE_LOG(LogTemp, Error, TEXT("Could not resolve Cesium Georeference"));
        return;
    }

    FString FileContent;
    FString FullPath = FPaths::ProjectContentDir() + FileName;

    if (!FPaths::FileExists(FullPath))
    {
        UE_LOG(LogTemp, Error, TEXT("GeoJSON file does not exist: %s"), *FullPath);
        return;
    }

    if (!FFileHelper::LoadFileToString(FileContent, *FullPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to read GeoJSON file: %s"), *FullPath);
        return;
    }

    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileContent);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to parse JSON from file: %s"), *FullPath);
        return;
    }

    TArray<TSharedPtr<FJsonValue>> CoordinatesArray;
FString JsonType;
    JsonObject->TryGetStringField(TEXT("type"), JsonType);
    
    const TSharedPtr<FJsonObject>* GeometryObject = nullptr;

    if (JsonType == TEXT("FeatureCollection"))
    {
        const TArray<TSharedPtr<FJsonValue>>* Features;
        if (JsonObject->TryGetArrayField(TEXT("features"), Features) && Features->Num() > 0)
        {
            TSharedPtr<FJsonObject> FirstFeature = (*Features)[0]->AsObject();
            if (FirstFeature.IsValid())
            {
                FirstFeature->TryGetObjectField(TEXT("geometry"), GeometryObject);
            }
    }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("No features found in FeatureCollection"));
            return;
        }
    }
    else
    {
        JsonObject->TryGetObjectField(TEXT("geometry"), GeometryObject);
    }
    
    if (GeometryObject && GeometryObject->IsValid())
    {
        const TArray<TSharedPtr<FJsonValue>>* Rings;
        if ((*GeometryObject)->TryGetArrayField(TEXT("coordinates"), Rings) && Rings->Num() > 0)
        {
            CoordinatesArray = (*Rings)[0]->AsArray();
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("No coordinates array found in geometry"));
            return;
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("No valid geometry found in GeoJSON"));
        return;
    }

    if (CoordinatesArray.Num() == 0) 
    {
        UE_LOG(LogTemp, Error, TEXT("No coordinate points found"));
        return;
    }

    Spline->ClearSplinePoints();

    int32 ProcessedPoints = 0;
    for (const TSharedPtr<FJsonValue>& PointVal : CoordinatesArray)
    {
        const TArray<TSharedPtr<FJsonValue>>& PointCoords = PointVal->AsArray();
        if (PointCoords.Num() >= 2)
        {
            double Longitude = PointCoords[0]->AsNumber();
            double Latitude = PointCoords[1]->AsNumber();
            double Height = 0.0;

            FVector UECoords = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(
                FVector(Longitude, Latitude, Height)
            );

            Spline->AddSplinePoint(
                UECoords,
                ESplineCoordinateSpace::World
            );
            
            ProcessedPoints++;
        }
    }

    Spline->SetClosedLoop(true);
    CesiumPoly->RerunConstructionScripts();
    
    UE_LOG(LogTemp, Log, TEXT("Successfully loaded GeoJSON polygon from '%s' with %d points"), *FileName, ProcessedPoints);
}
