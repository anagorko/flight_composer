#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CesiumCartographicPolygon.h"
#include "GeoJsonLoaderComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class EPBC_API UGeoJsonLoaderComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UGeoJsonLoaderComponent();

    // 1. The parameter to change in editor
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "GeoJSON")
    FString FileName;

    // Function to trigger the load (can be called from Blutility or BeginPlay)
    UFUNCTION(BlueprintCallable, CallInEditor, Category = "GeoJSON")
    void LoadPolygonData();


protected:
    virtual void BeginPlay() override;
};
