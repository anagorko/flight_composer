#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h" // Required include
#include "MoebiusStripActor.generated.h"

UCLASS()
class EPBC_API AMoebiusStripActor : public AActor
{
    GENERATED_BODY()

public:
    AMoebiusStripActor();

    // Called when properties are changed in the editor
    virtual void OnConstruction(const FTransform& Transform) override;

protected:
    // The component that handles the actual mesh generation
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Mesh")
    UProceduralMeshComponent* ProceduralMesh;

    // Radius of the flight path circle
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "10.0"))
    float PathRadius = 500.0f;

    // The wingspan (width of the ribbon)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "1.0"))
    float Wingspan = 100.0f;

    // How many discrete ticks/steps to generate
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Glider Path", meta = (ClampMin = "3"))
    int32 NumSegments = 100;

    // Helper to generate the geometry
    void GenerateMoebiusStrip();
};
