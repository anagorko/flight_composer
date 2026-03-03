#include "MoebiusStripActor.h"
#include "KismetProceduralMeshLibrary.h"

AMoebiusStripActor::AMoebiusStripActor()
{
    PrimaryActorTick.bCanEverTick = false; // Static geometry doesn't need to tick

    ProceduralMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("ProceduralMesh"));
    RootComponent = ProceduralMesh;

    // Default material ensures we can see it (Two-sided is best for ribbons)
    ProceduralMesh->bUseAsyncCooking = true;
}

void AMoebiusStripActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    GenerateMoebiusStrip();
}

void AMoebiusStripActor::GenerateMoebiusStrip()
{
    ProceduralMesh->ClearAllMeshSections();

    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UVs;
    TArray<FProcMeshTangent> Tangents;

    // --- STEP 1: Generate Raw Positions & UVs ---
    const float HalfWidth = Wingspan * 0.5f;
    const float AngleStep = (2.0f * PI) / NumSegments;

    for (int32 i = 0; i <= NumSegments; i++)
    {
        float Theta = i * AngleStep;

        // Position on Circle
        FVector CenterPos = FVector(FMath::Cos(Theta) * PathRadius, FMath::Sin(Theta) * PathRadius, 0.0f);

        // Orientation (Möbius Twist)
        FVector Radial = FVector(FMath::Cos(Theta), FMath::Sin(Theta), 0.0f);
        FVector Up = FVector(0.0f, 0.0f, 1.0f);
        float TwistAngle = Theta * 0.5f; // 180 deg twist over 360 deg circle

        // Calculate "Right" vector
        FVector CurrentRight = (Radial * FMath::Cos(TwistAngle)) + (Up * FMath::Sin(TwistAngle));

        // Vertices
        Vertices.Add(CenterPos - (CurrentRight * HalfWidth)); // 0: Left
        Vertices.Add(CenterPos + (CurrentRight * HalfWidth)); // 1: Right

        // UVs
        float U = (float)i / (float)NumSegments;
        UVs.Add(FVector2D(U, 0.0f));
        UVs.Add(FVector2D(U, 1.0f));
    }

    // --- STEP 2: Generate Geometric Normals ---
    // We calculate normals based on the Vector Cross Product of the ribbon's actual edges.
    for (int32 i = 0; i <= NumSegments; i++)
    {
        // 1. Get the vector across the wing (Left to Right)
        int32 VertIndexLeft = i * 2;
        int32 VertIndexRight = i * 2 + 1;
        FVector VectorAcross = Vertices[VertIndexRight] - Vertices[VertIndexLeft];

        // 2. Get the vector along the path (Current to Next)
        // Note: For the last segment, we look backward to keep continuity
        int32 NextIndex = (i == NumSegments) ? (i - 1) * 2 : (i + 1) * 2;
        int32 CurrIndex = (i == NumSegments) ? i * 2 : i * 2;

        // If we are at the very end, reverse the vector so math stays consistent
        float DirectionMult = (i == NumSegments) ? -1.0f : 1.0f;
        FVector VectorAlong = (Vertices[NextIndex] - Vertices[CurrIndex]) * DirectionMult;

        // 3. Cross Product = The True Surface Normal
        FVector GeoNormal = FVector::CrossProduct(VectorAlong, VectorAcross).GetSafeNormal();

        // Add for both vertices of this strip segment
        Normals.Add(GeoNormal);
        Normals.Add(GeoNormal);

        // Accurate Tangent is the vector along the path
        Tangents.Add(FProcMeshTangent(VectorAlong, false));
        Tangents.Add(FProcMeshTangent(VectorAlong, false));
    }

    // --- STEP 3: Triangles (Same as before) ---
    for (int32 i = 0; i < NumSegments; i++)
    {
        int32 BottomLeft  = i * 2;
        int32 BottomRight = i * 2 + 1;
        int32 TopLeft     = (i + 1) * 2;
        int32 TopRight    = (i + 1) * 2 + 1;

        Triangles.Add(BottomLeft);
        Triangles.Add(TopLeft);
        Triangles.Add(BottomRight);

        Triangles.Add(BottomRight);
        Triangles.Add(TopLeft);
        Triangles.Add(TopRight);
    }

    ProceduralMesh->CreateMeshSection(0, Vertices, Triangles, Normals, UVs, TArray<FColor>(), Tangents, true);
}
