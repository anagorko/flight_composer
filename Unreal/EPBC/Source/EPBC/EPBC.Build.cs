// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class EPBC : ModuleRules
{
	public EPBC(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "MovieRenderPipelineCore", "MovieScene", "LevelSequence" });

  		PublicDependencyModuleNames.AddRange(new string[] {
            "Json", "JsonUtilities",   // Required for parsing by UGeoJsonLoaderComponent
            "CesiumRuntime",           // Required for Cesium types by UGeoJsonLoaderComponent
            "ProceduralMeshComponent"  // Required by MoebiusStripActor
        });

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
	}
}
