// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MLXAudio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(
            name: "MLXAudio",
            targets: ["MLXAudio"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
        .package(url: "https://github.com/espeak-ng/espeak-ng-spm.git", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "MLXAudio",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "ESpeakNG", package: "espeak-ng-spm")
            ],
            path: "mlx_audio_swift/tts/MLXAudio",
            exclude: ["Preview Content", "Assets.xcassets", "MLXAudioApp.swift", "MLXAudio.entitlements"],
            resources: [
                .process("Kokoro/Resources") // Kokoro voices
            ]
        ),
        .testTarget(
            name: "MLXAudioTests",
            dependencies: ["MLXAudio"],
            path: "Tests"
        ),
    ]
)
