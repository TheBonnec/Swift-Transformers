// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS("18.0"), .macOS("15.0")],
    products: [
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", .upToNextMinor(from: "1.4.0")),
        .package(url: "https://github.com/johnmai-dev/Jinja", .upToNextMinor(from: "1.1.0"))
    ],
    targets: [
        .target(name: "Configurations"),
        .target(name: "Tokenizers", dependencies: [.product(name: "Jinja", package: "Jinja")]),
        .target(name: "Generation", dependencies: ["Tokenizers"]),
        .target(name: "Models", dependencies: ["Configurations", "Tokenizers", "Generation"]),
        .testTarget(name: "TokenizersTests", dependencies: ["Tokenizers", "Models"], resources: [.process("Resources"), .process("Vocabs")]),
        .testTarget(name: "PreTokenizerTests", dependencies: ["Tokenizers"]),
        .testTarget(name: "NormalizerTests", dependencies: ["Tokenizers"]),
        .testTarget(name: "PostProcessorTests", dependencies: ["Tokenizers"]),
    ]
)
