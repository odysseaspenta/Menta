# Menta iOS Deployment Project Overview

This project is the **iOS-specific** deployment arm of Menta, focusing on on-device inference using GGUF-quantized models and `llama.cpp`.

## ⚠️ Critical Note: Gemma 4 Support
To support **Gemma 4** models (used in the Android parallel project), the `llamacpp-framework` submodule must be updated to version **b8953** or newer. See the root `GEMINI.md` for update instructions.

## Core Technologies
- **Inference Engine**: `llama.cpp` (compiled as an XCFramework).
- **Frontend**: SwiftUI.
- **Model Format**: GGUF (optimized for mobile/Apple Silicon).
- **Metal Performance Shaders (MPS)**: Utilized via `llama.cpp` for GPU acceleration on iOS.

## Key Components
- `LlamaState.swift`: The central `ObservableObject` managing model lifecycle, inference, and metrics.
- `Tasks.swift`: Defines the 6 mental health tasks, including prompt templates and expected labels.
- `PromptGenerator.swift`: Handles ChatML formatting for Qwen3 and Phi-4-mini models.
- `PredictionParser.swift`: Robust parsing logic for extracting classification results from LLM outputs.
- `BatchProcessor.swift`: Manages memory during long-running benchmark sessions to prevent OOM.

## Building and Running

### Prerequisites
- Xcode 15.0+
- iOS 16.0+ device (A14 Bionic or newer recommended).
- `git submodule update --init --recursive` to pull `llama.cpp`.

### Setup Steps
1.  **Build XCFramework**:
    ```bash
    cd llamacpp-framework
    ./build-xcframework.sh
    ```
2.  **Download Models**:
    Place `.gguf` files (Menta, Qwen3, Phi-4-mini) into the `Menta/` directory.
3.  **Xcode**:
    Open `Menta.xcodeproj` and run on a physical device.

## Development Conventions
- **Memory Management**: iOS devices have strict RAM limits (~3GB usable for apps). Use `BatchProcessor` and explicit `autoreleasepool` blocks if extending inference logic.
- **Prompting**: Always use the defined templates in `Tasks.swift` to ensure consistency with the training phase.
- **Metrics**: New features should ideally integrate with the performance tracking in `LlamaState` (TTFT, ITPS, OTPS).

## Testing
The app serves as its own test harness. Use the "Start Evaluation" button to run a task on a subset of the bundled datasets and verify accuracy against expected benchmarks.
