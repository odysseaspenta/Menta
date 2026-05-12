# Menta Project Overview

Menta is a privacy-preserving mental health monitoring system that leverages Small Language Models (SLMs) for on-device inference. The project includes a multi-task fine-tuning pipeline and cross-platform deployment for iOS and Android.

## System Architecture

1.  **Menta Pretraining (`Menta_pretraining_code/`)**:
    *   **Core Technology**: Fine-tunes **Qwen3-4B-Instruct-2507** and **Gemma-4-E4B-it** using LoRA.
    *   **Multi-Task Learning**: Handles 6 classification tasks (Stress, Depression Binary/Severity, Suicide Ideation, Suicide Risk Binary/Severity).
    *   **Innovation**: Implements a novel **Log-Probability Evaluation** method and **BACC (Balanced Accuracy) Surrogate Loss**.
    *   **Optimization**: Uses 8-bit quantization to enable training on consumer-grade GPUs (~16GB VRAM).

2.  **iOS Deployment (`Menta_deployment/`)**:
    *   **Core Technology**: iOS SwiftUI application using **llama.cpp** for on-device inference.
    *   **Performance**: Optimized for Apple Silicon (Metal), achieving real-time inference on iPhone 15 Pro Max.

3.  **Android Deployment (`Menta_Android/`)**:
    *   **Core Technology**: Kotlin/Compose application using **llama.cpp** (JNI) and **Vulkan** for GPU acceleration.
    *   **Target Model**: Optimized for **Gemma-4-E4B** on Snapdragon 8 Gen 2+ devices.

---

## 📁 Project Structure

```text
.
├── Menta_deployment/        # iOS application and deployment code
├── Menta_Android/           # Android application (Kotlin/Compose)
├── Menta_pretraining_code/  # Python training and fine-tuning pipeline
│   ├── dataset/             # CSV datasets for training
│   ├── config.yaml          # Hyperparameters (Qwen3)
│   ├── config_gemma4.yaml   # Hyperparameters (Gemma 4)
│   └── *.py                 # Training and merging scripts
├── README.md                # Root project documentation
├── CLAUDE.md                # AI agent guidance
└── GEMINI.md                # This file
```

---

## 🚀 Quick Start

### Python Pretraining & Model Merging
```bash
cd Menta_pretraining_code
# Training (e.g., Qwen3)
python Menta_lora_config1_logprob.py
# Merging (e.g., Gemma 4)
python merge_lora_gemma4.py --adapter_dir ./gemma4_trained_model --output_dir ./gemma4_merged
```

### Cross-Platform Deployment
*   **iOS**: Open `Menta_deployment/Menta.xcodeproj` in Xcode. Requires `llamacpp-framework` build.
*   **Android**: Open `Menta_Android/` in Android Studio. Requires NDK and Vulkan support.

### ⚠️ Critical Dependency: llama.cpp Framework
The project requires `llama.cpp` version **b8953** or newer for **Gemma 4** support.
```bash
# To update/convert to proper submodule:
cd Menta_deployment
git rm -r --cached llamacpp-framework/
rm -rf llamacpp-framework/
git submodule add https://github.com/ggerganov/llama.cpp.git llamacpp-framework
git -C llamacpp-framework checkout b8953
```

---

## 📊 Mental Health Tasks & Datasets

| Task | Type | Dataset |
| :--- | :--- | :--- |
| **Stress Detection** | Binary | Dreaddit Stress Analysis |
| **Depression Detection** | Binary | Reddit Depression Dataset |
| **Depression Severity** | 4-class | Reddit Depression Dataset |
| **Suicide Ideation** | Binary | SDCNL |
| **Suicide Risk** | Binary | 500 Reddit Users |
| **Suicide Risk Severity** | 5-class | 500 Reddit Users |

---

## 🛠️ Development Conventions

### Coding Standards
*   **Python**: Follows standard PyTorch/Transformers patterns. Use `config.yaml` for all hyperparameters.
*   **Swift**: Uses SwiftUI with the `ObservableObject` pattern for state management (`LlamaState.swift`).

### Testing & Validation
*   **Pretraining**: Evaluation includes standard metrics (Accuracy, F1, BACC) and Log-Prob scoring. Use `example_usage.py` for sanity checks.
*   **Deployment**: The app includes a benchmarking harness to track TTFT (Time-to-First-Token), ITPS/OTPS (Tokens Per Second), and memory usage.

### Important Files
*   `GEMINI.md`: (This file) Root project overview.
*   `Menta_pretraining_code/GEMINI.md`: Detailed instructions for the training pipeline.
*   `Menta_deployment/GEMINI.md`: Detailed instructions for the iOS deployment.
*   `Menta_Android/GEMINI.md`: Detailed instructions for the Android deployment.
*   `CLAUDE.md`: Quick reference for commands and project structure.
*   `Menta_deployment/SETUP.md`: Detailed iOS environment setup.

## ⚠️ Security & Privacy
*   **Never commit `.gguf` or large model files.**
*   **Never commit raw dataset CSVs** if they contain sensitive or PII data (though public research datasets are used here).
*   **Privacy-First**: Always ensure changes to the iOS app maintain the "no cloud" processing guarantee.
