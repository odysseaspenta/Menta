# Plan: Deploy Fine-Tuned Gemma4 on Android

> **Prerequisites: llamacpp-framework update is required before any other step.**
> See Step 0 below.

## Step 0: Update the llamacpp-framework (BLOCKING — must do first)

### Why this is required

`google/gemma-4-E4B-it` has `model_type: "gemma4"` and `architectures: ["Gemma4ForConditionalGeneration"]` in its `config.json`. The **embedded** llamacpp-framework only implements `LLM_ARCH_GEMMA3N` (Gemma 3.5 / E-variants of Gemma 3) — it does **not** have `LLM_ARCH_GEMMA4`. Without this update:
- `convert_hf_to_gguf.py` will fail because it has no `Gemma4Model` class
- `llama-cli` / Android runtime cannot load the resulting GGUF because the architecture is unrecognised

The upstream llama.cpp release **b8953** (2026-04-28, one day newer than the current commit) adds full Gemma4 support with `Gemma4Model` and `Gemma4ForConditionalGeneration`.

### Current state of the framework

The `.gitmodules` file declares `llamacpp-framework` as a submodule pointing to `https://github.com/ggerganov/llama.cpp.git`, but the code is currently committed **directly** into the Menta repo as regular files (no `.git` inside the directory, `git submodule status` returns nothing). The embedded gguf-py version is 0.17.1; upstream is 0.18.0.

### Recommended: convert to a proper git submodule

This reduces repo size, makes future updates trivial, and matches the intent of `.gitmodules`.

```bash
# From the repo root (/sysnet/projects/Menta)

# 1. Remove the tracked files (keeps the files on disk temporarily)
git rm -r --cached Menta_deployment/llamacpp-framework/

# 2. Remove the directory entirely
rm -rf Menta_deployment/llamacpp-framework/

# 3. Re-add as a proper submodule (uses the URL already in .gitmodules)
git submodule add https://github.com/ggerganov/llama.cpp.git Menta_deployment/llamacpp-framework

# 4. Pin to the latest release tag that includes Gemma4 support
#    Check current tags: git -C Menta_deployment/llamacpp-framework tag --sort=-v:refname | head -5
git -C Menta_deployment/llamacpp-framework fetch --tags origin
git -C Menta_deployment/llamacpp-framework checkout b8953   # or the latest tag

# 5. Stage and commit
git add Menta_deployment/.gitmodules Menta_deployment/llamacpp-framework
git commit -m "Convert llamacpp-framework to proper submodule at b8953 (adds Gemma4 support)"
```

### Verify the update

```bash
# Confirm Gemma4 architecture is present
grep -r "GEMMA4\|Gemma4" Menta_deployment/llamacpp-framework/src/llama-arch.h | head -5
# Expected: LLM_ARCH_GEMMA4 entry

# Confirm the converter has Gemma4Model
grep -n "Gemma4Model\|Gemma4ForConditional" \
    Menta_deployment/llamacpp-framework/convert_hf_to_gguf.py | head -5
# Expected: class Gemma4Model(...) and register("Gemma4ForConditionalGeneration")

# Confirm gguf-py is 0.18.0
grep "version" Menta_deployment/llamacpp-framework/gguf-py/pyproject.toml
```

### What changed in the upstream Android example

The upstream Android example underwent a major rewrite in December 2025 (Hilt DI, Room database, HuggingFace model browser, GGUF metadata display). **Do not use this as the base for the Menta Android app** — it adds complexity that the Menta benchmark harness doesn't need. Instead, start from the simpler pre-rewrite `examples/llama.android/` structure but link against the updated (post-b8953) C++ runtime. Alternatively, cherry-pick just the JNI `.kt` and `.cpp` files you need.

---

## Context

The project already runs fine-tuned Qwen3-4B on iOS via llama.cpp/GGUF (in `Menta_deployment/`). This plan mirrors that pipeline for the Gemma4-E4B fine-tuned model, targeting Android. The training code saves only the LoRA adapter — so the first step is merging it back into the base model before any conversion.

The iOS process is: HuggingFace checkpoint → `convert_hf_to_gguf.py` → GGUF → llama.cpp XCFramework → Swift app. The Android process is identical up to the GGUF file, then diverges: Android uses the NDK + JNI instead of an XCFramework, and Vulkan instead of Metal for GPU acceleration.

---

## Step 1: Merge the LoRA Adapter into the Base Model

**Why:** `convert_hf_to_gguf.py` reads a full HuggingFace model directory. The training code saves only the PEFT adapter (adapter_config.json + adapter_model.bin), not a merged model.

**Important complication — `Gemma4ClippableLinear`:** The trainer in `gemma4_lora_trainer.py` applies LoRA to the inner `.linear` sub-module (e.g., `q_proj.linear` instead of `q_proj`). Standard `merge_and_unload()` should still work because PEFT resolves the full parameter path correctly, but verify the merged state_dict contains proper weight keys afterward.

**New file to create:** `Menta_pretraining_code/merge_lora_gemma4.py`

> Note: Use the `convert_hf_to_gguf.py` from the **updated** llamacpp-framework (Step 0). The embedded version does not have the Gemma4 converter.

```python
# Usage: python merge_lora_gemma4.py --adapter_dir ./gemma4_trained_model_config1 --output_dir ./gemma4_merged
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_dir",  required=True)
parser.add_argument("--output_dir",   required=True)
parser.add_argument("--base_model",   default="google/gemma-4-E4B-it")
args = parser.parse_args()

# Must use float32 — bitsandbytes 8-bit is incompatible with merge_and_unload()
model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, trust_remote_code=True)
model = PeftModel.from_pretrained(model, args.adapter_dir)
merged = model.merge_and_unload()

os.makedirs(args.output_dir, exist_ok=True)
merged.save_pretrained(args.output_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(args.adapter_dir).save_pretrained(args.output_dir)
print("Done:", args.output_dir)
```

**Memory requirement:** Loading Gemma4-E4B at float32 on CPU needs ~16 GB RAM. Use `torch_dtype=torch.bfloat16` on memory-constrained machines; the converter accepts bfloat16 safetensors.

**Fallback if merge_and_unload() fails due to ClippableLinear path mismatch:** Manually apply the delta `(lora_B @ lora_A) * (alpha / r)` to each target weight key in the base model's state_dict, remapping `.linear` suffix back to the HF parameter name.

---

## Step 2: Convert to GGUF

llama.cpp's `convert_hf_to_gguf.py` (from the **updated** framework) auto-detects `architectures: ["Gemma4ForConditionalGeneration"]` from `config.json` and routes to the `Gemma4Model` converter, which maps to `LLM_ARCH_GEMMA4`.

```bash
# Use the llama.cpp conversion script from the framework submodule
python Menta_deployment/llamacpp-framework/convert_hf_to_gguf.py \
    ./Menta_pretraining_code/gemma4_merged \
    --outfile gemma4-menta-f16.gguf \
    --outtype f16 \
    --vocab-type bpe
```

**Multimodal handling:** The upstream `Gemma4Model` converter handles text-only conversion and a separate `Gemma4VisionAudioModel` handles the vision path. Running `convert_hf_to_gguf.py` on the merged directory should automatically select the text-only path. If the converter still includes vision tensors or if you want a smaller GGUF, strip them before saving the merged model:

```python
# Add to merge_lora_gemma4.py before save_pretrained():
vision_prefixes = ("vision_tower.", "multi_modal_projector.", "image_newline")
clean_state = {k: v for k, v in merged.state_dict().items()
               if not any(k.startswith(p) for p in vision_prefixes)}
```

**Verify the GGUF:**
```bash
python Menta_deployment/llamacpp-framework/gguf-py/scripts/gguf-dump.py gemma4-menta-f16.gguf | grep "general.architecture"
# Should print: general.architecture = gemma4   (NOT gemma3n)
```

---

## Step 3: Quantize to Q4_K_M

```bash
# Build llama.cpp tools first (or use the existing framework build)
./Menta_deployment/llamacpp-framework/build/bin/llama-quantize \
    gemma4-menta-f16.gguf \
    gemma4-menta-Q4_K_M.gguf \
    Q4_K_M
```

**Recommended:** Q4_K_M (~2.5 GB). For Gemma4-E4B, single-token classification is not sensitive to 4-bit quantization. The AltUp routing coefficients (`altup_correct_coef`, `altup_predict_coef`) are small tensors quantized separately with K-quant's mixed strategy. Q6_K is a safe alternative if classification accuracy degrades.

**Verify inference on desktop before moving to Android:**
```bash
./llamacpp-framework/build/bin/llama-cli \
    -m gemma4-menta-Q4_K_M.gguf \
    -p "<|turn>system\nYou are a stress detection expert.<turn|>\n<|turn>user\nDoes this post show stress? 'I can't stop worrying about everything'\n\nRespond with only 0 or 1.<turn|>\n<|turn>model\n" \
    -n 4 --no-display-prompt
```

---

## Step 4: Android App — Project Structure

Create `Menta_Android/` as a new Android Studio project alongside `Menta_deployment/`. The llama.cpp JNI module (`llama/`) references the existing `llamacpp-framework` submodule — no duplication needed.

```
Menta_Android/
├── settings.gradle.kts
├── build.gradle.kts
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/datasets/           ← same 4 CSV files as iOS
│       └── java/com/sysnet/menta/
│           ├── MainActivity.kt
│           ├── Tasks.kt               ← analog of Tasks.swift
│           ├── PromptGenerator.kt     ← analog of PromptGenerator.swift (Gemma4 template!)
│           ├── PredictionParser.kt    ← analog of PredictionParser.swift
│           ├── DatasetLoader.kt       ← analog of DatasetLoader.swift
│           ├── BatchProcessor.kt      ← analog of BatchProcessor.swift
│           ├── ModelState.kt          ← ViewModel analog of LlamaState.swift
│           └── ui/EvalScreen.kt       ← Compose UI
└── llama/                             ← JNI library module
    ├── build.gradle.kts
    └── src/main/
        ├── cpp/
        │   ├── CMakeLists.txt
        │   └── llama-android.cpp      ← extend from examples/llama.android/
        └── java/android/llama/cpp/
            └── LLamaAndroid.kt        ← extend from examples/llama.android/
```

**Start from the existing example:** Copy `Menta_deployment/llamacpp-framework/examples/llama.android/` as the base and extend it. The example already has working JNI wiring, a `Downloadable.kt` for large model downloads, and a `send()` Flow API for streaming tokens.

---

## Step 5: Key Android-Specific Components

### CMakeLists.txt — Enable Vulkan
```cmake
option(GGML_VULKAN "Enable Vulkan backend" OFF)   # set to ON in build.gradle.kts

set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../llamacpp-framework")
add_subdirectory(${LLAMA_SOURCE_DIR} build-llama)
```

In `llama/build.gradle.kts`:
```kotlin
arguments += "-DGGML_VULKAN=ON"
arguments += "-DCMAKE_BUILD_TYPE=Release"
ndk { abiFilters += listOf("arm64-v8a") }
```

Start with `n_gpu_layers = 10` to test Vulkan stability on device, then increase. Full offload (`n_gpu_layers = 99`) requires ~3 GB VRAM (Snapdragon 8 Gen 2+ or equivalent).

### Model Delivery
A 2.5 GB GGUF cannot be bundled in an APK. Two options:
1. **`adb push` for development:** `adb push gemma4-menta-Q4_K_M.gguf /sdcard/Android/data/com.sysnet.menta/files/`
2. **Runtime download for production:** Use the existing `Downloadable.kt` pattern from `llama.android` via `DownloadManager` to `getExternalFilesDir(null)`.

### PromptGenerator.kt — Gemma4 Chat Template
**Critical difference from iOS.** The iOS app uses Qwen3's ChatML format. Gemma4 uses pipe-delimited turn tokens that are completely different from Gemma3's `<start_of_turn>` format:

| Model | Turn format |
|-------|-------------|
| Qwen3 (iOS) | `<\|im_start\|>role\n...<\|im_end\|>` |
| Gemma3/3N | `<start_of_turn>role\n...<end_of_turn>` |
| **Gemma4** | `<\|turn>role\n...<turn\|>` |

Gemma4 also supports a proper system role (not merged into the user turn):

```kotlin
fun generate(taskType: TaskType, text: String): String {
    val (systemContent, userQuestion, constraint) = getComponents(taskType)
    return buildString {
        append("<|turn>system\n")
        append(systemContent)
        append("<turn|>\n")
        append("<|turn>user\n")
        append(userQuestion.replace("{text}", text))
        append("\n\n$constraint")
        append("<turn|>\n")
        append("<|turn>model\n")
    }
}
```

> **Known community issue:** Gemma4 GGUFs sometimes ship with the wrong chat template baked into the GGUF metadata, causing the model to output `---` repeatedly. If this happens, the prompt tokens above bypass the template and force the correct format. Verify single-sample output is a digit before running full evaluations.

### ModelState.kt — Metrics Collection
Mirror `LlamaState.swift`'s metrics: TTFT, ITPS (input tokens/sec), OTPS (output tokens/sec), peak memory via `Debug.getNativeHeapAllocatedSize()`, OOM detection via try/catch on `llama.load()`.

For single-token classification, add a `classify()` method to `LLamaAndroid.kt` that stops after the first valid digit token rather than streaming the full response.

### PredictionParser.kt
Direct port of `PredictionParser.swift` — same keyword matching logic, same label mappings for all 6 tasks.

### DatasetLoader.kt
Read CSVs from `context.assets.open(path)` instead of `Bundle.main`. The label remapping logic (e.g., `"minimum" → 0` for depression) is identical to `DatasetLoader.swift`.

---

## Step 6: Key Differences from iOS Summary

| Aspect | iOS | Android |
|--------|-----|---------|
| Native bridge | XCFramework / `import llama` | NDK JNI / `System.loadLibrary("llama-android")` |
| GPU backend | Metal (always on) | Vulkan (opt-in, `-DGGML_VULKAN=ON`) |
| **Prompt format** | Qwen3 ChatML (`<\|im_start\|>`) | Gemma4 (`<\|turn>role\n...<turn\|>`) |
| Model bundling | Xcode asset group | `assets/` (small) or `DownloadManager` (large) |
| Memory monitoring | `mach_task_basic_info` | `Debug.getNativeHeapAllocatedSize()` |
| Build | Xcode + `build-xcframework.sh` | Android Studio + NDK + Gradle |
| Threading | Swift Actors | Kotlin Coroutines + single-thread Executor |
| minSdk | iOS 16 | API 33 (Android 13) — required by `llama.android` example |

---

## Potential Issues

1. **Vision tensors in conversion:** Gemma4-E4B includes SigLIP vision encoder weights. Strip them before conversion if `convert_hf_to_gguf.py` errors.
2. **ClippableLinear merge paths:** If `merge_and_unload()` produces zero-filled weights, use the manual delta-application fallback.
3. **AltUp architecture stability:** `LLM_ARCH_GEMMA4` uses AltUp/LAUREL layers — if NaN outputs occur, update the `llamacpp-framework` submodule to a later upstream commit.
4. **Memory on Android:** Q4_K_M at 2048 context needs ~3 GB total. Reduce `n_ctx = 1024` for 6 GB devices. Use `Q3_K_M` as a last resort.

---

## Verification

1. Run `merge_lora_gemma4.py` and confirm merged model generates plausible single-digit outputs in Python.
2. Verify `gguf-dump.py` shows `general.architecture = gemma4`.
3. Test `llama-cli` on desktop with Gemma4 prompt format — confirm digit outputs.
4. Build Android APK: `./gradlew :llama:assembleDebug` (no compile errors).
5. Push GGUF via adb, load in app, run Task 1 (stress) with 3 samples, confirm valid predictions.
6. Compare accuracy on same 10 samples across iOS (Qwen3 fine-tuned) and Android (Gemma4 fine-tuned) to establish baseline.

---

## Critical Files to Modify / Create

| File | Action |
|------|--------|
| `Menta_pretraining_code/merge_lora_gemma4.py` | Create (new script) |
| `Menta_Android/llama/src/main/cpp/CMakeLists.txt` | Create (from example, add Vulkan) |
| `Menta_Android/llama/src/main/cpp/llama-android.cpp` | Create (extend from `examples/llama.android/`) |
| `Menta_Android/llama/src/main/java/android/llama/cpp/LLamaAndroid.kt` | Create (extend from example, add `classify()`) |
| `Menta_Android/app/src/main/java/com/sysnet/menta/ModelState.kt` | Create |
| `Menta_Android/app/src/main/java/com/sysnet/menta/Tasks.kt` | Create |
| `Menta_Android/app/src/main/java/com/sysnet/menta/PromptGenerator.kt` | Create (Gemma4 template) |
| `Menta_Android/app/src/main/java/com/sysnet/menta/PredictionParser.kt` | Create (port from Swift) |
| `Menta_Android/app/src/main/java/com/sysnet/menta/DatasetLoader.kt` | Create (port from Swift) |

**Reference implementations to port from:**
- `Menta_deployment/llamacpp-framework/examples/llama.android/` — JNI module base
- `Menta_deployment/Menta/PromptGenerator.swift` — prompt logic (adapt template)
- `Menta_deployment/Menta/PredictionParser.swift` — parsing logic (direct port)
- `Menta_deployment/Menta/DatasetLoader.swift` — CSV loading (direct port)
- `Menta_deployment/Menta/LlamaState.swift` — model state / metrics (direct port)
