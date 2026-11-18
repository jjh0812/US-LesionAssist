# US-LesionAssist — PyTorch → ONNX Runtime → C++ / OpenCV

**Purpose:** Convert the existing trained BUSI U-Net v1 model into a standalone C++ ultrasound image segmentation demo. It is a **research prototype only, not a diagnostic device**. No clinical validation, safety certification, DICOM acquisition, live ultrasound integration, or regulatory approval is claimed.

## What is already implemented vs. what requires local execution

- Existing project: U-Net training/evaluation, Streamlit viewer, frozen `train/val/test` split.
- This upgrade contains: `export_onnx.py`, `verify_onnx.py`, `validate_deployment.py`, `prepare_demo.py`, a C++ ONNX Runtime CPU + OpenCV inference executable, benchmarking, and CMake build scripts.
- **Not shipped:** your own trained `outputs/best_model.pt`, BUSI images, ONNX exported weights, compiled Windows executable, or observed C++ benchmark results. These must be generated on your laptop.
- The original v1 checkpoint and Streamlit demo remain unchanged.

## A. Start with Python (Windows 11, VS Code PowerShell)

1. Extract the ZIP somewhere, then **open the extracted `US-LesionAssist-Cpp-ONNX-Upgrade` folder and copy its contents** (the `.py` files, `cpp` folder, and README) into your existing `C:\Projects\US-LesionAssist-VSCode-Starter\US-LesionAssist` folder. Check that `export_onnx.py` sits next to `train.py` and `app_v2.py`. Do not leave these files inside a second nested folder.
2. Open a terminal with your existing `.venv` activated and project folder as the current directory:

   ```powershell
   Get-Location
   Test-Path .\outputs\best_model.pt
   ```

   The second line must show `True`.

3. Install export and runtime dependencies:

   ```powershell
   python -m pip install -r requirements-deploy.txt
   ```

4. Export the **existing selected v1** checkpoint (trained at 160 px) to an ONNX file, without re-training:

   ```powershell
   python export_onnx.py
   ```

   Check for `ONNX graph check: PASSED`; output: `exports/unet_v1.onnx`. The output is **logits**, not probabilities, malignancy labels, or diagnostic results.

5. Copy the known **validation** example `malignant (63).png` to an ASCII-only demo path and run parity test. This avoids some Windows OpenCV issues with a non-ASCII user profile directory:

   ```powershell
   python prepare_demo.py
   python verify_onnx.py --image samples/demo.png
   ```

   To explore the known failure example later, use `python prepare_demo.py --filename 'malignant (106).png' --out samples/failure.png`; never present one chosen image as average performance.

   The report `exports/onnx_parity_report.json` must say `"status": "PASS"`. This compares PyTorch and ONNX Runtime on the **same OpenCV-preprocessed input tensor**. It outputs `exports/parity_input.bin` and `exports/parity_ort_logits.bin` for a later C++ comparison.

6. Evaluate the change in image preprocessing on the **validation** split (not the test split):

   ```powershell
   python validate_deployment.py
   ```

   Review `exports/deployment_val_report.json`, especially the total Dice and the malignant subset. It compares the original training pipeline (PIL image preprocessing) against the C++-matching OpenCV preprocessing; image preprocessing differences can affect masks even when ONNX numerical parity is excellent.

## B. Build the C++ app on Windows

You need **Microsoft Visual C++ Build Tools** with C++ workload, **CMake**, **Git**, **OpenCV C++ development libraries**, and **ONNX Runtime C/C++ Windows x64** binaries. `pip install opencv-python` / `pip install onnxruntime` alone does **not** install the C++ header and linker SDKs.

Official instructions:
- [ONNX Runtime C++ getting started](https://onnxruntime.ai/docs/get-started/with-cpp.html)
- [ONNX Runtime Windows guidance](https://onnxruntime.ai/docs/get-started/with-windows.html)
- [ONNX Runtime Windows x64 releases](https://github.com/microsoft/onnxruntime/releases)
- [Microsoft vcpkg](https://github.com/microsoft/vcpkg)

Example Windows setup (adjust versions and paths to your installed software):

```powershell
# A) Set up vcpkg and OpenCV C++ once:
git clone https://github.com/microsoft/vcpkg C:\Projects\vcpkg
C:\Projects\vcpkg\bootstrap-vcpkg.bat
C:\Projects\vcpkg\vcpkg.exe install opencv4:x64-windows

# B) Download a CPU Windows x64 ONNX Runtime ZIP from the official GitHub
# release page and unzip it, for example, under:
# C:\Projects\deps\onnxruntime-win-x64-VERSION\
# Confirm that include\onnxruntime_cxx_api.h and lib\onnxruntime.lib exist.

# C) Configure and build from the project ROOT (Visual Studio 2022 C++ tools shown):
cmake -S cpp -B cpp\build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="C:/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DONNXRUNTIME_ROOT="C:/Projects/deps/onnxruntime-win-x64-VERSION"
cmake --build cpp\build --config Release
```

If you have a different MSVC version, replace the generator with the corresponding one (`cmake -G` lists those available). If an OpenCV DLL cannot be found at run time, ensure the vcpkg `installed/x64-windows/bin` directory is on `PATH`; the CMake target copies the ONNX Runtime DLL from its SDK `lib` folder, if found.

## C. Run the C++ app and save an overlay and benchmark

From the **project root** in VS Code terminal:

```powershell
.\cpp\build\Release\lesion_infer.exe --model exports/unet_v1.onnx --image samples/demo.png --out cpp_results --warmup 10 --runs 100
```

Outputs: `cpp_results/overlay.png`, `mask.png`, `cpp_logits.bin`, `benchmark.json` with median and p95 **CPU ORT model-only** latency. Results are measured locally; no prefilled or invented numbers.

Confirm C++ and Python ONNX outputs agree on **exactly the same tensor**:

```powershell
.\cpp\build\Release\lesion_infer.exe --model exports/unet_v1.onnx --input-bin exports/parity_input.bin --reference-logits exports/parity_ort_logits.bin --out cpp_parity
```

Open `cpp_parity/benchmark.json` and inspect `max_abs_diff_vs_reference_logits` and `binary_mask_agreement_vs_reference`. The reference binaries were generated by `verify_onnx.py` with `samples/demo.png`. The result is meaningful only when the exact image and ONNX model are unchanged.

**Benchmark honesty:** The previous `1.77 ms` number came from **PyTorch on CUDA** and is **not** comparable to this C++ CPU ONNX Runtime benchmark. Compare like-for-like providers, machines and input dimensions, and distinguish inference latency from end-to-end latency. Python `verify_onnx.py` also records ONNX Runtime CPU timing for comparison.

## Suggested report / portfolio artifacts

- Screenshots: `cpp_results/overlay.png`, `cpp_results/mask.png`, existing Streamlit ground-truth viewer.
- `exports/onnx_parity_report.json`: verified numerical agreement.
- `exports/deployment_val_report.json`: validation Dice/IoU after C++-compatible preprocessing.
- `cpp_results/benchmark.json`: C++ CPU inference median/p95.
- `outputs/test_metrics.json`: selected v1 held-out test Dice/IoU, with limitations.
- `outputs_v2*/comparison_val.json`, `outputs_v3/comparison_val.json`, `outputs_control160/comparison_val.json`: five experiment comparisons (do not confuse validation with held-out test results).

Do **not** claim it runs in a real Samsung ultrasound system, performs medical diagnosis, or is product-ready. BUSI masks are region annotations, not malignant classification probabilities; no patient-level split guarantee is available.

## Troubleshooting

- `ModuleNotFoundError: onnx` → `python -m pip install -r requirements-deploy.txt` in the project `.venv`.
- `No ONNX file` → export first; make sure `outputs/best_model.pt` is your trained v1 checkpoint.
- `ONNXRUNTIME_ROOT` missing → download the **C++ native** distribution; Python wheel is insufficient.
- `OpenCVConfig.cmake` not found → configure CMake using the vcpkg toolchain, and ensure OpenCV C++ is installed.
- DLL cannot be found → copy the required `onnxruntime.dll` and OpenCV DLLs beside `lesion_infer.exe` or put their directories on `PATH`.
- Missing `samples/demo.png` → `python prepare_demo.py` from project root.
- Non-ASCII Windows username causes `cv::imread` error → use `samples/demo.png` (ASCII relative path).
