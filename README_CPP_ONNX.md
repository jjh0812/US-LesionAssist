# US-LesionAssist: C++ / ONNX Runtime 실행 가이드

PyTorch U-Net 모델을 Windows C++17, OpenCV, ONNX Runtime CPU 환경에서 실행하기 위한 가이드입니다.

## 1. 모델 준비

프로젝트 루트에서 아래 명령어를 실행합니다.

```bat
.venv\Scripts\python.exe export_onnx.py
.venv\Scripts\python.exe prepare_demo.py
.venv\Scripts\python.exe verify_onnx.py --image samples/demo.png
```

필요한 로컬 파일:

- `outputs/best_model.pt` : 학습 완료된 PyTorch 모델
- `exports/unet_v1.onnx` : 변환된 ONNX 모델
- `samples/demo.png` : 데모 초음파 영상
- `exports/parity_input.bin` : 수치 검증 입력
- `exports/parity_ort_logits.bin` : Python 측 기준 출력

모델 가중치와 데이터셋은 GitHub에 포함하지 않으므로 직접 생성해야 합니다.

## 2. C++ 개발 환경

- Windows 11
- Visual Studio C++ Build Tools
- CMake
- OpenCV C++ 라이브러리 (vcpkg)
- ONNX Runtime Windows x64 C++ SDK

검증 당시 Visual Studio 2026 Build Tools, OpenCV 4.14.0, ONNX Runtime 1.32.0 CPU를 사용했습니다.

## 3. CMake 빌드

MSVC 개발자 명령 프롬프트에서 실행합니다. 경로는 각 PC 환경에 맞게 수정해야 합니다.

```bat
cmake -S cpp -B cpp\build_132 -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake -DONNXRUNTIME_ROOT=C:/Projects/deps/onnxruntime-win-x64-1.32.0/onnxruntime-win-x64-1.32.0 -DVCPKG_APPLOCAL_DEPS=OFF -DUSLA_COPY_ORT_DLL=OFF
cmake --build cpp\build_132 --config Release
```

현재 설정은 빌드 후 DLL 자동 복사를 비활성화합니다. ONNX Runtime DLL을 실행 파일 옆에 배치하고 OpenCV DLL 검색 경로를 설정해야 합니다.

```bat
copy /Y "C:\Projects\deps\onnxruntime-win-x64-1.32.0\onnxruntime-win-x64-1.32.0\lib\onnxruntime.dll" "cpp\build_132\Release\onnxruntime.dll"
set "PATH=C:\Projects\vcpkg\installed\x64-windows\bin;%PATH%"
```

## 4. 실제 초음파 영상 분석

```bat
cpp\build_132\Release\lesion_infer.exe --model exports/unet_v1.onnx --image samples/demo.png --out cpp_results --warmup 10 --runs 100
```

생성 결과:

- `cpp_results/overlay.png` : 병변 오버레이 이미지
- `cpp_results/mask.png` : 예측 마스크
- `cpp_results/cpp_logits.bin` : 모델 출력
- `cpp_results/benchmark.json` : 추론 성능 기록

## 5. Python-C++ 결과 비교

```bat
cpp\build_132\Release\lesion_infer.exe --model exports/unet_v1.onnx --input-bin exports/parity_input.bin --reference-logits exports/parity_ort_logits.bin --out cpp_parity --warmup 10 --runs 100
```

실제 검증 결과:

```text
Numerical parity max_abs_diff: 0
mask agreement: 1
Median/p95 inference: 3.8 / 4.0 ms
```

모델 실행 시간은 영상 입출력과 전후처리를 제외합니다.

## 6. 주의사항

Python과 C++의 동일 입력 텐서 추론은 일치했지만, PIL과 OpenCV의 전처리가 완전히 동일한 것은 아닙니다.

실제 의료기기 연동, 외부 환자 집단에 대한 검증 및 임상적 안전성 평가는 수행하지 않았습니다.

[메인 README로 돌아가기](README.md)
