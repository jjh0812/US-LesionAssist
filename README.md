# US-LesionAssist

**유방 초음파 병변 자동 분할 및 C++ 기반 AI 추론 프로젝트**

PyTorch U-Net으로 유방 초음파 영상의 병변 영역을 분할하고, 학습한 모델을 ONNX로 변환하여 C++ / OpenCV / ONNX Runtime 환경에 통합한 연구용 프로젝트입니다.

> 연구용 프로토타입입니다. 양성·악성 진단 기능이 아니며, 실제 의료기기나 초음파 장비에 통합된 제품이 아닙니다.

## 핵심 성과

| 항목 | 결과 |
|---|---|
| 데이터셋 | BUSI 유방 초음파 이미지 780장 |
| 데이터 분할 | Train 546 / Validation 117 / Test 117 |
| 채택 모델 | U-Net v1, 160x160, 20 epochs |
| Test Dice | **0.6087** |
| Test IoU | **0.5351** |
| C++ CPU 추론 중앙값 | **3.8ms** |
| C++ CPU 추론 p95 | **4.0ms** |
| Python-C++ 최대 출력 차이 | **0** |
| 이진 마스크 일치율 | **100%** |

추론 시간은 모델 실행만 측정한 값으로, 영상 입출력과 전후처리를 포함하지 않습니다. 마스크 일치율 100%는 동일 입력에서 두 런타임의 결과가 일치했다는 의미이지 의료적 정확도를 의미하지 않습니다.

## 시스템 구조

BUSI 초음파 데이터 → PyTorch U-Net 학습 → ONNX 변환 → C++ / OpenCV 전처리 → ONNX Runtime CPU 추론 → 병변 마스크 및 오버레이 생성

**기술 스택:** Python, PyTorch, NumPy, Streamlit, ONNX, ONNX Runtime, C++17, OpenCV, CMake, Git

## 모델 실험 비교

동일한 검증 영상 117장을 공통 기준으로 평가한 결과입니다.

| 실험 | 해상도 | 증강 | Validation Dice |
|---|---|---|---|
| **v1 (최종 채택)** | 160x160 | 없음 | **0.6771** |
| v2 | 256x256 | 적용 | 0.5841 |
| v2 (증강 제거) | 256x256 | 없음 | 0.5878 |
| v3 | 160x160 | 적용 | 0.6442 |
| 대조 실험 | 160x160 | 없음 | 0.6690 |

더 높은 해상도와 데이터 증강이 항상 성능 개선으로 이어지지는 않았습니다. 해당 조건에서 가장 우수한 검증 결과를 보인 v1 모델을 채택했습니다.

## C++ 애플리케이션 구현

1. PyTorch 학습 모델을 ONNX 형식으로 변환
2. OpenCV를 이용한 초음파 영상 로딩 및 전처리
3. ONNX Runtime C++ API로 CPU 추론 수행
4. 예측 병변 마스크와 오버레이 이미지 생성
5. 추론 시간 측정 및 Python-C++ 출력 일치 검증

검증 결과:

- Numerical parity max_abs_diff: **0**
- Mask agreement: **1.0**
- Median inference: **3.8ms**
- p95 inference: **4.0ms**

이는 동일 입력 텐서 한 건에서 확인한 수치 일치 결과입니다. 전체 데이터셋에 대한 완전한 일치나 임상 정확도를 의미하지 않습니다.

## 실행 방법

Windows 11 / Python 3.11 기준입니다.

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-deploy.txt
.venv\Scripts\python.exe download_data.py
.venv\Scripts\python.exe train.py --epochs 20 --img-size 160 --batch-size 8
.venv\Scripts\python.exe evaluate.py
```

ONNX 변환 및 검증:

```bat
.venv\Scripts\python.exe export_onnx.py
.venv\Scripts\python.exe prepare_demo.py
.venv\Scripts\python.exe verify_onnx.py --image samples/demo.png
.venv\Scripts\python.exe validate_deployment.py
```

시각화 애플리케이션:

```bat
.venv\Scripts\python.exe -m streamlit run app_v2.py
```

C++ 빌드 및 실행은 [C++ ONNX Runtime 가이드](README_CPP_ONNX.md)를 참고하세요.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `models/unet.py` | U-Net 모델 정의 |
| `dataset.py` | 초음파 이미지 및 마스크 로딩 |
| `train.py` | 기본 모델 학습 |
| `train_v2.py` | 해상도 및 증강 실험 |
| `compare_v2.py` | 모델 비교 |
| `evaluate.py` | 분할 성능 평가 |
| `app_v2.py` | Streamlit 결과 시각화 |
| `export_onnx.py` | ONNX 모델 변환 |
| `verify_onnx.py` | ONNX 출력 검증 |
| `validate_deployment.py` | 전처리 차이 평가 |
| `cpp/src/main.cpp` | C++ 병변 분할 추론 |
| `cpp/CMakeLists.txt` | C++ 빌드 설정 |

## 한계 및 향후 개선

- BUSI 데이터에 신뢰할 수 있는 환자 ID가 없어 환자 단위 독립 분할을 보장하지 못합니다.
- 단일 공개 데이터셋과 제한된 실험 조건에서 얻은 성능입니다.
- Python PIL과 C++ OpenCV 전처리 방식 사이에 차이가 존재합니다.
- 일부 영상에서는 병변 분할에 실패합니다.
- 실제 초음파 장비 연동, DICOM 처리, 임상 검증은 수행하지 않았습니다.
- 전체 파이프라인 지연시간은 별도로 검증해야 합니다.

## 데이터 출처

Al-Dhabyani et al. (2020), Dataset of breast ultrasound images, Data in Brief.

- https://doi.org/10.1016/j.dib.2019.104863
- https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
