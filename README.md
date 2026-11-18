# US-LesionAssist

유방 초음파 이미지에서 병변 마스크를 예측하는 **연구용 U-Net 데모**입니다. 실제 의료 진단에 사용할 수 없습니다.

## 환경 (Windows 11 / VS Code / Python 3.11)

1. 압축을 풀고 VS Code에서 **File → Open Folder → US-LesionAssist**.
2. VS Code의 **Terminal → New Terminal**을 열고 PowerShell에서:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

별도의 PowerShell 실행 정책 변경 없이 `.venv\Scripts\python.exe`를 직접 호출합니다. VS Code에서 `Ctrl+Shift+P → Python: Select Interpreter → .venv` 선택 후, 터미널을 새로 열면 `python` 명령만 사용해도 됩니다.

### 데이터 다운로드 (인터넷 연결 필요)

```powershell
.\.venv\Scripts\python.exe download_data.py
```

원본 Kaggle BUSI 데이터셋 (`aryashah2k/breast-ultrasound-images-dataset`)을 `kagglehub`로 내려받고 캐시 경로를 `data_path.txt`에 기록합니다. 일반적으로 공개 데이터셋은 로그인 없이 접근 가능합니다. 네트워크 제한 등이 있으면 [Kaggle 원본 데이터셋](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)에서 직접 받아 압축 해제한 `Dataset_BUSI_with_GT` 경로를 `data_path.txt`에 적어도 됩니다.

### 1) 빠른 동작 테스트 (성능 측정용 **아님**)

```powershell
.\.venv\Scripts\python.exe train.py --epochs 2 --img-size 128 --batch-size 4 --limit 36
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 2) 전체 데이터 학습 (실제 결과 측정)

```powershell
.\.venv\Scripts\python.exe train.py --epochs 20 --img-size 160 --batch-size 8
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

- `outputs/best_model.pt`: 검증 Dice 기준 가장 좋은 모델
- `outputs/splits.json`: 고정된 train/val/test 이미지 분할
- `outputs/train_history.json`: 학습 기록
- `outputs/test_metrics.json`: 보류된 테스트셋 Dice/IoU/추론 시간

## 학습과 평가 설계

- U-Net, BCEWithLogits + soft Dice loss. `models/unet.py`, `metrics.py` 확인.
- 이미지-정답 마스크를 1:1 매칭하고 다수 병변 마스크는 OR 결합.
- 이미지 픽셀 기준 *완전히 동일한* 중복 이미지는 같은 split에 배정합니다.
- 학습 중에는 검증셋만 사용해 최고 모델을 선택합니다. 테스트셋은 `evaluate.py`에만 사용합니다.
- **제약:** BUSI에는 신뢰 가능한 환자 ID가 없어 환자 단위 분할을 보장하지 못합니다. 이미지의 유사 중복, 라벨 오류, 임상 적용 한계도 있습니다. Dice/IoU를 독립적인 임상 성능으로 주장하면 안 됩니다.
- **Normal image:** 정답이 비어 있을 경우 올바르게 비어있는 예측은 Dice/IoU 1.0으로 처리합니다. 최종 보고 시 정상/병변군 별도 성능도 검토하세요.
- `--limit`은 작동 확인용으로만 사용하며 결과를 이력서에 기재하지 않습니다.

## 데이터 인용

Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images.* Data in Brief, 28:104863 (2020). DOI: 10.1016/j.dib.2019.104863.

**데이터 원본:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

**참고 미러의 데이터 품질 주의:** https://huggingface.co/datasets/MedOtter/BUSI

## 다음 단계

전체 학습 성능을 확인한 뒤, 영상 노이즈 및 블러에 대한 강건성 실험을 추가할 계획입니다. 처음부터 정상/양성/악성을 분류한다고 주장하지 마세요. 이 모델은 **병변의 위치 및 경계만** 예측합니다.
