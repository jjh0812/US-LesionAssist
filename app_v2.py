"""US-LesionAssist v2: visualize prediction, BUSI ground truth, and per-image metrics.

Run from the project root: python -m streamlit run app_v2.py
Research only; not intended for clinical diagnosis.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

from models.unet import UNet

st.set_page_config(page_title='US-LesionAssist v2', layout='wide')
st.title('US-LesionAssist v2')
st.caption('유방 초음파 병변 분할 연구용 데모 | 실제 의료 진단에 사용하지 마세요')


@st.cache_resource

def load_model():
    checkpoint_path = Path('outputs/best_model.pt')
    if not checkpoint_path.exists():
        return None, None, None
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint, device


def find_busi_ground_truth(original: Image.Image, filename: str):
    """Find masks only when a local BUSI source image is pixel-identical to the upload."""
    pointer = Path('data_path.txt')
    if not pointer.exists():
        return None
    root = Path(pointer.read_text(encoding='utf-8').strip())
    if not root.is_dir():
        return None

    matches = []
    name = Path(filename).name
    for category in ('benign', 'malignant', 'normal'):
        source = root / category / name
        if source.is_file() and '_mask' not in source.stem:
            try:
                with Image.open(source) as img:
                    same_image = img.size == original.size and np.array_equal(
                        np.asarray(img.convert('RGB')), np.asarray(original)
                    )
            except (OSError, ValueError):
                same_image = False
            if same_image:
                matches.append((source, category))

    if len(matches) != 1:
        return None  # Don't silently match on filename if ambiguous or not identical.
    source, category = matches[0]
    mask_files = sorted(source.parent.glob(f'{source.stem}_mask*.png'))
    ground_truth = Image.new('L', original.size, 0)
    mask_union = np.zeros((original.height, original.width), dtype=np.uint8)
    for path in mask_files:
        with Image.open(path) as mask_img:
            if mask_img.size != original.size:
                # BUSI source and GT should match dimensions; ignore inconsistent examples.
                return None
            pixels = np.asarray(mask_img.convert('L'), dtype=np.uint8)
            mask_union = np.maximum(mask_union, (pixels > 127).astype(np.uint8))
    ground_truth = Image.fromarray(mask_union * 255)

    split_name = '확인 불가'
    split_file = Path('outputs/splits.json')
    if split_file.exists():
        try:
            splits = json.loads(split_file.read_text(encoding='utf-8'))
            for split, samples in splits.items():
                if any(Path(item['image']).resolve() == source.resolve() for item in samples):
                    split_name = split
                    break
        except (ValueError, OSError, KeyError, TypeError):
            pass

    return {'mask': ground_truth, 'category': category, 'split': split_name}


def tint_mask(image: Image.Image, mask: np.ndarray, rgb: tuple[int, int, int]):
    """Overlay a binary mask and contour without changing the base image in-place."""
    base = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if mask_u8.any():
        mixed = base.copy()
        mixed[mask_u8 > 0] = rgb
        base = cv2.addWeighted(base, 0.78, mixed, 0.22, 0)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, contours, -1, rgb, thickness=max(2, image.width // 200))
    return base


def compute_metrics(pred: np.ndarray, truth: np.ndarray):
    """Matches the current evaluate.py binary Dice/IoU empty-mask convention."""
    p, t = pred.astype(bool), truth.astype(bool)
    intersection = int(np.logical_and(p, t).sum())
    p_size, t_size = int(p.sum()), int(t.sum())
    denom = p_size + t_size
    union = int(np.logical_or(p, t).sum())
    return (2 * intersection / denom if denom else 1.0,
            intersection / union if union else 1.0)


model, checkpoint, device = load_model()
if model is None:
    st.error('`outputs/best_model.pt`가 없습니다. 먼저 train.py 학습을 완료하세요.')
    st.stop()

st.caption(f'학습 체크포인트: {checkpoint["img_size"]}×{checkpoint["img_size"]} | 실행 장치: {device}')
if checkpoint.get('smoke_test', False):
    st.warning('이 모델은 --limit 옵션으로 학습된 기능 점검용 체크포인트입니다.')

uploaded = st.file_uploader('유방 초음파 원본 이미지 업로드 (PNG / JPG)', type=['png', 'jpg', 'jpeg'])
threshold = st.slider('병변 마스크 임계값', min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if uploaded is not None:
    original = Image.open(uploaded).convert('RGB')
    size = int(checkpoint['img_size'])
    resized = original.convert('L').resize((size, size), Image.Resampling.BILINEAR)
    input_tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32)[None, None] / 255.0).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(input_tensor))[0, 0].cpu().numpy()
    pred_small = probabilities >= threshold
    pred_full = np.asarray(
        Image.fromarray(pred_small.astype(np.uint8) * 255).resize(original.size, Image.Resampling.NEAREST)
    ) > 0

    gt_data = find_busi_ground_truth(original, uploaded.name)
    if gt_data is None:
        st.warning('이 이미지와 정확하게 일치하는 로컬 BUSI 원본/정답 마스크를 찾지 못했습니다. 예측만 표시합니다.')
        first, second = st.columns(2)
        with first:
            st.image(original, caption='원본 초음파 영상', use_container_width=True)
        with second:
            st.image(tint_mask(original, pred_full, (0, 235, 120)), caption='AI 예측 (초록)', use_container_width=True)
    else:
        gt_full = np.asarray(gt_data['mask'], dtype=np.uint8) > 127
        # Evaluate at checkpoint resolution, using the same resizing rules as dataset.py.
        gt_small = np.asarray(
            gt_data['mask'].resize((size, size), Image.Resampling.NEAREST)
        ) > 127
        dice, iou = compute_metrics(pred_small, gt_small)

        st.write(f"BUSI 라벨: **{gt_data['category']}** | 데이터 분할: **{gt_data['split']}**")
        if gt_data['split'] == 'train':
            st.warning('학습셋 이미지입니다. 여기서 계산한 점수는 처음 보는 데이터에 대한 성능이 아닙니다.')
        elif gt_data['split'] == 'val':
            st.info('검증셋 이미지입니다. 이 데이터는 학습 중 모델 선택에 사용됐습니다.')
        elif gt_data['split'] == 'test':
            st.info('테스트셋 이미지입니다. 아래 점수는 이 한 이미지의 참고값이며 전체 테스트셋 평균을 대체하지 않습니다.')
        else:
            st.warning('학습/검증/테스트 분할을 확인하지 못했습니다. 점수를 일반화 성능으로 해석하지 마세요.')

        a, b, c = st.columns(3)
        with a:
            st.image(original, caption='① 원본 영상', use_container_width=True)
        with b:
            st.image(tint_mask(original, gt_full, (255, 175, 30)), caption='② 정답 마스크 (주황)', use_container_width=True)
        with c:
            st.image(tint_mask(original, pred_full, (0, 235, 120)), caption='③ AI 예측 (초록)', use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric('이 이미지의 Dice', f'{dice:.4f}')
        m2.metric('이 이미지의 IoU', f'{iou:.4f}')
        with st.expander('오류 영역 확인 (파랑 = 놓친 정답, 빨강 = 잘못 표시한 영역)'):
            tp = pred_full & gt_full
            fp = pred_full & ~gt_full
            fn = ~pred_full & gt_full
            comparison = np.asarray(original).copy()
            comparison[tp] = [0, 210, 100]
            comparison[fp] = [255, 65, 65]
            comparison[fn] = [45, 150, 255]
            st.image(comparison, caption='초록: 겹침 / 빨강: 과검출 / 파랑: 미검출', use_container_width=True)

    st.caption(f'AI 추정 마스크 면적 비율: {pred_full.mean()*100:.2f}% (이미지 전체 픽셀 대비)')
    st.info('본 모델은 병변 영역 분할 연구용입니다. 양성/악성 판별이나 의료 진단에는 사용할 수 없습니다. BUSI에는 환자 ID가 없어 환자 단위 데이터 분할을 보장하지 못합니다.')
