"""Research-only breast ultrasound lesion segmentation viewer."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

from models.unet import UNet

st.set_page_config(page_title='US-LesionAssist', layout='wide')
st.title('US-LesionAssist')
st.caption('유방 초음파 병변 분할 연구용 데모 — 의료 진단 목적 사용 금지')


@st.cache_resource
def load_model():
    path = Path('outputs/best_model.pt')
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = UNet()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


model, checkpoint = load_model()
if model is None:
    st.warning('학습된 모델이 없습니다. 먼저 `python train.py`를 실행하세요.')
    st.stop()
if checkpoint.get('smoke_test'):
    st.warning('현재는 샘플 데이터로 빠르게 학습한 모델입니다. 성능 검증 전입니다.')

uploaded = st.file_uploader('유방 초음파 이미지 업로드 (PNG / JPG)', type=['png', 'jpg', 'jpeg'])
threshold = st.slider('병변 확률 마스크 임계값', 0.1, 0.9, 0.5, 0.05)
if uploaded is not None:
    original = Image.open(uploaded).convert('RGB')
    gray = original.convert('L').resize((checkpoint['img_size'], checkpoint['img_size']), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(gray, dtype=np.float32)[None, None] / 255.0)
    with torch.inference_mode():
        prediction = torch.sigmoid(model(x))[0, 0].numpy()
    full_mask = Image.fromarray(np.uint8(prediction >= threshold) * 255).resize(original.size, Image.Resampling.NEAREST)
    mask = np.asarray(full_mask, dtype=np.uint8)
    overlay = np.asarray(original).copy()
    if mask.any():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 120), thickness=max(2, original.width // 200))
        color_mask = overlay.copy()
        color_mask[mask > 0] = (0, 220, 110)
        overlay = cv2.addWeighted(overlay, 0.78, color_mask, 0.22, 0)
    a, b = st.columns(2)
    with a:
        st.image(original, caption='원본 초음파 영상', use_container_width=True)
    with b:
        st.image(overlay, caption='AI 추정 병변 영역 (초록색)', use_container_width=True)
    st.write(f'마스크 영역 비율: {(mask > 0).mean() * 100:.2f}%')
    st.info('이 값은 병변의 양성/악성 확률이나 실제 병변 크기가 아닙니다. 모델 예측은 임상적으로 검증되지 않았습니다.')
