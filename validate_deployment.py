"""Evaluate ONNX Runtime + OpenCV preprocessing on the frozen validation split.

This checks whether deployment preprocessing changes segmentation Dice/IoU.
Do not use held-out test results to iterate on the model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from dataset import BUSIDataset
from models.unet import UNet


def metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = a.astype(bool), b.astype(bool)
    both = int((a & b).sum())
    union = int((a | b).sum())
    denom = int(a.sum()) + int(b.sum())
    return (2 * both / denom if denom else 1.0, both / union if union else 1.0)


def image_cv2_unicode_safe(path: str) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f'Failed to load image: {path}')
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val'], default='val',
                        help='Only frozen validation split is allowed for development diagnostics')
    parser.add_argument('--split-file', type=Path, default=Path('outputs/splits.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/best_model.pt'))
    parser.add_argument('--model', type=Path, default=Path('exports/unet_v1.onnx'))
    parser.add_argument('--out', type=Path, default=Path('exports/deployment_val_report.json'))
    args = parser.parse_args()
    splits = json.loads(args.split_file.read_text(encoding='utf-8'))
    samples = splits['val']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    size = int(checkpoint['img_size'])
    pt = UNet()
    pt.load_state_dict(checkpoint['model_state'])
    pt.eval()
    ort_session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    dataset = BUSIDataset(samples, img_size=size)

    results = []
    diffs = []
    for i, sample in enumerate(samples):
        pil_tensor, gt_tensor = dataset[i]
        pil_input = pil_tensor.unsqueeze(0).numpy()
        truth = gt_tensor[0].numpy() >= 0.5
        bgr = image_cv2_unicode_safe(sample['image'])
        cv_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cv_resized = cv2.resize(cv_gray, (size, size), interpolation=cv2.INTER_LINEAR)
        cv_input = np.ascontiguousarray(cv_resized.astype(np.float32)[None, None] / 255.0)
        diffs.append(float(np.abs(pil_input - cv_input).mean()))
        with torch.inference_mode():
            pil_logits = pt(torch.from_numpy(pil_input))[0, 0].numpy()
        ort_logits = ort_session.run(['logits'], {'images': cv_input})[0][0, 0]
        pil_dice, pil_iou = metrics(pil_logits >= 0.0, truth)
        deploy_dice, deploy_iou = metrics(ort_logits >= 0.0, truth)
        results.append({'label': sample['label'], 'image': Path(sample['image']).name,
                        'pytorch_pil_dice': pil_dice, 'pytorch_pil_iou': pil_iou,
                        'deployment_cv2_dice': deploy_dice, 'deployment_cv2_iou': deploy_iou})

    def aggregate(rows):
        if not rows:
            return None
        return {'n': len(rows),
                'pytorch_pil_dice_mean': float(np.mean([r['pytorch_pil_dice'] for r in rows])),
                'deployment_cv2_dice_mean': float(np.mean([r['deployment_cv2_dice'] for r in rows])),
                'pytorch_pil_iou_mean': float(np.mean([r['pytorch_pil_iou'] for r in rows])),
                'deployment_cv2_iou_mean': float(np.mean([r['deployment_cv2_iou'] for r in rows])),
                'pytorch_pil_zero_dice': sum(r['pytorch_pil_dice'] == 0 for r in rows),
                'deployment_cv2_zero_dice': sum(r['deployment_cv2_dice'] == 0 for r in rows)}

    report = {'split': 'val', 'image_count': len(samples),
              'baseline': 'PyTorch + PIL preprocessing (training pipeline)',
              'deployment': 'ONNX Runtime CPU + OpenCV preprocessing (C++ pipeline)',
              'mean_abs_normalized_input_difference_pil_vs_opencv': float(np.mean(diffs)),
              'overall': aggregate(results),
              'by_category': {label: aggregate([r for r in results if r['label'] == label])
                              for label in ('benign', 'malignant', 'normal')},
              'warning': 'BUSI may contain near duplicates across splits; no patient IDs provided.',
              'notes': 'For development diagnostics only. No test-split tuning.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
