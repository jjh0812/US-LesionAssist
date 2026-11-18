"""Evaluate the saved best-validation checkpoint exactly once on held-out test images."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset
from metrics import scores_from_logits
from models.unet import UNet


def main():
    out = Path('outputs')
    split = json.loads((out / 'splits.json').read_text(encoding='utf-8'))
    ckpt = torch.load(out / 'best_model.pt', map_location='cpu', weights_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    loader = DataLoader(BUSIDataset(split['test'], ckpt['img_size']), batch_size=1, num_workers=0)
    dice_values, iou_values, latencies = [], [], []
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            dice, iou = scores_from_logits(logits.cpu(), masks)
            dice_values.extend(dice.tolist())
            iou_values.extend(iou.tolist())
    result = {
        'n_test_images': len(dice_values),
        'test_dice_mean': float(np.mean(dice_values)),
        'test_iou_mean': float(np.mean(iou_values)),
        'inference_latency_median_ms': float(np.median(latencies)),
        'inference_device': str(device),
        'smoke_test': bool(ckpt.get('smoke_test', False)),
        'caveat': 'Image-exact duplicates grouped, but patient-level IDs are unavailable; near-duplicate leakage remains possible.',
    }
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if result['smoke_test']:
        print('주의: 빠른 기능 테스트 결과이며 최종 성능으로 해석하지 마세요.')


if __name__ == '__main__':
    main()
