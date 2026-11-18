"""Compare U-Net v1 and v2 on an identical frozen split.

Validation first. Only run --split test for the final model comparison.
Reports native-resolution scores and fair common-256px scores (not directly
identical to historical v1 native-160px metrics).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dataset import BUSIDataset
from models.unet import UNet


def mask_scores(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    pred, true = pred.astype(bool), true.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    pred_count, true_count = pred.sum(), true.sum()
    union = np.logical_or(pred, true).sum()
    return (
        float(2 * intersection / (pred_count + true_count)) if pred_count + true_count else 1.0,
        float(intersection / union) if union else 1.0,
    )


def ground_truth_at(sample: dict, size: int) -> np.ndarray:
    truth = np.zeros((size, size), dtype=bool)
    for filename in sample['masks']:
        with Image.open(filename) as img:
            resized = img.convert('L').resize((size, size), Image.Resampling.NEAREST)
            truth |= np.asarray(resized) > 127
    return truth


def check_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


def eval_model(model, checkpoint, samples, device, canonical_size=256):
    native_size = int(checkpoint['img_size'])
    ds = BUSIDataset(samples, img_size=native_size)
    rows = []
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            image, native_truth = ds[i]
            logits = model(image.unsqueeze(0).to(device))
            pred_native = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5)
            native_true = native_truth[0].numpy() >= 0.5
            native_dice, native_iou = mask_scores(pred_native, native_true)
            # Both models are compared at exactly the same pixel resolution here.
            pred_common = np.asarray(Image.fromarray(pred_native.astype(np.uint8) * 255).resize(
                (canonical_size, canonical_size), Image.Resampling.NEAREST)) > 127
            true_common = ground_truth_at(sample, canonical_size)
            common_dice, common_iou = mask_scores(pred_common, true_common)
            rows.append({
                'image': sample['image'],
                'label': sample['label'],
                'native_dice': native_dice,
                'native_iou': native_iou,
                'common_dice': common_dice,
                'common_iou': common_iou,
            })
    return rows


def aggregate(rows):
    def avg(subset, field):
        return float(np.mean([r[field] for r in subset])) if subset else None

    return {
        'n_images': len(rows),
        'mean_native_dice': avg(rows, 'native_dice'),
        'mean_native_iou': avg(rows, 'native_iou'),
        'mean_common_dice': avg(rows, 'common_dice'),
        'mean_common_iou': avg(rows, 'common_iou'),
        'zero_common_dice_count': sum(r['common_dice'] == 0 for r in rows),
        'by_category': {
            category: {
                'n': sum(r['label'] == category for r in rows),
                'mean_common_dice': avg([r for r in rows if r['label'] == category], 'common_dice'),
                'mean_common_iou': avg([r for r in rows if r['label'] == category], 'common_iou'),
            }
            for category in ('benign', 'malignant', 'normal')
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--split-file', type=Path, default=Path('outputs/splits.json'))
    parser.add_argument('--baseline', type=Path, default=Path('outputs/best_model.pt'))
    parser.add_argument('--v2', type=Path, default=Path('outputs_v2/best_model.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs_v2'))
    args = parser.parse_args()
    split_bytes = args.split_file.read_bytes()
    digest = hashlib.sha256(split_bytes).hexdigest()
    splits = json.loads(split_bytes.decode('utf-8'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v1, old_ckpt = check_checkpoint(args.baseline, device)
    v2, new_ckpt = check_checkpoint(args.v2, device)
    if new_ckpt.get('split_sha256') != digest:
        raise RuntimeError('The original data split changed since v2 training. Abort comparison.')
    print(f'Device: {device} | split: {args.split} | images: {len(splits[args.split])}')
    rows_v1 = eval_model(v1, old_ckpt, splits[args.split], device)
    rows_v2 = eval_model(v2, new_ckpt, splits[args.split], device)
    old = aggregate(rows_v1)
    new = aggregate(rows_v2)
    result = {
        'split': args.split,
        'image_count': len(rows_v1),
        'comparison_note': 'Same samples and 256px common mask evaluation. Native metrics differ in resolution. No patient-level splitting guarantee.',
        'v1': old,
        'v2': new,
        'delta_common_dice_v2_minus_v1': new['mean_common_dice'] - old['mean_common_dice'],
        'delta_common_iou_v2_minus_v1': new['mean_common_iou'] - old['mean_common_iou'],
        'per_image': [
            {'image': row1['image'], 'label': row1['label'],
             'v1_common_dice': row1['common_dice'],
             'v2_common_dice': row2['common_dice'],
             'v1_common_iou': row1['common_iou'],
             'v2_common_iou': row2['common_iou']}
            for row1, row2 in zip(rows_v1, rows_v2)
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / f'comparison_{args.split}.json'
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_image'}, indent=2, ensure_ascii=False))
    print('Saved:', dest)
    if args.split == 'val':
        print('Validation results can guide next changes. Reserve test for the final locked comparison.')
    else:
        print('Test results should be reported without further tuning on this test set.')


if __name__ == '__main__':
    main()
