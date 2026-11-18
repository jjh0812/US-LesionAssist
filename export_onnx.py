"""Export the exact selected US-LesionAssist U-Net v1 checkpoint to ONNX.

Usage (from project root): python export_onnx.py
Requires the existing models/unet.py and outputs/best_model.pt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.unet import UNet


def main() -> None:
    p = argparse.ArgumentParser(description='Export the selected PyTorch v1 model to ONNX')
    p.add_argument('--checkpoint', type=Path, default=Path('outputs/best_model.pt'))
    p.add_argument('--out', type=Path, default=Path('exports/unet_v1.onnx'))
    args = p.parse_args()

    if not args.checkpoint.is_file():
        p.error(f'Checkpoint not found: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('smoke_test', False):
        p.error('This is a smoke-test checkpoint; use your trained full-data v1 model.')
    size = int(ckpt['img_size'])
    if size <= 0 or size % 8:
        p.error(f'Invalid input image size in checkpoint: {size}')
    model = UNet()
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    dummy = torch.zeros(1, 1, size, size, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Static input dims simplify C++ deployment. 'dynamo=False' keeps the
    # lightweight legacy exporter path explicit and supports opset 17.
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(args.out),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['logits'],
            dynamo=False,
        )

    import onnx
    graph = onnx.load(str(args.out))
    onnx.checker.check_model(graph)

    metadata = {
        'selected_checkpoint': str(args.checkpoint.as_posix()),
        'onnx_model': str(args.out.as_posix()),
        'input_name': 'images',
        'input_shape_nchw': [1, 1, size, size],
        'input_type': 'float32',
        'preprocessing': 'RGB image -> grayscale -> bilinear resize -> float32 / 255',
        'output_name': 'logits',
        'output_shape_nchw': [1, 1, size, size],
        'postprocessing': 'sigmoid(logits) >= 0.5, then nearest-neighbor resize for display',
        'model_purpose': 'Binary lesion segmentation; not malignancy classification',
        'clinical_use': 'Not validated for clinical use',
    }
    meta_path = args.out.with_suffix('.json')
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print('ONNX export completed:', args.out)
    print('ONNX graph check: PASSED')
    print('Input/output:', metadata['input_shape_nchw'], '->', metadata['output_shape_nchw'])
    print('Metadata:', meta_path)


if __name__ == '__main__':
    main()
