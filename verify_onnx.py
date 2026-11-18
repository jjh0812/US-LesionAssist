"""Check PyTorch-vs-ONNX Runtime numerical parity on exactly the same float tensor.

Also exports a float32 NCHW input and the ONNX logits for C++ parity testing.
Usage: python verify_onnx.py --image samples/demo.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

from models.unet import UNet


def preprocess_cv2(image_path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.astype(np.float32)[None, None] / 255.0)


def median_and_p95(series: list[float]) -> dict[str, float]:
    return {
        'median_ms': float(np.median(series)),
        'p95_ms': float(np.percentile(series, 95)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=Path('outputs/best_model.pt'))
    p.add_argument('--model', type=Path, default=Path('exports/unet_v1.onnx'))
    p.add_argument('--image', type=Path, default=Path('samples/demo.png'))
    p.add_argument('--out-dir', type=Path, default=Path('exports'))
    p.add_argument('--runs', type=int, default=50)
    args = p.parse_args()
    if args.runs < 1:
        p.error('--runs must be >= 1')
    if not args.model.is_file():
        p.error('No ONNX file. Run python export_onnx.py first.')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    size = int(ckpt['img_size'])
    x = preprocess_cv2(args.image, size)
    model = UNet()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    with torch.inference_mode():
        pt_logits = model(torch.from_numpy(x)).numpy()

    # CPU-only provider to allow an apples-to-apples C++ ORT CPU comparison.
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].name != 'images' or session.get_outputs()[0].name != 'logits':
        raise RuntimeError('Unexpected ONNX input/output names')
    actual = session.run(['logits'], {'images': x})[0]
    if actual.shape != pt_logits.shape:
        raise RuntimeError(f'Shape mismatch: torch={pt_logits.shape}, ort={actual.shape}')
    abs_diff = np.abs(pt_logits - actual)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    torch_mask = pt_logits >= 0.0  # sigmoid(x) >= 0.5, stable comparison
    ort_mask = actual >= 0.0
    mask_agreement = float(np.mean(torch_mask == ort_mask))
    ok = bool(np.allclose(pt_logits, actual, rtol=1e-3, atol=1e-4))
    if not ok:
        print('WARNING: numerical parity exceeds tolerance; do not claim equivalent inference.')

    # Warm up before measuring model.Run(), excluding loading and preprocessing.
    for _ in range(10):
        session.run(['logits'], {'images': x})
    ort_ms = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(['logits'], {'images': x})
        ort_ms.append((time.perf_counter() - start) * 1000)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'parity_input.bin').write_bytes(x.astype('<f4').tobytes(order='C'))
    (args.out_dir / 'parity_ort_logits.bin').write_bytes(actual.astype('<f4').tobytes(order='C'))
    report = {
        'status': 'PASS' if ok else 'FAIL',
        'image': str(args.image),
        'input_shape': list(x.shape),
        'max_abs_logit_diff_pytorch_vs_onnx': max_abs,
        'mean_abs_logit_diff_pytorch_vs_onnx': mean_abs,
        'mask_agreement_at_threshold_0_5': mask_agreement,
        'rtol': 1e-3, 'atol': 1e-4,
        'onnxruntime_execution_provider': session.get_providers(),
        'onnxruntime_model_only_cpu_latency': median_and_p95(ort_ms),
        'warmup_runs': 10, 'timed_runs': args.runs,
        'note': 'Model-only CPU inference; excludes image I/O, preprocessing and visualization.',
    }
    dest = args.out_dir / 'onnx_parity_report.json'
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Saved exact-input binary:', args.out_dir / 'parity_input.bin')
    print('Saved ONNX-reference logits:', args.out_dir / 'parity_ort_logits.bin')
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
