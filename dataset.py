"""Data discovery, duplicate-aware splitting, and mask loading for the BUSI dataset."""
from collections import defaultdict
import hashlib
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CLASSES = ('benign', 'malignant', 'normal')


def locate_dataset(folder: str | Path | None = None) -> Path:
    candidates = []
    if folder:
        candidates.append(Path(folder))
    pointer = Path('data_path.txt')
    if pointer.exists():
        candidates.append(Path(pointer.read_text(encoding='utf-8').strip()))
    candidates.append(Path('data'))
    for candidate in candidates:
        if not candidate.exists():
            continue
        if all((candidate / c).is_dir() for c in CLASSES):
            return candidate.resolve()
        for nested in candidate.rglob('Dataset_BUSI_with_GT'):
            if all((nested / c).is_dir() for c in CLASSES):
                return nested.resolve()
    raise FileNotFoundError(
        'BUSI 데이터를 찾지 못했습니다. 먼저 `python download_data.py`를 실행하세요.'
    )


def collect_samples(root: Path) -> list[dict[str, Any]]:
    samples = []
    for label in CLASSES:
        for image_path in sorted((root / label).glob('*.png')):
            if '_mask' in image_path.stem:
                continue
            masks = sorted(image_path.parent.glob(f'{image_path.stem}_mask*.png'))
            if not masks and label != 'normal':
                raise FileNotFoundError(f'정답 마스크가 없습니다: {image_path}')
            # Hash raw pixels after canonical resizing to group exact image duplicates.
            with Image.open(image_path) as im:
                pixels = im.convert('L').resize((128, 128), Image.Resampling.BILINEAR).tobytes()
            image_hash = hashlib.sha256(pixels).hexdigest()
            samples.append({
                'image': str(image_path.resolve()),
                'masks': [str(m.resolve()) for m in masks],
                'label': label,
                'image_hash': image_hash,
            })
    if not samples:
        raise RuntimeError(f'이미지 없음: {root}')
    return samples


def split_samples(samples: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Keep pixel-identical images together; not a patient-level guarantee."""
    by_hash = defaultdict(list)
    for sample in samples:
        by_hash[sample['image_hash']].append(sample)
    by_label = defaultdict(list)
    for group in by_hash.values():
        # When duplicate images have differing labels, keep them in one split.
        by_label[group[0]['label']].append(group)
    rng = random.Random(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for label, groups in sorted(by_label.items()):
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * 0.15)) if n >= 3 else 0
        n_val = max(1, round(n * 0.15)) if n >= 3 else 0
        for i, group in enumerate(groups):
            dest = 'test' if i < n_test else 'val' if i < n_test + n_val else 'train'
            splits[dest].extend(group)
    for key in splits:
        rng.shuffle(splits[key])
    return splits


class BUSIDataset(Dataset):
    def __init__(self, samples: list[dict], img_size: int = 160):
        self.samples = samples
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample['image']) as img:
            image = np.asarray(
                img.convert('L').resize((self.img_size, self.img_size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for path in sample['masks']:
            with Image.open(path) as item:
                binary = np.asarray(
                    item.convert('L').resize((self.img_size, self.img_size), Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            mask = np.maximum(mask, (binary > 127).astype(np.uint8))
        return (
            torch.from_numpy(image[None, ...].copy()),
            torch.from_numpy(mask[None, ...].astype(np.float32)),
        )
