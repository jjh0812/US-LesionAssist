from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from dataset import BUSIDataset, collect_samples, split_samples
from metrics import scores_from_logits, segmentation_loss
from models.unet import UNet


def test_unet_shape_and_gradient():
    model = UNet(base_channels=8)
    x = torch.rand(2, 1, 64, 64)
    y = torch.zeros(2, 1, 64, 64)
    prediction = model(x)
    assert prediction.shape == y.shape
    segmentation_loss(prediction, y).backward()
    assert model.out.weight.grad is not None


def test_empty_mask_score():
    d, i = scores_from_logits(torch.full((1, 1, 8, 8), -10.0), torch.zeros(1, 1, 8, 8))
    assert d.item() == 1.0 and i.item() == 1.0


def test_dataset_and_splits():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label in ('benign', 'malignant', 'normal'):
            (root / label).mkdir()
            for n in range(6):
                base = np.full((32, 32), (n * 30 + len(label) * 3) % 255, dtype=np.uint8)
                base[5:12, 6:13] = n * 20
                img = root / label / f'{label} ({n}).png'
                Image.fromarray(base).save(img)
                if label != 'normal':
                    mask = np.zeros((32, 32), dtype=np.uint8)
                    mask[5:12, 6:13] = 255
                    Image.fromarray(mask).save(root / label / f'{label} ({n})_mask.png')
        samples = collect_samples(root)
        assert len(samples) == 18
        splits = split_samples(samples)
        assert len(splits['train']) + len(splits['val']) + len(splits['test']) == 18
        hashes = {k: {s['image_hash'] for s in v} for k, v in splits.items()}
        assert not (hashes['train'] & hashes['test'])
        assert not (hashes['train'] & hashes['val'])
        x, y = BUSIDataset(splits['train'], img_size=64)[0]
        assert x.shape == y.shape == (1, 64, 64)
