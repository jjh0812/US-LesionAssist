"""Train U-Net v2 on the *original frozen* BUSI split.

Place next to original train.py, dataset.py, metrics.py and models/unet.py.
Does not overwrite the v1 checkpoint or outputs/splits.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import BUSIDataset
from metrics import scores_from_logits, segmentation_loss
from models.unet import UNet


class AugmentedBUSIDataset(Dataset):
    """Apply *paired* spatial transforms to an image and its segmentation mask.

    Photometric changes affect only the image. Validation/test have no augmentation.
    """

    def __init__(self, samples: list[dict], img_size: int, augment: bool = True):
        self.base = BUSIDataset(samples, img_size)
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask = self.base[index]  # Both shape [1, H, W], in [0, 1]
        if not self.augment:
            return image, mask

        if random.random() < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        if random.random() < 0.5:
            image, mask = image.flip(-2), mask.flip(-2)
        if random.random() < 0.65:
            turns = random.randint(1, 3)
            image = torch.rot90(image, turns, dims=(-2, -1))
            mask = torch.rot90(mask, turns, dims=(-2, -1))

        if random.random() < 0.75:
            contrast = random.uniform(0.90, 1.10)
            brightness = random.uniform(-0.06, 0.06)
            image = ((image - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
        if random.random() < 0.35:
            noise_std = random.uniform(0.005, 0.02)
            image = (image + torch.randn_like(image) * noise_std).clamp(0, 1)
        return image.contiguous(), mask.contiguous()


def validate(model, loader, device):
    model.eval()
    dices, ious = [], []
    with torch.inference_mode():
        for image, mask in loader:
            logits = model(image.to(device))
            dice, iou = scores_from_logits(logits, mask.to(device))
            dices.extend(dice.cpu().tolist())
            ious.extend(iou.cpu().tolist())
    return float(np.mean(dices)), float(np.mean(ious))


def main():
    parser = argparse.ArgumentParser(description='U-Net v2: augmentation + higher input resolution')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-file', type=Path, default=Path('outputs/splits.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs_v2'))
    parser.add_argument('--no-augment', action='store_true', help='Ablation: keep resolution but disable augmentation')
    args = parser.parse_args()
    if args.img_size <= 0 or args.img_size % 8:
        parser.error('--img-size must be a positive multiple of 8')
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error('--epochs and --batch-size must be positive')
    if not args.split_file.is_file():
        parser.error(f'Missing frozen split: {args.split_file}. Complete original train.py first.')
    if args.output_dir.resolve() == Path('outputs').resolve():
        parser.error('Use a new output directory; outputs/ contains v1 baseline files.')

    split_bytes = args.split_file.read_bytes()
    split_digest = hashlib.sha256(split_bytes).hexdigest()
    splits = json.loads(split_bytes.decode('utf-8'))
    if not all(splits.get(k) for k in ('train', 'val', 'test')):
        raise RuntimeError('Frozen split must contain nonempty train/val/test lists')
    hashsets = {k: {row['image_hash'] for row in splits[k]} for k in ('train', 'val', 'test')}
    if any(hashsets[a] & hashsets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise RuntimeError('Split integrity failed: identical-image hashes cross split boundaries')
    print('Frozen split SHA256:', split_digest)
    print('Data split:', {k: len(v) for k, v in splits.items()})

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_set = AugmentedBUSIDataset(splits['train'], args.img_size, augment=not args.no_augment)
    val_set = BUSIDataset(splits['val'], args.img_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, image_count = 0.0, 0
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = segmentation_loss(logits, mask)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(image)
            image_count += len(image)
        val_dice, val_iou = validate(model, val_loader, device)
        record = {
            'epoch': epoch,
            'train_loss': loss_sum / max(1, image_count),
            'val_dice': val_dice,
            'val_iou': val_iou,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss={record['train_loss']:.4f} | "
            f"val Dice={val_dice:.4f} | val IoU={val_iou:.4f} | "
            f"{time.perf_counter()-started:.1f}s",
            flush=True,
        )
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'model_state': model.state_dict(),
                'img_size': args.img_size,
                'seed': args.seed,
                'epochs_completed': epoch,
                'best_val_dice': best_dice,
                'split_sha256': split_digest,
                'augmentation': not args.no_augment,
                'smoke_test': False,
            }, out / 'best_model.pt')

    (out / 'train_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (out / 'run_config.json').write_text(json.dumps({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'lr': args.lr,
        'seed': args.seed,
        'augmentation': not args.no_augment,
        'best_val_dice': best_dice,
        'split_sha256': split_digest,
        'note': 'Paired 90-degree rotation/flips, image-only brightness/contrast/noise; BCE+Dice remains the v1 loss.',
    }, indent=2), encoding='utf-8')
    print(f'Best validation Dice: {best_dice:.4f}')
    print('Saved:', out / 'best_model.pt')
    print('Next: python compare_v2.py --split val')


if __name__ == '__main__':
    main()
