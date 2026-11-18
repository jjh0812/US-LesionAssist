"""Train a small U-Net using disjoint image groups and validation-only model selection."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset, collect_samples, locate_dataset, split_samples
from metrics import scores_from_logits, segmentation_loss
from models.unet import UNet


def select_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_loader(model, loader, device):
    model.eval()
    losses, dice_scores, iou_scores = [], [], []
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            losses.append((segmentation_loss(logits, masks).item(), images.size(0)))
            dice, iou = scores_from_logits(logits, masks)
            dice_scores.extend(dice.cpu().tolist())
            iou_scores.extend(iou.cpu().tolist())
    return {
        'loss': sum(x * n for x, n in losses) / max(sum(n for _, n in losses), 1),
        'dice': float(np.mean(dice_scores)) if dice_scores else 0.0,
        'iou': float(np.mean(iou_scores)) if iou_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=160)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='Small smoke test: max number of images, 0=all')
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()
    if args.img_size % 8 != 0:
        parser.error('--img-size must be divisible by 8')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = locate_dataset(args.data)
    samples = collect_samples(root)
    if args.limit:
        # For a smoke test, keep all three categories represented.
        groups = {label: [s for s in samples if s['label'] == label] for label in ('benign', 'malignant', 'normal')}
        rng = random.Random(args.seed)
        samples = []
        for label, group in groups.items():
            rng.shuffle(group)
            samples.extend(group[:max(3, args.limit // 3)])
    split = split_samples(samples, seed=args.seed)
    if not split['train'] or not split['val'] or not split['test']:
        raise RuntimeError('학습/검증/테스트 데이터가 충분하지 않습니다. --limit을 늘려주세요.')
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    (out / 'splits.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    print('데이터 개수:', {k: len(v) for k, v in split.items()})
    device = select_device()
    print('학습 장치:', device)
    train_loader = DataLoader(BUSIDataset(split['train'], args.img_size), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(BUSIDataset(split['val'], args.img_size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float('-inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        train_losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        validation = evaluate_loader(model, val_loader, device)
        record = {'epoch': epoch, 'train_loss': float(np.mean(train_losses)), 'val_loss': validation['loss'], 'val_dice': validation['dice'], 'val_iou': validation['iou']}
        history.append(record)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss={record['train_loss']:.4f} | val Dice={validation['dice']:.4f} | val IoU={validation['iou']:.4f} | {time.monotonic()-started:.1f}s", flush=True)
        if validation['dice'] > best:
            best = validation['dice']
            torch.save({'model_state': model.state_dict(), 'img_size': args.img_size, 'seed': args.seed, 'epochs_completed': epoch, 'smoke_test': bool(args.limit)}, out / 'best_model.pt')
    (out / 'train_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print('저장 완료: outputs/best_model.pt')
    if args.limit:
        print('주의: --limit 사용 결과는 기능 확인용이며 포트폴리오 성능 지표로 사용하면 안 됩니다.')
    print('최종 테스트: python evaluate.py')


if __name__ == '__main__':
    main()
