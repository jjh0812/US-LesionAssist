"""Copy a publicly available BUSI validation image to an ASCII-only local path.

The demo is for development; do not use the test split for tuning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split-file', type=Path, default=Path('outputs/splits.json'))
    p.add_argument('--out', type=Path, default=Path('samples/demo.png'))
    p.add_argument('--filename', type=str, default='malignant (63).png',
                   help='Example must exist in the frozen validation split')
    args = p.parse_args()
    split = json.loads(args.split_file.read_text(encoding='utf-8'))
    candidates = [s for s in split['val'] if s['label'] == 'malignant'
                  and Path(s['image']).name == args.filename]
    if not candidates:
        p.error(f'No matching malignant validation sample: {args.filename}')
    # Known validation demonstration example only; never report its score as an average.
    candidate = candidates[0]
    path = Path(candidate['image'])
    if not path.is_file():
        p.error(f'Image missing: {path}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, args.out)
    print('Copied validation demo:', path.name, '->', args.out)
    print('Category:', candidate['label'], '| split: val')


if __name__ == '__main__':
    main()
