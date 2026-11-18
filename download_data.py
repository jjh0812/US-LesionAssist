"""Download the public original BUSI images from KaggleHub (network required)."""
from pathlib import Path
import kagglehub

from dataset import locate_dataset


def main():
    print('BUSI 데이터셋 다운로드 중...')
    location = kagglehub.dataset_download('aryashah2k/breast-ultrasound-images-dataset')
    root = locate_dataset(location)
    Path('data_path.txt').write_text(str(root), encoding='utf-8')
    print('다운로드 완료:', root)
    print('데이터 위치를 data_path.txt에 기록했습니다.')


if __name__ == '__main__':
    main()
