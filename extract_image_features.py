#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image


def load_resnet18():
    """ResNet18（事前学習済み）をロードし、最終FC層を除去。"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 最終FC層を除去 → 512次元の特徴量ベクトルを出力
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


def extract_features(model, image_path):
    """1枚の画像から512次元の特徴量を抽出。"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        features = model(tensor)  # (1, 512, 1, 1)
        features = features.squeeze()  # (512,)

    return features.numpy()


def compute_frame_stats(feature_vector):
    """512次元の特徴量から4統計量を計算。"""
    return {
        'mean': float(feature_vector.mean()),
        'std': float(feature_vector.std()),
        'min': float(feature_vector.min()),
        'max': float(feature_vector.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description='空の写真から16特徴量を抽出してCSVに出力'
    )
    parser.add_argument('images_dir', type=str,
                        help='画像が入っているディレクトリ')
    parser.add_argument('output_csv', type=str,
                        help='出力先CSVファイルパス')
    parser.add_argument('--times', type=int, default=4,
                        help='使用する画像の数（デフォルト4）')
    parser.add_argument('--split', type=str, default='diagnostic',
                        help='split列の値 (train/validation/diagnostic)')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f'Error: {images_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # 画像ファイルを取得（時系列順にソート）
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    ])

    if len(image_files) < args.times:
        print(f'Error: need at least {args.times} images, found {len(image_files)}',
              file=sys.stderr)
        sys.exit(1)

    print(f'Loading ResNet18 model...')
    model = load_resnet18()

    # GPUがあれば使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'Using device: {device}')

    # 各画像から特徴量を抽出
    rows = []
    for i in range(args.times):
        image_path = image_files[i]
        print(f'Processing {i+1}/{args.times}: {image_path.name}')

        feature_vector = extract_features(model, image_path)
        stats = compute_frame_stats(feature_vector)

        row = {
            'sample_id': f'img_frame{i}',
            'split': args.split,
        }
        for key in ['mean', 'std', 'min', 'max']:
            row[f'frame{i}_{key}'] = stats[key]

        rows.append(row)

    # 出力CSVを作成
    fieldnames = [
        'sample_id', 'split',
        'frame0_mean', 'frame0_std', 'frame0_min', 'frame0_max',
        'frame1_mean', 'frame1_std', 'frame1_min', 'frame1_max',
        'frame2_mean', 'frame2_std', 'frame2_min', 'frame2_max',
        'frame3_mean', 'frame3_std', 'frame3_min', 'frame3_max',
    ]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(rows[0])  # 1行だけ出力（単一予測用）

    print(f'Output written to: {output_path}')
    print(f'Features extracted: {args.times} frames × 4 stats = {args.times * 4} features')


if __name__ == '__main__':
    main()
