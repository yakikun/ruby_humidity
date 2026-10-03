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
        description='単一画像から16特徴量を抽出してCSVに出力'
    )
    parser.add_argument('image_path', type=str,
                        help='画像ファイルパス')
    parser.add_argument('output_csv', type=str,
                        help='出力先CSVファイルパス')
    parser.add_argument('--split', type=str, default='diagnostic',
                        help='split列の値 (train/validation/diagnostic)')
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.is_file():
        print(f'Error: {image_path} is not a file', file=sys.stderr)
        sys.exit(1)

    print(f'Loading ResNet18 model...')
    model = load_resnet18()

    # GPUがあれば使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'Using device: {device}')

    print(f'Processing: {image_path.name}')
    feature_vector = extract_features(model, image_path)
    stats = compute_frame_stats(feature_vector)

    # 単一画像を4フレームに複製して1行にまとめる
    row = {
        'sample_id': 'img_0',
        'split': args.split,
    }
    for i in range(4):
        stats = compute_frame_stats(feature_vector)
        for key in ['mean', 'std', 'min', 'max']:
            row[f'frame{i}_{key}'] = stats[key]

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
        writer.writerow(row)

    print(f'Output written to: {output_path}')
    print(f'Features extracted: 1 image × 4 frames × 4 stats = 16 features')


if __name__ == '__main__':
    main()
