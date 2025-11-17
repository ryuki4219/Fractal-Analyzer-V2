"""
IMG_5023.jpg 品質分析レポート
"""

import cv2
import os
import numpy as np
from PIL import Image

filepath = r"E:\画質別頬画像(元画像＋10段階)\IMG_5023.jpg"

# ファイル情報
file_size = os.path.getsize(filepath)
print("=" * 80)
print("IMG_5023.jpg 詳細分析レポート")
print("=" * 80)
print(f"\nファイルパス: {filepath}")
print(f"ファイルサイズ: {file_size:,} バイト ({file_size/1024:.2f} KB)")

# 画像読み込み（日本語パス対応）
with open(filepath, 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

if img is None:
    print("エラー: 画像を読み込めませんでした")
    exit(1)

height, width = img.shape[:2]
total_pixels = width * height

print(f"\n画像情報:")
print(f"  解像度: {width} × {height} = {total_pixels:,} 画素")
print(f"  カラー: {img.shape[2]} チャンネル")

# 基準解像度との比較
base_width, base_height = 1920, 1080
base_pixels = base_width * base_height
resolution_ratio = (total_pixels / base_pixels) * 100

print(f"\n解像度比較:")
print(f"  基準解像度: {base_width}×{base_height} = {base_pixels:,} 画素")
print(f"  現在の画像: {width}×{height} = {total_pixels:,} 画素")
print(f"  比率: {resolution_ratio:.2f}%")

if resolution_ratio < 1:
    print(f"\n⚠️ この画像は基準解像度の1%未満です")
    print(f"   → 元画像の{1/resolution_ratio*100:.0f}分の1以下のサイズです")

# 画像の特徴
bytes_per_pixel = file_size / total_pixels
print(f"\n圧縮情報:")
print(f"  Byte/画素: {bytes_per_pixel:.2f}")
if bytes_per_pixel < 0.5:
    print(f"  → 高圧縮率（元画像から大きく圧縮されています）")

print(f"\n判定:")
if width < 500 and height < 500:
    print(f"  ❌ この画像は切り出し/縮小された画像です")
    print(f"  ❌ フラクタル次元の解析には不適切です")
    print(f"\n推奨:")
    print(f"  ✅ 元の撮影画像（数千×数千ピクセル）を使用してください")
    print(f"  ✅ スマホやカメラで撮影した生の画像が必要です")
else:
    print(f"  ✅ 使用可能な解像度です")

print("=" * 80)
