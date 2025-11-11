import cv2
import os
import sys

path = r"E:\画質別頬画像(元画像＋10段階)"

print("=" * 70)
print("元画像の品質確認")
print("=" * 70)

# 元画像を探す(lowが含まれていないファイル)
files = [f for f in os.listdir(path) if f.endswith('.jpg') and 'low' not in f.lower()]

print(f"\n元画像数: {len(files)}枚")
print("\n最初の5枚の詳細:\n")

for i, filename in enumerate(files[:5], 1):
    filepath = os.path.join(path, filename)
    
    # ファイルサイズ
    file_size = os.path.getsize(filepath)
    
    # OpenCVで読み込み(日本語パス対応)
    import numpy as np
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img is not None:
        height, width = img.shape[:2]
        total_pixels = width * height
        bytes_per_pixel = file_size / total_pixels
        
        print(f"{i}. {filename}")
        print(f"   解像度: {width}x{height} ({total_pixels:,}画素)")
        print(f"   ファイルサイズ: {file_size:,}バイト ({file_size/1024:.1f}KB)")
        print(f"   Byte/pixel: {bytes_per_pixel:.2f}")
        print()

print("=" * 70)
