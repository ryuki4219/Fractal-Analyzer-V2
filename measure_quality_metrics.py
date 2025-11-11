import cv2
import os
import numpy as np
import image_quality_assessor as iqa

path = r"E:\画質別頬画像(元画像＋10段階)"

print("=" * 70)
print("元画像の品質メトリクス測定")
print("=" * 70)

# 元画像を探す
files = [f for f in os.listdir(path) if f.endswith('.jpg') and 'low' not in f.lower()]

print(f"\n元画像数: {len(files)}枚")
print("\n最初の10枚の品質評価:\n")

for i, filename in enumerate(files[:10], 1):
    filepath = os.path.join(path, filename)
    
    # 日本語パス対応
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img is not None:
        # 品質メトリクス計算
        height, width = img.shape[:2]
        total_pixels = width * height
        # 基準解像度: 1920x1080 = 2,073,600画素
        resolution_score = (total_pixels / 2073600) * 100
        
        sharpness = iqa.calculate_sharpness(img)
        noise = iqa.estimate_noise_level(img)
        jpeg_quality = iqa.estimate_jpeg_quality(filepath)
        
        quality_level = iqa.classify_quality_level(
            resolution_score, sharpness, noise, jpeg_quality
        )
        
        print(f"{i}. {filename}")
        print(f"   判定: {quality_level}")
        print(f"   解像度: {width}x{height} (スコア: {resolution_score:.1f})")
        print(f"   シャープネス: {sharpness:.2f}")
        print(f"   ノイズレベル: {noise:.2f}")
        print(f"   JPEG品質推定: {jpeg_quality}")
        print()

print("=" * 70)
print("\n現在の品質判定基準:")
print("\n【high】")
print("  - 解像度スコア ≥ 90")
print("  - シャープネス ≥ 200")
print("  - ノイズ最大値 ≤ 30")
print("  - JPEG品質 ≥ 75")
print("\n【low4-7】")
print("  - 解像度スコア ≥ 50")
print("  - シャープネス ≥ 40")
print("  - ノイズ最大値 ≤ 50")
print("  - JPEG品質 40-75")
print("=" * 70)
