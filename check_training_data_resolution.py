import cv2
import os
import numpy as np
from collections import defaultdict

skin_data_dir = r"c:\Users\iikrk\OneDrive - 神奈川工科大学\ドキュメント\GitHub\Fractal-Analyzer-V2\SKIN_DATA"

print("=" * 80)
print("学習データ（SKIN_DATA）の解像度確認")
print("=" * 80)

# 各フォルダの画像を確認
resolutions = []
file_count = 0

for root, dirs, files in os.walk(skin_data_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(root, file)
            img = cv2.imread(filepath)
            
            if img is not None:
                height, width = img.shape[:2]
                total_pixels = width * height
                resolutions.append({
                    'file': os.path.relpath(filepath, skin_data_dir),
                    'width': width,
                    'height': height,
                    'pixels': total_pixels,
                    'resolution_score': (total_pixels / 2073600) * 100  # 基準: 1920x1080
                })
                file_count += 1

# 解像度でソート
resolutions.sort(key=lambda x: x['pixels'])

print(f"\n総ファイル数: {file_count}枚\n")

# 統計情報
pixels_list = [r['pixels'] for r in resolutions]
scores_list = [r['resolution_score'] for r in resolutions]

print("📊 解像度統計:")
print(f"   最小: {min(pixels_list):,} 画素 ({min(scores_list):.1f}%)")
print(f"   最大: {max(pixels_list):,} 画素 ({max(scores_list):.1f}%)")
print(f"   平均: {np.mean(pixels_list):,.0f} 画素 ({np.mean(scores_list):.1f}%)")
print(f"   中央値: {np.median(pixels_list):,.0f} 画素 ({np.median(scores_list):.1f}%)")

# 分布確認
print(f"\n📈 解像度スコア分布:")
ranges = [
    ("90%以上 (high基準)", 90, 1000),
    ("50-90% (low4-7上限)", 50, 90),
    ("25-50%", 25, 50),
    ("10-25%", 10, 25),
    ("10%未満 (very low)", 0, 10)
]

for label, min_score, max_score in ranges:
    count = sum(1 for s in scores_list if min_score <= s < max_score)
    percentage = (count / file_count) * 100
    print(f"   {label}: {count}枚 ({percentage:.1f}%)")

# 最小・最大の画像例を表示
print(f"\n🔍 最小解像度の画像（サンプル5枚）:")
for i, r in enumerate(resolutions[:5], 1):
    print(f"   {i}. {r['file']}")
    print(f"      {r['width']}×{r['height']} = {r['pixels']:,}画素 (スコア: {r['resolution_score']:.1f}%)")

print(f"\n🔍 最大解像度の画像（サンプル5枚）:")
for i, r in enumerate(resolutions[-5:], 1):
    print(f"   {i}. {r['file']}")
    print(f"      {r['width']}×{r['height']} = {r['pixels']:,}画素 (スコア: {r['resolution_score']:.1f}%)")

# 50%以上の画像を確認
high_res_images = [r for r in resolutions if r['resolution_score'] >= 50]
print(f"\n✅ 解像度スコア50%以上の画像: {len(high_res_images)}枚")
if high_res_images:
    print(f"   最小解像度: {min(r['width'] for r in high_res_images)}×{min(r['height'] for r in high_res_images)}")
    print(f"   平均解像度: {np.mean([r['width'] for r in high_res_images]):.0f}×{np.mean([r['height'] for r in high_res_images]):.0f}")

# 1000x1000以上の画像を確認
very_high_res = [r for r in resolutions if r['width'] >= 1000 and r['height'] >= 1000]
print(f"\n🔍 1000×1000以上の画像: {len(very_high_res)}枚")
if very_high_res:
    for r in very_high_res[:5]:
        print(f"   - {r['file']}: {r['width']}×{r['height']}")

# 実際の解像度分布を確認
print(f"\n📏 実際の解像度（幅×高さ）の分布:")
resolution_groups = defaultdict(int)
for r in resolutions:
    key = f"{r['width']}×{r['height']}"
    resolution_groups[key] += 1

# 上位10個を表示
sorted_groups = sorted(resolution_groups.items(), key=lambda x: x[1], reverse=True)
for resolution, count in sorted_groups[:10]:
    percentage = (count / file_count) * 100
    print(f"   {resolution}: {count}枚 ({percentage:.1f}%)")

print("=" * 80)
