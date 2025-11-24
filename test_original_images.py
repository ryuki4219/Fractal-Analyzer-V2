# -*- coding: utf-8 -*-
import os
import glob
import sys
import io
from image_quality_assessor import assess_image_quality

# Windows環境でUTF-8出力を強制
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 元画像のみ評価（_lowが含まれないファイル）
all_files = glob.glob(r'E:\画質別頬画像(元画像＋10段階)\*.jpg')
original_files = [f for f in all_files if '_low' not in os.path.basename(f).lower()]

print(f"元画像数: {len(original_files)}枚\n")
print("=" * 100)

quality_counts = {'high': 0, 'low4-7': 0, 'low1-3': 0, 'low8-10': 0}

for f in sorted(original_files):
    result = assess_image_quality(f)
    if 'error' not in result:
        m = result['metrics']
        ql = result['quality_level']
        quality_counts[ql] += 1
        
        status = "[OK]" if ql == "high" else "[AI]" if ql == "low4-7" else "[NG]"
        print(f"{status} {os.path.basename(f):30s} | {ql:8s} | {m['resolution']:12s} | res={m['resolution_score']:5.1f}% | sharp={m['sharpness']:5.1f} | jpeg={m['estimated_jpeg_quality']}")

print("\n" + "=" * 100)
print("\n[元画像の品質分布]")
print(f"  [OK] high (直接解析):      {quality_counts['high']:3d}枚 ({quality_counts['high']/len(original_files)*100:5.1f}%)")
print(f"  [AI] low4-7 (AI予測):      {quality_counts['low4-7']:3d}枚 ({quality_counts['low4-7']/len(original_files)*100:5.1f}%)")
print(f"  [OK] low1-3 (直接解析推奨): {quality_counts['low1-3']:3d}枚 ({quality_counts['low1-3']/len(original_files)*100:5.1f}%)")
print(f"  [NG] low8-10 (解析不可):    {quality_counts['low8-10']:3d}枚 ({quality_counts['low8-10']/len(original_files)*100:5.1f}%)")
print(f"\n合計直接解析可能: {quality_counts['high'] + quality_counts['low1-3']}枚 / {len(original_files)}枚 ({(quality_counts['high'] + quality_counts['low1-3'])/len(original_files)*100:.1f}%)")
