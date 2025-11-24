# -*- coding: utf-8 -*-
import os
import glob
import sys
import io
from image_quality_assessor import assess_image_quality

# Windows環境でUTF-8出力を強制
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 最初の30枚をサンプリング
files = glob.glob(r'E:\画質別頬画像(元画像＋10段階)\*.jpg')[:30]

for f in files:
    result = assess_image_quality(f)
    if 'error' not in result:
        m = result['metrics']
        print(f"{os.path.basename(f):40s} | {result['quality_level']:8s} | {m['resolution']:12s} | res_score={m['resolution_score']:5.1f}% | sharp={m['sharpness']:5.1f}")
