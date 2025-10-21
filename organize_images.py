"""
画像整理スクリプト
解凍した画像をBIGDATEフォルダにimage1, image2...という名前でコピー
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# パス設定
base_dir = Path(r"c:\Users\iikrk\OneDrive - 神奈川工科大学\ドキュメント\GitHub\Fractal-Analyzer-V2")
temp_dir = base_dir / "BIGDATE" / "temp"
output_dir = base_dir / "BIGDATE" / "images"

# 出力ディレクトリ作成
output_dir.mkdir(parents=True, exist_ok=True)

# 画像ファイルを取得
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
image_files = []

print("画像ファイルを検索中...")
for ext in image_extensions:
    image_files.extend(temp_dir.rglob(f'*{ext}'))

print(f"見つかった画像: {len(image_files)}枚")

# リネームしてコピー
print("画像をコピー中...")
for idx, img_path in enumerate(tqdm(image_files), start=1):
    # 拡張子を取得
    ext = img_path.suffix
    # 新しい名前
    new_name = f"image{idx}{ext}"
    new_path = output_dir / new_name
    # コピー
    shutil.copy2(img_path, new_path)

print(f"\n完了！{len(image_files)}枚の画像を整理しました。")
print(f"保存先: {output_dir}")
