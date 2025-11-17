import cv2
import os
import numpy as np
from PIL import Image
import image_quality_assessor as iqa

filepath = r"E:\画質別頬画像(元画像＋10段階)\IMG_5023.jpg"

print("=" * 80)
print("IMG_5023.jpg の詳細確認")
print("=" * 80)

# ファイル情報
file_size = os.path.getsize(filepath)
print(f"\n📁 ファイル情報:")
print(f"   パス: {filepath}")
print(f"   ファイルサイズ: {file_size:,} バイト ({file_size/1024:.2f} KB)")

# 画像を読み込み（日本語パス対応）
with open(filepath, 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

if img is not None:
    height, width = img.shape[:2]
    total_pixels = width * height
    
    print(f"\n🖼️  画像情報:")
    print(f"   解像度: {width} × {height}")
    print(f"   総画素数: {total_pixels:,} 画素")
    print(f"   カラーチャンネル: {img.shape[2]}")
    print(f"   データ型: {img.dtype}")
    
    # PILで詳細情報取得
    pil_img = Image.open(filepath)
    print(f"\n📊 EXIF情報:")
    print(f"   フォーマット: {pil_img.format}")
    print(f"   モード: {pil_img.mode}")
    
    # 品質メトリクス計算
    resolution_score = (total_pixels / 2073600) * 100  # 基準: 1920×1080
    sharpness = iqa.calculate_sharpness(img)
    noise = iqa.estimate_noise_level(img)
    jpeg_quality = iqa.estimate_jpeg_quality(filepath)
    
    print(f"\n🔍 品質メトリクス:")
    print(f"   解像度スコア: {resolution_score:.2f}% (基準: 90%以上)")
    print(f"   シャープネス: {sharpness:.2f} (基準: 200以上)")
    print(f"   ノイズレベル: {noise:.2f} (基準: 30以下)")
    print(f"   JPEG品質推定: {jpeg_quality} (基準: 75以上)")
    
    # 品質判定
    quality_level = iqa.classify_quality_level(
        resolution_score, sharpness, noise, jpeg_quality
    )
    
    print(f"\n⚖️  品質判定結果: {quality_level}")
    
    # 詳細分析
    print(f"\n📈 各メトリクスの詳細分析:")
    
    # 解像度
    base_resolution = 1920 * 1080  # 2,073,600 画素
    print(f"\n   1. 解像度:")
    print(f"      - 現在: {total_pixels:,} 画素")
    print(f"      - 基準: {base_resolution:,} 画素 (1920×1080)")
    print(f"      - 比率: {(total_pixels/base_resolution)*100:.2f}%")
    print(f"      - 判定: {'✅ PASS' if resolution_score >= 90 else '❌ FAIL'} (high基準: 90%)")
    
    # シャープネス
    print(f"\n   2. シャープネス (ラプラシアン分散):")
    print(f"      - 現在: {sharpness:.2f}")
    print(f"      - 基準: 200 (high), 40 (low4-7)")
    print(f"      - 判定: {'✅ PASS (high)' if sharpness >= 200 else '⚠️ PASS (low4-7)' if sharpness >= 40 else '❌ FAIL'}")
    
    # ノイズ
    print(f"\n   3. ノイズレベル:")
    print(f"      - 現在: {noise:.2f}")
    print(f"      - 基準: 30以下 (high), 50以下 (low4-7)")
    print(f"      - 判定: {'✅ PASS (high)' if noise <= 30 else '⚠️ PASS (low4-7)' if noise <= 50 else '❌ FAIL'}")
    
    # JPEG品質
    print(f"\n   4. JPEG品質:")
    print(f"      - 現在: {jpeg_quality}")
    print(f"      - 基準: 75以上 (high), 40-75 (low4-7)")
    print(f"      - 判定: {'✅ PASS (high)' if jpeg_quality >= 75 else '⚠️ PASS (low4-7)' if jpeg_quality >= 40 else '❌ FAIL'}")
    
    # 総合判定理由
    print(f"\n🎯 判定理由の詳細:")
    
    reasons = []
    if resolution_score < 50:
        reasons.append(f"❌ 解像度が極端に低い ({resolution_score:.1f}% < 50%)")
    elif resolution_score < 90:
        reasons.append(f"⚠️ 解像度が不足 ({resolution_score:.1f}% < 90%)")
    
    if sharpness < 40:
        reasons.append(f"❌ シャープネスが低すぎる ({sharpness:.2f} < 40)")
    elif sharpness < 200:
        reasons.append(f"⚠️ シャープネスが不足 ({sharpness:.2f} < 200)")
    
    if noise > 50:
        reasons.append(f"❌ ノイズが多すぎる ({noise:.2f} > 50)")
    elif noise > 30:
        reasons.append(f"⚠️ ノイズがやや多い ({noise:.2f} > 30)")
    
    if jpeg_quality < 40:
        reasons.append(f"❌ JPEG品質が低すぎる ({jpeg_quality} < 40)")
    elif jpeg_quality < 75:
        reasons.append(f"⚠️ JPEG品質が不足 ({jpeg_quality} < 75)")
    
    if reasons:
        for reason in reasons:
            print(f"   {reason}")
    else:
        print(f"   ✅ すべての基準をクリア - 高品質画像")
    
    # この画像の用途判定
    print(f"\n💡 推奨される使用方法:")
    if quality_level == 'high':
        print(f"   ✅ 直接Box-counting解析に使用可能")
        print(f"   ✅ 高精度なフラクタル次元測定が可能")
    elif quality_level == 'low4-7':
        print(f"   ⚠️ AI予測モードでの使用を推奨")
        print(f"   ⚠️ 直接解析は精度が低下する可能性あり")
    elif quality_level == 'low1-3':
        print(f"   ⚠️ 実験的使用のみ（過学習リスクあり）")
        print(f"   ⚠️ 結果の信頼性は限定的")
    else:  # low8-10
        print(f"   ❌ この画像は使用不可")
        print(f"   ❌ 解像度が低すぎて正確な解析ができません")
        print(f"   💡 推奨: 元の高解像度画像を使用してください")
    
    # 画像の特性推測
    print(f"\n🔬 画像の特性推測:")
    if width < 300 and height < 300:
        print(f"   📌 この画像は切り出し/リサイズされた可能性が高い")
        print(f"   📌 元画像のサイズ: おそらく数百万画素以上")
        print(f"   📌 現在のサイズ: {width}×{height} = {total_pixels:,}画素")
        print(f"   💡 元の撮影画像があれば、そちらを使用してください")
    
    bytes_per_pixel = file_size / total_pixels
    if bytes_per_pixel < 0.5:
        print(f"   📌 圧縮率が高い (Byte/pixel: {bytes_per_pixel:.2f})")
        print(f"   💡 元画像はより高品質だった可能性があります")

print("\n" + "=" * 80)
