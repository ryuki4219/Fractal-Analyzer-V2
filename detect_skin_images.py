"""
肌画像検出プログラム
画像群から肌の画像を自動的に検出してピックアップします
"""
import cv2
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict

def detect_skin_color(image):
    """
    肌色を検出する関数
    HSVとYCrCbの2つの色空間を使用して肌色領域を検出
    
    Returns:
        skin_ratio: 画像全体に占める肌色の割合 (0.0～1.0)
    """
    # BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # BGR to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # HSVでの肌色範囲 (一般的な肌色の範囲)
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    
    # YCrCbでの肌色範囲
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    
    # 2つの色空間でマスクを作成
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # 2つのマスクを結合（AND演算で精度向上）
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    # ノイズ除去（モルフォロジー演算）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 肌色ピクセルの割合を計算
    total_pixels = image.shape[0] * image.shape[1]
    skin_pixels = np.count_nonzero(mask)
    skin_ratio = skin_pixels / total_pixels
    
    return skin_ratio

def analyze_image_features(image):
    """
    画像の特徴を分析（補助的な判定）
    
    Returns:
        features: dict with various features
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 明るさの平均と標準偏差
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # エッジ検出（肌は比較的滑らかなのでエッジが少ない）
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
    
    return {
        'brightness': mean_brightness,
        'brightness_std': std_brightness,
        'edge_ratio': edge_ratio
    }

def is_skin_image(image_path, skin_threshold=0.15, verbose=False):
    """
    画像が肌画像かどうかを判定
    
    Args:
        image_path: 画像ファイルのパス
        skin_threshold: 肌色判定の閾値(デフォルト15%以上)
        verbose: 詳細情報を表示
    
    Returns:
        is_skin: 肌画像かどうか (True/False)
        skin_ratio: 肌色の割合
    """
    try:
        # 画像読み込み（日本語パス対応）
        with open(image_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return False, 0.0
        
        # 肌色検出
        skin_ratio = detect_skin_color(image)
        
        # 追加の特徴分析
        features = analyze_image_features(image)
        
        # 判定ロジック（閾値を緩和）
        is_skin = skin_ratio >= skin_threshold
        
        # エッジが多すぎる場合は除外（テクスチャ画像など）
        # ただし、肌色が多い場合は許容
        if features['edge_ratio'] > 0.5 and skin_ratio < 0.30:
            is_skin = False
        
        if verbose:
            print(f"{image_path.name}: 肌色割合={skin_ratio:.2%}, エッジ={features['edge_ratio']:.2%}, 判定={'肌画像' if is_skin else '非肌画像'}")
        
        return is_skin, skin_ratio
    
    except Exception as e:
        print(f"エラー ({image_path.name}): {e}")
        return False, 0.0

def main():
    """メイン処理"""
    print("=" * 60)
    print("🔍 肌画像検出プログラム")
    print("=" * 60)
    
    # パス設定
    base_dir = Path(r"c:\Users\iikrk\OneDrive - 神奈川工科大学\ドキュメント\GitHub\Fractal-Analyzer-V2")
    temp_dir = base_dir / "BIGDATE" / "temp"
    output_dir = base_dir / "BIGDATE" / "skin_images"
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 検索フォルダ: {temp_dir}")
    print(f"📁 出力フォルダ: {output_dir}")
    
    # 画像ファイルを検索
    print("\n🔍 画像ファイルを検索中...")
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        image_files.extend(list(temp_dir.rglob(f'*{ext}')))
    
    print(f"✅ {len(image_files)}枚の画像が見つかりました")
    
    if len(image_files) == 0:
        print("❌ 画像が見つかりませんでした")
        return
    
    # 肌画像を検出
    print("\n🔬 肌画像を検出中...")
    print("-" * 60)
    
    skin_images = []
    skin_ratios = []
    
    for idx, img_path in enumerate(image_files, 1):
        # 進捗表示
        if idx % 100 == 0 or idx == len(image_files):
            print(f"進捗: {idx}/{len(image_files)} ({idx/len(image_files)*100:.1f}%)")
        
        is_skin, skin_ratio = is_skin_image(img_path, skin_threshold=0.05, verbose=False)
        
        if is_skin:
            skin_images.append(img_path)
            skin_ratios.append(skin_ratio)
    
    print("-" * 60)
    print(f"\n✅ 肌画像を {len(skin_images)}枚 検出しました！")
    
    if len(skin_images) == 0:
        print("❌ 肌画像が見つかりませんでした")
        print("\n💡 ヒント:")
        print("  - 閾値を下げる（デフォルト15%）")
        print("  - 画像のサンプルを確認して調整が必要か判断")
        return
    
    # 肌画像をコピー
    print(f"\n📋 肌画像を {output_dir} にコピー中...")
    
    # 肌色割合で降順ソート
    sorted_indices = np.argsort(skin_ratios)[::-1]
    
    for idx, sort_idx in enumerate(sorted_indices, 1):
        src_path = skin_images[sort_idx]
        ratio = skin_ratios[sort_idx]
        
        # ファイル名: skin_001_85.2%.png (肌色割合を含む)
        dst_name = f"skin_{idx:04d}_{ratio*100:.1f}%{src_path.suffix}"
        dst_path = output_dir / dst_name
        
        shutil.copy2(src_path, dst_path)
        
        if idx <= 10:  # 最初の10枚を表示
            print(f"  {idx}. {src_path.name} → {dst_name} (肌色: {ratio:.1%})")
    
    print("\n" + "=" * 60)
    print("✅ 完了！")
    print("=" * 60)
    print(f"\n📊 結果:")
    print(f"  - 総画像数: {len(image_files)}枚")
    print(f"  - 肌画像数: {len(skin_images)}枚 ({len(skin_images)/len(image_files)*100:.1f}%)")
    print(f"  - 保存先: {output_dir}")
    print(f"\n💡 肌色割合の範囲:")
    print(f"  - 最大: {max(skin_ratios)*100:.1f}%")
    print(f"  - 最小: {min(skin_ratios)*100:.1f}%")
    print(f"  - 平均: {np.mean(skin_ratios)*100:.1f}%")
    
    # 統計情報をテキストファイルに保存
    stats_file = output_dir / "detection_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("肌画像検出結果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"総画像数: {len(image_files)}枚\n")
        f.write(f"肌画像数: {len(skin_images)}枚 ({len(skin_images)/len(image_files)*100:.1f}%)\n")
        f.write(f"\n肌色割合:\n")
        f.write(f"  最大: {max(skin_ratios)*100:.1f}%\n")
        f.write(f"  最小: {min(skin_ratios)*100:.1f}%\n")
        f.write(f"  平均: {np.mean(skin_ratios)*100:.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("検出された画像一覧:\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, sort_idx in enumerate(sorted_indices, 1):
            src_path = skin_images[sort_idx]
            ratio = skin_ratios[sort_idx]
            f.write(f"{idx:4d}. {src_path.name} (肌色: {ratio:.1%})\n")
    
    print(f"\n📄 統計情報を保存しました: {stats_file}")

if __name__ == "__main__":
    main()
