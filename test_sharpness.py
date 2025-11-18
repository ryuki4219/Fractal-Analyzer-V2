import cv2
import numpy as np
from PIL import Image
import os

def calculate_sharpness_original(image):
    """現在の実装"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness

def calculate_sharpness_normalized(image):
    """正規化版 - 画像サイズに影響されにくい"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # ラプラシアンフィルタを適用
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 絶対値の平均を使用（正規化）
    sharpness = np.mean(np.abs(laplacian))
    
    return sharpness

def calculate_sharpness_tenengrad(image):
    """Tenengrad法 - より安定的"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sobelフィルタで勾配を計算
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 勾配の大きさ
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # 平均勾配強度
    sharpness = np.mean(gradient_magnitude)
    
    return sharpness

def calculate_sharpness_variance_of_laplacian(image):
    """分散法の改良版"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 正規化
    gray = gray.astype(np.float32) / 255.0
    
    # ラプラシアンフィルタを適用
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    
    # 分散を計算
    sharpness = laplacian.var() * 1000  # スケーリング
    
    return sharpness

# テスト用画像を生成
def create_test_images():
    """異なる鮮明度のテスト画像を生成"""
    # 高鮮明度画像（エッジが多い）
    sharp = np.zeros((1000, 1000), dtype=np.uint8)
    for i in range(0, 1000, 20):
        sharp[i:i+10, :] = 255
        sharp[:, i:i+10] = 255
    
    # 中鮮明度（ランダムノイズ）
    medium = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    medium = cv2.GaussianBlur(medium, (5, 5), 0)
    
    # 低鮮明度（ぼかし）
    blurred = cv2.GaussianBlur(sharp, (51, 51), 0)
    
    # 肌のようなテクスチャ
    skin = np.ones((1000, 1000), dtype=np.uint8) * 180
    noise = np.random.randint(-10, 10, (1000, 1000), dtype=np.int16)
    skin = np.clip(skin + noise, 0, 255).astype(np.uint8)
    
    return {
        'sharp': sharp,
        'medium': medium,
        'blurred': blurred,
        'skin': skin
    }

# 実際の画像でテスト
def test_real_images():
    """実際のSKIN_DATAでテスト"""
    results = []
    
    for i in range(1, 10):
        img_path = f"SKIN_DATA/{i}/front.jpg"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                
                original = calculate_sharpness_original(img)
                normalized = calculate_sharpness_normalized(img)
                tenengrad = calculate_sharpness_tenengrad(img)
                variance = calculate_sharpness_variance_of_laplacian(img)
                
                results.append({
                    'file': img_path,
                    'size': f"{w}x{h}",
                    'pixels': w*h,
                    'original': round(original, 2),
                    'normalized': round(normalized, 2),
                    'tenengrad': round(tenengrad, 2),
                    'variance': round(variance, 2)
                })
    
    return results

# メイン実行
if __name__ == "__main__":
    print("=" * 80)
    print("シャープネス計算方法の比較テスト")
    print("=" * 80)
    
    # 合成画像でテスト
    print("\n【合成画像テスト】")
    test_images = create_test_images()
    
    for name, img in test_images.items():
        original = calculate_sharpness_original(img)
        normalized = calculate_sharpness_normalized(img)
        tenengrad = calculate_sharpness_tenengrad(img)
        variance = calculate_sharpness_variance_of_laplacian(img)
        
        print(f"\n{name.upper()}:")
        print(f"  Original (var):   {original:.2f}")
        print(f"  Normalized (abs): {normalized:.2f}")
        print(f"  Tenengrad:        {tenengrad:.2f}")
        print(f"  Variance (norm):  {variance:.2f}")
    
    # 実画像でテスト
    print("\n" + "=" * 80)
    print("【実画像テスト - SKIN_DATA】")
    print("=" * 80)
    
    results = test_real_images()
    
    if results:
        print(f"\n{'ファイル':<30} {'サイズ':<15} {'Original':<12} {'Normalized':<12} {'Tenengrad':<12} {'Variance':<12}")
        print("-" * 100)
        for r in results:
            print(f"{r['file']:<30} {r['size']:<15} {r['original']:<12} {r['normalized']:<12} {r['tenengrad']:<12} {r['variance']:<12}")
    else:
        print("SKIN_DATAが見つかりませんでした")
