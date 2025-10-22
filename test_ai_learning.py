"""
AI学習機能のテストプログラム
フラクタル次元のAI予測が正しく動作するか確認します
"""
import cv2
import numpy as np
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# GPU対応
USE_CUPY = False
try:
    import cupy as cp
    _ = cp.zeros(1)
    USE_CUPY = True
    xp = cp
except:
    USE_CUPY = False
    xp = np

def to_xp(arr):
    return cp.asarray(arr) if USE_CUPY else np.asarray(arr)

def to_host(arr):
    return cp.asnumpy(arr) if USE_CUPY else arr

def calculate_fd(img_bgr, scales=(2,4,8,16,32,64)):
    """フラクタル次元を計算"""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = img_gray.shape
    
    Nh_vals = []
    valid_scales = []
    
    for h in scales:
        Hc = (H // h) * h
        Wc = (W // h) * h
        if Hc < h or Wc < h:
            continue
        
        gray_crop = img_gray[:Hc, :Wc]
        arr = to_xp(gray_crop)
        
        new_shape = (Hc//h, h, Wc//h, h)
        blocks = arr.reshape(new_shape).transpose(0,2,1,3)
        
        mean_blk = blocks.mean(axis=(2,3))
        sq_mean = (blocks**2).mean(axis=(2,3))
        std_blk = xp.sqrt(xp.maximum(0, sq_mean - mean_blk**2))
        
        nh = std_blk / float(h)
        nh_total = float(to_host(nh.sum()))
        
        Nh_vals.append(nh_total + 1e-12)
        valid_scales.append(h)
    
    if len(valid_scales) < 3:
        return None, [], []
    
    log_h = np.log(np.array(valid_scales, dtype=np.float64))
    log_Nh = np.log(np.array(Nh_vals, dtype=np.float64))
    
    coeffs = np.polyfit(log_h, log_Nh, 1)
    D = abs(coeffs[0])
    
    return float(D), valid_scales, Nh_vals

def extract_features(img_bgr):
    """特徴量抽出"""
    gray = cv2.cvtColor(cv2.resize(img_bgr, (256, 256)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mean = float(np.mean(np.sqrt(gx**2 + gy**2)))
    
    noise_level = float(np.mean(np.abs(gray - cv2.GaussianBlur(gray, (3,3), 1))))
    
    probs, _ = np.histogram(gray.flatten(), bins=256, range=(0,255), density=True)
    probs = probs + 1e-12
    entropy = -np.sum(probs * np.log2(probs))
    
    return [mean_val, std_val, edge_mean, noise_level, entropy]

def create_test_images(n_samples=10):
    """テスト用の画像ペアを生成"""
    print("テスト画像を生成中...")
    
    high_imgs = []
    low_imgs = []
    
    for i in range(n_samples):
        # 高画質画像（512x512）
        img_high = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # ランダムなパターンを追加
        for _ in range(50):
            x, y = np.random.randint(0, 450), np.random.randint(0, 450)
            size = np.random.randint(20, 80)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.rectangle(img_high, (x, y), (x+size, y+size), color, -1)
        
        # 低画質画像（ダウンサンプリング+ノイズ）
        img_low = cv2.resize(img_high, (128, 128))
        img_low = cv2.resize(img_low, (512, 512))
        
        # ノイズ追加
        noise = np.random.normal(0, 20, img_low.shape).astype(np.int16)
        img_low = np.clip(img_low.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # ガウシアンブラー
        img_low = cv2.GaussianBlur(img_low, (5, 5), 1.5)
        
        high_imgs.append(img_high)
        low_imgs.append(img_low)
    
    print(f"✅ {n_samples}組の画像ペアを生成しました")
    return high_imgs, low_imgs

def test_ai_learning():
    """AI学習のテスト"""
    print("="*60)
    print("🔬 AI学習機能テスト開始")
    print("="*60)
    
    # テスト画像生成
    n_train = 15
    n_test = 5
    
    print(f"\n📊 学習データ: {n_train}組")
    print(f"📊 テストデータ: {n_test}組")
    
    high_train, low_train = create_test_images(n_train)
    high_test, low_test = create_test_images(n_test)
    
    # 学習データのFDを計算
    print("\n🔢 学習データのFD計算中...")
    X_train = []
    y_train = []
    
    for i, (low, high) in enumerate(zip(low_train, high_train)):
        feat = extract_features(low)
        D_high, _, _ = calculate_fd(high)
        
        if D_high is not None:
            X_train.append(feat)
            y_train.append(D_high)
            print(f"  サンプル{i+1}: 特徴量={[f'{f:.2f}' for f in feat[:3]]}..., 高画質FD={D_high:.4f}")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\n✅ 学習データ準備完了: {len(X_train)}サンプル")
    print(f"   FD範囲: {y_train.min():.4f} ～ {y_train.max():.4f}")
    print(f"   FD平均: {y_train.mean():.4f} ± {y_train.std():.4f}")
    
    # モデル学習
    print("\n🤖 LightGBMモデルを学習中...")
    model = LGBMRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.05,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    print("✅ 学習完了")
    
    # テストデータで評価
    print("\n📈 テストデータで評価中...")
    D_high_test = []
    D_low_test = []
    D_pred_test = []
    
    for i, (low, high) in enumerate(zip(low_test, high_test)):
        D_high, _, _ = calculate_fd(high)
        D_low, _, _ = calculate_fd(low)
        
        feat = extract_features(low)
        D_pred = float(model.predict([feat])[0])
        
        if D_high is not None and D_low is not None:
            D_high_test.append(D_high)
            D_low_test.append(D_low)
            D_pred_test.append(D_pred)
            
            error_low = abs(D_high - D_low)
            error_pred = abs(D_high - D_pred)
            improvement = ((error_low - error_pred) / error_low * 100) if error_low > 0 else 0
            
            print(f"\nサンプル{i+1}:")
            print(f"  高画質FD: {D_high:.4f}")
            print(f"  低画質FD: {D_low:.4f} (誤差: {error_low:.4f})")
            print(f"  AI予測FD: {D_pred:.4f} (誤差: {error_pred:.4f})")
            print(f"  改善度: {improvement:+.1f}%")
    
    # 統計情報
    D_high_arr = np.array(D_high_test)
    D_low_arr = np.array(D_low_test)
    D_pred_arr = np.array(D_pred_test)
    
    mae_low = np.mean(np.abs(D_high_arr - D_low_arr))
    mae_pred = np.mean(np.abs(D_high_arr - D_pred_arr))
    improvement_avg = ((mae_low - mae_pred) / mae_low * 100) if mae_low > 0 else 0
    
    print("\n" + "="*60)
    print("📊 総合評価")
    print("="*60)
    print(f"MAE (低画質):     {mae_low:.4f}")
    print(f"MAE (AI補正):     {mae_pred:.4f}")
    print(f"平均改善度:       {improvement_avg:+.1f}%")
    
    # グラフ表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 散布図
    ax1.scatter(D_high_arr, D_low_arr, label='低画質', alpha=0.6, s=100)
    ax1.scatter(D_high_arr, D_pred_arr, label='AI補正', alpha=0.9, s=100)
    ax1.plot([D_high_arr.min(), D_high_arr.max()], 
             [D_high_arr.min(), D_high_arr.max()], 'k--', alpha=0.5)
    ax1.set_xlabel('高画質FD')
    ax1.set_ylabel('予測FD')
    ax1.set_title('FD予測結果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 誤差比較
    errors_low = np.abs(D_high_arr - D_low_arr)
    errors_pred = np.abs(D_high_arr - D_pred_arr)
    x = np.arange(len(D_high_test))
    
    ax2.bar(x - 0.2, errors_low, 0.4, label='低画質誤差', alpha=0.7)
    ax2.bar(x + 0.2, errors_pred, 0.4, label='AI補正誤差', alpha=0.7)
    ax2.set_xlabel('サンプル番号')
    ax2.set_ylabel('誤差 (絶対値)')
    ax2.set_title('誤差比較')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ai_test_result.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 グラフを保存しました: ai_test_result.png")
    plt.show()
    
    print("\n" + "="*60)
    if improvement_avg > 5:
        print("✅ AI学習は正常に動作しています！")
    elif improvement_avg > 0:
        print("⚠️ AI学習は動作していますが、改善度が低いです")
    else:
        print("❌ AI学習に問題がある可能性があります")
    print("="*60)

if __name__ == "__main__":
    test_ai_learning()
