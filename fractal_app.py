# fractal_fd_app_optimized.py
# ============================================================
# 低画質特化型 フラクタル次元解析＋AI補正（高速化版）
# - CuPy がある場合は GPU を自動検出して使用
# - ブロック演算をベクトル化して box-counting を高速化
# - LightGBM を使った低画質->高画質FD予測（並列化）
# ============================================================

import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import streamlit as st
from lightgbm import LGBMRegressor
import time
import pickle

# Try import cupy for GPU acceleration (optional)
USE_CUPY = False
xp = np  # alias for numpy/cupy
try:
    import cupy as cp
    # quick check: is CUDA visible?
    _ = cp.zeros(1)
    USE_CUPY = True
    xp = cp
except Exception:
    USE_CUPY = False
    xp = np

# ============================================================
# Data Augmentation (データ拡張) - 画像が少ない場合の対策
# ============================================================
def augment_image(img, augmentation_type):
    """
    画像にデータ拡張を適用
    
    Args:
        img: 入力画像 (BGR)
        augmentation_type: 拡張タイプ
    
    Returns:
        拡張された画像
    """
    if augmentation_type == 'flip_h':
        # 水平反転
        return cv2.flip(img, 1)
    elif augmentation_type == 'flip_v':
        # 垂直反転
        return cv2.flip(img, 0)
    elif augmentation_type == 'rotate_90':
        # 90度回転
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif augmentation_type == 'rotate_180':
        # 180度回転
        return cv2.rotate(img, cv2.ROTATE_180)
    elif augmentation_type == 'rotate_270':
        # 270度回転
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif augmentation_type == 'brightness_up':
        # 明るさ増加
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * 1.2  # 明るさを20%増加
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif augmentation_type == 'brightness_down':
        # 明るさ減少
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * 0.8  # 明るさを20%減少
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif augmentation_type == 'contrast_up':
        # コントラスト増加
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = l.astype(np.float32)
        l = (l - 128) * 1.3 + 128  # コントラストを30%増加
        l = np.clip(l, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif augmentation_type == 'contrast_down':
        # コントラスト減少
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = l.astype(np.float32)
        l = (l - 128) * 0.7 + 128  # コントラストを30%減少
        l = np.clip(l, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif augmentation_type == 'gamma_bright':
        # ガンマ補正 (明るく)
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    elif augmentation_type == 'gamma_dark':
        # ガンマ補正 (暗く)
        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    elif augmentation_type == 'noise':
        # ガウシアンノイズ追加
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    elif augmentation_type == 'blur':
        # ガウシアンぼかし
        return cv2.GaussianBlur(img, (5, 5), 1.0)
    elif augmentation_type == 'sharpen':
        # シャープ化
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)
    elif augmentation_type == 'saturation_up':
        # 彩度増加
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # 彩度を30%増加
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif augmentation_type == 'saturation_down':
        # 彩度減少
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.7  # 彩度を30%減少
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif augmentation_type == 'hue_shift':
        # 色相シフト
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180  # 色相を10度シフト
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif augmentation_type == 'equalize':
        # ヒストグラム均等化
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # ============================================================
    # 🎯 AI学習に特に有効な追加拡張 (フラクタル次元学習最適化)
    # ============================================================
    elif augmentation_type == 'scale_up':
        # スケール変換 (拡大 110%) - フラクタル次元のスケール不変性学習
        h, w = img.shape[:2]
        new_h, new_w = int(h * 1.1), int(w * 1.1)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 中央クロップで元のサイズに戻す
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return scaled[start_h:start_h+h, start_w:start_w+w]
    
    elif augmentation_type == 'scale_down':
        # スケール変換 (縮小 90%) - フラクタル次元のスケール不変性学習
        h, w = img.shape[:2]
        new_h, new_w = int(h * 0.9), int(w * 0.9)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # パディングで元のサイズに戻す
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return cv2.copyMakeBorder(scaled, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                                  cv2.BORDER_REFLECT)
    
    elif augmentation_type == 'clahe':
        # CLAHE (適応的ヒストグラム均等化) - 局所的なテクスチャ強調
        # フラクタル構造の詳細を保持しながらコントラスト向上
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif augmentation_type == 'bilateral':
        # バイラテラルフィルタ - エッジ保存平滑化
        # フラクタル構造のエッジを保ちながらノイズ除去
        return cv2.bilateralFilter(img, 9, 75, 75)
    
    elif augmentation_type == 'median':
        # メディアンフィルタ - ノイズ除去
        # 塩胡椒ノイズに強く、フラクタル構造を保持
        return cv2.medianBlur(img, 5)
    
    elif augmentation_type == 'temp_warm':
        # 温度調整 (暖色化) - 照明条件の変化に対するロバスト性
        # 肌画像は照明で色温度が変わるため重要
        warm_lut = np.array([[[i, 
                                int(np.clip(i * 0.9, 0, 255)), 
                                int(np.clip(i * 0.8, 0, 255))] 
                              for i in range(256)]], dtype=np.uint8)
        b, g, r = cv2.split(img)
        b = cv2.LUT(b, warm_lut[0, :, 2])
        g = cv2.LUT(g, warm_lut[0, :, 1])
        r = cv2.LUT(r, warm_lut[0, :, 0])
        return cv2.merge([b, g, r])
    
    elif augmentation_type == 'temp_cool':
        # 温度調整 (寒色化) - 照明条件の変化に対するロバスト性
        cool_lut = np.array([[[int(np.clip(i * 0.8, 0, 255)), 
                                int(np.clip(i * 0.9, 0, 255)), 
                                i] 
                              for i in range(256)]], dtype=np.uint8)
        b, g, r = cv2.split(img)
        b = cv2.LUT(b, cool_lut[0, :, 0])
        g = cv2.LUT(g, cool_lut[0, :, 1])
        r = cv2.LUT(r, cool_lut[0, :, 2])
        return cv2.merge([b, g, r])
    
    elif augmentation_type == 'rotate_small_cw':
        # 微小回転 (時計回り5度) - 方向不変性の学習
        # フラクタル次元は回転に対して不変であるべき
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        angle = 5
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT)
    
    elif augmentation_type == 'rotate_small_ccw':
        # 微小回転 (反時計回り5度) - 方向不変性の学習
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        angle = -5
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT)
    
    elif augmentation_type == 'unsharp':
        # アンシャープマスク - エッジ強調
        # フラクタル構造の境界を明確化
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        return unsharp
    
    elif augmentation_type == 'crop_zoom':
        # 中央クロップ&ズーム (90%を拡大)
        h, w = img.shape[:2]
        crop_size = int(min(h, w) * 0.9)
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        cropped = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        return img

def apply_data_augmentation(high_imgs, low_imgs, high_names, low_names, augmentation_methods):
    """
    データ拡張を適用して画像ペア数を増やす
    
    Args:
        high_imgs: 高画質画像リスト
        low_imgs: 低画質画像リスト
        high_names: 高画質画像名リスト
        low_names: 低画質画像名リスト
        augmentation_methods: 適用する拡張手法のリスト
    
    Returns:
        拡張後の画像リストと名前リスト
    """
    augmented_high = high_imgs.copy()
    augmented_low = low_imgs.copy()
    augmented_high_names = high_names.copy()
    augmented_low_names = low_names.copy()
    
    for method in augmentation_methods:
        for high, low, h_name, l_name in zip(high_imgs, low_imgs, high_names, low_names):
            aug_high = augment_image(high, method)
            aug_low = augment_image(low, method)
            augmented_high.append(aug_high)
            augmented_low.append(aug_low)
            augmented_high_names.append(f"{h_name}_{method}")
            augmented_low_names.append(f"{l_name}_{method}")
    
    return augmented_high, augmented_low, augmented_high_names, augmented_low_names


# Helper to move array to xp (cupy or numpy)
def to_xp(arr):
    if USE_CUPY:
        return cp.asarray(arr)
    else:
        return np.asarray(arr)

def to_host(arr):
    if USE_CUPY:
        return cp.asnumpy(arr)
    else:
        return arr

# ------------------------------------------------------------
# Utility: ensure image is color BGR uint8
def read_bgr_from_buffer(buf):
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def read_bgr_from_path(filepath):
    """日本語パスに対応した画像読み込み"""
    try:
        # OpenCVは日本語パスを直接扱えないため、numpyを経由
        with open(filepath, 'rb') as f:
            buf = f.read()
        arr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

# ============================================================
# Fast vectorized standard-deviation box-counting (中川式ベース)
# ============================================================
def fast_fractal_std_boxcount_batched(img_bgr, scales=(2,4,8,16,32,64), use_gpu=None):
    """
    img_bgr: HxWx3 uint8 (OpenCV BGR)
    scales: iterable of block sizes (h)
    use_gpu: None => auto (global USE_CUPY), True/False to force
    returns: D, scales_used, Nh_values (host numpy arrays)
    """
    if use_gpu is None:
        use_gpu = USE_CUPY

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = img_gray.shape

    Nh_vals = []
    valid_scales = []
    for h in scales:
        # crop to multiple of h for clean reshaping
        Hc = (H // h) * h
        Wc = (W // h) * h
        if Hc < h or Wc < h:
            continue

        gray_crop = img_gray[:Hc, :Wc]

        # move to xp
        arr = to_xp(gray_crop)

        # reshape to blocks: (Hc//h, h, Wc//h, h) then transpose to (Hc//h, Wc//h, h, h)
        new_shape = (Hc//h, h, Wc//h, h)
        try:
            blocks = arr.reshape(new_shape).transpose(0,2,1,3)
        except Exception:
            # fallback per-block (rare)
            blocks_list = []
            for i in range(0, Hc, h):
                row=[]
                for j in range(0, Wc, h):
                    row.append(arr[i:i+h, j:j+h])
                blocks_list.append(row)
            blocks = to_xp(np.array(blocks_list))

        # compute std over last two axes (h,h) -> shape (Hc//h, Wc//h)
        # note: xp.std uses different dtype; do manually for numerical stability
        mean_blk = blocks.mean(axis=(2,3))
        sq_mean = (blocks**2).mean(axis=(2,3))
        std_blk = xp.sqrt(xp.maximum(0, sq_mean - mean_blk**2))

        # nh per block: sigma/h
        nh = std_blk / float(h)

        # sum across blocks and convert to host
        nh_total = float(to_host(nh.sum()))
        Nh_vals.append(nh_total + 1e-12)
        valid_scales.append(h)

    if len(valid_scales) < 3:
        return None, np.array(scales), np.array([1]*len(scales))

    log_h = np.log(np.array(valid_scales, dtype=np.float64))
    log_Nh = np.log(np.array(Nh_vals, dtype=np.float64))

    # linear fit
    coeffs = np.polyfit(log_h, log_Nh, 1)
    D = abs(coeffs[0])

    return float(D), np.array(valid_scales), np.array(Nh_vals)

# ============================================================
# 3D DBC fast version (vectorized)
# ============================================================
def fast_fractal_3d_dbc(img_bgr, scales=None, max_size=256, use_gpu=None):
    """
    Convert grayscale intensity to height and perform vectorized DBC counting.
    Returns (FD_3d, used_scales, counts)
    """
    if use_gpu is None:
        use_gpu = USE_CUPY

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    # resize for speed
    scale_factor = 1.0
    if max(H, W) > max_size:
        scale_factor = max_size / max(H, W)
        gray = cv2.resize(gray, (int(W*scale_factor), int(H*scale_factor)), interpolation=cv2.INTER_AREA)
        H, W = gray.shape

    if scales is None:
        max_box = max(2, min(H, W)//4)
        scales = []
        s = 2
        while s <= max_box:
            scales.append(s)
            s *= 2
        if len(scales) < 3:
            scales = [2,4,8,16]

    counts = []
    arr_host = gray / 255.0
    arr = to_xp(arr_host)
    for r in scales:
        nh = (H // r)
        nw = (W // r)
        if nh < 1 or nw < 1:
            counts.append(0)
            continue

        Hc = nh * r
        Wc = nw * r
        arr_crop = arr[:Hc, :Wc]
        # shape (nh, r, nw, r) -> (nh, nw, r, r)
        blocks = arr_crop.reshape((nh, r, nw, r)).transpose(0,2,1,3)
        # min and max per block
        bmin = blocks.min(axis=(2,3))
        bmax = blocks.max(axis=(2,3))

        # G as in original (small quantization step): use 1/r to scale
        G = max(0.001, 1.0 / r)
        # l = floor(min/G), k = ceil(max/G)
        l = xp.floor(bmin / G)
        k = xp.ceil(bmax / G)
        # number of boxes per block (k-l)
        nr = (k - l).astype(xp.int32)
        # sum, ensure >=1 per block
        nr = xp.maximum(nr, 1)
        total_nr = int(to_host(nr.sum()))
        counts.append(total_nr)

    # check validity
    valid_sizes = []
    valid_counts = []
    for s,c in zip(scales, counts):
        if c > 0:
            valid_sizes.append(s)
            valid_counts.append(c)
    if len(valid_counts) < 3:
        return None, np.array(scales), np.array(counts)

    log_sizes = np.log(np.array(valid_sizes, dtype=np.float64))
    log_counts = np.log(np.array(valid_counts, dtype=np.float64))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    slope = coeffs[0]
    # FD = 3 - |slope|
    fd3 = 3.0 - abs(slope)
    fd3 = float(np.clip(fd3, 2.0, 3.0))
    return fd3, np.array(valid_sizes), np.array(valid_counts)

# ============================================================
# Feature extraction (vectorized, batch-friendly)
# ============================================================
def extract_feature_vector(img_bgr, size=256, use_gpu=None):
    if use_gpu is None:
        use_gpu = USE_CUPY
    gray = cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # move to xp for possible GPU ops
    arr = to_xp(gray)
    mean_val = float(to_host(arr.mean()))
    std_val = float(to_host(arr.std()))
    # Sobel edges
    gx = to_host(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = to_host(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_mean = float(np.mean(np.sqrt(gx**2 + gy**2)))
    noise_level = float(np.mean(np.abs(gray - cv2.GaussianBlur(gray, (3,3), 1))))
    # entropy
    probs, _ = np.histogram(gray.flatten(), bins=256, range=(0,255), density=True)
    probs = probs + 1e-12
    entropy = -np.sum(probs * np.log2(probs))
    return [mean_val, std_val, edge_mean, noise_level, entropy]

# ============================================================
# Train FD predictor (low->high) using LightGBM (fast, parallel)
# ============================================================
def train_fd_predictor_fast(low_imgs, high_imgs, n_estimators=400, max_depth=8):
    # サンプル数チェック
    if len(low_imgs) < 2 or len(high_imgs) < 2:
        raise ValueError(
            f"❌ **学習に必要な画像ペア数が不足しています**\n\n"
            f"- 検出された画像ペア数: {len(low_imgs)}\n"
            f"- 必要な最小ペア数: 2\n\n"
            f"💡 **解決方法:**\n"
            f"1. フォルダ内に少なくとも2組以上の画像ペアがあることを確認してください\n"
            f"2. ファイル名パターンが正しいか確認してください\n"
            f"   - 例: `IMG_0001.jpg` と `IMG_0001_low1.jpg`\n"
            f"3. 画像が正しく読み込めているか確認してください"
        )
    
    X = []
    y = []
    for low, high in zip(low_imgs, high_imgs):
        feat = extract_feature_vector(low)
        X.append(feat)
        D_high, *_ = fast_fractal_std_boxcount_batched(high, use_gpu=False)  # computing target on CPU for stability
        if D_high is None:
            # fallback to classic fractal_dimension naive
            D_high, *_ = fractal_dimension_naive(high)
        y.append(D_high)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05, n_jobs=-1)
    model.fit(X, y)
    return model

# ============================================================
# Model Save/Load (モデルの保存・読み込み)
# ============================================================
def save_model(model, filepath="trained_fd_model.pkl"):
    """学習済みモデルを保存"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    return filepath

def load_model(filepath="trained_fd_model.pkl"):
    """学習済みモデルを読み込み"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_fd_from_low_quality(low_img, model):
    """
    低画質画像だけから高画質相当のフラクタル次元を予測
    
    Args:
        low_img: 低画質画像 (BGR)
        model: 学習済みLightGBMモデル
    
    Returns:
        予測されたフラクタル次元
    """
    feat = extract_feature_vector(low_img)
    D_pred = float(model.predict([feat])[0])
    return D_pred

# ============================================================
# 信頼度計算機能 (Confidence Scoring)
# ============================================================
def calculate_prediction_confidence(low_img, model, predicted_fd):
    """
    予測値の信頼度を計算
    
    信頼度指標:
    1. 特徴量品質スコア (0-100): 入力画像の品質評価
    2. モデル信頼度 (0-100): 予測の安定性
    3. 総合信頼度 (0-100): 全体的な信頼性
    
    Args:
        low_img: 低画質画像
        model: 学習済みモデル
        predicted_fd: 予測されたFD値
    
    Returns:
        dict: 信頼度情報
    """
    feat = extract_feature_vector(low_img)
    
    # 1. 特徴量品質スコア (Feature Quality Score)
    # エッジ強度、ノイズレベル、エントロピーから評価
    mean_val, std_val, edge_strength, noise_level, entropy = feat
    
    # エッジ強度が高い = 明確な構造 = 良い (0-40点)
    edge_score = min(edge_strength / 30.0 * 40, 40)
    
    # ノイズレベルが低い = 良い (0-30点)
    noise_score = max(30 - noise_level / 10.0 * 30, 0)
    
    # エントロピーが適度 (5-7が理想) = 良い (0-30点)
    entropy_diff = abs(entropy - 6.0)
    entropy_score = max(30 - entropy_diff * 10, 0)
    
    feature_quality = edge_score + noise_score + entropy_score
    feature_quality = np.clip(feature_quality, 0, 100)
    
    # 2. モデル信頼度 (Model Confidence)
    # 予測値が妥当な範囲内か (2.0-3.0)
    if 2.0 <= predicted_fd <= 3.0:
        range_score = 50
    elif 1.9 <= predicted_fd <= 3.1:
        range_score = 30
    else:
        range_score = 10
    
    # 予測値の安定性 (LightGBMの場合、木の予測のばらつきを推定)
    # 簡易版: 予測値が極端でないほど高スコア
    stability_score = 50 - abs(predicted_fd - 2.5) * 20
    stability_score = np.clip(stability_score, 0, 50)
    
    model_confidence = range_score + stability_score
    model_confidence = np.clip(model_confidence, 0, 100)
    
    # 3. 総合信頼度 (Overall Confidence)
    # 特徴量品質 60%, モデル信頼度 40%
    overall_confidence = feature_quality * 0.6 + model_confidence * 0.4
    overall_confidence = np.clip(overall_confidence, 0, 100)
    
    # 信頼度レベルの判定
    if overall_confidence >= 80:
        confidence_level = "非常に高い"
        level_emoji = "🟢"
        level_color = "success"
    elif overall_confidence >= 60:
        confidence_level = "高い"
        level_emoji = "🔵"
        level_color = "info"
    elif overall_confidence >= 40:
        confidence_level = "中程度"
        level_emoji = "🟡"
        level_color = "warning"
    else:
        confidence_level = "低い"
        level_emoji = "🔴"
        level_color = "error"
    
    # 予測区間の推定 (簡易版)
    # 信頼度が低いほど区間が広い
    uncertainty = (100 - overall_confidence) / 100 * 0.1
    lower_bound = predicted_fd - uncertainty
    upper_bound = predicted_fd + uncertainty
    
    return {
        'overall_confidence': overall_confidence,
        'feature_quality': feature_quality,
        'model_confidence': model_confidence,
        'confidence_level': confidence_level,
        'level_emoji': level_emoji,
        'level_color': level_color,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'uncertainty': uncertainty,
        'feature_details': {
            'edge_strength': edge_strength,
            'noise_level': noise_level,
            'entropy': entropy,
            'edge_score': edge_score,
            'noise_score': noise_score,
            'entropy_score': entropy_score
        }
    }

# fallback naive fractal (simple binary box count) used only if needed
def fractal_dimension_naive(img_bgr, scales=(2,4,8,16,32)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    Nh = []
    valid_scales = []
    for s in scales:
        Hc = (H // s) * s
        Wc = (W // s) * s
        if Hc < s or Wc < s:
            continue
        cropped = gray[:Hc, :Wc]
        blocks = (cropped.reshape(Hc//s, s, Wc//s, s).mean(axis=(1,3)) > 127).astype(np.int32)
        Nh.append(blocks.sum() + 1e-9)
        valid_scales.append(s)
    if len(valid_scales) < 3:
        return None, np.array(scales), np.array([1]*len(scales))
    log_h = np.log(np.array(valid_scales))
    log_Nh = np.log(np.array(Nh))
    coeffs = np.polyfit(log_h, log_Nh, 1)
    return float(abs(coeffs[0])), np.array(valid_scales), np.array(Nh)

# ============================================================
# Evaluate pairs and show metrics & plots
# ============================================================
def evaluate_and_plot(high_imgs, low_imgs, model, use_gpu=None):
    if use_gpu is None:
        use_gpu = USE_CUPY

    D_high_list = []
    D_low_list = []
    D_pred_list = []
    t0 = time.time()
    for high, low in zip(high_imgs, low_imgs):
        # compute FD (fast vectorized)
        D_high, *_ = fast_fractal_std_boxcount_batched(high, use_gpu=use_gpu)
        D_low, *_ = fast_fractal_std_boxcount_batched(low, use_gpu=use_gpu)
        # predicted FD
        feat = extract_feature_vector(low)
        D_pred = float(model.predict([feat])[0])
        D_high_list.append(D_high)
        D_low_list.append(D_low)
        D_pred_list.append(D_pred)
    t1 = time.time()

    D_high_arr = np.array(D_high_list, dtype=np.float32)
    D_low_arr = np.array(D_low_list, dtype=np.float32)
    D_pred_arr = np.array(D_pred_list, dtype=np.float32)

    # metrics
    valid_mask = ~np.isnan(D_high_arr) & ~np.isnan(D_low_arr) & ~np.isnan(D_pred_arr)
    if valid_mask.sum() < 2:
        st.warning("解析可能な高画質FDが少ないため評価できません。")
        return D_high_list, D_low_list, D_pred_list

    # 安全に相関係数を計算
    r_low = 0.0
    r_pred = 0.0
    
    try:
        # 標準偏差が0の場合はnanになるので対策
        std_low = np.std(D_low_arr[valid_mask])
        std_high = np.std(D_high_arr[valid_mask])
        std_pred = np.std(D_pred_arr[valid_mask])
        
        if std_low > 1e-10 and std_high > 1e-10:
            r_low_val, _ = pearsonr(D_high_arr[valid_mask], D_low_arr[valid_mask])
            # nanチェック
            if not np.isnan(r_low_val):
                r_low = r_low_val
            else:
                st.warning("⚠️ 低画質の相関係数がnanです(標準偏差が0に近い)")
        else:
            st.warning(f"⚠️ 低画質FDの分散が0に近いため相関係数を計算できません (std_low={std_low:.6f}, std_high={std_high:.6f})")
        
        if std_pred > 1e-10 and std_high > 1e-10:
            r_pred_val, _ = pearsonr(D_high_arr[valid_mask], D_pred_arr[valid_mask])
            # nanチェック
            if not np.isnan(r_pred_val):
                r_pred = r_pred_val
            else:
                st.warning("⚠️ AI補正の相関係数がnanです(標準偏差が0に近い)")
                st.info(f"AI予測値の統計: 平均={np.mean(D_pred_arr[valid_mask]):.4f}, 標準偏差={std_pred:.6f}")
        else:
            st.warning(f"⚠️ AI予測値の分散が0に近いため相関係数を計算できません (std_pred={std_pred:.6f}, std_high={std_high:.6f})")
            st.error("🔴 **問題**: AIが全て同じ値(またはほぼ同じ値)を予測しています!")
            st.info("💡 **原因**: 学習データのバリエーション不足、または特徴量が効果的でない可能性があります")
    except Exception as e:
        st.error(f"相関係数の計算エラー: {e}")
        r_low = 0.0
        r_pred = 0.0
    
    mae_low = mean_absolute_error(D_high_arr[valid_mask], D_low_arr[valid_mask])
    mae_pred = mean_absolute_error(D_high_arr[valid_mask], D_pred_arr[valid_mask])
    
    try:
        r2_val = r2_score(D_high_arr[valid_mask], D_pred_arr[valid_mask])
        if not np.isnan(r2_val) and not np.isinf(r2_val):
            r2 = r2_val
        else:
            r2 = 0.0
            st.warning("⚠️ R²スコアがnanまたはinfです")
    except Exception as e:
        st.error(f"R²スコアの計算エラー: {e}")
        r2 = 0.0
    
    # 改善度の計算
    improvement = ((mae_low - mae_pred) / mae_low) * 100 if mae_low > 0 else 0
    
    # デバッグ情報
    with st.expander("🔍 計算値の詳細 (デバッグ用)"):
        st.write("### 基本統計")
        st.write(f"**相関係数 (低画質):** r_low = {r_low}")
        st.write(f"**相関係数 (AI補正):** r_pred = {r_pred}")
        st.write(f"**MAE (低画質):** mae_low = {mae_low}")
        st.write(f"**MAE (AI補正):** mae_pred = {mae_pred}")
        st.write(f"**R² スコア:** r2 = {r2}")
        st.write(f"**改善度:** {improvement}%")
        st.write(f"**有効サンプル数:** {valid_mask.sum()} / {len(D_high_arr)}")
        
        st.write("### AI予測値の分析")
        st.write(f"**予測値の平均:** {np.mean(D_pred_arr[valid_mask]):.4f}")
        st.write(f"**予測値の標準偏差:** {np.std(D_pred_arr[valid_mask]):.4f}")
        st.write(f"**予測値の最小値:** {np.min(D_pred_arr[valid_mask]):.4f}")
        st.write(f"**予測値の最大値:** {np.max(D_pred_arr[valid_mask]):.4f}")
        st.write(f"**予測値の範囲:** {np.max(D_pred_arr[valid_mask]) - np.min(D_pred_arr[valid_mask]):.4f}")
        
        st.write("### 高画質FDの分析")
        st.write(f"**高画質の平均:** {np.mean(D_high_arr[valid_mask]):.4f}")
        st.write(f"**高画質の標準偏差:** {np.std(D_high_arr[valid_mask]):.4f}")
        st.write(f"**高画質の最小値:** {np.min(D_high_arr[valid_mask]):.4f}")
        st.write(f"**高画質の最大値:** {np.max(D_high_arr[valid_mask]):.4f}")
        
        # R²が0になる理由を説明
        if r2 <= 0.01:
            st.error("⚠️ **R²スコアが0に近い理由:**")
            if np.std(D_pred_arr[valid_mask]) < 0.001:
                st.write("- AIが**ほぼ同じ値**ばかり予測しています(予測値の標準偏差が0に近い)")
                st.write("- これは学習データの多様性不足、または特徴量が効果的でない可能性があります")
            else:
                st.write("- AIの予測が正解値と全く相関していません")
                st.write("- モデルの学習が適切に行われていない可能性があります")

    # 評価指標を見やすく表示
    st.subheader("📊 AI性能評価")
    st.markdown("""
    **各指標の意味:**
    - 🎯 **改善度**: 低画質の誤差からどれだけ改善したか (高いほど良い)
    - 📈 **相関係数**: 予測値と正解値の一致度 (1.0で完全一致、0で無相関)
    - 📉 **MAE**: 平均絶対誤差 (小さいほど正確)
    - 🔢 **R²**: モデルの説明力 (1.0で完璧、0以下でランダム以下)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="🎯 改善度",
            value=f"{improvement:.1f}%",
            delta=f"{mae_low-mae_pred:.4f}",
            help="低画質からAI補正でどれだけ誤差が減ったか。正の値は改善、負の値は悪化を意味します。"
        )
        if improvement > 50:
            st.success("✅ 大幅改善")
        elif improvement > 20:
            st.info("👍 良好な改善")
        elif improvement > 0:
            st.warning("⚠️ わずかな改善")
        else:
            st.error("❌ 改善なし")
            
    with col2:
        # nanチェック
        r_pred_display = "N/A" if np.isnan(r_pred) else f"{r_pred:.4f}"
        r_low_safe = 0.0 if np.isnan(r_low) else r_low
        r_pred_safe = 0.0 if np.isnan(r_pred) else r_pred
        delta_r = r_pred_safe - r_low_safe
        
        st.metric(
            label="📈 相関係数 (AI)",
            value=r_pred_display,
            delta=f"+{delta_r:.4f}" if delta_r > 0 else f"{delta_r:.4f}" if not np.isnan(delta_r) else "N/A",
            help="AI補正後の値と高画質FDの相関。1.0に近いほど予測が正確です。"
        )
        
        if np.isnan(r_pred):
            st.error("❌ 計算不可 (nanエラー)")
        elif r_pred > 0.9:
            st.success("✅ 非常に高い相関")
        elif r_pred > 0.7:
            st.info("👍 良好な相関")
        elif r_pred > 0.5:
            st.warning("⚠️ 中程度の相関")
        else:
            st.error("❌ 低い相関")
            
    with col3:
        # nanチェック
        mae_display = "N/A" if np.isnan(mae_pred) else f"{mae_pred:.4f}"
        mae_low_safe = mae_low if not np.isnan(mae_low) else 0.0
        mae_pred_safe = mae_pred if not np.isnan(mae_pred) else 0.0
        delta_mae = mae_low_safe - mae_pred_safe
        
        st.metric(
            label="📉 MAE (AI補正)",
            value=mae_display,
            delta=f"-{delta_mae:.4f}" if not np.isnan(delta_mae) else "N/A",
            delta_color="inverse",
            help="AI補正後の平均絶対誤差。小さいほど正確な予測です。"
        )
        
        if np.isnan(mae_pred):
            st.error("❌ 計算不可 (nanエラー)")
        elif mae_pred < 0.01:
            st.success("✅ 非常に正確")
        elif mae_pred < 0.05:
            st.info("👍 良好な精度")
        elif mae_pred < 0.1:
            st.warning("⚠️ 中程度の精度")
        else:
            st.error("❌ 低い精度")
            
    with col4:
        # nanチェック
        r2_display = "N/A" if (np.isnan(r2) or np.isinf(r2)) else f"{r2:.4f}"
        st.metric(
            label="🔢 R-squared",
            value=r2_display,
            help=f"決定係数。モデルがデータをどれだけ説明できるか。1.0で完璧、0以下はランダム予測以下です。"
        )
        
        if np.isnan(r2) or np.isinf(r2):
            st.error("❌ 計算不可 (nanまたはinfエラー)")
        elif r2 > 0.8:
            st.success("✅ 優れたモデル")
        elif r2 > 0.5:
            st.info("👍 良好なモデル")
        elif r2 > 0.2:
            st.warning("⚠️ 改善の余地あり")
        else:
            st.error("❌ モデル性能不足")
    
    # 🔍 自動診断と改善提案
    st.markdown("---")
    st.subheader("🔍 結果の診断と改善提案")
    
    problems = []
    suggestions = []
    
    # 診断1: 改善度
    if improvement < 0:
        problems.append("❌ **改善度が負**: AIが低画質よりも悪い予測をしています")
        suggestions.append("📌 **対策**: 画像ペア数を増やす (現在: {}組 → 推奨: 20組以上)".format(len(high_imgs)))
        suggestions.append("📌 **対策**: 異なる品質レベル (low2, low3) を試す")
        suggestions.append("📌 **対策**: より多様なシーン・被写体の画像を追加")
    elif improvement < 20:
        problems.append("⚠️ **改善度が低い**: AI補正の効果が限定的です")
        suggestions.append("📌 **対策**: 画像の多様性を増やす (異なるシーン・被写体)")
        suggestions.append("📌 **対策**: より低品質な画像レベル (low2, low3) を試す")
    
    # 診断2: 相関係数
    if np.isnan(r_pred) or r_pred <= 0.0:
        problems.append("❌ **相関係数が0またはN/A**: AIが有効な予測をしていません")
        suggestions.append("📌 **対策**: 画像ペア数を大幅に増やす (推奨: 30組以上)")
        suggestions.append("📌 **対策**: 高画質と低画質の差が明確なペアを使用")
    elif r_pred < 0.5:
        problems.append("⚠️ **相関係数が低い**: 予測精度が不十分です")
        suggestions.append("📌 **対策**: より多くの画像ペアで学習 (推奨: 15組以上)")
    
    # 診断3: R²スコア
    if r2 <= 0:
        problems.append("❌ **R²スコアが0以下**: モデルがランダム予測以下の性能")
        suggestions.append("📌 **対策**: 学習データの質を見直す (同じような画像ばかりになっていないか)")
        suggestions.append("📌 **対策**: 画像ペア数を増やす")
    elif r2 < 0.3:
        problems.append("⚠️ **R²スコアが低い**: モデルの説明力が不足")
        suggestions.append("📌 **対策**: データの多様性を増やす")
    
    # 診断4: MAE
    if mae_pred > 0.1:
        problems.append("⚠️ **MAEが大きい**: 予測誤差が大きいです")
        suggestions.append("📌 **対策**: より多くのサンプルで学習")
    
    # 診断5: データの多様性
    if len(high_imgs) < 10:
        problems.append(f"⚠️ **画像ペア数が少ない**: 現在{len(high_imgs)}組 (推奨: 10組以上)")
        suggestions.append("📌 **対策**: より多くの画像ペアを追加してください")
    
    # 結果表示
    if problems:
        st.warning("### ⚠️ 検出された問題")
        for problem in problems:
            st.markdown(problem)
        
        st.info("### 💡 推奨される改善策")
        for suggestion in suggestions:
            st.markdown(suggestion)
        
        # 具体的な次のステップ
        st.success("""
        ### 📝 次に試すこと (優先順位順)
        
        1. **画像ペア数を増やす**
           - 目標: 20組以上 (現在: {}組)
           - より多様なシーン・被写体を含める
        
        2. **品質レベルを変更**
           - 現在使用中のレベルで効果が薄い場合
           - low1 → low2 → low3 の順に試す
        
        3. **画像の質を確認**
           - 高画質と低画質の差が明確か
           - 同じような画像ばかりになっていないか
        
        4. **データの多様性を増やす**
           - 異なる照明条件
           - 異なる被写体
           - 異なるアングル
        """.format(len(high_imgs)))
    else:
        st.success("""
        ### ✅ 良好な結果です!
        
        現在の設定で十分な性能が出ています。
        
        **さらに改善したい場合:**
        - より多くの画像ペアを追加 (精度向上)
        - 異なる品質レベルを試す (汎用性向上)
        """)
    
    st.markdown("---")
    
    # 比較表 (詳細説明付き)
    st.subheader("📋 低画質 vs AI補正 比較")
    st.markdown("""
    **この表の見方:**
    - **低画質(補正なし)**: 低画質画像から直接計算したフラクタル次元の性能
    - **AI補正後**: AIが低画質画像から高画質相当のFDを予測した結果
    - **改善**: AI補正によってどれだけ性能が向上したか (プラスは改善、マイナスは悪化)
    """)
    
    import pandas as pd
    comparison_df = pd.DataFrame({
        "指標": ["相関係数 (r)", "平均絶対誤差 (MAE)", "R-squared", "処理時間"],
        "低画質(補正なし)": [f"{r_low:.4f}", f"{mae_low:.4f}", "-", "-"],
        "AI補正後": [f"{r_pred:.4f}", f"{mae_pred:.4f}", f"{r2:.4f}", f"{t1-t0:.2f}秒"],
        "改善": [
            f"+{r_pred-r_low:.4f}" if r_pred > r_low else f"{r_pred-r_low:.4f}",
            f"-{mae_low-mae_pred:.4f}" if mae_pred < mae_low else f"+{mae_pred-mae_low:.4f}",
            "-",
            "-"
        ]
    })
    
    # 表を見やすく表示
    st.dataframe(
        comparison_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "指標": st.column_config.TextColumn("指標", width="medium"),
            "低画質(補正なし)": st.column_config.TextColumn("低画質(補正なし)", width="medium"),
            "AI補正後": st.column_config.TextColumn("AI補正後", width="medium"),
            "改善": st.column_config.TextColumn("改善", width="medium"),
        }
    )

    # scatter plot (詳細説明付き)
    st.subheader("📈 フラクタル次元 比較グラフ")
    
    st.markdown("""
    ### グラフの見方
    
    **横軸 (X軸)**: 高画質フラクタル次元 = **正解値** (目標とする値)
    
    **縦軸 (Y軸)**: 予測フラクタル次元 = 低画質から推定した値
    
    **🔵 青い丸**: 低画質画像から直接計算したFD (補正なし)
    - 正解値から大きくずれている = 低画質では正確に測定できない
    
    **🔺 赤い三角**: AIが低画質から予測したFD (AI補正後)
    - 黒い点線に近いほど = 高画質相当の正確な値を予測できている
    
    **⚫ 黒い点線**: 完全一致ライン (予測=正解となる理想的な状態)
    - この線上にあれば完璧な予測
    
    **理想的な結果**: 赤い三角が黒い点線に沿って並び、青い丸よりも点線に近い
    """)
    
    # 日本語フォント設定(文字化け対策)
    try:
        import matplotlib
        matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # グラフサイズを小さく調整
    fig = plt.figure(figsize=(7,5))
    
    # 低画質をプロット
    plt.scatter(D_high_arr, D_low_arr, 
                label='低画質 (補正なし)', 
                alpha=0.6, s=80, c='#1f77b4', 
                edgecolors='darkblue', linewidth=1.2)
    
    # AI補正をプロット
    plt.scatter(D_high_arr, D_pred_arr, 
                label='AI補正後', 
                alpha=0.9, s=100, c='#ff7f0e', 
                marker='^', edgecolors='darkred', linewidth=1.2)
    
    # 理想的な一致ライン
    plt.plot([2.0,3.0],[2.0,3.0],'k--', linewidth=1.5, label='完全一致ライン', alpha=0.5)
    
    plt.xlabel('高画質フラクタル次元 (正解値)', fontsize=11, fontweight='bold')
    plt.ylabel('予測フラクタル次元', fontsize=11, fontweight='bold')
    plt.title(f'AI補正効果\n相関: {r_pred:.4f} | MAE: {mae_pred:.4f} | R²: {r2:.4f}', 
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=9, loc='upper left', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tick_params(labelsize=10)
    
    # 軸の範囲を自動調整
    all_vals = np.concatenate([D_high_arr, D_low_arr, D_pred_arr])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (vmax - vmin) * 0.1
    plt.xlim(vmin - margin, vmax + margin)
    plt.ylim(vmin - margin, vmax + margin)
    
    plt.tight_layout()
    
    # グラフを中央寄せで表示 (コンパクト)
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    return D_high_list, D_low_list, D_pred_list

# ============================================================
# Streamlit app
# ============================================================
def app():
    st.set_page_config(layout="wide", page_title="高速フラクタル解析（GPU最適化版）")
    st.title("🚀 高速フラクタル解析 + AI補正（GPU 最適化版）")
    st.markdown("CuPy が利用可能な場合は GPU を自動で使います。無ければ CPU (NumPy) で処理します。")

    # ============================================================
    # 🔄 自動モデル読み込み機能 - アプリ起動時に実行
    # ============================================================
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
        st.session_state['persistent_model'] = None
        st.session_state['model_info'] = None
        
        # 保存されたモデルを探す
        default_model_path = "trained_fd_model.pkl"
        if os.path.exists(default_model_path):
            try:
                model = load_model(default_model_path)
                st.session_state['persistent_model'] = model
                st.session_state['model_loaded'] = True
                st.session_state['model_info'] = {
                    'path': default_model_path,
                    'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': '自動読み込み'
                }
            except Exception as e:
                pass  # 読み込み失敗時は無視

    gpu_auto = USE_CUPY
    st.sidebar.header("設定")
    st.sidebar.write(f"GPU 利用可能: {USE_CUPY}")
    use_gpu_checkbox = st.sidebar.checkbox("GPU を使う(自動判定)", value=USE_CUPY)
    st.sidebar.write("※ GPU が無い場合は自動的に CPU にフォールバックします。")
    
    # ============================================================
    # 📊 現在のモデル状態を表示
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AIモデル状態")
    if st.session_state.get('model_loaded', False):
        model_info = st.session_state.get('model_info', {})
        st.sidebar.success("✅ モデル読み込み済み")
        st.sidebar.write(f"📁 {model_info.get('source', '不明')}")
        st.sidebar.write(f"🕒 {model_info.get('loaded_at', '不明')}")
        
        # モデルをリセットするボタン
        if st.sidebar.button("🔄 モデルをリセット"):
            st.session_state['persistent_model'] = None
            st.session_state['model_loaded'] = False
            st.session_state['model_info'] = None
            st.rerun()
    else:
        st.sidebar.warning("⚠️ モデル未読み込み")
        st.sidebar.write("学習モードで学習するか、")
        st.sidebar.write("推論モードでモデルをアップロードしてください")
    
    # 改善方法ガイド
    with st.sidebar.expander("💡 結果を改善する方法"):
        st.markdown("""
        ### 🎯 良い結果を得るためのポイント
        
        #### 1️⃣ **画像の質と多様性**
        - ✅ **最低でも10組以上**の画像ペアを用意
        - ✅ **異なる被写体・シーン**を含める
        - ✅ 高画質と低画質の**差が明確**なペアを使用
        - 🔄 **画像が少ない場合**: データ拡張機能を使用
        
        #### 2️⃣ **画質レベルの選択**
        - 📌 `low1` (最も高品質) → 差が小さい
        - 📌 `low2` (中程度) → **推奨**
        - 📌 `low3` (最も低品質) → 差が大きい
        
        **💡 ヒント**: まず `low2` を試して、結果が悪ければ `low3` を試してください
        
        #### 3️⃣ **画像が足りない場合の対策**
        
        **🔄 データ拡張 (Data Augmentation) を使用**
        - アプリ内で自動的に画像を増やせます
        - 水平反転、回転、明るさ調整など
        - 例: 5組 → 拡張後 15組以上
        
        **📸 追加の画像を用意**
        - 異なる角度から撮影
        - 異なる照明条件で撮影
        - 異なる被写体を追加
        
        **⚠️ 最低限必要な枚数**
        - 学習には最低2組必要
        - 推奨: 10組以上 (拡張前)
        - 理想: 20組以上
        
        #### 4️⃣ **改善度が低い/負の場合**
        - ❌ **原因**: 画像の多様性不足
        - ✅ **対策1**: データ拡張を使用して画像を増やす
        - ✅ **対策2**: 異なる品質レベル (low2, low3) を試す
        - ✅ **対策3**: 異なるシーンの画像を追加
        
        #### 5️⃣ **相関係数が0.0またはN/Aの場合**
        - ❌ **原因**: AIが同じ値を予測している
        - ✅ **対策1**: データ拡張で画像の多様性を増やす
        - ✅ **対策2**: 画像ペア数を増やす (20組以上推奨)
        - ✅ **対策3**: 異なる品質レベルを試す
        
        #### 6️⃣ **R²スコアが0以下の場合**
        - ❌ **原因**: モデルがランダム予測以下
        - ✅ **対策**: 上記1〜4を全て実施
        
        ---
        
        ### 📊 良い結果の目安
        - ✅ **改善度**: 30%以上
        - ✅ **相関係数 (AI)**: 0.7以上
        - ✅ **MAE (AI補正)**: 0.05以下
        - ✅ **R²スコア**: 0.5以上
        """)
    
    st.sidebar.markdown("---")

    # アプリケーションモード選択
    st.sidebar.header("アプリケーションモード")
    app_mode = st.sidebar.radio(
        "モードを選択",
        ["🎓 学習モード (画像ペアが必要)", "🔮 推論モード (低画質画像のみで予測)"],
        help="学習モード: 高画質+低画質ペアでAIを学習\n推論モード: 学習済みモデルで低画質画像から予測"
    )
    
    st.sidebar.markdown("---")

    # 推論モード
    if app_mode == "🔮 推論モード (低画質画像のみで予測)":
        st.header("🔮 推論モード - 低画質画像だけで高品質FDを予測")
        
        st.markdown("""
        ### このモードについて
        
        **学習済みモデルを使って、低画質の肌画像だけから高品質相当のフラクタル次元を予測します。**
        
        #### 📋 使い方
        1. まず「学習モード」で画像ペアを使ってAIを学習
        2. モデルを保存
        3. このモードで低画質画像だけをアップロード
        4. **AIが自動的に高品質相当のFDを予測**
        
        #### ✨ メリット
        - 低画質画像だけでOK (高画質画像不要)
        - 高速処理
        - 学習済みモデルは再利用可能
        """)
        
        # モデルの読み込み
        st.subheader("📂 モデルの読み込み")
        
        # 永続化されたモデルがあるか確認
        if st.session_state.get('model_loaded', False):
            model = st.session_state['persistent_model']
            model_info = st.session_state.get('model_info', {})
            st.success(f"✅ モデル読み込み済み ({model_info.get('source', '不明')})")
            
            st.info(f"""
            **モデル情報:**
            - 種類: {type(model).__name__}
            - 推定器数: {model.n_estimators if hasattr(model, 'n_estimators') else 'N/A'}
            - 最大深度: {model.max_depth if hasattr(model, 'max_depth') else 'N/A'}
            - 読み込み日時: {model_info.get('loaded_at', '不明')}
            """)
        else:
            model = None
            st.warning("⚠️ モデルが読み込まれていません")
        
        # 追加でモデルをアップロードする機能
        with st.expander("📤 別のモデルをアップロード"):
            model_file = st.file_uploader(
                "学習済みモデルをアップロード (.pkl)",
                type=['pkl'],
                help="学習モードで保存したモデルファイルをアップロード",
                key="inference_model_uploader"
            )
            
            if model_file is not None:
                try:
                    new_model = pickle.load(model_file)
                    st.success("✅ 新しいモデルを読み込みました!")
                    
                    # 永続化
                    st.session_state['persistent_model'] = new_model
                    st.session_state['model_loaded'] = True
                    st.session_state['model_info'] = {
                        'path': model_file.name,
                        'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'アップロード'
                    }
                    
                    st.info(f"""
                    **新しいモデル情報:**
                    - 種類: {type(new_model).__name__}
                    - 推定器数: {new_model.n_estimators if hasattr(new_model, 'n_estimators') else 'N/A'}
                    - 最大深度: {new_model.max_depth if hasattr(new_model, 'max_depth') else 'N/A'}
                    """)
                    
                    model = new_model
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ モデルの読み込みに失敗: {e}")
        
        if model is not None:
            # 低画質画像のアップロード
            st.subheader("📤 低画質画像をアップロード")
            
            st.success("🤖 モデルが読み込まれています。予測の準備完了!")
            
            low_quality_imgs = st.file_uploader(
                "低画質の肌画像",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="フラクタル次元を予測したい低画質画像",
                key="inference_image_uploader"
            )
            
            if low_quality_imgs:
                st.success(f"✅ {len(low_quality_imgs)}枚の画像を読み込みました")
                
                # 予測実行ボタン
                if st.button("🔮 フラクタル次元を予測"):
                    st.info("予測を開始します...")
                    
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, img_file in enumerate(low_quality_imgs):
                        # 画像読み込み
                        img = read_bgr_from_buffer(img_file.read())
                        
                        if img is not None:
                            # 予測
                            predicted_fd = predict_fd_from_low_quality(img, model)
                            
                            # 信頼度計算
                            confidence_info = calculate_prediction_confidence(img, model, predicted_fd)
                            
                            results.append({
                                'filename': img_file.name,
                                'predicted_fd': predicted_fd,
                                'image': img,
                                'confidence': confidence_info
                            })
                        
                        progress_bar.progress((idx + 1) / len(low_quality_imgs))
                    
                    st.success("✅ 予測完了!")
                    
                    # 結果表示
                    st.subheader("📊 予測結果と信頼度")
                    
                    st.markdown("""
                    **予測されたフラクタル次元と信頼度:**
                    - **予測FD**: AIが推定した高画質相当のフラクタル次元
                    - **信頼度**: 予測値の信頼性 (0-100%)
                    - **予測区間**: 予測値の推定範囲
                    
                    💡 **信頼度が高いほど、予測値の精度が高いと期待できます**
                    """)
                    
                    # 結果テーブル (信頼度付き)
                    import pandas as pd
                    df = pd.DataFrame({
                        "No.": range(1, len(results) + 1),
                        "画像名": [r['filename'] for r in results],
                        "予測FD": [f"{r['predicted_fd']:.4f}" for r in results],
                        "信頼度": [f"{r['confidence']['overall_confidence']:.1f}%" for r in results],
                        "信頼度レベル": [f"{r['confidence']['level_emoji']} {r['confidence']['confidence_level']}" for r in results],
                        "予測区間": [f"{r['confidence']['lower_bound']:.4f} - {r['confidence']['upper_bound']:.4f}" for r in results]
                    })
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # 統計情報
                    predicted_fds = [r['predicted_fd'] for r in results]
                    avg_confidence = np.mean([r['confidence']['overall_confidence'] for r in results])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"""
                        **予測値の統計:**
                        - 平均FD: {np.mean(predicted_fds):.4f}
                        - 標準偏差: {np.std(predicted_fds):.4f}
                        - 最小値: {np.min(predicted_fds):.4f}
                        - 最大値: {np.max(predicted_fds):.4f}
                        """)
                    
                    with col2:
                        st.info(f"""
                        **信頼度の統計:**
                        - 平均信頼度: {avg_confidence:.1f}%
                        - 高信頼度(≥80%): {sum(1 for r in results if r['confidence']['overall_confidence'] >= 80)}枚
                        - 中信頼度(60-80%): {sum(1 for r in results if 60 <= r['confidence']['overall_confidence'] < 80)}枚
                        - 低信頼度(<60%): {sum(1 for r in results if r['confidence']['overall_confidence'] < 60)}枚
                        """)
                    
                    # 詳細な信頼度情報 (展開可能)
                    with st.expander("🔍 信頼度の詳細情報"):
                        st.markdown("""
                        ### 信頼度の計算方法
                        
                        **総合信頼度**は以下の2つの要素から計算されます:
                        
                        1. **特徴量品質スコア (60%の重み)**
                           - エッジ強度: 画像の構造が明確か
                           - ノイズレベル: ノイズが少ないか
                           - エントロピー: 情報量が適切か
                        
                        2. **モデル信頼度 (40%の重み)**
                           - 範囲妥当性: 予測値が正常範囲内か (2.0-3.0)
                           - 予測安定性: 予測値が極端でないか
                        
                        **信頼度レベル:**
                        - 🟢 非常に高い (80%以上): 予測値は非常に信頼できる
                        - 🔵 高い (60-80%): 予測値は信頼できる
                        - 🟡 中程度 (40-60%): 予測値は参考程度
                        - 🔴 低い (40%未満): 予測値は慎重に扱うべき
                        """)
                        
                        # 各画像の詳細
                        for idx, result in enumerate(results):
                            conf = result['confidence']
                            st.markdown(f"---")
                            st.markdown(f"### {idx+1}. {result['filename']}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "総合信頼度",
                                    f"{conf['overall_confidence']:.1f}%",
                                    delta=None
                                )
                            with col2:
                                st.metric(
                                    "特徴量品質",
                                    f"{conf['feature_quality']:.1f}%"
                                )
                            with col3:
                                st.metric(
                                    "モデル信頼度",
                                    f"{conf['model_confidence']:.1f}%"
                                )
                            
                            # 特徴量の詳細
                            feat_details = conf['feature_details']
                            st.write(f"""
                            **特徴量の詳細:**
                            - エッジ強度: {feat_details['edge_strength']:.2f} (スコア: {feat_details['edge_score']:.1f}/40)
                            - ノイズレベル: {feat_details['noise_level']:.2f} (スコア: {feat_details['noise_score']:.1f}/30)
                            - エントロピー: {feat_details['entropy']:.2f} (スコア: {feat_details['entropy_score']:.1f}/30)
                            """)
                    
                    # 画像プレビュー (信頼度付き)
                    st.subheader("📷 画像プレビュー (上位3枚)")
                    cols = st.columns(min(3, len(results)))
                    for idx, result in enumerate(results[:3]):
                        with cols[idx]:
                            conf = result['confidence']
                            st.image(
                                cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB),
                                caption=f"{result['filename']}",
                                use_container_width=True
                            )
                            st.markdown(f"""
                            **FD:** {result['predicted_fd']:.4f}  
                            **信頼度:** {conf['level_emoji']} {conf['overall_confidence']:.1f}%  
                            **区間:** {conf['lower_bound']:.4f} - {conf['upper_bound']:.4f}
                            """)
                    
                    # CSV出力 (信頼度情報含む)
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 結果をCSVでダウンロード (信頼度含む)",
                        data=csv,
                        file_name="predicted_fractal_dimensions_with_confidence.csv",
                        mime="text/csv"
                    )
        
        return  # 推論モードはここで終了

    # ============================================================
    # 学習モード (既存のコード)
    # ============================================================
    st.header("🎓 学習モード - AIを学習させる")
    
    # 既存モデルがある場合は通知
    if st.session_state.get('model_loaded', False):
        model_info = st.session_state.get('model_info', {})
        st.info(f"""
        ℹ️ 既にモデルが読み込まれています ({model_info.get('source', '不明')})
        
        - このまま新しく学習すると、**既存モデルは上書き**されます
        - 既存モデルを保持したい場合は、先にダウンロードしてください
        - 推論モードに切り替えれば、既存モデルで予測できます
        """)
    
    # モード選択
    mode = st.radio(
        "画像読み込みモード",
        ["📁 フォルダから自動ペアリング", "📤 手動アップロード"],
        help="フォルダモード: 同じ名前の画像を自動的にペアリング\n手動モード: 個別にアップロード"
    )

    if mode == "📁 フォルダから自動ペアリング":
        st.markdown("""
        ### フォルダ選択ガイド
        画像ペアの検出パターン:
        1. **IMG_XXX.jpg + IMG_XXX_low1.jpg** 形式 (例: `E:\\頬画像　画質別\\画質別＿頬画像`)
        2. **高画質/低画質フォルダ分離** 形式 (今後対応予定)
        3. **その他のパターン** - 手動モードをご利用ください
        """)
        
        folder_path = st.text_input(
            "画像フォルダのパス",
            value=r"E:\頬画像　画質別\画質別＿頬画像",
            help="高画質と低画質の画像が入ったフォルダパスを指定してください"
        )
        
        # ファイル名パターン選択
        col1, col2 = st.columns(2)
        with col1:
            file_pattern = st.selectbox(
                "ファイル名パターン",
                ["IMG_*.jpg", "*.jpg", "*.png", "カスタム"],
                help="検出する画像ファイルのパターン"
            )
            if file_pattern == "カスタム":
                file_pattern = st.text_input("カスタムパターン", value="*.jpg")
        
        with col2:
            quality_level = st.selectbox(
                "低画質レベルを選択",
                ["low1", "low2", "low3", "カスタム"],
                help="比較する低画質レベルを選択 (low1が最も高品質、low3が最も低品質)"
            )
            if quality_level == "カスタム":
                quality_level = st.text_input("カスタムサフィックス", value="low1")
        
        if folder_path and os.path.exists(folder_path):
            # フォルダから画像ペアを自動検出
            all_files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
            
            # 高画質画像を検出(_lowがついていないもの)
            high_files = [f for f in all_files if f"_{quality_level}" not in os.path.basename(f) 
                          and not any(f"_low{i}" in os.path.basename(f) for i in ["1", "2", "3"])]
            
            if len(all_files) > 0:
                st.info(f"📂 検出された全画像: {len(all_files)}枚")
            
            if len(high_files) > 0:
                st.success(f"✅ {len(high_files)}枚の高画質画像を検出しました")
                
                # デバッグ: 最初のファイルパスを表示
                with st.expander("🔍 検出された画像パス (デバッグ情報)"):
                    st.write(f"**フォルダ:** {folder_path}")
                    st.write(f"**パターン:** {file_pattern}")
                    st.write(f"**全ファイル数:** {len(all_files)}")
                    st.write(f"**高画質ファイル数:** {len(high_files)}")
                    st.write(f"**高画質例:** {os.path.basename(high_files[0]) if high_files else 'なし'}")
                    if len(high_files) > 1:
                        st.write(f"**他の例:** {', '.join([os.path.basename(f) for f in high_files[1:min(4, len(high_files))]])}")
                
                # 対応する低画質画像を検索
                low_files = []
                missing_files = []
                for hf in high_files:
                    base_name = os.path.splitext(os.path.basename(hf))[0]
                    ext = os.path.splitext(os.path.basename(hf))[1]
                    low_file = os.path.join(folder_path, f"{base_name}_{quality_level}{ext}")
                    if os.path.exists(low_file):
                        low_files.append(low_file)
                    else:
                        missing_files.append(f"{base_name}_{quality_level}{ext}")
                
                # デバッグ: 低画質ファイルパスも表示
                if low_files:
                    with st.expander("🔍 ペア画像パス (デバッグ情報)"):
                        st.write(f"**低画質ファイル数:** {len(low_files)}")
                        st.write(f"**低画質例:** {os.path.basename(low_files[0])}")
                        if missing_files:
                            st.warning(f"**見つからないファイル:** {len(missing_files)}件")
                            st.write(f"例: {', '.join(missing_files[:3])}")
                
                if len(low_files) == len(high_files):
                    st.success(f"✅ {len(low_files)}組の完全なペアを検出しました")
                    
                    # 画像を読み込み
                    uploaded_high = high_files
                    uploaded_low = low_files
                    auto_mode = True
                else:
                    st.error(f"❌ ペアが不完全です (高画質: {len(high_files)}枚, 低画質: {len(low_files)}枚)")
                    if len(low_files) > 0:
                        st.warning(f"一部のペアのみ使用しますか? (完全なペア: {len(low_files)}組)")
                        if st.checkbox("不完全でも続行する"):
                            # 完全なペアのみ使用
                            valid_high = []
                            valid_low = []
                            for hf in high_files:
                                base_name = os.path.splitext(os.path.basename(hf))[0]
                                ext = os.path.splitext(os.path.basename(hf))[1]
                                low_file = os.path.join(folder_path, f"{base_name}_{quality_level}{ext}")
                                if os.path.exists(low_file):
                                    valid_high.append(hf)
                                    valid_low.append(low_file)
                            uploaded_high = valid_high
                            uploaded_low = valid_low
                            auto_mode = True
                            st.info(f"✅ {len(valid_high)}組の完全なペアを使用します")
                        else:
                            uploaded_high = None
                            uploaded_low = None
                            auto_mode = False
                    else:
                        uploaded_high = None
                        uploaded_low = None
                        auto_mode = False
            else:
                st.warning(f"⚠️ フォルダ内に'{file_pattern}'パターンの画像が見つかりません")
                uploaded_high = None
                uploaded_low = None
                auto_mode = False
        elif folder_path:
            st.error(f"❌ **フォルダパスが無効です**")
            st.info(f"指定されたパス: `{folder_path}`")
            st.info(f"フォルダが存在するか確認してください。")
            
            # パスの存在確認の詳細
            parent_dir = os.path.dirname(folder_path)
            if os.path.exists(parent_dir):
                st.warning(f"親フォルダは存在します: `{parent_dir}`")
                # 親フォルダ内のサブフォルダ一覧を表示
                try:
                    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                    if subdirs:
                        st.info(f"利用可能なサブフォルダ: {', '.join(subdirs[:5])}")
                except:
                    pass
            else:
                st.error(f"親フォルダも存在しません: `{parent_dir}`")
            
            uploaded_high = None
            uploaded_low = None
            auto_mode = False
        else:
            st.info("👆 フォルダパスを入力してください")
            uploaded_high = None
            uploaded_low = None
            auto_mode = False
    else:
        uploaded_high = st.file_uploader("高画質画像をペアでアップロード(同枚数)", type=['png','jpg','jpeg'], accept_multiple_files=True)
        uploaded_low = st.file_uploader("低画質画像をペアでアップロード(同枚数)", type=['png','jpg','jpeg'], accept_multiple_files=True)
        auto_mode = False


    if uploaded_high and uploaded_low:
        if not auto_mode and len(uploaded_high) != len(uploaded_low):
            st.error("高画質と低画質の枚数を揃えてください(ペアで解析します)。")
            return

        # read images
        if auto_mode:
            # ファイルパスから直接読み込み(日本語パス対応)
            high_imgs = []
            low_imgs = []
            failed_files = []
            
            for hf, lf in zip(uploaded_high, uploaded_low):
                h_img = read_bgr_from_path(hf)
                l_img = read_bgr_from_path(lf)
                
                if h_img is None:
                    failed_files.append(f"高画質: {os.path.basename(hf)}")
                if l_img is None:
                    failed_files.append(f"低画質: {os.path.basename(lf)}")
                
                if h_img is not None and l_img is not None:
                    high_imgs.append(h_img)
                    low_imgs.append(l_img)
            
            if failed_files:
                st.error(f"以下のファイルの読み込みに失敗しました:\n" + "\n".join(failed_files[:5]))
                if len(failed_files) > 5:
                    st.error(f"...他 {len(failed_files)-5} 件")
                return
            
            # ファイル名を取得
            high_names = [os.path.basename(f) for f in uploaded_high]
            low_names = [os.path.basename(f) for f in uploaded_low]
        else:
            # アップロードされたファイルから読み込み
            high_imgs = [read_bgr_from_buffer(f.read()) for f in uploaded_high]
            low_imgs = [read_bgr_from_buffer(f.read()) for f in uploaded_low]
            high_names = [f.name for f in uploaded_high]
            low_names = [f.name for f in uploaded_low]

        if len(high_imgs) == 0:
            st.error("❌ 画像の読み込みに失敗しました。")
            return
        
        # データ拡張オプション
        st.markdown("---")
        st.subheader("🔄 データ拡張 (Data Augmentation)")
        
        if len(high_imgs) < 10:
            st.warning(f"""
            ⚠️ **画像ペア数が少ないです** (現在: {len(high_imgs)}組)
            
            データ拡張を使用すると、少ない画像から多くの学習サンプルを生成できます。
            """)
        
        st.markdown("""
        **データ拡張とは？**
        
        画像に変換を加えて学習サンプル数を増やす手法です。画像が少ない場合に有効です。
        
        **📋 利用可能な変換 (全28種類):**
        
        **🔄 幾何学変換 (7種類)**
        - 水平反転、垂直反転
        - 90度回転、180度回転、270度回転
        - 微小回転 (±5度) - 方向不変性学習
        
        **💡 明るさ・コントラスト (6種類)**
        - 明るさ増加/減少
        - コントラスト増加/減少
        - ガンマ補正 (明るく/暗く)
        
        **🎨 色調整 (5種類)**
        - 彩度増加/減少
        - 色相シフト
        - 温度調整 (暖色/寒色) - 照明条件対応 🌟
        
        **🔧 画質処理 (6種類)**
        - ノイズ追加、ぼかし
        - シャープ化、ヒストグラム均等化
        - メディアンフィルタ、バイラテラルフィルタ 🌟
        
        **🎯 AI学習最適化 (4種類) - フラクタル次元学習に特化 🌟**
        - スケール変換 (拡大/縮小) - スケール不変性学習
        - CLAHE - 局所的テクスチャ強調
        - アンシャープマスク - エッジ強調
        
        **🌟 = AI学習に特に効果的**
        
        **注意**: 拡張により処理時間が増加します
        """)
        
        use_augmentation = st.checkbox(
            "データ拡張を使用する",
            value=len(high_imgs) < 10,
            help="チェックすると画像ペア数を増やします。画像が10組未満の場合に推奨"
        )
        
        if use_augmentation:
            st.info("🔄 データ拡張オプション - 使用する変換を選択")
            
            # ============================================================
            # 🎯 全選択ボタン機能
            # ============================================================
            col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 3])
            with col_btn1:
                if st.button("✅ 全て選択", use_container_width=True, help="全ての拡張機能をオンにします", type="primary"):
                    st.session_state['select_all_augmentation'] = True
                    # タブごとの状態もリセット
                    st.session_state.pop('geo_select_all', None)
                    st.session_state.pop('bright_select_all', None)
                    st.session_state.pop('color_select_all', None)
                    st.session_state.pop('quality_select_all', None)
                    st.session_state.pop('ai_select_all', None)
                    st.rerun()
            with col_btn2:
                if st.button("❌ 全て解除", use_container_width=True, help="全ての拡張機能をオフにします"):
                    st.session_state['select_all_augmentation'] = False
                    # タブごとの状態もリセット
                    st.session_state.pop('geo_select_all', None)
                    st.session_state.pop('bright_select_all', None)
                    st.session_state.pop('color_select_all', None)
                    st.session_state.pop('quality_select_all', None)
                    st.session_state.pop('ai_select_all', None)
                    st.rerun()
            with col_btn3:
                # 現在の状態を表示
                select_all_state = st.session_state.get('select_all_augmentation', None)
                if select_all_state == True:
                    st.success("✅ 全選択中 (28種類)")
                elif select_all_state == False:
                    st.warning("全解除中")
            
            # 全選択/解除の状態を取得
            select_all = st.session_state.get('select_all_augmentation', None)
            
            # タブで分類 - 5つのタブに拡張
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔄 幾何学変換", 
                "💡 明るさ・コントラスト", 
                "🎨 色調整", 
                "🔧 画質処理",
                "🎯 AI学習最適化 🌟"
            ])
            
            with tab1:
                st.markdown("**幾何学変換 - 画像の向きや角度を変更**")
                
                # タブごとの全選択ボタン
                col_tab_btn1, col_tab_btn2 = st.columns([1, 3])
                with col_tab_btn1:
                    if st.button("✅ 全選択", key="geo_all", help="幾何学変換を全てオン"):
                        st.session_state['geo_select_all'] = True
                        st.rerun()
                with col_tab_btn2:
                    if st.button("❌ 全解除", key="geo_clear", help="幾何学変換を全てオフ"):
                        st.session_state['geo_select_all'] = False
                        st.rerun()
                
                geo_select = st.session_state.get('geo_select_all', None)
                default_geo = True if (select_all or geo_select) else (False if (select_all == False or geo_select == False) else True)
                default_geo_off = False if (select_all == False or geo_select == False) else (True if (select_all or geo_select) else False)
                
                col1, col2 = st.columns(2)
                with col1:
                    use_flip_h = st.checkbox("🔄 水平反転 (左右反転)", value=default_geo, help="画像を左右反転", key="aug_flip_h")
                    use_flip_v = st.checkbox("🔃 垂直反転 (上下反転)", value=default_geo_off, help="画像を上下反転", key="aug_flip_v")
                    use_rotate_90 = st.checkbox("↩️ 90度回転", value=default_geo, help="時計回りに90度回転", key="aug_rot90")
                    use_rotate_180 = st.checkbox("🔁 180度回転", value=default_geo_off, help="180度回転", key="aug_rot180")
                with col2:
                    use_rotate_270 = st.checkbox("↪️ 270度回転", value=default_geo_off, help="時計回りに270度回転", key="aug_rot270")
                    use_rotate_small_cw = st.checkbox("🔄 微小回転(+5°) 🌟", value=default_geo_off, help="時計回りに5度回転 - 方向不変性学習に効果的", key="aug_rot_small_cw")
                    use_rotate_small_ccw = st.checkbox("🔄 微小回転(-5°) 🌟", value=default_geo_off, help="反時計回りに5度回転 - 方向不変性学習に効果的", key="aug_rot_small_ccw")
            
            with tab2:
                st.markdown("**明るさ・コントラスト - 画像の明るさやコントラストを調整**")
                
                # タブごとの全選択ボタン
                col_tab_btn1, col_tab_btn2 = st.columns([1, 3])
                with col_tab_btn1:
                    if st.button("✅ 全選択", key="bright_all", help="明るさ・コントラストを全てオン"):
                        st.session_state['bright_select_all'] = True
                        st.rerun()
                with col_tab_btn2:
                    if st.button("❌ 全解除", key="bright_clear", help="明るさ・コントラストを全てオフ"):
                        st.session_state['bright_select_all'] = False
                        st.rerun()
                
                bright_select = st.session_state.get('bright_select_all', None)
                default_bright = False if (select_all == False or bright_select == False) else (True if (select_all or bright_select) else False)
                
                col1, col2 = st.columns(2)
                with col1:
                    use_brightness_up = st.checkbox("☀️ 明るさ増加 (+20%)", value=default_bright, help="画像を20%明るく", key="aug_br_up")
                    use_brightness_down = st.checkbox("🌙 明るさ減少 (-20%)", value=default_bright, help="画像を20%暗く", key="aug_br_down")
                    use_contrast_up = st.checkbox("📈 コントラスト増加", value=default_bright, help="コントラストを強く", key="aug_cont_up")
                with col2:
                    use_contrast_down = st.checkbox("📉 コントラスト減少", value=default_bright, help="コントラストを弱く", key="aug_cont_down")
                    use_gamma_bright = st.checkbox("✨ ガンマ補正 (明るく)", value=default_bright, help="ガンマ補正で明るく", key="aug_gamma_br")
                    use_gamma_dark = st.checkbox("🌑 ガンマ補正 (暗く)", value=default_bright, help="ガンマ補正で暗く", key="aug_gamma_dk")
            
            with tab3:
                st.markdown("**色調整 - 画像の色合いや彩度を変更**")
                
                # タブごとの全選択ボタン
                col_tab_btn1, col_tab_btn2 = st.columns([1, 3])
                with col_tab_btn1:
                    if st.button("✅ 全選択", key="color_all", help="色調整を全てオン"):
                        st.session_state['color_select_all'] = True
                        st.rerun()
                with col_tab_btn2:
                    if st.button("❌ 全解除", key="color_clear", help="色調整を全てオフ"):
                        st.session_state['color_select_all'] = False
                        st.rerun()
                
                color_select = st.session_state.get('color_select_all', None)
                default_color = False if (select_all == False or color_select == False) else (True if (select_all or color_select) else False)
                
                col1, col2 = st.columns(2)
                with col1:
                    use_saturation_up = st.checkbox("🌈 彩度増加", value=default_color, help="色を鮮やかに", key="aug_sat_up")
                    use_saturation_down = st.checkbox("🌫️ 彩度減少", value=default_color, help="色を淡く", key="aug_sat_down")
                    use_hue_shift = st.checkbox("🎨 色相シフト", value=default_color, help="色合いを変更", key="aug_hue")
                with col2:
                    use_temp_warm = st.checkbox("🔥 温度調整(暖色) 🌟", value=default_color, help="照明条件の変化に対応 - AI学習に効果的", key="aug_temp_warm")
                    use_temp_cool = st.checkbox("❄️ 温度調整(寒色) 🌟", value=default_color, help="照明条件の変化に対応 - AI学習に効果的", key="aug_temp_cool")
            
            with tab4:
                st.markdown("**画質処理 - ノイズやぼかし、シャープ化などの処理**")
                
                # タブごとの全選択ボタン
                col_tab_btn1, col_tab_btn2 = st.columns([1, 3])
                with col_tab_btn1:
                    if st.button("✅ 全選択", key="quality_all", help="画質処理を全てオン"):
                        st.session_state['quality_select_all'] = True
                        st.rerun()
                with col_tab_btn2:
                    if st.button("❌ 全解除", key="quality_clear", help="画質処理を全てオフ"):
                        st.session_state['quality_select_all'] = False
                        st.rerun()
                
                quality_select = st.session_state.get('quality_select_all', None)
                default_quality = False if (select_all == False or quality_select == False) else (True if (select_all or quality_select) else False)
                
                col1, col2 = st.columns(2)
                with col1:
                    use_noise = st.checkbox("📡 ノイズ追加", value=default_quality, help="ガウシアンノイズを追加", key="aug_noise")
                    use_blur = st.checkbox("🌀 ぼかし", value=default_quality, help="ガウシアンぼかしを適用", key="aug_blur")
                    use_sharpen = st.checkbox("🔪 シャープ化", value=default_quality, help="エッジを強調", key="aug_sharp")
                with col2:
                    use_equalize = st.checkbox("📊 ヒストグラム均等化", value=default_quality, help="コントラストを自動調整", key="aug_eq")
                    use_median = st.checkbox("🔲 メディアンフィルタ 🌟", value=default_quality, help="ノイズ除去 - フラクタル構造保持に効果的", key="aug_median")
                    use_bilateral = st.checkbox("🎭 バイラテラル 🌟", value=default_quality, help="エッジ保存平滑化 - AI学習に効果的", key="aug_bilateral")
            
            # 🎯 AI学習最適化タブ (新規追加)
            with tab5:
                st.markdown("**AI学習最適化 - フラクタル次元学習に特化した拡張 🌟**")
                st.info("これらの拡張は、フラクタル次元のAI学習に特に効果的です。スケール不変性、局所的な特徴抽出、エッジ保存などを強化します。")
                
                # タブごとの全選択ボタン
                col_tab_btn1, col_tab_btn2 = st.columns([1, 3])
                with col_tab_btn1:
                    if st.button("✅ 全選択", key="ai_all", help="AI学習最適化を全てオン"):
                        st.session_state['ai_select_all'] = True
                        st.rerun()
                with col_tab_btn2:
                    if st.button("❌ 全解除", key="ai_clear", help="AI学習最適化を全てオフ"):
                        st.session_state['ai_select_all'] = False
                        st.rerun()
                
                ai_select = st.session_state.get('ai_select_all', None)
                default_ai = False if (select_all == False or ai_select == False) else (True if (select_all or ai_select) else False)
                
                col1, col2 = st.columns(2)
                with col1:
                    use_scale_up = st.checkbox("📐 スケール拡大 🌟", value=default_ai, help="110%に拡大 - スケール不変性学習", key="aug_scale_up")
                    use_scale_down = st.checkbox("📐 スケール縮小 🌟", value=default_ai, help="90%に縮小 - スケール不変性学習", key="aug_scale_down")
                with col2:
                    use_clahe = st.checkbox("🔆 CLAHE 🌟", value=default_ai, help="適応的ヒストグラム均等化 - 局所的テクスチャ強調", key="aug_clahe")
                    use_unsharp = st.checkbox("🔍 アンシャープマスク 🌟", value=default_ai, help="エッジ強調 - フラクタル構造の境界明確化", key="aug_unsharp")
            
            # 選択された拡張手法を収集
            augmentation_methods = []
            
            # 幾何学変換
            if use_flip_h:
                augmentation_methods.append('flip_h')
            if use_flip_v:
                augmentation_methods.append('flip_v')
            if use_rotate_90:
                augmentation_methods.append('rotate_90')
            if use_rotate_180:
                augmentation_methods.append('rotate_180')
            if use_rotate_270:
                augmentation_methods.append('rotate_270')
            if use_rotate_small_cw:
                augmentation_methods.append('rotate_small_cw')
            if use_rotate_small_ccw:
                augmentation_methods.append('rotate_small_ccw')
            
            # 明るさ・コントラスト
            if use_brightness_up:
                augmentation_methods.append('brightness_up')
            if use_brightness_down:
                augmentation_methods.append('brightness_down')
            if use_contrast_up:
                augmentation_methods.append('contrast_up')
            if use_contrast_down:
                augmentation_methods.append('contrast_down')
            if use_gamma_bright:
                augmentation_methods.append('gamma_bright')
            if use_gamma_dark:
                augmentation_methods.append('gamma_dark')
            
            # 色調整
            if use_saturation_up:
                augmentation_methods.append('saturation_up')
            if use_saturation_down:
                augmentation_methods.append('saturation_down')
            if use_hue_shift:
                augmentation_methods.append('hue_shift')
            if use_temp_warm:
                augmentation_methods.append('temp_warm')
            if use_temp_cool:
                augmentation_methods.append('temp_cool')
            
            # 画質処理
            if use_noise:
                augmentation_methods.append('noise')
            if use_blur:
                augmentation_methods.append('blur')
            if use_sharpen:
                augmentation_methods.append('sharpen')
            if use_equalize:
                augmentation_methods.append('equalize')
            if use_median:
                augmentation_methods.append('median')
            if use_bilateral:
                augmentation_methods.append('bilateral')
            
            # AI学習最適化
            if use_scale_up:
                augmentation_methods.append('scale_up')
            if use_scale_down:
                augmentation_methods.append('scale_down')
            if use_clahe:
                augmentation_methods.append('clahe')
            if use_unsharp:
                augmentation_methods.append('unsharp')
            
            if augmentation_methods:
                # データ拡張を適用
                original_count = len(high_imgs)
                
                # 選択された拡張方法の情報を表示
                st.info(f"""
                **選択された拡張方法: {len(augmentation_methods)}種類**
                
                - 元の画像ペア数: {original_count}組
                - 予想される拡張後: {original_count * (len(augmentation_methods) + 1)}組 (元画像 + 拡張版)
                """)
                
                high_imgs, low_imgs, high_names, low_names = apply_data_augmentation(
                    high_imgs, low_imgs, high_names, low_names, augmentation_methods
                )
                augmented_count = len(high_imgs)
                
                st.success(f"""
                ✅ データ拡張完了
                - 元の画像ペア数: {original_count}組
                - 拡張後の画像ペア数: {augmented_count}組
                - 増加率: {((augmented_count / original_count - 1) * 100):.0f}%
                - 使用した拡張方法: {len(augmentation_methods)}種類
                """)
            else:
                st.warning("⚠️ 少なくとも1つの拡張手法を選択してください")
        
        st.markdown("---")
        
        # サンプル数チェック
        if len(high_imgs) < 2:
            st.error(f"""
            ❌ **画像ペア数が不足しています**
            
            - 検出された画像ペア数: **{len(high_imgs)}**
            - 必要な最小ペア数: **2**
            
            💡 **解決方法:**
            1. フォルダ内に少なくとも**2組以上**の画像ペアがあることを確認してください
            2. ファイル名パターンが正しいか確認してください
               - 例: `IMG_0001.jpg` と `IMG_0001_low1.jpg`
               - 例: `photo1.png` と `photo1_low1.png`
            3. 「デバッグ情報を表示」で検出状況を確認してください
            """)
            return
            
        st.success(f"✅ {len(high_imgs)} 組の画像ペアを読み込みました。")

        # Quick preview first pair (説明付き)
        st.subheader("📷 プレビュー (1枚目)")
        st.markdown("""
        **これから解析する画像ペアの例:**
        - **左 (低画質)**: AIがこの画像から高画質相当のFDを予測します
        - **右 (高画質)**: AIの予測の正解値として使用します (学習・評価用)
        
        💡 AIは低画質画像の特徴を学習し、高画質相当の正確なフラクタル次元を推定します。
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(low_imgs[0], cv2.COLOR_BGR2RGB), caption=f"低画質: {low_names[0]}", width=300)
        with col2:
            st.image(cv2.cvtColor(high_imgs[0], cv2.COLOR_BGR2RGB), caption=f"高画質: {high_names[0]}", width=300)

        # Train button
        if st.button("🔧 AI を学習して解析を実行"):
            try:
                st.info("学習を開始します...")
                start = time.time()
                model = train_fd_predictor_fast(low_imgs, high_imgs)
                st.success("学習完了しました。")
                
                # モデルを保存
                model_path = save_model(model, "trained_fd_model.pkl")
                st.success(f"💾 モデルを保存しました: {model_path}")
                
                # ============================================================
                # 🔄 モデルを永続化 - アプリ全体で使用可能に
                # ============================================================
                st.session_state['persistent_model'] = model
                st.session_state['model_loaded'] = True
                st.session_state['model_info'] = {
                    'path': model_path,
                    'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': '学習モード'
                }
                st.info("✅ モデルを永続化しました。推論モードでも使用できます。")
                
                # モデルダウンロードボタン
                with open(model_path, 'rb') as f:
                    model_data = f.read()
                st.download_button(
                    label="📥 学習済みモデルをダウンロード",
                    data=model_data,
                    file_name="trained_fd_model.pkl",
                    mime="application/octet-stream",
                    help="このモデルを保存して、後で推論モードで使用できます"
                )

                # Evaluate & show metrics
                st.info("解析・比較を行います...")
                D_high, D_low, D_pred = evaluate_and_plot(high_imgs, low_imgs, model, use_gpu=use_gpu_checkbox)
                
                # 結果をsession_stateに保存
                st.session_state['analysis_results'] = {
                    'D_high': D_high,
                    'D_low': D_low,
                    'D_pred': D_pred,
                    'high_names': high_names,
                    'low_names': low_names,
                    'model': model,  # モデルも保存
                    'completed': True
                }
            except ValueError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"❌ **エラーが発生しました:** {str(e)}")
                st.stop()
        
        # 結果が保存されている場合は表示
        if 'analysis_results' in st.session_state and st.session_state['analysis_results'].get('completed'):
            results = st.session_state['analysis_results']
            D_high = results['D_high']
            D_low = results['D_low']
            D_pred = results['D_pred']
            high_names = results['high_names']
            low_names = results['low_names']
            
            # show detailed table
            st.subheader("📋 詳細データ一覧")
            
            st.markdown("""
            ### 表の各列の意味
            
            - **No.**: 画像の番号
            - **画像名**: 処理した画像のファイル名
            - **高画質FD**: 高画質画像から計算した正解のフラクタル次元 (**目標値**)
            - **低画質FD**: 低画質画像から直接計算したFD (**補正なし、通常は不正確**)
            - **AI補正FD**: AIが低画質から予測した高画質相当のFD (**AI補正後**)
            - **低画質誤差**: |高画質FD - 低画質FD| = 補正なしの誤差 (大きいほど不正確)
            - **AI補正誤差**: |高画質FD - AI補正FD| = AI補正後の誤差 (**小さいほど優秀**)
            - **改善率**: (低画質誤差 - AI補正誤差) / 低画質誤差 × 100% (**高いほどAIが効果的**)
            
            💡 **見方のポイント**: 
            - AI補正誤差が低画質誤差より小さければAI補正が成功
            - 改善率がプラスならAIによる改善あり、マイナスなら悪化
            """)
            
            import pandas as pd
            df = pd.DataFrame({
                "No.": range(1, len(D_high)+1),
                "画像名": [name.replace('.jpg', '').replace('IMG_', '') for name in high_names],
                "高画質FD": [f"{x:.4f}" if x is not None else "N/A" for x in D_high],
                "低画質FD": [f"{x:.4f}" if x is not None else "N/A" for x in D_low],
                "AI補正FD": [f"{x:.4f}" if x is not None else "N/A" for x in D_pred],
                "低画質誤差": [f"{abs(h-l):.4f}" if h is not None and l is not None else "N/A" 
                          for h, l in zip(D_high, D_low)],
                "AI補正誤差": [f"{abs(h-p):.4f}" if h is not None and p is not None else "N/A" 
                           for h, p in zip(D_high, D_pred)],
                "改善率": [f"{((abs(h-l)-abs(h-p))/abs(h-l)*100):.1f}%" 
                        if h is not None and l is not None and p is not None and abs(h-l) > 0
                        else "N/A"
                        for h, l, p in zip(D_high, D_low, D_pred)]
            })
            
            # カラム幅を指定して表示
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=350,
                column_config={
                    "No.": st.column_config.NumberColumn("No.", width="small"),
                    "画像名": st.column_config.TextColumn("画像名", width="medium"),
                    "高画質FD": st.column_config.TextColumn("高画質FD", width="small"),
                    "低画質FD": st.column_config.TextColumn("低画質FD", width="small"),
                    "AI補正FD": st.column_config.TextColumn("AI補正FD", width="small"),
                    "低画質誤差": st.column_config.TextColumn("低画質誤差", width="small"),
                    "AI補正誤差": st.column_config.TextColumn("AI補正誤差", width="small"),
                    "改善率": st.column_config.TextColumn("改善率", width="small"),
                }
            )
            
            # 統計サマリー
            with st.expander("📊 統計サマリー - 全データの統計情報"):
                st.markdown("""
                **各統計の意味:**
                - **平均**: 全画像のフラクタル次元の平均値
                - **標準偏差**: データのばらつき (小さいほど均一、大きいほど多様)
                - **最小/最大**: データの範囲
                
                💡 **比較のポイント**: AI補正FDの統計が高画質FDに近いほど、AIの予測が正確です。
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("### 高画質FD統計")
                    st.caption("(正解値)")
                    valid_high = [x for x in D_high if x is not None]
                    st.write(f"**平均:** {np.mean(valid_high):.4f}")
                    st.write(f"**標準偏差:** {np.std(valid_high):.4f}")
                    st.write(f"**最小:** {np.min(valid_high):.4f}")
                    st.write(f"**最大:** {np.max(valid_high):.4f}")
                
                with col2:
                    st.write("### 低画質FD統計")
                    st.caption("(補正なし)")
                    valid_low = [x for x in D_low if x is not None]
                    st.write(f"**平均:** {np.mean(valid_low):.4f}")
                    st.write(f"**標準偏差:** {np.std(valid_low):.4f}")
                    st.write(f"**最小:** {np.min(valid_low):.4f}")
                    st.write(f"**最大:** {np.max(valid_low):.4f}")
                
                with col3:
                    st.write("### AI補正FD統計")
                    st.caption("(AI予測値)")
                    valid_pred = [x for x in D_pred if x is not None]
                    st.write(f"**平均:** {np.mean(valid_pred):.4f}")
                    st.write(f"**標準偏差:** {np.std(valid_pred):.4f}")
                    st.write(f"**最小:** {np.min(valid_pred):.4f}")
                    st.write(f"**最大:** {np.max(valid_pred):.4f}")

    else:
        st.info("📁 フォルダモード: フォルダパスを入力すると自動的に画像ペアを検出します\n📤 手動モード: 高画質と低画質のペア画像を同数アップロードしてください")

if __name__ == "__main__":
    app()
