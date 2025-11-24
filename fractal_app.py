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
import json
import re
import pandas as pd

# Plotlyのインポート（研究報告用）
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not found. Install with: pip install plotly")

# 肌品質評価モジュールをインポート
try:
    from skin_quality_evaluator import SkinQualityEvaluator
    SKIN_EVALUATOR_AVAILABLE = True
except ImportError:
    SKIN_EVALUATOR_AVAILABLE = False
    print("Warning: skin_quality_evaluator.py not found. Skin quality evaluation will be disabled.")

# 画像品質判定モジュールをインポート
try:
    from image_quality_assessor import (
        assess_image_quality,
        check_device_compatibility,
        RECOMMENDED_DEVICES,
        HIGH_QUALITY_CRITERIA
    )
    IMAGE_QUALITY_ASSESSOR_AVAILABLE = True
except ImportError:
    IMAGE_QUALITY_ASSESSOR_AVAILABLE = False
    print("Warning: image_quality_assessor.py not found. Image quality assessment will be disabled.")

# 肌分析モジュールをインポート
try:
    from skin_analysis import (
        detect_face_landmarks,
        extract_face_regions,
        detect_skin_troubles,
        create_trouble_report,
        REGION_NAMES_JP,
        TROUBLE_NAMES_JP
    )
    SKIN_ANALYSIS_AVAILABLE = True
except ImportError:
    SKIN_ANALYSIS_AVAILABLE = False
    print("Warning: skin_analysis.py not found. Face region analysis will be disabled.")

# 実験データ分析モジュールをインポート
try:
    from experiment_analysis import (
        ExperimentDataManager,
        calculate_correlations,
        create_scatter_plot,
        create_correlation_heatmap,
        generate_experiment_summary
    )
    EXPERIMENT_ANALYSIS_AVAILABLE = True
except ImportError:
    EXPERIMENT_ANALYSIS_AVAILABLE = False
    print("Warning: experiment_analysis.py not found. Experimental data collection will be disabled.")

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
    slope = coeffs[0]
    
    # フラクタル次元は傾きの絶対値として計算
    # ただし、標準偏差法では傾きがフラクタル次元に対応
    # 2D画像の場合、フラクタル次元は2〜3の範囲
    D = abs(slope)
    
    # 異常値チェック: 傾きが大きすぎる場合は計算失敗とみなす
    if D > 5.0 or D < 0.5:
        print(f"Warning: fast_fractal_std_boxcount_batched で異常な傾き検出: {D}")
        print(f"  scales: {valid_scales}")
        print(f"  Nh_vals: {Nh_vals}")
        print(f"  log_h: {log_h}")
        print(f"  log_Nh: {log_Nh}")
        return None, np.array(scales), np.array([1]*len(scales))
    
    # 2D画像のフラクタル次元は2〜3の範囲に制限
    D = np.clip(D, 2.0, 3.0)

    return float(D), np.array(valid_scales), np.array(Nh_vals), log_h, log_Nh, coeffs

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

# 特徴量名のリスト
FEATURE_NAMES = ['mean', 'std', 'edge_strength', 'noise_level', 'entropy']

# ============================================================
# 最小二乗法フィッティングのグラフを描画
# ============================================================
def plot_least_squares_fit(log_h, log_Nh, coeffs, fd_value):
    """
    最小二乗法による線形フィッティングのグラフを描画
    
    Args:
        log_h: log(スケール)の配列
        log_Nh: log(カウント値)の配列
        coeffs: polyfitの係数 [slope, intercept]
        fd_value: 計算されたフラクタル次元
    
    Returns:
        matplotlib figure
    """
    # 日本語フォント設定(文字化け対策)
    import matplotlib
    matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 実測値をプロット
    ax.scatter(log_h, log_Nh, s=100, alpha=0.6, color='blue', 
               label='実測値', edgecolors='navy', linewidth=2)
    
    # 最小二乗法によるフィッティング直線
    fit_line = coeffs[0] * log_h + coeffs[1]
    ax.plot(log_h, fit_line, 'r-', linewidth=2, 
            label=f'最小二乗法フィット\n傾き = {coeffs[0]:.4f}')
    
    # グリッド
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # ラベルとタイトル
    ax.set_xlabel('log(スケール)', fontsize=12, fontweight='bold')
    ax.set_ylabel('log(カウント値)', fontsize=12, fontweight='bold')
    ax.set_title(f'Box-Counting法：最小二乗法フィッティング\nフラクタル次元 = {fd_value:.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 凡例
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # 統計情報を追加
    residuals = log_Nh - fit_line
    r_squared = 1 - (np.sum(residuals**2) / np.sum((log_Nh - np.mean(log_Nh))**2))
    
    info_text = f'決定係数 R² = {r_squared:.4f}\n切片 = {coeffs[1]:.4f}\nデータ点数 = {len(log_h)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

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
    
    # 特徴量名を付けてDataFrameに変換
    import pandas as pd
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    
    model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05, n_jobs=-1)
    model.fit(X_df, y)
    return model

# ============================================================
# Model Save/Load (モデルの保存・読み込み)
# ============================================================
def save_model(model, filepath="trained_fd_model.pkl"):
    """学習済みモデルを保存"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    return filepath

def save_training_history(history, filepath="training_history.json"):
    """学習履歴をJSON形式で保存"""
    try:
        # 既存の履歴を読み込む
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
        else:
            all_history = []
        
        # 新しい履歴を追加
        all_history.append(history)
        
        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_history, f, indent=2, ensure_ascii=False)
        
        return filepath
    except Exception as e:
        print(f"履歴保存エラー: {e}")
        return None

def load_training_history(filepath="training_history.json"):
    """学習履歴を読み込み"""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"履歴読み込みエラー: {e}")
        return []

def calculate_ai_readiness(history):
    """
    AIの実用準備状況を評価
    
    Returns:
        dict: {
            'ready': bool (実用可能か),
            'confidence': float (0-100, 信頼度),
            'level': str (初級/中級/上級/プロ/マスター),
            'recommendations': list (改善アドバイス),
            'stats': dict (統計情報),
            'next_milestone': dict (次の目標)
        }
    """
    if not history or len(history) == 0:
        return {
            'ready': False,
            'confidence': 0,
            'level': '未学習',
            'recommendations': [
                '📚 まずは学習を開始してください',
                '🎯 推奨: 20組以上の画像ペアで学習',
                '🔄 データ拡張を使用して多様性を確保'
            ],
            'stats': {
                'total_sessions': 0,
                'total_pairs': 0,
                'best_correlation': 0,
                'best_improvement': 0,
                'avg_mae': 0
            },
            'next_milestone': {
                'target': '初回学習完了',
                'progress': 0,
                'needed': '学習開始'
            }
        }
    
    # 統計計算
    total_sessions = len(history)
    total_pairs = sum(h.get('num_pairs', 0) for h in history)
    
    # metricsから正しいキーで取得
    correlations = []
    improvements = []
    maes = []
    
    for h in history:
        metrics = h.get('metrics', {})
        if metrics:
            # correlation_pred または correlation
            corr = metrics.get('correlation_pred', metrics.get('correlation', 0))
            if corr > 0:
                correlations.append(corr)
            
            # improvement または improvement_rate
            imp = metrics.get('improvement', metrics.get('improvement_rate', 0))
            if imp != 0:
                improvements.append(imp)
            
            # mae_pred
            mae = metrics.get('mae_pred', 1.0)
            if mae < 1.0:
                maes.append(mae)
    
    best_correlation = max(correlations) if correlations else 0
    avg_correlation = sum(correlations) / len(correlations) if correlations else 0
    best_improvement = max(improvements) if improvements else 0
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    best_mae = min(maes) if maes else 1.0
    avg_mae = sum(maes) / len(maes) if maes else 1.0
    
    # 最新セッションの性能 - metricsから取得
    latest = history[-1]
    latest_metrics = latest.get('metrics', {})
    latest_corr = latest_metrics.get('correlation_pred', latest_metrics.get('correlation', 0))
    latest_mae = latest_metrics.get('mae_pred', 1.0)
    latest_improvement = latest_metrics.get('improvement', latest_metrics.get('improvement_rate', 0))
    
    # デバッグ情報を追加
    print(f"DEBUG - Latest metrics: correlation={latest_corr}, mae={latest_mae}, improvement={latest_improvement}")
    print(f"DEBUG - History length: {len(history)}, Total pairs: {total_pairs}")
    
    # 信頼度計算 (0-100)
    confidence = 0
    
    # 1. 相関係数による評価 (40点満点)
    if latest_corr >= 0.95:
        confidence += 40
    elif latest_corr >= 0.90:
        confidence += 35
    elif latest_corr >= 0.85:
        confidence += 30
    elif latest_corr >= 0.80:
        confidence += 25
    elif latest_corr >= 0.70:
        confidence += 15
    else:
        confidence += latest_corr * 20
    
    # 2. MAEによる評価 (30点満点)
    # フラクタル次元は通常2.0-3.0の範囲なので、MAEの基準を調整
    # 注意: 全レベル (low1-10) を使用すると、品質差が大きすぎてMAEが悪化します
    #       → 推奨: low4-7 (中度劣化) で学習すると、MAE 0.02-0.03 を達成可能
    if latest_mae < 0.01:
        confidence += 30  # 非常に優秀 (0.3-0.5%の相対誤差)
    elif latest_mae < 0.02:
        confidence += 28  # 優秀 (0.7-1.0%の相対誤差)
    elif latest_mae < 0.03:
        confidence += 25  # 良好 (1.0-1.5%の相対誤差)
    elif latest_mae < 0.05:
        confidence += 20  # 実用レベル (1.7-2.5%の相対誤差) ← 全レベル使用時はここに留まる
    elif latest_mae < 0.08:
        confidence += 15  # 許容範囲
    elif latest_mae < 0.10:
        confidence += 10  # 要改善
    else:
        confidence += max(0, 10 - latest_mae * 50)  # スケール調整
    
    # 3. 改善率による評価 (20点満点)
    if latest_improvement > 50:
        confidence += 20
    elif latest_improvement > 30:
        confidence += 15
    elif latest_improvement > 10:
        confidence += 10
    elif latest_improvement > 0:
        confidence += 5
    
    # 4. 学習回数と安定性 (10点満点)
    if total_sessions >= 10:
        confidence += 10
    elif total_sessions >= 5:
        confidence += 7
    elif total_sessions >= 3:
        confidence += 5
    else:
        confidence += total_sessions * 1.5
    
    confidence = min(100, confidence)
    
    # レベル判定
    if confidence >= 90:
        level = '🏆 マスター'
        level_desc = '実用レベル - 高精度予測が可能'
    elif confidence >= 75:
        level = '⭐ プロ'
        level_desc = '実用可能 - 信頼できる予測'
    elif confidence >= 60:
        level = '🥇 上級'
        level_desc = '実用に近い - さらなる改善推奨'
    elif confidence >= 40:
        level = '🥈 中級'
        level_desc = '学習中 - 追加データが必要'
    elif confidence >= 20:
        level = '🥉 初級'
        level_desc = '学習開始 - 継続が重要'
    else:
        level = '🌱 入門'
        level_desc = 'データ収集段階'
    
    # 実用可能判定
    ready = (confidence >= 75 and 
             latest_corr >= 0.85 and 
             latest_mae < 0.01 and
             total_sessions >= 3)
    
    # 改善アドバイス
    recommendations = []
    
    if latest_corr < 0.85:
        recommendations.append('📊 相関係数が低い → より多様なデータが必要です')
    if latest_mae > 0.01:
        recommendations.append('🎯 誤差が大きい → データ拡張を増やしてください')
    if latest_improvement < 20:
        recommendations.append('⚡ 改善率が低い → 品質レベルを見直してください')
    if total_pairs < 100:
        recommendations.append(f'📈 データ量不足 → 現在{total_pairs}組、目標100組以上')
    if total_sessions < 5:
        recommendations.append(f'🔄 学習回数 → 現在{total_sessions}回、推奨5回以上')
    
    if not recommendations:
        recommendations.append('✅ 優れた性能です！継続して学習を重ねましょう')
        if confidence < 90:
            recommendations.append('🎖️ マスターレベルを目指して、さらなるデータ追加を')
    
    # 次のマイルストーン
    if confidence < 40:
        next_milestone = {
            'target': '中級レベル到達',
            'progress': confidence / 40 * 100,
            'needed': f'信頼度を{40-confidence:.0f}ポイント向上 (データ追加と品質改善)'
        }
    elif confidence < 60:
        next_milestone = {
            'target': '上級レベル到達',
            'progress': (confidence - 40) / 20 * 100,
            'needed': f'信頼度を{60-confidence:.0f}ポイント向上 (精度向上が必要)'
        }
    elif confidence < 75:
        next_milestone = {
            'target': 'プロレベル到達 (実用化)',
            'progress': (confidence - 60) / 15 * 100,
            'needed': f'信頼度を{75-confidence:.0f}ポイント向上 (安定性向上)'
        }
    elif confidence < 90:
        next_milestone = {
            'target': 'マスターレベル到達',
            'progress': (confidence - 75) / 15 * 100,
            'needed': f'信頼度を{90-confidence:.0f}ポイント向上 (最高精度を目指す)'
        }
    else:
        next_milestone = {
            'target': '完璧な維持',
            'progress': 100,
            'needed': '現在の高水準を維持してください'
        }
    
    return {
        'ready': ready,
        'confidence': confidence,
        'level': level,
        'level_desc': level_desc,
        'recommendations': recommendations,
        'stats': {
            'total_sessions': total_sessions,
            'total_pairs': total_pairs,
            'best_correlation': best_correlation,
            'avg_correlation': avg_correlation,
            'best_improvement': best_improvement,
            'avg_improvement': avg_improvement,
            'best_mae': best_mae,
            'avg_mae': avg_mae,
            'latest_correlation': latest_corr,
            'latest_mae': latest_mae,
            'latest_improvement': latest_improvement
        },
        'next_milestone': next_milestone
    }

def evaluate_ai_performance(correlation, improvement, mae):
    """
    AI学習の性能を評価
    
    Args:
        correlation: 相関係数 (0-1)
        improvement: 改善率 (%)
        mae: 平均絶対誤差
    
    Returns:
        dict: 評価結果
    """
    # 相関係数ベースの評価
    if correlation >= 0.95:
        corr_grade = "S"
        corr_points = 100
    elif correlation >= 0.90:
        corr_grade = "A"
        corr_points = 90
    elif correlation >= 0.85:
        corr_grade = "B"
        corr_points = 80
    elif correlation >= 0.75:
        corr_grade = "C"
        corr_points = 70
    elif correlation >= 0.60:
        corr_grade = "D"
        corr_points = 50
    else:
        corr_grade = "F"
        corr_points = 30
    
    # 改善率ベースの評価
    if improvement >= 80:
        improve_points = 100
    elif improvement >= 60:
        improve_points = 80
    elif improvement >= 40:
        improve_points = 60
    elif improvement >= 20:
        improve_points = 40
    elif improvement > 0:
        improve_points = 20
    else:
        improve_points = 0
    
    # MAEベースの評価
    if mae <= 0.01:
        mae_points = 100
    elif mae <= 0.02:
        mae_points = 90
    elif mae <= 0.03:
        mae_points = 80
    elif mae <= 0.05:
        mae_points = 70
    elif mae <= 0.08:
        mae_points = 50
    else:
        mae_points = 30
    
    # 総合スコア(重み付け: 相関50%, 改善30%, MAE20%)
    total_score = (corr_points * 0.5 + improve_points * 0.3 + mae_points * 0.2)
    
    # 総合評価
    if total_score >= 90:
        grade = "S (優秀)"
        emoji = "🌟"
        comment = "素晴らしい性能です！"
    elif total_score >= 80:
        grade = "A (良好)"
        emoji = "⭐"
        comment = "良好な性能です"
    elif total_score >= 70:
        grade = "B (普通)"
        emoji = "👍"
        comment = "標準的な性能です"
    elif total_score >= 60:
        grade = "C (改善の余地あり)"
        emoji = "📈"
        comment = "さらなる改善が期待できます"
    else:
        grade = "D (要改善)"
        emoji = "⚠️"
        comment = "データ量や品質の見直しが必要です"
    
    return {
        'grade': grade,
        'emoji': emoji,
        'score': total_score,
        'comment': comment,
        'correlation_grade': corr_grade,
        'details': {
            'correlation': correlation,
            'improvement': improvement,
            'mae': mae,
            'corr_points': corr_points,
            'improve_points': improve_points,
            'mae_points': mae_points
        }
    }

def analyze_learning_growth(history):
    """
    学習履歴から成長を分析
    
    Args:
        history: 学習履歴のリスト
    
    Returns:
        dict: 成長分析結果
    """
    if len(history) < 2:
        return {
            'trend': '不明',
            'trend_emoji': '❓',
            'correlation_change': 0,
            'improvement_change': 0,
            'best_correlation': 0,
            'recommendation': 'まだ学習回数が少ないため、トレンドを判定できません'
        }
    
    # メトリクスを含む履歴のみを抽出
    valid_history = [h for h in history if 'metrics' in h]
    
    if len(valid_history) < 2:
        return {
            'trend': '不明',
            'trend_emoji': '❓',
            'correlation_change': 0,
            'improvement_change': 0,
            'best_correlation': 0,
            'recommendation': '評価メトリクスが不足しています'
        }
    
    # 最新と前回の比較
    latest = valid_history[-1]['metrics']
    previous = valid_history[-2]['metrics']
    
    corr_change = latest.get('correlation_pred', 0) - previous.get('correlation_pred', 0)
    improve_change = latest.get('improvement', 0) - previous.get('improvement', 0)
    
    # 全履歴の最高記録
    best_corr = max([h['metrics'].get('correlation_pred', 0) for h in valid_history])
    best_improve = max([h['metrics'].get('improvement', 0) for h in valid_history])
    
    # トレンド判定
    if corr_change > 0.05:
        trend = "大幅改善"
        trend_emoji = "🚀"
        recommendation = "素晴らしい成長です！この調子で学習を続けてください。"
    elif corr_change > 0.02:
        trend = "改善中"
        trend_emoji = "📈"
        recommendation = "順調に性能が向上しています。データ拡張や品質レベルの調整でさらに改善できます。"
    elif corr_change > -0.02:
        trend = "横ばい"
        trend_emoji = "➡️"
        recommendation = "性能が安定しています。異なるデータや設定を試してみると良いでしょう。"
    elif corr_change > -0.05:
        trend = "やや低下"
        trend_emoji = "📉"
        recommendation = "前回より性能が下がっています。データの質や多様性を見直してください。"
    else:
        trend = "大幅低下"
        trend_emoji = "⚠️"
        recommendation = "性能が大きく低下しています。データセットを変更したか、外れ値が含まれている可能性があります。"
    
    return {
        'trend': trend,
        'trend_emoji': trend_emoji,
        'correlation_change': corr_change,
        'improvement_change': improve_change,
        'best_correlation': best_corr,
        'best_improvement': best_improve,
        'recommendation': recommendation,
        'num_sessions': len(valid_history)
    }

def load_model(filepath="trained_fd_model.pkl"):
    """学習済みモデルを読み込み"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def calculate_fractal_dimension(img):
    """
    ボックスカウンティング法を使用してフラクタル次元を直接計算
    
    高品質画像や品質過剰画像に使用。
    AI予測を使わず、実際の画像から直接フラクタル次元を計算する。
    
    Args:
        img: 入力画像 (BGR)
    
    Returns:
        dict: {
            'fd': フラクタル次元値,
            'confidence': 信頼度 (95% - 直接計算のため高い),
            'method': '直接解析',
            'range': 推定範囲 [min, max]
        }
    """
    try:
        # 高速ベクトル化されたボックスカウンティング法を使用
        fd_value, scales, counts, log_h, log_Nh, coeffs = fast_fractal_std_boxcount_batched(img, use_gpu=False)
        
        # 計算失敗時はナイーブ法にフォールバック
        if fd_value is None:
            fd_value, scales, counts = fractal_dimension_naive(img)
            log_h, log_Nh, coeffs = None, None, None
        
        # まだNoneの場合はエラー
        if fd_value is None:
            raise ValueError("フラクタル次元の計算に失敗しました")
        
        # フラクタル次元は2.0〜3.0の範囲に制限
        # 異常値の場合は警告を出してクリッピング
        if fd_value < 2.0 or fd_value > 3.0:
            print(f"Warning: 異常なフラクタル次元値を検出: {fd_value}, 2.0-3.0にクリッピングします")
            fd_value = np.clip(fd_value, 2.0, 3.0)
        
        # 直接計算のため、信頼度は95%と高く設定
        # 範囲は非常に狭い (±0.01程度)
        confidence = 95.0
        fd_min = max(2.0, fd_value - 0.01)
        fd_max = min(3.0, fd_value + 0.01)
        
        return {
            'fd': float(fd_value),
            'confidence': confidence,
            'method': '直接解析 (Box-Counting法)',
            'range': [fd_min, fd_max],
            'fitting_data': {
                'log_h': log_h,
                'log_Nh': log_Nh,
                'coeffs': coeffs
            } if log_h is not None else None
        }
    except Exception as e:
        # エラー時は低信頼度の結果を返す
        print(f"Error in calculate_fractal_dimension: {e}")
        return {
            'fd': 2.5,  # デフォルト値
            'confidence': 10.0,
            'method': '直接解析 (エラー)',
            'range': [2.0, 3.0],
            'error': str(e)
        }

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
    # DataFrameに変換して特徴量名を付与
    import pandas as pd
    feat_df = pd.DataFrame([feat], columns=FEATURE_NAMES)
    D_pred = float(model.predict(feat_df)[0])
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
    # DataFrameに変換して特徴量名を付与
    import pandas as pd
    feat_df = pd.DataFrame([feat], columns=FEATURE_NAMES)
    
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
        # DataFrameに変換して特徴量名を付与
        import pandas as pd
        feat_df = pd.DataFrame([feat], columns=FEATURE_NAMES)
        D_pred = float(model.predict(feat_df)[0])
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
    
    # メトリクスをsession_stateに保存
    st.session_state['metrics'] = {
        'correlation_low': float(r_low),
        'correlation_pred': float(r_pred),
        'mae_low': float(mae_low),
        'mae_pred': float(mae_pred),
        'r2_score': float(r2),
        'improvement': float(improvement),
        'num_samples': int(valid_mask.sum())
    }

    return D_high_list, D_low_list, D_pred_list

# ============================================================
# 研究報告・品質ガイド表示関数
# ============================================================
def show_quality_optimization_report():
    """品質レベル最適化研究報告を表示"""
    
    st.header("📊 品質レベル最適化研究報告")
    st.markdown("**フラクタル次元予測における最適JPEG品質レベルの科学的検証**")
    
    # タブで構成
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 結論・推奨事項",
        "📊 完全検証データ", 
        "🔬 重要な発見",
        "💡 実用ガイド",
        "📈 詳細分析",
        "📚 研究詳細"
    ])
    
    # タブ1: 結論・推奨事項
    with tab1:
        st.markdown("## 🎯 研究の結論と実用推奨")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最優秀レベル", "Low5", "MAE 0.0094")
        with col2:
            st.metric("Golden Zone", "Low4-7", "平均MAE 0.0124")
        with col3:
            st.metric("信頼性", "100%", "誤差<0.05達成")
        
        st.markdown("---")
        
        st.markdown("### 📌 何が分かったか?")
        st.success("""
        **5つの重要な発見:**
        
        1. **最優秀レベル**: Low5が全レベル中最高精度 (MAE 0.0094, 0.94%)
        2. **Golden Zone**: Low4-7が最適動作範囲 (全ケースで誤差<0.05を達成)
        3. **U字型カーブ**: 高画質・低画質の両端で性能劣化
        4. **臨界境界**: Low7→Low8で6.62倍の性能悪化
        5. **常識の否定**: Low1(最高画質)がLow5より6.84倍悪い
        """)
        
        st.markdown("### 💡 どう生かせるか?")
        
        # 用途別推奨表
        st.markdown("#### 用途別ベストプラクティス")
        recommendations_df = {
            "用途": ["🏥 臨床・研究", "💼 商用アプリ", "🔍 スクリーニング", "📚 ベンチマーク", "❌ 避けるべき", "🚫 使用禁止"],
            "推奨レベル": ["Low5", "Low4-6", "Low3-7", "Low4-6", "Low1-2", "Low8-10"],
            "期待MAE": ["0.0094", "< 0.015", "< 0.04", "< 0.015", "> 0.055", "> 0.10"],
            "期待誤差%": ["0.94%", "< 1.5%", "< 4%", "< 1.5%", "> 5.5%", "> 10%"],
            "理由": [
                "最高精度・最高信頼性",
                "精度とコストのバランス",
                "大規模処理に適合",
                "標準化・再現性重視",
                "過学習リスク",
                "情報損失深刻"
            ]
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(recommendations_df), use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 実装への影響")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **すぐに実践できること:**
            
            ✅ 品質レベルのデフォルトを**Low5**に設定
            
            ✅ Golden Zone (Low4-7)を推奨範囲として強調
            
            ✅ Low1-2, Low8-10使用時に警告表示
            
            ✅ 予測時に期待精度を提示
            """)
        
        with col2:
            st.warning("""
            **開発における注意点:**
            
            ⚠️ 高画質 ≠ 高精度 (常識の否定)
            
            ⚠️ Low7/Low8境界は絶対に超えない
            
            ⚠️ 入力画像を自動的にLow5相当に最適化
            
            ⚠️ 品質レベル選択UIを改善
            """)
        
        st.markdown("---")
        
        st.markdown("### 📈 期待される効果")
        st.success("""
        **この知見を活用することで:**
        
        🎯 **精度向上**: 最適品質レベル選択により予測精度が最大6.84倍改善
        
        💰 **コスト削減**: 不要な高画質化処理を削減、ストレージ最適化
        
        ⚡ **処理高速化**: 適切な品質レベルで処理時間短縮
        
        🔬 **再現性向上**: 科学的根拠に基づく標準化プロトコル
        
        📊 **信頼性確保**: 100%の許容誤差達成(Golden Zone)
        """)
    
    # タブ2: 完全検証データ
    with tab2:
        st.markdown("## 📊 全品質レベルの性能一覧")
        
        # 完全データテーブル
        quality_data = {
            "品質レベル": ["Low1", "Low2", "Low3", "Low4", "Low5", "Low6", "Low7", "Low8", "Low9", "Low10"],
            "MAE": [0.0643, 0.0559, 0.0356, 0.0099, 0.0094, 0.0142, 0.0160, 0.1059, 0.1119, 0.1205],
            "平均誤差%": [2.42, 2.10, 1.34, 0.37, 0.35, 0.53, 0.60, 3.98, 4.21, 4.53],
            "最大誤差": [0.1548, 0.1724, 0.1282, 0.0338, 0.0281, 0.0420, 0.0373, 0.3614, 0.3473, 0.4609],
            "最悪誤差%": [6.16, 6.90, 5.13, 1.31, 1.02, 1.52, 1.35, 12.64, 12.15, 16.12],
            "Low5比": ["6.84×", "5.95×", "3.79×", "1.05×", "1.00×", "1.51×", "1.70×", "11.27×", "11.90×", "12.82×"],
            "評価": ["⚠️ 中", "⚠️ 中", "✨ 良", "🌟 優秀", "🌟 最優秀", "🌟 優秀", "🌟 優秀", "❌ 不良", "❌ 不良", "❌ 最不良"]
        }
        df = pd.DataFrame(quality_data)
        
        # 背景色付きで表示
        def highlight_quality(row):
            if row["品質レベル"] == "Low5":
                return ['background-color: #90EE90'] * len(row)  # 最優秀: 緑
            elif row["品質レベル"] in ["Low4", "Low6", "Low7"]:
                return ['background-color: #FFE4B5'] * len(row)  # Golden Zone: 薄オレンジ
            elif row["品質レベル"] in ["Low8", "Low9", "Low10"]:
                return ['background-color: #FFB6C1'] * len(row)  # 不良: 薄赤
            elif row["品質レベル"] in ["Low1", "Low2"]:
                return ['background-color: #FFFFE0'] * len(row)  # 過学習: 薄黄
            else:
                return [''] * len(row)
        
        st.dataframe(df.style.apply(highlight_quality, axis=1), use_container_width=True)
        
        st.caption("""
        **凡例:**
        🟢 緑: 最優秀(Low5) | 🟡 薄オレンジ: Golden Zone(Low4,6,7) | 
        🔴 薄赤: 使用禁止(Low8-10) | 🟡 薄黄: 避けるべき(Low1-2)
        """)
        
        st.markdown("---")
        
        st.markdown("### 📈 精度カーブの可視化")
        
        # グラフ描画
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        quality_levels = list(range(1, 11))
        mae_values = [0.0643, 0.0559, 0.0356, 0.0099, 0.0094, 0.0142, 0.0160, 0.1059, 0.1119, 0.1205]
        
        # 背景領域を追加 (Golden Zone)
        fig.add_shape(
            type="rect", 
            x0=3.5, x1=7.5, 
            y0=0, y1=max(mae_values),
            fillcolor="lightgreen", 
            opacity=0.2, 
            line_width=0
        )
        
        # Golden Zoneのラベルを追加
        fig.add_annotation(
            x=5.5,
            y=max(mae_values) * 0.95,
            text="Golden Zone",
            showarrow=False,
            font=dict(size=14, color="darkgreen"),
            bgcolor="rgba(255, 255, 255, 0.7)"
        )
        
        # MAEライン
        fig.add_trace(go.Scatter(
            x=quality_levels,
            y=mae_values,
            mode='lines+markers',
            name='MAE',
            marker=dict(size=12, color=mae_values, colorscale='RdYlGn_r', showscale=True),
            line=dict(width=3)
        ))
        
        # Low5を強調
        fig.add_trace(go.Scatter(
            x=[5],
            y=[0.0094],
            mode='markers+text',
            name='Best (Low5)',
            marker=dict(size=20, color='green', symbol='star'),
            text=['Best!'],
            textposition='top center'
        ))
        
        fig.update_layout(
            title="品質レベル vs MAE (U字型カーブ)",
            xaxis_title="品質レベル (Low1=最高画質, Low10=最低画質)",
            yaxis_title="MAE (Mean Absolute Error)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 誤差分布の詳細")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 高精度ケース (誤差 < 0.02)")
            precision_data = {
                "レベル": ["Low3", "Low4", "Low5", "Low6", "Low7"],
                "割合": ["91%", "91%", "97%", "88%", "85%"],
                "ケース数": ["30/33", "30/33", "32/33", "29/33", "28/33"]
            }
            st.dataframe(pd.DataFrame(precision_data), use_container_width=True)
        
        with col2:
            st.markdown("#### 許容範囲 (誤差 < 0.05)")
            acceptable_data = {
                "レベル": ["Low3", "Low4", "Low5", "Low6", "Low7", "Low8"],
                "割合": ["91%", "100%", "100%", "100%", "100%", "36%"],
                "ケース数": ["30/33", "33/33 ✓", "33/33 ✓", "33/33 ✓", "33/33 ✓", "12/33"]
            }
            st.dataframe(pd.DataFrame(acceptable_data), use_container_width=True)
        
        st.info("**重要**: Golden Zone(Low4-7)は全33ケースで誤差<0.05を達成!")
    
    # タブ3: 重要な発見
    with tab3:
        st.markdown("## 🔬 3つの重要な発見")
        
        # 発見1: 臨界境界
        st.markdown("### 1️⃣ 臨界境界の発見")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **境界A: Low2 → Low3**
            
            **過学習からの脱却**
            
            - Low2: MAE 0.0559 (5.59%)
            - Low3: MAE 0.0356 (3.56%)
            - **改善**: 1.57倍
            
            **解釈**: 過剰な画質(過学習リスク領域)から適正な情報量への移行点
            """)
        
        with col2:
            st.error("""
            **境界B: Low7 → Low8** ⚠️ **CRITICAL**
            
            **情報損失の開始**
            
            - Low7: MAE 0.0160 (1.60%)
            - Low8: MAE 0.1059 (10.59%)
            - **悪化**: 6.62倍 ← 性能崖!
            
            **解釈**: JPEG圧縮による情報損失が予測精度に致命的影響を与える閾値
            
            **この境界を超えてはならない**
            """)
        
        st.markdown("---")
        
        # 発見2: Low1の逆説
        st.markdown("### 2️⃣ Low1(最高画質)の逆説的劣化")
        
        st.warning("""
        **常識を覆す発見: 高画質 ≠ 高精度**
        
        従来の常識: 「画質は高ければ高いほど良い」
        
        実際の結果: **Low1(最高画質)がLow5(中程度)より6.84倍も悪い!**
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low1 (最高画質)", "MAE 0.0643", "6.84× 悪い ⬇️", delta_color="inverse")
        with col2:
            st.metric("Low5 (中程度)", "MAE 0.0094", "最良 ⬆️")
        with col3:
            st.metric("性能比", "6.84倍", "Low1はLow5より悪い")
        
        st.markdown("#### 原因の考察")
        
        st.markdown("""
        **1. 訓練データとの品質ミスマッチ**
        - モデルはLow4-7程度の品質で訓練された可能性が高い
        - 訓練時に見なかった品質領域では汎化性能が低下
        
        **2. 微細な違いへの過敏性**
        - 高画質すぎると訓練時に存在しなかった微細なパターンに反応
        - ノイズと信号の区別が困難になる
        
        **3. 過学習の証拠**
        - 入力データの品質による過学習の新しい視点
        - データ前処理・標準化の重要性を実証
        """)
        
        st.markdown("#### 問題画像の事例")
        
        case_data = {
            "画像": ["IMG_5049", "IMG_5050"],
            "実測FD": [2.4978, 2.5180],
            "Low1予測": [2.6517, 2.6728],
            "Low1誤差": ["+0.1538 (6.16%)", "+0.1548 (6.15%)"],
            "Low5予測": [2.5701, 2.5321],
            "Low5誤差": ["+0.0723 (2.89%)", "+0.0141 (0.56%)"],
            "改善": ["2.13倍", "10.94倍"]
        }
        st.dataframe(pd.DataFrame(case_data), use_container_width=True)
        
        st.markdown("---")
        
        # 発見3: Low8-10の情報損失
        st.markdown("### 3️⃣ Low8-10の深刻な情報損失")
        
        st.error("""
        **JPEG圧縮による不可逆的な情報損失**
        
        Low8以降では、フラクタル解析に必要な情報が失われ、実用不可能。
        """)
        
        st.markdown("#### IMG_5039の挙動分析 (高フラクタル次元画像)")
        
        img5039_data = {
            "品質レベル": ["Low1", "Low5", "Low7", "Low8", "Low9", "Low10"],
            "予測FD": [2.8910, 2.8384, 2.8397, 2.4975, 2.5116, 2.3979],
            "誤差": ["+0.0321", "-0.0204", "-0.0192", "-0.3614", "-0.3473", "-0.4609"],
            "相対誤差%": ["1.12%", "0.71%", "0.67%", "12.64%", "12.15%", "16.12%"],
            "評価": ["✓ 良好", "✓ 優秀", "✓ 優秀", "✗ 完全崩壊", "✗ 深刻", "✗ 予測失敗"]
        }
        st.dataframe(pd.DataFrame(img5039_data), use_container_width=True)
        st.caption("実測FD: 2.8589 (高複雑度)")
        
        st.markdown("#### 情報損失のメカニズム")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **JPEG圧縮の影響:**
            
            1. 高周波成分の消失 (DCT変換で優先削減)
            2. 8×8ブロックノイズ (境界アーティファクト)
            3. テクスチャの劣化 (微細パターン破壊)
            4. エッジの曖昧化 (境界情報不明瞭)
            """)
        
        with col2:
            st.markdown("""
            **フラクタル解析への影響:**
            
            - Box-counting法の小スケール情報欠落
            - 自己相似性を示す構造が失われる
            - フラクタル次元が実際より低く見積もられる
            - 予測が不安定・信頼性喪失
            """)
        
        st.info("**結論**: Low8以降では、情報が不可逆的に失われており、実用に耐えない。")
    
    # タブ4: 実用ガイド
    with tab4:
        st.markdown("## 💡 実用ガイドライン")
        
        st.markdown("### 🏥 臨床・医学研究用途")
        st.success("""
        **推奨レベル: Low5**
        
        **性能:**
        - MAE: 0.0094 (0.94%)
        - 最大誤差: 1.02%
        - 高精度率: 97% (誤差<0.02)
        
        **メリット:**
        ✓ 診断精度の最大化
        ✓ 最も安定した予測
        ✓ 再現性が高い
        ✓ 標準化しやすい
        
        **適用例:**
        - がん組織のフラクタル解析
        - 病理診断支援
        - 疾患進行モニタリング
        - 学術研究・論文発表
        """)
        
        st.markdown("---")
        
        st.markdown("### 💼 商用アプリケーション")
        st.info("""
        **推奨レベル: Low4-6**
        
        **性能:**
        - 平均MAE: 0.0128 (1.28%)
        - 許容誤差達成率: 100%
        
        **メリット:**
        ✓ 精度とコストのバランス
        ✓ 処理速度が適切
        ✓ ストレージ効率良好
        ✓ 実用性が高い
        
        **適用例:**
        - 品質管理システム
        - 製造プロセス監視
        - リアルタイム解析
        - モバイルアプリ
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔍 スクリーニング・大規模解析")
        st.info("""
        **推奨レベル: Low3-7**
        
        **性能:**
        - MAE範囲: 0.009-0.036
        - 実用的精度を確保
        
        **メリット:**
        ✓ 大量データの効率的処理
        ✓ ストレージコスト削減
        ✓ 処理速度の向上
        ✓ 十分な精度を維持
        
        **適用例:**
        - 疫学調査
        - 人口ベース研究
        - ビッグデータ解析
        - 予備スクリーニング
        """)
        
        st.markdown("---")
        
        st.markdown("### 📚 研究・ベンチマーク用途")
        st.info("""
        **推奨レベル: Low4, Low5, Low6**
        
        **性能:**
        - Golden Zoneの中核
        - 標準化に最適
        
        **メリット:**
        ✓ 研究間の比較可能性
        ✓ 再現性の確保
        ✓ プロトコルの標準化
        ✓ 国際的な互換性
        
        **適用例:**
        - 学術論文
        - 国際共同研究
        - ベンチマークデータセット
        - 方法論の比較
        """)
        
        st.markdown("---")
        
        st.markdown("### ❌ 避けるべきレベル")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("""
            **Low1-2: 過学習リスク領域**
            
            **問題点:**
            ✗ 高画質なのに精度が低い（逆説的）
            ✗ MAE 0.056-0.064 (Golden Zoneの4-6倍悪い)
            ✗ 画像ごとのばらつきが大きい
            ✗ 予測が不安定
            
            **技術的理由:**
            - 訓練データの品質範囲外
            - 微細な変動にモデルが過敏反応
            - 汎化性能の低下
            - ノイズと信号の区別困難
            
            **推奨: 使用しない**
            """)
        
        with col2:
            st.error("""
            **Low8-10: 情報損失領域**
            
            **問題点:**
            ✗ 深刻な精度劣化
            ✗ MAE 0.106-0.121 (Golden Zoneの7-13倍悪い)
            ✗ 一部画像で完全な予測失敗（誤差 > 10%）
            ✗ 信頼性が著しく低い
            
            **技術的理由:**
            - JPEG圧縮による情報の不可逆的損失
            - フラクタル構造の破壊
            - 高周波成分の消失
            - ブロックノイズの影響
            
            **推奨: 絶対に使用しない（実用不可）**
            """)
    
    # タブ5: 詳細分析
    with tab5:
        st.markdown("## 📈 Golden Zone (Low4-7) 詳細分析")
        
        st.markdown("### Golden Zoneの統計的特徴")
        
        golden_stats = {
            "指標": ["平均MAE", "MAE範囲", "全レベルでMAE", "最悪ケースでも", "許容誤差達成率"],
            "値": ["0.0124 (1.24%)", "0.0094 - 0.0160", "< 0.02 ✓", "相対誤差 < 2% ✓", "100% (全33ケースで誤差 < 0.05)"]
        }
        st.table(pd.DataFrame(golden_stats))
        
        st.markdown("---")
        
        # 各レベルの詳細
        st.markdown("### 各レベルの詳細特性")
        
        level_col1, level_col2 = st.columns(2)
        
        with level_col1:
            st.markdown("#### 🥇 Low5 (最優秀)")
            st.success("""
            **性能指標:**
            - MAE: 0.0094 (0.94%)
            - 最大誤差: 0.0281 (1.02%)
            - 高精度ケース率: 97% (誤差 < 0.02)
            - 許容誤差達成率: 100% (誤差 < 0.05)
            
            **特徴:**
            ✓ 全レベル中最良の平均精度
            ✓ 最も安定した予測（誤差分布が集中）
            ✓ 最小の最大誤差
            ✓ 臨床・研究用途に最適
            """)
            
            st.markdown("#### 🥉 Low6 (優秀)")
            st.info("""
            **性能指標:**
            - MAE: 0.0142 (1.42%)
            - 最大誤差: 0.0420 (1.52%)
            - 高精度ケース率: 88% (誤差 < 0.02)
            - 許容誤差達成率: 100% (誤差 < 0.05)
            
            **特徴:**
            ✓ 実用上十分な精度
            ✓ ファイルサイズとのバランス良好
            ✓ 商用アプリに適合
            """)
        
        with level_col2:
            st.markdown("#### 🥈 Low4 (準最優秀)")
            st.success("""
            **性能指標:**
            - MAE: 0.0099 (0.99%)
            - 最大誤差: 0.0338 (1.31%)
            - 高精度ケース率: 91% (誤差 < 0.02)
            - 許容誤差達成率: 100% (誤差 < 0.05)
            
            **特徴:**
            ✓ Low5とほぼ同等の性能
            ✓ 外れ値が少なく安定
            ✓ 標準的な選択肢として優秀
            """)
            
            st.markdown("#### ⭐ Low7 (Golden Zone上限)")
            st.info("""
            **性能指標:**
            - MAE: 0.0160 (1.60%)
            - 最大誤差: 0.0373 (1.35%)
            - 高精度ケース率: 85% (誤差 < 0.02)
            - 許容誤差達成率: 100% (誤差 < 0.04)
            
            **特徴:**
            ✓ Golden Zoneの境界
            ✓ この品質以下は推奨しない
            ✓ コスト重視の用途に
            """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 実装推奨コード")
        
        st.code('''
# 品質レベル推奨設定
QUALITY_RECOMMENDATIONS = {
    # ベストプラクティス
    "best": "low5",  # MAE 0.0094 - 最高精度
    
    # Golden Zone（推奨範囲）
    "golden_zone": ["low4", "low5", "low6", "low7"],  # MAE < 0.02
    
    # 許容範囲
    "acceptable": ["low3", "low4", "low5", "low6", "low7"],  # MAE < 0.04
    
    # 使用禁止
    "avoid": ["low1", "low2", "low8", "low9", "low10"]
}

# 用途別推奨
USE_CASE_MAPPING = {
    "clinical_research": "low5",  # 臨床・研究: 最高精度
    "commercial": ["low4", "low5", "low6"],  # 商用: バランス
    "screening": ["low3", "low4", "low5", "low6", "low7"],  # スクリーニング
    "benchmark": ["low4", "low5", "low6"]  # ベンチマーク: 標準化
}

# 警告メッセージ
WARNING_MESSAGES = {
    "low1": "⚠️ 過学習リスク - Low5推奨 (6.84倍精度悪化)",
    "low2": "⚠️ 過学習リスク - Low5推奨 (5.95倍精度悪化)",
    "low8": "❌ 情報損失深刻 - 使用禁止 (11.27倍精度悪化)",
    "low9": "❌ 情報損失深刻 - 使用禁止 (11.90倍精度悪化)",
    "low10": "❌ 情報損失深刻 - 使用禁止 (12.82倍精度悪化)"
}

# 精度期待値
EXPECTED_ACCURACY = {
    "low5": {"mae": 0.0094, "avg_error": "0.35%", "reliability": "最優秀"},
    "low4": {"mae": 0.0099, "avg_error": "0.37%", "reliability": "優秀"},
    "low6": {"mae": 0.0142, "avg_error": "0.53%", "reliability": "優秀"},
    # ... 他のレベル
}
''', language='python')
    
    # タブ6: 研究詳細
    with tab6:
        st.markdown("## 📚 研究の詳細情報")
        
        st.markdown("### 研究の背景と目的")
        st.markdown("""
        医療画像解析やフラクタル次元予測において、**入力画像の品質が予測精度に与える影響**は長年の課題でした。
        
        **従来の常識:**
        - 「高画質 = 高精度」という暗黙の仮定
        - 画質は高ければ高いほど良いという考え
        - 科学的根拠に基づく品質基準の不在
        
        **この研究の目的:**
        1. **定量的評価**: 10段階の品質レベルで系統的に精度を測定
        2. **最適範囲の特定**: どの品質レベルが最も高精度か?
        3. **実用ガイドライン**: 用途別の推奨レベルを明確化
        4. **科学的根拠**: 過学習・情報損失のメカニズムを解明
        """)
        
        st.markdown("---")
        
        st.markdown("### 研究方法")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **検証データセット:**
            - 検証画像数: 33枚
            - 品質レベル: Low1-10 (全10段階)
            - 画像: IMG_5023-5052, IMG_5202, IMG_5204, IMG_5205
            
            **評価指標:**
            - MAE (Mean Absolute Error)
            - 平均相対誤差 (%)
            - 最大絶対誤差
            - 最悪ケース相対誤差 (%)
            - 誤差分布 (高精度ケースの割合)
            """)
        
        with col2:
            st.markdown("""
            **検証プロセス:**
            1. 各品質レベルで高画質→低画質ペアを作成
            2. モデルによるフラクタル次元予測を実行
            3. 実測値との誤差を統計的に分析
            4. 10段階全てのデータを収集・比較
            5. U字型カーブと臨界境界を発見
            6. 用途別推奨事項を策定
            """)
        
        st.markdown("---")
        
        st.markdown("### 学術的意義")
        
        st.success("""
        **1. 画質と予測精度の非線形関係の実証**
        - 従来の仮定: 高画質 = 高精度
        - 本研究の発見: U字型の関係性（両端で劣化）
        - 意義: 入力データの品質最適化の重要性を実証
        
        **2. 過学習の新しい視点**
        - 従来: データ量や複雑さによる過学習
        - 本研究: 入力品質による過学習を発見
        - 意義: データ前処理・標準化の再評価が必要
        
        **3. JPEG圧縮の影響の定量化**
        - 従来: 定性的な理解のみ
        - 本研究: 定量的な閾値を特定(Low7/Low8境界)
        - 意義: 医療画像解析での標準プロトコル策定に貢献
        
        **4. 実用的ガイドラインの提供**
        - 従来: 経験則ベース
        - 本研究: データ駆動の具体的基準
        - 意義: 再現性のある研究プロトコルの確立
        """)
        
        st.markdown("---")
        
        st.markdown("### 論文化への提案")
        
        st.info("""
        **タイトル案:**
        
        英語: "JPEG Compression Quality Optimization for Fractal Dimension Prediction: 
        Discovery of U-shaped Accuracy Curve and Golden Zone"
        
        日本語: "フラクタル次元予測におけるJPEG圧縮品質最適化: 
        U字型精度カーブとゴールデンゾーンの発見"
        
        **投稿先候補:**
        1. Medical Image Analysis (IF: 10.7) - 医療画像解析のトップジャーナル
        2. IEEE Transactions on Medical Imaging (IF: 10.6) - 画像処理と医療の融合
        3. Pattern Recognition (IF: 8.0) - パターン認識全般
        4. Journal of Digital Imaging (IF: 4.4) - デジタル医療画像
        
        **論文構成案:**
        1. Abstract: U字カーブ、Golden Zone、2つの臨界境界
        2. Introduction: 画質と予測精度の関係性の重要性
        3. Methods: 10段階検証プロトコル、統計的評価手法
        4. Results: 詳細な統計データ、可視化、ケーススタディ
        5. Discussion: 過学習と情報損失のメカニズム、臨床的意義
        6. Conclusion: 実用的ガイドライン、今後の展望
        """)
        
        st.markdown("---")
        
        st.markdown("### 今後の研究課題")
        
        st.markdown("""
        **1. 原因の深掘り調査**
        - なぜLow5が最適なのか? (訓練データの品質分布を調査)
        - Low7/Low8境界で何が失われるのか? (周波数解析)
        - 特異画像の原因は?
        
        **2. モデル改善の可能性**
        - マルチ品質学習 (Low1-10全てを訓練データに含める)
        - データ拡張戦略 (品質レベルごとのデータ拡張)
        - アンサンブル予測 (複数品質レベルの予測を統合)
        - 品質自動選択 (入力画像の品質を自動検出)
        
        **3. 他のデータセットでの検証**
        - 異なる組織タイプ (皮膚、肺、脳など)
        - 異なる撮影条件 (顕微鏡タイプ、倍率、染色)
        - 異なる被験者集団 (年齢層、疾患タイプ、人種)
        
        **4. リアルタイム最適化**
        - 品質自動検出アルゴリズム
        - 適応的前処理パイプライン
        - 予測信頼度の推定手法
        - エラー検出・補正機能
        """)
        
        st.markdown("---")
        
        st.markdown("### 研究環境・データ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **検証環境:**
            - プロジェクト: Fractal-Analyzer-V2
            - モデル: CNN-based FD Predictor
            - 検証日: 2025年11月11日
            - 画像数: 33枚
            """)
        
        with col2:
            st.markdown("""
            **データファイル:**
            - Low1-10の検証結果CSV
            - 学習済みモデル (pkl)
            - 研究レポート (MD)
            """)
        
        st.markdown("---")
        
        st.markdown("### 引用")
        st.code("""
Fractal-Analyzer-V2 Quality Optimization Study (2025)
"JPEG Compression Quality Optimization for Fractal Dimension Prediction"
GitHub: ryuki4219/Fractal-Analyzer-V2
        """, language='text')

# ============================================================
# Streamlit app
# ============================================================
def app():
    # ============================================================
    # 📱 レイアウトモードの初期化（最優先で実行）
    # ============================================================
    if 'layout_mode' not in st.session_state:
        st.session_state['layout_mode'] = 'モバイル版'  # デフォルトはモバイル版
    
    st.set_page_config(layout="centered", page_title="Fractal Analyzer V2 - フラクタル次元解析")
    st.title("� Fractal Analyzer V2 - フラクタル次元解析システム")
    
    # ============================================================
    # ⚠️ 重要な注意事項・免責事項（コンパクト版）
    # ============================================================
    with st.expander("⚠️ 重要な注意事項（必ずお読みください）", expanded=False):
        st.warning("""
        **本サービスは研究・教育目的のツールであり、医療診断を目的としたものではありません。**
        
        - 🔬 フラクタル幾何学の研究・学習用ツールです
        - ❌ 医療診断・健康判定には使用しないでください
        - ⚕️ 医療に関する判断は、必ず医療機関・医師にご相談ください
        - 📊 解析結果は参考値としてご利用ください
        
        本サービスを利用することで、[利用規約](https://github.com/ryuki4219/Fractal-Analyzer-V2/blob/main/TERMS_OF_SERVICE.md)に同意したものとみなされます。
        """)
        
        # ドキュメントへのリンク（2列×2行でモバイル対応）
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("📖 [使い方ガイド](https://github.com/ryuki4219/Fractal-Analyzer-V2/blob/main/USER_GUIDE.md)")
            st.markdown("🔒 [プライバシーポリシー](https://github.com/ryuki4219/Fractal-Analyzer-V2/blob/main/PRIVACY_POLICY.md)")
        with col2:
            st.markdown("📜 [利用規約](https://github.com/ryuki4219/Fractal-Analyzer-V2/blob/main/TERMS_OF_SERVICE.md)")
            st.markdown("💻 [GitHubリポジトリ](https://github.com/ryuki4219/Fractal-Analyzer-V2)")
    
    # システム情報もコンパクトに
    with st.expander("ℹ️ システム情報", expanded=False):
        st.info("CuPy が利用可能な場合は GPU を自動で使います。無ければ CPU (NumPy) で処理します。")
    
    # ============================================================
    # � 自動モデル読み込み機能 - アプリ起動時に実行（最初に実行）
    # ============================================================
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
        st.session_state['persistent_model'] = None
        st.session_state['model_info'] = None
        st.session_state['auto_load_attempted'] = False
        
        # 保存されたモデルを探す
        default_model_path = "trained_fd_model.pkl"
        history_path = "training_history.json"
        
        # モデルファイルと学習履歴の存在確認
        model_exists = os.path.exists(default_model_path)
        history_exists = os.path.exists(history_path)
        
        if model_exists:
            try:
                model = load_model(default_model_path)
                st.session_state['persistent_model'] = model
                st.session_state['model_loaded'] = True
                
                # ファイルの更新日時を取得
                model_mtime = os.path.getmtime(default_model_path)
                model_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_mtime))
                
                st.session_state['model_info'] = {
                    'path': default_model_path,
                    'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'trained_at': model_date,
                    'source': '自動読み込み（前回の学習結果）',
                    'file_size': os.path.getsize(default_model_path)
                }
                st.session_state['auto_load_attempted'] = True
            except Exception as e:
                st.session_state['auto_load_error'] = str(e)
                pass  # 読み込み失敗時は無視
        
        # 学習履歴の統計を取得
        if history_exists:
            try:
                history = load_training_history()
                st.session_state['history_stats'] = {
                    'total_sessions': len(history),
                    'last_trained': history[-1].get('timestamp', '不明') if history else '不明',
                    'total_samples': sum(h.get('total_samples', 0) for h in history)
                }
            except:
                pass
    
    # ============================================================
    # �🔔 起動時の継続性通知（session_state初期化後に実行）
    # ============================================================
    if 'startup_notification_shown' not in st.session_state:
        st.session_state['startup_notification_shown'] = True
        
        # モデルと学習履歴の状態を確認
        model_loaded = st.session_state.get('model_loaded', False)
        history_stats = st.session_state.get('history_stats', {})
        total_sessions = history_stats.get('total_sessions', 0)
        
        if model_loaded and total_sessions > 0:
            # 前回の学習が継続されている場合
            st.success(f"""
            ✅ **前回の学習状態を復元しました！**
            
            - 📚 学習回数: {total_sessions}回
            - 🎓 累計学習データ: {history_stats.get('total_samples', 0):,}組
            - 📅 最終学習: {history_stats.get('last_trained', '不明')}
            - 🤖 モデル: 自動読み込み完了
            
            💡 前回の学習結果がそのまま使えます。推論モードですぐに解析できます！
            """)
        elif total_sessions > 0:
            # 学習履歴はあるがモデルが無い場合
            st.warning(f"""
            ⚠️ **学習履歴を検出しましたが、モデルファイルが見つかりません**
            
            - 📚 学習履歴: {total_sessions}回
            - 📅 最終学習: {history_stats.get('last_trained', '不明')}
            
            💡 学習モードで再学習すると、AIが復活します。
            """)
        else:
            # 初めての起動
            st.info("""
            👋 **ようこそ！フラクタル解析AIへ**
            
            このアプリは、あなたの学習履歴とAIモデルを自動的に保存します。
            
            **次回起動時も:**
            - ✅ 学習した知識が継続
            - ✅ AIモデルが自動読み込み
            - ✅ 学習履歴が保持
            
            💡 まずは「学習モード」でAIを学習させましょう！
            """)
    
    # ============================================================
    # 📱 レイアウト選択機能（サイドバー）
    # ============================================================
    with st.sidebar:
        st.markdown("### 📱 表示レイアウト設定")
        layout_mode = st.radio(
            "レイアウトモードを選択",
            options=['モバイル版', 'デスクトップ版'],
            index=0 if st.session_state['layout_mode'] == 'モバイル版' else 1,
            help="""モバイル版: 2列表示、縦スクロール最適化
デスクトップ版: 4-5列表示、横幅最大活用"""
        )
        
        # レイアウトモードが変更された場合は再読み込み
        if layout_mode != st.session_state['layout_mode']:
            st.session_state['layout_mode'] = layout_mode
            st.info(f"💡 {layout_mode}に切り替えました。ページを再読み込みしてください。")
            st.button("🔄 再読み込み", on_click=lambda: st.rerun())
        
        st.divider()
    
    # ============================================================
    # 🎯 AI成長状況レポート（トップに表示）
    # ============================================================
    training_history_preview = load_training_history()
    ai_status = calculate_ai_readiness(training_history_preview)
    
    # レイアウトモードによって列数を変更
    is_mobile = st.session_state['layout_mode'] == 'モバイル版'
    
    # AIステータス表示（レイアウトモードに応じて列数変更）
    if is_mobile:
        # モバイル版: 2×2グリッド
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 AI信頼度", f"{ai_status['confidence']:.0f}%", 
                     delta=ai_status['level'])
        with col2:
            status_emoji = "✅" if ai_status['ready'] else "⚠️"
            status_text = "実用可能" if ai_status['ready'] else "学習中"
            st.metric("📊 実用化状況", f"{status_emoji} {status_text}", 
                     delta=f"{ai_status['stats']['total_sessions']}回学習")
        
        col3, col4 = st.columns(2)
        with col3:
            if ai_status['stats']['total_sessions'] > 0:
                st.metric("📈 最新相関係数", 
                         f"{ai_status['stats']['latest_correlation']:.3f}",
                         delta=f"目標: 0.850+")
            else:
                st.metric("📈 最新相関係数", "未学習", delta="学習開始してください")
        with col4:
            if ai_status['stats']['total_sessions'] > 0:
                st.metric("🎯 最新誤差(MAE)", 
                         f"{ai_status['stats']['latest_mae']:.4f}",
                         delta=f"目標: 0.010以下",
                         delta_color="inverse")
            else:
                st.metric("🎯 最新誤差(MAE)", "未学習", delta="")
    else:
        # デスクトップ版: 1×4横並び
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 AI信頼度", f"{ai_status['confidence']:.0f}%", 
                     delta=ai_status['level'])
        with col2:
            status_emoji = "✅" if ai_status['ready'] else "⚠️"
            status_text = "実用可能" if ai_status['ready'] else "学習中"
            st.metric("📊 実用化状況", f"{status_emoji} {status_text}", 
                     delta=f"{ai_status['stats']['total_sessions']}回学習")
        with col3:
            if ai_status['stats']['total_sessions'] > 0:
                st.metric("📈 最新相関係数", 
                         f"{ai_status['stats']['latest_correlation']:.3f}",
                         delta=f"目標: 0.850+")
            else:
                st.metric("📈 最新相関係数", "未学習", delta="学習開始してください")
        with col4:
            if ai_status['stats']['total_sessions'] > 0:
                st.metric("🎯 最新誤差(MAE)", 
                         f"{ai_status['stats']['latest_mae']:.4f}",
                         delta=f"目標: 0.010以下",
                         delta_color="inverse")
            else:
                st.metric("🎯 最新誤差(MAE)", "未学習", delta="")
    
    # 詳細レポートはエクスパンダーで
    if ai_status['stats']['total_sessions'] > 0:
        with st.expander("📊 AI成長レポート（詳細）", expanded=False):
            tab1, tab2, tab3 = st.tabs(["📈 成長状況", "📚 学習履歴", "🎯 改善アドバイス"])
            
            with tab1:
                st.markdown("### 🎯 AI実用化進捗")
                
                # プログレスバー
                st.progress(ai_status['confidence'] / 100, 
                           text=f"信頼度: {ai_status['confidence']:.1f}% - {ai_status['level']}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("#### 📊 現在の性能")
                    stats = ai_status['stats']
                    perf_data = {
                        "指標": ["相関係数", "平均誤差", "改善率", "学習回数", "総データ数"],
                        "現在値": [
                            f"{stats['latest_correlation']:.4f}",
                            f"{stats['latest_mae']:.4f}",
                            f"{stats['latest_improvement']:.1f}%",
                            f"{stats['total_sessions']}回",
                            f"{stats['total_pairs']}組"
                        ],
                        "目標値": ["0.850+", "0.010以下", "20%+", "5回+", "100組+"],
                        "達成": [
                            "✅" if stats['latest_correlation'] >= 0.85 else "🔄",
                            "✅" if stats['latest_mae'] <= 0.01 else "🔄",
                            "✅" if stats['latest_improvement'] >= 20 else "🔄",
                            "✅" if stats['total_sessions'] >= 5 else "🔄",
                            "✅" if stats['total_pairs'] >= 100 else "🔄"
                        ]
                    }
                    st.dataframe(perf_data, use_container_width=True, hide_index=True)
                
                with col_b:
                    st.markdown("#### 🎯 次のマイルストーン")
                    milestone = ai_status['next_milestone']
                    st.write(f"**目標:** {milestone['target']}")
                    st.progress(milestone['progress'] / 100)
                    st.info(f"💡 {milestone['needed']}")
                    
                    st.markdown("#### 📈 ベスト記録")
                    st.write(f"**最高相関係数:** {stats['best_correlation']:.4f}")
                    st.write(f"**最小誤差:** {stats['best_mae']:.4f}")
                    st.write(f"**最大改善率:** {stats['best_improvement']:.1f}%")
                
                # 実用化判定
                st.markdown("---")
                if ai_status['ready']:
                    st.success("""
                    ### ✅ 実用化可能レベルに到達！
                    
                    **おめでとうございます！** このAIは本番アプリに搭載できる水準です。
                    
                    **次のステップ:**
                    1. 🎯 モデルをエクスポート（`trained_fd_model.pkl`）
                    2. 🚀 本番アプリケーションに統合
                    3. 📊 実運用でのパフォーマンス監視
                    4. 🔄 定期的な再学習で精度維持
                    """)
                else:
                    st.warning(f"""
                    ### ⚠️ 現在は学習中 - 信頼度 {ai_status['confidence']:.0f}%
                    
                    実用化には **信頼度75%以上** が必要です。
                    あと **{75 - ai_status['confidence']:.0f}ポイント** の改善が必要です。
                    
                    **改善方法は「改善アドバイス」タブをご確認ください。**
                    """)
            
            with tab2:
                st.markdown("### 📚 学習履歴一覧")
                if len(training_history_preview) > 0:
                    history_data = []
                    for i, record in enumerate(training_history_preview[-10:], 1):  # 最新10件
                        metrics = record.get('metrics', {})
                        history_data.append({
                            "回": len(training_history_preview) - 10 + i,
                            "日時": record.get('timestamp', '')[:16],
                            "データ数": record.get('num_pairs', 0),
                            "品質レベル": record.get('quality_level', '不明'),
                            "拡張": record.get('augmentation_count', 0),
                            "相関": f"{metrics.get('correlation_pred', 0):.3f}",
                            "誤差": f"{metrics.get('mae_pred', 0):.4f}",
                            "改善": f"{metrics.get('improvement', 0):.1f}%"
                        })
                    
                    import pandas as pd
                    st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
                    
                    # グラフで成長を可視化
                    st.markdown("#### 📈 成長グラフ")
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    sessions = list(range(1, len(training_history_preview) + 1))
                    correlations = [h.get('metrics', {}).get('correlation_pred', 0) for h in training_history_preview]
                    maes = [h.get('metrics', {}).get('mae_pred', 0) for h in training_history_preview]
                    
                    ax1.plot(sessions, correlations, marker='o', linewidth=2, markersize=6)
                    ax1.axhline(y=0.85, color='g', linestyle='--', label='目標: 0.85')
                    ax1.set_xlabel('学習回数')
                    ax1.set_ylabel('相関係数')
                    ax1.set_title('相関係数の推移')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(sessions, maes, marker='s', linewidth=2, markersize=6, color='orange')
                    ax2.axhline(y=0.01, color='g', linestyle='--', label='目標: 0.01')
                    ax2.set_xlabel('学習回数')
                    ax2.set_ylabel('平均絶対誤差 (MAE)')
                    ax2.set_title('誤差の推移')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("まだ学習履歴がありません。学習モードで学習を開始してください。")
            
            with tab3:
                st.markdown("### 💡 改善アドバイス")
                for i, rec in enumerate(ai_status['recommendations'], 1):
                    st.write(f"**{i}.** {rec}")
                
                st.markdown("---")
                st.markdown("""
                ### 📖 効果的な学習のポイント
                
                #### 1. データ量を増やす
                - **目標:** 100組以上の画像ペア
                - **方法:** データ拡張を活用（現在28種類利用可能）
                - **効果:** 精度向上、過学習防止
                
                #### 2. 品質レベルを調整
                - **low1-3:** 軽度劣化 - 学習が容易
                - **low4-7:** 中度劣化 - バランスが良い（推奨）
                - **low8-10:** 重度劣化 - 難易度高、実用的
                
                #### 3. データの多様性
                - 異なる照明条件の画像
                - 異なる被写体
                - 異なるアングル
                
                #### 4. 定期的な学習
                - 週1回以上の学習を推奨
                - 新しいデータを追加して再学習
                - 性能の安定性を確認
                """)
    else:
        st.info("💡 学習を開始すると、AIの成長状況がここに表示されます。下の「学習モード」でデータを学習させてください。")
    
    st.markdown("---")

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
        st.sidebar.success("✅ **モデル読み込み済み**")
        
        with st.sidebar.expander("📋 モデル詳細情報", expanded=True):
            st.write(f"**読み込み元:** {model_info.get('source', '不明')}")
            st.write(f"**学習日時:** {model_info.get('trained_at', '不明')}")
            st.write(f"**読み込み時刻:** {model_info.get('loaded_at', '不明')}")
            
            # ファイルサイズを表示
            file_size = model_info.get('file_size', 0)
            if file_size > 0:
                size_mb = file_size / (1024 * 1024)
                st.write(f"**ファイルサイズ:** {size_mb:.2f} MB")
            
            # 学習履歴統計
            if 'history_stats' in st.session_state:
                stats = st.session_state['history_stats']
                st.write("---")
                st.write("**📚 学習履歴統計:**")
                st.write(f"- 学習回数: {stats.get('total_sessions', 0)}回")
                st.write(f"- 累計サンプル数: {stats.get('total_samples', 0):,}組")
                st.write(f"- 最終学習: {stats.get('last_trained', '不明')}")
        
        st.sidebar.info("💡 このモデルは次回起動時も自動的に読み込まれます")
        
        # モデルをリセットするボタン
        if st.sidebar.button("🔄 モデルをリセット"):
            st.session_state['persistent_model'] = None
            st.session_state['model_loaded'] = False
            st.session_state['model_info'] = None
            st.rerun()
    else:
        st.sidebar.warning("⚠️ **モデル未読み込み**")
        
        # 自動読み込みエラーの場合
        if 'auto_load_error' in st.session_state:
            st.sidebar.error(f"エラー: {st.session_state['auto_load_error']}")
        
        st.sidebar.write("**モデルを準備する方法:**")
        st.sidebar.write("1. 学習モードで新規学習")
        st.sidebar.write("2. 推論モードでモデルをアップロード")
        
        # 学習履歴がある場合の表示
        if 'history_stats' in st.session_state:
            stats = st.session_state['history_stats']
            if stats.get('total_sessions', 0) > 0:
                st.sidebar.info(f"📚 学習履歴: {stats['total_sessions']}回\n最終学習: {stats['last_trained']}")
                st.sidebar.write("💡 モデルファイルが見つかりません。再学習してください。")
    
    # ============================================================
    # 📚 学習履歴を表示
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.header("📚 学習履歴")
    training_history = load_training_history()
    
    # AI準備状況評価
    ai_readiness = calculate_ai_readiness(training_history)
    
    # AIステータスダッシュボード
    st.sidebar.markdown("### 🎯 AI実用化ステータス")
    
    # 信頼度メーター
    confidence = ai_readiness['confidence']
    confidence_color = (
        "🟢" if confidence >= 75 else
        "🟡" if confidence >= 50 else
        "🟠" if confidence >= 30 else
        "🔴"
    )
    st.sidebar.metric(
        label="信頼度",
        value=f"{confidence:.0f}%",
        delta=f"{ai_readiness['level']}"
    )
    
    # プログレスバー
    st.sidebar.progress(confidence / 100)
    
    # 実用可否の判定
    if ai_readiness['ready']:
        st.sidebar.success("✅ **実用可能** - 本番アプリに搭載できます")
    else:
        st.sidebar.warning("⚠️ **学習中** - さらなるデータが必要です")
    
    st.sidebar.write(f"**レベル:** {ai_readiness['level']}")
    st.sidebar.write(f"_{ai_readiness['level_desc']}_")
    
    # 統計情報
    with st.sidebar.expander("📊 詳細統計"):
        stats = ai_readiness['stats']
        st.write(f"**学習回数:** {stats['total_sessions']}回")
        st.write(f"**総データ数:** {stats['total_pairs']}組")
        st.write(f"**最高相関:** {stats['best_correlation']:.3f}")
        st.write(f"**平均相関:** {stats['avg_correlation']:.3f}")
        st.write(f"**最小誤差:** {stats['best_mae']:.4f}")
        st.write(f"**平均誤差:** {stats['avg_mae']:.4f}")
    
    # 次のマイルストーン
    milestone = ai_readiness['next_milestone']
    with st.sidebar.expander("🎯 次の目標"):
        st.write(f"**目標:** {milestone['target']}")
        st.progress(milestone['progress'] / 100)
        st.write(f"_{milestone['needed']}_")
    
    # 改善アドバイス
    with st.sidebar.expander("💡 改善アドバイス"):
        for rec in ai_readiness['recommendations']:
            st.write(f"• {rec}")
    
    # 🔍 デバッグ情報（開発者向け）
    with st.sidebar.expander("🔍 デバッグ情報"):
        st.write("### 最新学習データ")
        if training_history and len(training_history) > 0:
            latest = training_history[-1]
            st.json({
                'timestamp': latest.get('timestamp'),
                'num_pairs': latest.get('num_pairs'),
                'total_samples': latest.get('total_samples'),
                'quality_level': latest.get('quality_level'),
                'augmentation_count': latest.get('augmentation_count'),
                'metrics': latest.get('metrics', {})
            })
            
            st.write("### 信頼度計算に使用した値")
            metrics = latest.get('metrics', {})
            st.write(f"- correlation_pred: {metrics.get('correlation_pred', 'なし')}")
            st.write(f"- improvement: {metrics.get('improvement', 'なし')}")
            st.write(f"- mae_pred: {metrics.get('mae_pred', 'なし')}")
            st.write(f"**計算された信頼度:** {ai_readiness['confidence']:.1f}%")
            
            # 履歴ファイルをリセットするボタン
            st.write("---")
            if st.button("🗑️ 学習履歴をリセット", help="古いデータをクリアして新しく学習し直します"):
                try:
                    if os.path.exists("training_history.json"):
                        os.remove("training_history.json")
                        st.success("✅ 学習履歴をリセットしました")
                        st.rerun()
                except Exception as e:
                    st.error(f"エラー: {e}")
        else:
            st.write("学習履歴がありません")
    
    # 学習履歴詳細
    if training_history:
        st.sidebar.write(f"")
        st.sidebar.write(f"**学習記録:** {len(training_history)}回")
        
        # 最新の学習情報を表示
        if len(training_history) > 0:
            latest = training_history[-1]
            st.sidebar.write(f"**最新学習:** {latest.get('timestamp', '不明')[:16]}")
            if 'metrics' in latest:
                metrics = latest['metrics']
                corr = metrics.get('correlation_pred', 0)
                improve = metrics.get('improvement', 0)
                st.sidebar.write(f"📈 相関: {corr:.3f}")
                st.sidebar.write(f"🎯 改善: {improve:.1f}%")
                
                # AI評価を表示
                evaluation = evaluate_ai_performance(corr, improve, metrics.get('mae_pred', 0))
                st.sidebar.write(f"**総合評価:** {evaluation['grade']} {evaluation['emoji']}")
        
        # 学習成長分析
        if len(training_history) >= 2:
            with st.sidebar.expander("� AI成長分析"):
                growth_analysis = analyze_learning_growth(training_history)
                st.write(f"**成長トレンド:** {growth_analysis['trend']} {growth_analysis['trend_emoji']}")
                st.write(f"**相関係数の変化:** {growth_analysis['correlation_change']:+.3f}")
                st.write(f"**改善率の変化:** {growth_analysis['improvement_change']:+.1f}%")
                st.write(f"**最高記録 (相関):** {growth_analysis['best_correlation']:.3f}")
                st.write("")
                st.write(growth_analysis['recommendation'])
        
        # 履歴の詳細を展開可能に
        with st.sidebar.expander("📋 全履歴を表示"):
            for i, record in enumerate(reversed(training_history[-10:]), 1):
                idx = len(training_history) - i + 1
                st.write(f"**#{idx}** {record.get('timestamp', '不明')}")
                st.write(f"  - サンプル数: {record.get('total_samples', 0)}")
                st.write(f"  - 拡張: {record.get('augmentation_count', 0)}種類")
                if 'metrics' in record:
                    metrics = record['metrics']
                    corr = metrics.get('correlation_pred', 0)
                    improve = metrics.get('improvement', 0)
                    st.write(f"  - 相関: {corr:.3f}")
                    st.write(f"  - 改善: {improve:.1f}%")
                    # 各記録の評価
                    eval_result = evaluate_ai_performance(corr, improve, metrics.get('mae_pred', 0))
                    st.write(f"  - 評価: {eval_result['grade']} {eval_result['emoji']}")
                st.write("---")
    else:
        st.sidebar.info("まだ学習記録がありません")
    
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
    
    # 🚀 次のステップガイド（AI準備状況に応じて表示）
    if ai_status['stats']['total_sessions'] > 0:
        with st.sidebar.expander("🚀 本番アプリ開発について"):
            if ai_status['ready']:
                st.success("✅ AIは実用レベルです！")
            else:
                st.warning(f"⚠️ 信頼度 {ai_status['confidence']:.0f}% - あと{75-ai_status['confidence']:.0f}ポイント")
            
            st.markdown("""
            ### 🎯 アプリの最終目標
            
            低画質画像でも正確なフラクタル次元を予測できるAI搭載アプリ
            
            ---
            
            ### ✅ 推奨: このアプリを改良
            
            **メリット:**
            - 学習と推論が統合済み
            - モデル更新が容易
            - 開発時間が短い
            
            **改良の流れ:**
            1. 信頼度75%達成まで学習
            2. 推論モードをメインに
            3. 自動品質判定を追加
            4. UIをシンプル化
            
            ---
            
            ### 🔄 または: 新規アプリ作成
            
            **方法:**
            1. `trained_fd_model.pkl` をエクスポート
            2. 推論専用アプリを新規作成
            3. シンプルなUIで実装
            
            **デメリット:** 開発時間、メンテナンス2倍
            
            ---
            
            **💡 どちらを選ぶべき?**
            
            → **改良推奨** (効率的、柔軟性高)
            """)

    st.sidebar.markdown("---")

    # アプリケーションモード選択
    st.sidebar.header("アプリケーションモード")
    app_mode = st.sidebar.radio(
        "モードを選択",
        [
            "🔮 推論モード (低画質画像のみで予測)",
            "🎓 学習モード (画像ペアが必要)", 
            "📊 研究報告・品質ガイド",
            "🌸 顔全体分析モード",
            "🔬 実験データ収集",
            "📈 相関分析"
        ],
        index=0,  # デフォルトを推論モードに設定
        help="推論モード: 学習済みモデルで低画質画像から予測（メイン機能）\n学習モード: 高画質+低画質ペアでAIを学習\n研究報告: 品質最適化研究の結果と実用ガイド\n顔全体分析: 顔の各部位を自動検出して分析\n実験データ収集: 肌状態とFDの相関実験用データ収集\n相関分析: 収集したデータの統計分析"
    )
    
    st.sidebar.markdown("---")

    # 推論モード
    if app_mode == "🔮 推論モード (低画質画像のみで予測)":
        st.header("🔮 推論モード - 低画質画像だけで高品質FDを予測")
        
        # サブモード選択を追加
        st.markdown("### モード選択")
        inference_submode = st.radio(
            "実行モード",
            ["🔮 通常予測モード", "🎯 精度検証モード (高画質ペアで検証)"],
            help="通常予測: 低画質画像のみで予測\n精度検証: 高画質ペアと比較して予測精度を確認"
        )
        
        st.markdown("---")
        
        if inference_submode == "🔮 通常予測モード":
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
        else:
            st.markdown("""
            ### 🎯 精度検証モードについて
            
            **高画質・低画質のペア画像を使って、AIの予測精度を詳しく検証できます。**
            
            #### 📋 使い方
            1. 高画質画像と低画質画像をペアでアップロード
            2. AIが低画質から高品質FDを予測
            3. 実際の高画質画像のFDと比較
            4. **予測精度を定量的に評価**
            
            #### ✨ できること
            - 予測値 vs 実測値の比較
            - 誤差の統計分析
            - 相関係数・MAE・RMSE の計算
            - 散布図・誤差分布の可視化
            - 画像ごとの詳細な精度確認
            
            💡 **モデルの性能を客観的に評価し、改善点を見つけられます**
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
            # ========================================
            # 🔮 通常予測モード
            # ========================================
            if inference_submode == "🔮 通常予測モード":
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
                    
                    # 🆕 画像品質自動判定
                    if IMAGE_QUALITY_ASSESSOR_AVAILABLE:
                        with st.expander("🔍 画像品質チェック", expanded=False):
                            st.markdown("""
                            **アップロードされた画像の品質を自動判定します**
                            - ✅ **高品質・推奨**: very_high / high 信頼度で解析可能
                            - ✅ **品質過剰**: 直接解析推奨（AI予測には不向き）
                            - ⚠️ **信頼度低下**: 画質が低く、解析精度が下がる可能性
                            - ℹ️ **全ての画像が解析可能**: 信頼度は異なりますが、全て処理できます
                            """)
                            
                            quality_results = []
                            quality_check_progress = st.progress(0)
                            
                            for idx, img_file in enumerate(low_quality_imgs):
                                # 一時ファイルとして保存して品質評価
                                import tempfile
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file.name)[1]) as tmp:
                                    tmp.write(img_file.getvalue())
                                    tmp_path = tmp.name
                                
                                # 品質評価
                                quality_result = assess_image_quality(tmp_path)
                                quality_results.append({
                                    'filename': img_file.name,
                                    'result': quality_result
                                })
                                
                                # 一時ファイル削除
                                os.unlink(tmp_path)
                                
                                quality_check_progress.progress((idx + 1) / len(low_quality_imgs))
                            
                            # 品質判定結果の表示
                            st.markdown("### 📊 品質判定結果")
                            
                            quality_df_data = []
                            for qr in quality_results:
                                result = qr['result']
                                if 'error' not in result:
                                    rec = result['recommendation']
                                    metrics = result['metrics']
                                    quality_df_data.append({
                                        "画像名": qr['filename'],
                                        "判定": f"{rec['icon']} {rec['title']}",
                                        "品質レベル": result['quality_level'],
                                        "信頼度": rec['confidence'],
                                        "解像度": metrics['resolution'],
                                        "鮮明度": f"{metrics['sharpness']:.1f}",
                                        "JPEG品質": metrics['estimated_jpeg_quality']
                                    })
                                else:
                                    quality_df_data.append({
                                        "画像名": qr['filename'],
                                        "判定": "❌ エラー",
                                        "品質レベル": "-",
                                        "信頼度": "-",
                                        "解像度": "-",
                                        "鮮明度": "-",
                                        "JPEG品質": "-"
                                    })
                            
                            quality_df = pd.DataFrame(quality_df_data)
                            st.dataframe(quality_df, use_container_width=True, hide_index=True)
                            
                            # 統計サマリー
                            total_count = len(quality_results)
                            high_quality_count = sum(1 for qr in quality_results if qr['result'].get('quality_level') == 'high')
                            low47_count = sum(1 for qr in quality_results if qr['result'].get('quality_level') == 'low4-7')
                            low13_count = sum(1 for qr in quality_results if qr['result'].get('quality_level') == 'low1-3')
                            low810_count = sum(1 for qr in quality_results if qr['result'].get('quality_level') == 'low8-10')
                            
                            # レイアウトモードに応じて列数変更
                            if is_mobile:
                                # モバイル版: 2×2グリッド
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("総画像数", total_count)
                                with col2:
                                    st.metric("高品質", high_quality_count)
                                
                                col3, col4 = st.columns(2)
                                with col3:
                                    st.metric("Golden Zone", low47_count)
                                with col4:
                                    st.metric("低信頼度", low810_count)
                            else:
                                # デスクトップ版: 1×4横並び
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("総画像数", total_count)
                                with col2:
                                    st.metric("高品質", high_quality_count)
                                with col3:
                                    st.metric("Golden Zone", low47_count)
                                with col4:
                                    st.metric("低信頼度", low810_count)
                            
                            # 情報表示
                            if low810_count > 0:
                                st.info(f"""
                                ℹ️ **{low810_count}枚の画像は信頼度が低下する可能性があります**
                                
                                解析は可能ですが、より高品質な画像での再撮影を推奨します:
                                - iPhone 8以降
                                - Galaxy S8以降
                                - Pixel 2以降
                                - 一眼レフ・ミラーレスカメラ
                                """)
                            
                            if low13_count > 0:
                                st.info(f"""
                                ℹ️ **{low13_count}枚の画像は品質過剰です**
                                
                                JPEG品質が高すぎるため、AI予測には不向きですが、
                                **直接解析を使用すれば高精度な結果が得られます**。
                                
                                または、JPEG品質を70-85%程度に調整して再撮影し、
                                AI予測を使用することもできます。
                                """)
                    
                    # 予測実行ボタン
                    if st.button("🔮 フラクタル次元を予測"):
                        st.info("処理を開始します...")
                        
                        # session_stateに結果を保存
                        st.session_state['inference_results'] = []
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, img_file in enumerate(low_quality_imgs):
                            # 画像読み込み
                            img_file.seek(0)  # ファイルポインタをリセット
                            img = read_bgr_from_buffer(img_file.read())
                            
                            if img is not None:
                                # 品質レベルを取得（既に評価済み）
                                quality_info = quality_results[idx]['result']
                                quality_level = quality_info.get('quality_level', 'unknown')
                                processing_method = quality_info['recommendation']['processing_method']
                                
                                # 処理方法に応じて分岐
                                if processing_method == 'direct_analysis':
                                    # 直接解析(high, low1-3)
                                    st.info(f"📐 {img_file.name}: 直接解析を実行中...")
                                    fd_result_dict = calculate_fractal_dimension(img)
                                    fd_value = fd_result_dict['fd']
                                    
                                    results.append({
                                        'filename': img_file.name,
                                        'predicted_fd': fd_value,
                                        'image': img,
                                        'method': 'direct_analysis',
                                        'quality_level': quality_level,
                                        'fitting_data': fd_result_dict.get('fitting_data'),
                                        'confidence': {
                                            'overall_confidence': fd_result_dict['confidence'],
                                            'confidence_level': '高信頼度',
                                            'level_emoji': '✅',
                                            'lower_bound': fd_result_dict['range'][0],
                                            'upper_bound': fd_result_dict['range'][1]
                                        }
                                    })
                                else:
                                    # AI予測（low4-7, low8-10）
                                    predicted_fd = predict_fd_from_low_quality(img, model)
                                    
                                    # 信頼度計算
                                    confidence_info = calculate_prediction_confidence(img, model, predicted_fd)
                                    
                                    results.append({
                                        'filename': img_file.name,
                                        'predicted_fd': predicted_fd,
                                        'image': img,
                                        'method': 'ai_prediction',
                                        'quality_level': quality_level,
                                        'confidence': confidence_info
                                    })
                            
                            progress_bar.progress((idx + 1) / len(low_quality_imgs))
                        
                        # session_stateに結果を保存
                        st.session_state['inference_results'] = results
                        
                        st.success("✅ 処理完了!")
                    
                    # 結果が存在する場合に表示（ボタンの外で処理）
                    if 'inference_results' in st.session_state and st.session_state['inference_results']:
                        results = st.session_state['inference_results']
                        
                        # 結果表示
                        st.subheader("📊 解析・予測結果")
                        
                        # 処理方法別に分類
                        direct_analysis_results = [r for r in results if r['method'] == 'direct_analysis']
                        ai_prediction_results = [r for r in results if r['method'] == 'ai_prediction']
                        
                        st.markdown(f"""
                        **処理結果サマリー:**
                        - 📐 **直接解析**: {len(direct_analysis_results)}枚 (高品質・品質過剰の画像)
                        - 🔮 **AI予測**: {len(ai_prediction_results)}枚 (Golden Zoneの画像)
                        
                        💡 **直接解析は実測値、AI予測は推定値です**
                        """)
                        
                        # 結果テーブル (処理方法付き)
                        import pandas as pd
                        
                        def get_method_icon(method):
                            return "📐 直接解析" if method == 'direct_analysis' else "🔮 AI予測"
                        
                        df = pd.DataFrame({
                            "No.": range(1, len(results) + 1),
                            "画像名": [r['filename'] for r in results],
                            "処理方法": [get_method_icon(r['method']) for r in results],
                            "フラクタル次元": [f"{r['predicted_fd']:.4f}" for r in results],
                            "信頼度": [f"{r['confidence']['overall_confidence']:.1f}%" for r in results],
                            "信頼度レベル": [f"{r['confidence']['level_emoji']} {r['confidence']['confidence_level']}" for r in results],
                            "推定範囲": [f"{r['confidence']['lower_bound']:.4f} - {r['confidence']['upper_bound']:.4f}" for r in results]
                        })
                        
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # 処理方法別の詳細表示
                        if len(direct_analysis_results) > 0:
                            with st.expander("📐 直接解析の詳細", expanded=False):
                                st.markdown("""
                                **直接解析した画像:**
                                - 高品質画像（high）または品質過剰画像（low1-3）
                                - ボックスカウンティング法による実測値
                                - 信頼度: 95%以上（実測のため高精度）
                                """)
                                
                                direct_df = pd.DataFrame({
                                    "画像名": [r['filename'] for r in direct_analysis_results],
                                    "品質レベル": [r['quality_level'] for r in direct_analysis_results],
                                    "フラクタル次元": [f"{r['predicted_fd']:.4f}" for r in direct_analysis_results],
                                })
                                st.dataframe(direct_df, use_container_width=True, hide_index=True)
                        
                        if len(ai_prediction_results) > 0:
                            with st.expander("🔮 AI予測の詳細", expanded=False):
                                st.markdown("""
                                **AI予測した画像:**
                                - Golden Zone画像（low4-7）または低品質画像（low8-10）
                                - ニューラルネットワークによる推定値
                                - 信頼度: 画像品質とモデルの確信度に依存
                                """)
                                
                                ai_df = pd.DataFrame({
                                    "画像名": [r['filename'] for r in ai_prediction_results],
                                    "品質レベル": [r['quality_level'] for r in ai_prediction_results],
                                    "予測FD": [f"{r['predicted_fd']:.4f}" for r in ai_prediction_results],
                                    "信頼度": [f"{r['confidence']['overall_confidence']:.1f}%" for r in ai_prediction_results],
                                })
                                st.dataframe(ai_df, use_container_width=True, hide_index=True)
                        
                        # 🆕 肌品質評価の追加
                        if SKIN_EVALUATOR_AVAILABLE:
                            st.markdown("---")
                            st.subheader("🌟 肌品質評価（フラクタル次元ベース）")
                            
                            st.markdown("""
                            **フラクタル次元から肌の滑らかさを評価します:**
                            
                            【参考文献】中川匡弘「肌のフラクタル構造解析」光学 39巻11号 (2010)
                            
                            > 💡 **重要:** 滑らかな肌ほどフラクタル構造が複雑化し、FD値が3に近くなります。
                            
                            **評価基準:**
                            - **S (Superior)**: FD ≥ 2.90 - 非常に滑らか（フラクタル構造が非常に複雑）
                            - **A (Excellent)**: 2.80 ≤ FD < 2.90 - 滑らか（構造が複雑）
                            - **B (Good)**: 2.50 ≤ FD < 2.80 - 普通（構造は中程度）
                            - **C (Fair)**: 2.40 ≤ FD < 2.50 - やや粗い（構造がやや単純）
                            - **D (Poor)**: FD < 2.40 - 粗い（構造が単純）
                            """)
                            
                            evaluator = SkinQualityEvaluator()
                            skin_evaluation_data = []
                            
                            for r in results:
                                fd = r['predicted_fd']
                                grade = evaluator.get_grade(fd)
                                grade_info = evaluator.grade_criteria[grade]
                                
                                skin_evaluation_data.append({
                                    "画像名": r['filename'],
                                    "予測FD": f"{fd:.4f}",
                                    "評価": f"{grade_info['icon']} {grade}",
                                    "肌状態": grade_info['description'],
                                    "解釈": grade_info['interpretation'][:30] + "..."  # 短縮版
                                })
                            
                            skin_df = pd.DataFrame(skin_evaluation_data)
                            st.dataframe(skin_df, use_container_width=True, hide_index=True)
                            
                            # 評価グレード分布
                            grade_counts = {}
                            for r in results:
                                grade = evaluator.get_grade(r['predicted_fd'])
                                grade_counts[grade] = grade_counts.get(grade, 0) + 1
                            
                            st.markdown("### 📊 肌品質グレード分布")
                            # レイアウトモードに応じて表示変更
                            grade_list = ['S', 'A', 'B', 'C', 'D']
                            if is_mobile:
                                # モバイル版: 2列グリッド
                                for i in range(0, len(grade_list), 2):
                                    cols = st.columns(2)
                                    for j, col in enumerate(cols):
                                        if i + j < len(grade_list):
                                            grade = grade_list[i + j]
                                            count = grade_counts.get(grade, 0)
                                            icon = evaluator.grade_criteria[grade]['icon']
                                            grade_info = evaluator.grade_criteria[grade]
                                            with col:
                                                st.metric(f"{icon} グレード{grade}", f"{count}枚", delta=grade_info['description'])
                            else:
                                # デスクトップ版: 1×5横並び
                                cols = st.columns(5)
                                for idx, grade in enumerate(grade_list):
                                    count = grade_counts.get(grade, 0)
                                    icon = evaluator.grade_criteria[grade]['icon']
                                    grade_info = evaluator.grade_criteria[grade]
                                    with cols[idx]:
                                        st.metric(f"{icon} {grade}", f"{count}枚", delta=grade_info['description'])
                            
                            # 詳細な推奨ケア
                            with st.expander("💡 推奨ケアの詳細"):
                                for r in results:
                                    fd = r['predicted_fd']
                                    grade = evaluator.get_grade(fd)
                                    grade_info = evaluator.grade_criteria[grade]
                                    
                                    st.markdown(f"---")
                                    st.markdown(f"### {r['filename']}")
                                    st.markdown(f"**グレード:** {grade_info['icon']} {grade} - {grade_info['description']}")
                                    st.markdown(f"**FD値:** {fd:.4f}")
                                    st.markdown(f"**解釈:** {grade_info['interpretation']}")
                                    st.markdown(f"**推奨ケア:** {grade_info['recommendation']}")
                        
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
                                
                                # 処理方法によって表示内容を変更
                                if result['method'] == 'ai_prediction':
                                    # AI予測の場合: 3つの指標を表示
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
                                else:
                                    # 直接解析の場合: 信頼度のみ表示
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            "信頼度",
                                            f"{conf['overall_confidence']:.1f}%",
                                            delta=None
                                        )
                                    with col2:
                                        st.metric(
                                            "信頼度レベル",
                                            f"{conf['confidence_level']}"
                                        )
                                    
                                    st.write(f"""
                                    **推定範囲:** {conf['lower_bound']:.4f} - {conf['upper_bound']:.4f}
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
                        
                        # ============================================================
                        # 🌸 肌品質評価セクション
                        # ============================================================
                        if SKIN_EVALUATOR_AVAILABLE:
                            st.markdown("---")
                            st.subheader("🌸 肌品質評価 (フラクタル次元分析)")
                            
                            st.info("""
                            💡 **フラクタル次元と肌の関係**
                            
                            - **低いFD値 (2.0-2.4)**: きめ細かく、スムーズな肌
                            - **中程度のFD値 (2.4-2.6)**: 普通の肌質
                            - **高いFD値 (2.6-3.0)**: 粗い肌、毛穴やシワが目立つ
                            
                            このAIは低画質画像から高画質相当のFD値を予測し、正確な肌評価を可能にします。
                            """)
                            
                            # 評価モード選択
                            eval_mode = st.radio(
                                "評価モード",
                                ["総合評価", "個別評価", "年齢層比較"],
                                horizontal=True,
                                key="skin_evaluation_mode"
                            )
                            
                            evaluator = SkinQualityEvaluator()
                            
                            if eval_mode == "総合評価":
                                # 全画像の総合評価
                                fd_values = [r['predicted_fd'] for r in results]
                                labels = [r['filename'] for r in results]
                                
                                multi_eval = evaluator.evaluate_multiple(fd_values, labels)
                                
                                if multi_eval:
                                    st.markdown("### 📊 総合評価結果")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "総合評価",
                                            f"{multi_eval['overall']['grade_emoji']} {multi_eval['overall']['grade']}",
                                            delta=f"スコア: {multi_eval['overall']['score']:.1f}/100"
                                        )
                                    with col2:
                                        st.metric(
                                            "平均FD値",
                                            f"{multi_eval['statistics']['mean']:.4f}",
                                            delta=f"標準偏差: {multi_eval['statistics']['std']:.4f}"
                                        )
                                    with col3:
                                        st.metric(
                                            "一貫性",
                                            multi_eval['consistency']['level'],
                                            delta=multi_eval['consistency']['message']
                                        )
                                
                                # 解釈とアドバイス
                                st.markdown("#### 💭 解釈")
                                st.info(multi_eval['overall']['interpretation'])
                                
                                st.markdown("#### 📝 改善提案")
                                for rec in multi_eval['overall']['recommendations']:
                                    st.write(rec)
                                
                                # 統計情報
                                with st.expander("📈 詳細統計"):
                                    stats = multi_eval['statistics']
                                    st.write(f"**最小値:** {stats['min']:.4f}")
                                    st.write(f"**最大値:** {stats['max']:.4f}")
                                    st.write(f"**中央値:** {stats['median']:.4f}")
                                    st.write(f"**範囲:** {stats['range']:.4f}")
                            
                            elif eval_mode == "個別評価":
                                # 個別画像の詳細評価
                                st.markdown("### 📋 個別画像評価")
                                
                                selected_idx = st.selectbox(
                                    "評価する画像を選択",
                                    range(len(results)),
                                    format_func=lambda i: results[i]['filename'],
                                    key="selected_image_idx"
                                )
                                
                                if selected_idx is not None:
                                    result = results[selected_idx]
                                    fd_value = result['predicted_fd']
                                    
                                    single_eval = evaluator.evaluate_single(fd_value)
                                    
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        st.image(
                                            cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB),
                                            caption=result['filename'],
                                            use_container_width=True
                                        )
                                    
                                    with col2:
                                        st.markdown(f"### {single_eval['grade_emoji']} {single_eval['grade']}")
                                        st.metric("スコア", f"{single_eval['score']:.1f}/100")
                                        st.metric("FD値", f"{fd_value:.4f}")
                                        
                                        st.markdown("#### 特徴分析")
                                        features = single_eval['features']
                                        st.write(f"- **スムーズさ:** {features['smoothness']}")
                                        st.write(f"- **きめ細かさ:** {features['texture']}")
                                        st.write(f"- **複雑度:** {features['complexity']}")
                                    
                                    # 最小二乗法のグラフを表示するオプション
                                    show_fitting_graph = st.checkbox(
                                        "🔬 最小二乗法フィッティンググラフを表示", 
                                        value=False,
                                        key="show_least_squares_graph"
                                    )
                                    
                                    if show_fitting_graph:
                                        # デバッグ情報
                                        if 'fitting_data' not in result:
                                            st.warning("⚠️ この画像はAI予測で処理されたため、最小二乗法のデータがありません。")
                                            st.info("💡 直接解析（高品質画像またはlow1-3画像）のみグラフを表示できます。")
                                        elif result.get('fitting_data') is None:
                                            st.warning("⚠️ フィッティングデータが取得できませんでした。")
                                        else:
                                            fitting_data = result['fitting_data']
                                            if fitting_data.get('log_h') is None:
                                                st.warning("⚠️ 計算データが不完全です。")
                                            else:
                                                st.markdown("#### 📊 最小二乗法フィッティング解析")
                                                try:
                                                    fig = plot_least_squares_fit(
                                                        fitting_data['log_h'],
                                                        fitting_data['log_Nh'],
                                                        fitting_data['coeffs'],
                                                        fd_value
                                                    )
                                                    st.pyplot(fig)
                                                    plt.close(fig)
                                                    
                                                    st.caption("""
                                                    **グラフの見方:**
                                                    - 青い点: 実際の測定データ (log(スケール) vs log(ボックス数))
                                                    - 赤い線: 最小二乗法によるフィッティング直線
                                                    - 傾きの絶対値がフラクタル次元(FD)値になります
                                                    - R²値が1に近いほど、フィッティングの精度が高いことを示します
                                                    """)
                                                except Exception as e:
                                                    st.error(f"❌ グラフ描画エラー: {str(e)}")
                                    
                                    st.markdown("#### 💭 解釈")
                                    st.info(single_eval['interpretation'])
                                    
                                    st.markdown("#### 📝 改善提案")
                                    for rec in single_eval['recommendations']:
                                        st.write(rec)
                            
                            else:  # 年齢層比較
                                st.markdown("### 👥 年齢層との比較")
                                
                                age_group = st.selectbox(
                                    "あなたの年齢層を選択",
                                    ['10-20', '20-30', '30-40', '40-50', '50+'],
                                    format_func=lambda x: f"{x}代" if x != '50+' else '50代以上',
                                    key="age_group_selection"
                                )
                                
                                # 平均FD値を使用
                                fd_values = [r['predicted_fd'] for r in results]
                                avg_fd = np.mean(fd_values)
                                
                                comparison = evaluator.compare_with_age_group(avg_fd, age_group)
                                
                                if 'error' not in comparison:
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            "あなたのFD値",
                                            f"{comparison['your_value']:.4f}"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "年齢層平均",
                                            f"{comparison['age_average']:.4f}",
                                            delta=f"差: {comparison['difference']:+.4f}"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "パーセンタイル",
                                            f"{comparison['percentile']:.0f}%"
                                        )
                                    
                                    st.markdown("#### 💭 比較結果")
                                    st.info(comparison['interpretation'])
                                    
                                    # Z-scoreの解説
                                    with st.expander("📊 統計的解釈"):
                                        st.write(f"**Z-スコア:** {comparison['z_score']:.2f}")
                                        st.write("Z-スコアの意味:")
                                        st.write("- 0付近: 平均的")
                                        st.write("- -1～-2: 平均より良好")
                                        st.write("- -2以下: 非常に良好")
                                        st.write("- +1～+2: 平均より高め")
                                        st.write("- +2以上: 要改善")
            
            # ========================================
            # 🎯 精度検証モード
            # ========================================
            else:  # inference_submode == "🎯 精度検証モード (高画質ペアで検証)"
                st.subheader("📤 画像ペアを準備")
                
                st.success("🤖 モデルが読み込まれています。検証の準備完了!")
                
                # 読み込みモード選択
                validation_mode = st.radio(
                    "画像読み込みモード",
                    ["📁 フォルダから自動ペアリング", "📤 手動アップロード"],
                    help="フォルダモード: 同じフォルダ内の画像を自動でペアリング\n手動モード: 個別にアップロード"
                )
                
                if validation_mode == "📁 フォルダから自動ペアリング":
                    st.markdown("""
                    **フォルダパスを入力すると、自動的にペアを検出します。**
                    - 高画質画像: `IMG_XXXX.jpg`
                    - 低画質画像: `IMG_XXXX_low1.jpg` (low1-10に対応)
                    """)
                    
                    folder_path = st.text_input(
                        "📁 画像フォルダのパス",
                        value=r"E:\画質別頬画像(元画像＋10段階)",
                        help="高画質と低画質の画像が入ったフォルダパスを指定してください"
                    )
                    
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
                        quality_level_val = st.selectbox(
                            "低画質レベル",
                            ["low1", "low2", "low3", "low4", "low5", "low6", "low7", "low8", "low9", "low10", "自動検出"],
                            index=3,  # デフォルトでlow4
                            help="検証に使用する低画質レベル"
                        )
                    
                    if folder_path and os.path.exists(folder_path):
                        # フォルダから画像ペアを自動検出
                        all_files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
                        
                        # 高画質画像を検出(_lowがついていないもの)
                        high_files = [f for f in all_files if not re.search(r'_low\d+', os.path.basename(f))]
                        
                        if len(high_files) > 0:
                            st.info(f"📂 検出された高画質画像: {len(high_files)}枚")
                            
                            # 対応する低画質画像を検索してペア作成
                            pairs = []
                            
                            if quality_level_val == "自動検出":
                                # すべてのlow1-10を検出
                                for hf in high_files:
                                    base_name = os.path.splitext(os.path.basename(hf))[0]
                                    ext = os.path.splitext(os.path.basename(hf))[1]
                                    
                                    for lv in range(1, 11):
                                        low_file = os.path.join(folder_path, f"{base_name}_low{lv}{ext}")
                                        if os.path.exists(low_file):
                                            pairs.append({
                                                'base_name': base_name,
                                                'high_file': hf,
                                                'low_file': low_file,
                                                'quality_level': f"low{lv}"
                                            })
                            else:
                                # 指定された品質レベルのみ
                                for hf in high_files:
                                    base_name = os.path.splitext(os.path.basename(hf))[0]
                                    ext = os.path.splitext(os.path.basename(hf))[1]
                                    low_file = os.path.join(folder_path, f"{base_name}_{quality_level_val}{ext}")
                                    
                                    if os.path.exists(low_file):
                                        pairs.append({
                                            'base_name': base_name,
                                            'high_file': hf,
                                            'low_file': low_file,
                                            'quality_level': quality_level_val
                                        })
                            
                            if len(pairs) > 0:
                                st.success(f"✅ {len(pairs)}組のペアが見つかりました")
                                
                                # ペア一覧を表示
                                with st.expander("📋 検出されたペア一覧"):
                                    import pandas as pd
                                    pair_df = pd.DataFrame({
                                        "No.": range(1, len(pairs) + 1),
                                        "ベース名": [p['base_name'] for p in pairs],
                                        "品質レベル": [p['quality_level'] for p in pairs],
                                        "高画質画像": [os.path.basename(p['high_file']) for p in pairs],
                                        "低画質画像": [os.path.basename(p['low_file']) for p in pairs]
                                    })
                                    st.dataframe(pair_df, use_container_width=True, hide_index=True)
                                
                                # 検証実行に進む（後で実装）
                                validation_pairs = pairs
                                
                            else:
                                st.warning(f"⚠️ ペアが見つかりませんでした。品質レベル「{quality_level_val}」の画像があるか確認してください。")
                                validation_pairs = None
                        else:
                            st.warning("⚠️ 高画質画像が見つかりませんでした。フォルダパスとファイルパターンを確認してください。")
                            validation_pairs = None
                    elif folder_path:
                        st.error(f"❌ フォルダが見つかりません: {folder_path}")
                        validation_pairs = None
                    else:
                        validation_pairs = None
                
                else:  # 手動アップロード
                    st.markdown("""
                    **高画質画像と低画質画像をペアでアップロードしてください。**
                    - ファイル名が一致するものを自動でペアリングします
                    - 例: `IMG_001.jpg` (高画質) と `IMG_001_low4.jpg` (低画質)
                    """)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        high_quality_imgs = st.file_uploader(
                            "高画質画像",
                            type=['png', 'jpg', 'jpeg'],
                            accept_multiple_files=True,
                            help="正解となる高画質の肌画像",
                            key="validation_high_quality_uploader"
                        )
                    
                    with col2:
                        low_quality_imgs_val = st.file_uploader(
                            "低画質画像",
                            type=['png', 'jpg', 'jpeg'],
                            accept_multiple_files=True,
                            help="AIで予測する低画質の肌画像",
                            key="validation_low_quality_uploader"
                        )
                    
                    if high_quality_imgs and low_quality_imgs_val:
                        st.success(f"✅ 高画質: {len(high_quality_imgs)}枚, 低画質: {len(low_quality_imgs_val)}枚")
                        
                        # 🆕 画像品質自動判定（学習モード用）
                        if IMAGE_QUALITY_ASSESSOR_AVAILABLE:
                            with st.expander("🔍 画像品質チェック（高画質画像）", expanded=False):
                                st.markdown("""
                                **高画質画像の品質を確認します**
                                - ✅ 推奨: 解像度・鮮明度が十分
                                - ⚠️ 注意: 品質が基準を満たさない可能性
                                """)
                                
                                high_quality_results = []
                                for img_file in high_quality_imgs:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file.name)[1]) as tmp:
                                        tmp.write(img_file.getvalue())
                                        tmp_path = tmp.name
                                    
                                    quality_result = assess_image_quality(tmp_path)
                                    high_quality_results.append({
                                        'filename': img_file.name,
                                        'result': quality_result
                                    })
                                    os.unlink(tmp_path)
                                
                                # 簡易サマリー
                                high_count = sum(1 for r in high_quality_results if r['result'].get('quality_level') == 'high')
                                low47_count = sum(1 for r in high_quality_results if r['result'].get('quality_level') == 'low4-7')
                                low_count = len(high_quality_results) - high_count - low47_count
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("高品質", f"{high_count}/{len(high_quality_results)}")
                                with col2:
                                    st.metric("中品質", low47_count)
                                with col3:
                                    st.metric("低品質", low_count)
                                
                                if low_count > 0:
                                    st.warning(f"⚠️ {low_count}枚の高画質画像が基準を満たしていません。学習精度が低下する可能性があります。")
                        
                        # ペアリング処理
                        st.subheader("🔗 ペアリング")
                        
                        def extract_base_name(filename):
                            """ファイル名からベース名を抽出 (low1-10のサフィックスを除去)"""
                            # 拡張子を除去
                            name_without_ext = os.path.splitext(filename)[0]
                            # _low1 から _low10 を除去
                            base_name = re.sub(r'_low\d+$', '', name_without_ext)
                            return base_name
                        
                        # ペア作成
                        high_dict = {extract_base_name(f.name): f for f in high_quality_imgs}
                        low_dict = {}
                        
                        for f in low_quality_imgs_val:
                            base = extract_base_name(f.name)
                            if base not in low_dict:
                                low_dict[base] = []
                            low_dict[base].append(f)
                        
                        # マッチング
                        validation_pairs = []
                        for base_name in high_dict.keys():
                            if base_name in low_dict:
                                for low_file in low_dict[base_name]:
                                    validation_pairs.append({
                                        'base_name': base_name,
                                        'high_file': high_dict[base_name],
                                        'low_file': low_file
                                    })
                        
                        if validation_pairs:
                            st.success(f"✅ {len(validation_pairs)}組のペアが見つかりました")
                            
                            # ペア一覧を表示
                            with st.expander("📋 検出されたペア一覧"):
                                import pandas as pd
                                # 手動アップロードの場合はfile objectなので.nameを使用
                                pair_df = pd.DataFrame({
                                    "No.": range(1, len(validation_pairs) + 1),
                                    "ベース名": [p['base_name'] for p in validation_pairs],
                                    "高画質画像": [p['high_file'].name for p in validation_pairs],
                                    "低画質画像": [p['low_file'].name for p in validation_pairs]
                                })
                                st.dataframe(pair_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("⚠️ ペアが見つかりませんでした。ファイル名を確認してください。")
                            validation_pairs = None
                    else:
                        validation_pairs = None
                
                # ========================================
                # 検証実行（フォルダモード・手動アップロード共通）
                # ========================================
                if validation_pairs and len(validation_pairs) > 0:
                    if st.button("🎯 精度検証を実行", type="primary"):
                        st.info("検証を開始します...")
                        
                        validation_results = []
                        progress_bar = st.progress(0)
                        
                        # デバッグ情報表示
                        debug_info = st.empty()
                        
                        for idx, pair in enumerate(validation_pairs):
                            debug_info.info(f"処理中: {idx+1}/{len(validation_pairs)} - {pair['base_name']}")
                            
                            # 画像読み込み（フォルダモード=パス、手動アップロード=file buffer）
                            if isinstance(pair['high_file'], str):  # フォルダモード
                                high_img = read_bgr_from_path(pair['high_file'])
                                low_img = read_bgr_from_path(pair['low_file'])
                                high_filename = os.path.basename(pair['high_file'])
                                low_filename = os.path.basename(pair['low_file'])
                            else:  # 手動アップロード
                                high_img = read_bgr_from_buffer(pair['high_file'].read())
                                low_img = read_bgr_from_buffer(pair['low_file'].read())
                                high_filename = pair['high_file'].name
                                low_filename = pair['low_file'].name
                            
                            if high_img is not None and low_img is not None:
                                # 高画質画像の実際のFD計算（学習時と同じ方法を使用）
                                actual_fd, _, _ = fast_fractal_std_boxcount_batched(high_img, use_gpu=False)
                                if actual_fd is None:
                                    # フォールバック: naiveメソッドを試す
                                    actual_fd, _, _ = fractal_dimension_naive(high_img)
                                
                                debug_info.info(f"""
                                {pair['base_name']}:
                                - 高画質画像サイズ: {high_img.shape}
                                - 実測FD: {actual_fd}
                                """)
                                
                                if actual_fd is None:
                                    st.warning(f"⚠️ {pair['base_name']}: FD計算失敗（スキップ）")
                                    continue
                                
                                # 低画質画像からの予測FD
                                predicted_fd = predict_fd_from_low_quality(low_img, model)
                                
                                debug_info.info(f"""
                                {pair['base_name']}:
                                - 低画質画像サイズ: {low_img.shape}
                                - 予測FD: {predicted_fd}
                                """)
                                
                                # 誤差計算
                                error = predicted_fd - actual_fd
                                abs_error = abs(error)
                                relative_error = (abs_error / actual_fd) * 100 if actual_fd != 0 else 0
                                
                                validation_results.append({
                                    'base_name': pair['base_name'],
                                    'high_filename': high_filename,
                                    'low_filename': low_filename,
                                    'actual_fd': actual_fd,
                                    'predicted_fd': predicted_fd,
                                    'error': error,
                                    'abs_error': abs_error,
                                    'relative_error': relative_error,
                                    'high_img': high_img,
                                    'low_img': low_img
                                })
                            else:
                                st.warning(f"⚠️ {pair['base_name']}: 画像読み込み失敗")
                            
                            progress_bar.progress((idx + 1) / len(validation_pairs))
                        
                        debug_info.empty()  # デバッグ情報をクリア
                        
                        st.success(f"✅ 検証完了! {len(validation_results)}件のデータを取得")
                        
                        # データ数チェック
                        if len(validation_results) == 0:
                            st.error("❌ 有効なデータが取得できませんでした。画像ファイルを確認してください。")
                        elif len(validation_results) == 1:
                            st.warning("⚠️ データが1件のみです。統計分析には最低3件以上を推奨します。")
                        
                        # ========================================
                        # 📊 精度検証結果の表示
                        # ========================================
                        st.subheader("📊 精度検証結果")
                        
                        if validation_results:
                            # 統計指標の計算
                            actual_fds = np.array([r['actual_fd'] for r in validation_results])
                            predicted_fds = np.array([r['predicted_fd'] for r in validation_results])
                            errors = np.array([r['error'] for r in validation_results])
                            abs_errors = np.array([r['abs_error'] for r in validation_results])
                            
                            # データ数と変動チェック
                            n_samples = len(validation_results)
                            actual_std = np.std(actual_fds)
                            predicted_std = np.std(predicted_fds)
                            
                            st.info(f"""
                            **検証データ情報:**
                            - データ数: {n_samples}件
                            - 実測FDの変動: {actual_std:.4f}
                            - 予測FDの変動: {predicted_std:.4f}
                            """)
                            
                            # 相関係数（変動がない場合はnanになる）
                            if actual_std > 0 and predicted_std > 0:
                                correlation = np.corrcoef(actual_fds, predicted_fds)[0, 1]
                            else:
                                correlation = np.nan
                                st.warning("⚠️ データの変動がないため、相関係数を計算できません。異なる画像を使用してください。")
                            
                            # MAE (平均絶対誤差)
                            mae = np.mean(abs_errors)
                            
                            # RMSE (二乗平均平方根誤差)
                            rmse = np.sqrt(np.mean(errors ** 2))
                            
                            # R² (決定係数)
                            ss_res = np.sum(errors ** 2)
                            ss_tot = np.sum((actual_fds - np.mean(actual_fds)) ** 2)
                            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                            
                            # 総合評価
                            st.markdown("### 🎯 総合評価")
                            
                            # レイアウトモードに応じて列数変更
                            if is_mobile:
                                # モバイル版: 2×2グリッド
                                col1, col2 = st.columns(2)
                            else:
                                # デスクトップ版: 1×4横並び
                                col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                # 相関係数の評価
                                if np.isnan(correlation):
                                    corr_grade = "N/A"
                                    corr_emoji = "💫"
                                    corr_display = "nan"
                                elif correlation >= 0.95:
                                    corr_grade = "S"
                                    corr_emoji = "🌟"
                                    corr_display = f"{correlation:.4f}"
                                elif correlation >= 0.90:
                                    corr_grade = "A"
                                    corr_emoji = "⭐"
                                    corr_display = f"{correlation:.4f}"
                                elif correlation >= 0.85:
                                    corr_grade = "B"
                                    corr_emoji = "✨"
                                    corr_display = f"{correlation:.4f}"
                                else:
                                    corr_grade = "C"
                                    corr_emoji = "💫"
                                    corr_display = f"{correlation:.4f}"
                                
                                st.metric(
                                    "相関係数",
                                    corr_display,
                                    delta=f"{corr_grade}評価 {corr_emoji}"
                                )
                            
                            with col2:
                                # MAEの評価
                                if mae <= 0.03:
                                    mae_grade = "S"
                                    mae_emoji = "🌟"
                                elif mae <= 0.05:
                                    mae_grade = "A"
                                    mae_emoji = "⭐"
                                elif mae <= 0.08:
                                    mae_grade = "B"
                                    mae_emoji = "✨"
                                else:
                                    mae_grade = "C"
                                    mae_emoji = "💫"
                                
                                st.metric(
                                    "MAE (平均絶対誤差)",
                                    f"{mae:.4f}",
                                    delta=f"{mae_grade}評価 {mae_emoji}"
                                )
                            
                            # モバイル版の場合は2行目を作成
                            if is_mobile:
                                col3, col4 = st.columns(2)
                            
                            with col3:
                                st.metric(
                                    "RMSE",
                                    f"{rmse:.4f}",
                                    delta=f"±{rmse:.4f}"
                                )
                            
                            with col4:
                                # R²の評価
                                if r2 >= 0.90:
                                    r2_emoji = "🌟"
                                elif r2 >= 0.80:
                                    r2_emoji = "⭐"
                                elif r2 >= 0.70:
                                    r2_emoji = "✨"
                                else:
                                    r2_emoji = "💫"
                                
                                st.metric(
                                    "R² (決定係数)",
                                    f"{r2:.4f}",
                                    delta=r2_emoji
                                )
                            
                            # 詳細統計
                            st.markdown("### 📈 詳細統計")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.info(f"""
                                **実測値 (高画質FD) の統計:**
                                - 平均: {np.mean(actual_fds):.4f}
                                - 標準偏差: {np.std(actual_fds):.4f}
                                - 最小値: {np.min(actual_fds):.4f}
                                - 最大値: {np.max(actual_fds):.4f}
                                - 範囲: {np.max(actual_fds) - np.min(actual_fds):.4f}
                                """)
                            
                            with col2:
                                st.info(f"""
                                **予測値 (AI予測FD) の統計:**
                                - 平均: {np.mean(predicted_fds):.4f}
                                - 標準偏差: {np.std(predicted_fds):.4f}
                                - 最小値: {np.min(predicted_fds):.4f}
                                - 最大値: {np.max(predicted_fds):.4f}
                                - 範囲: {np.max(predicted_fds) - np.min(predicted_fds):.4f}
                                """)
                            
                            st.warning(f"""
                            **誤差の統計:**
                            - 平均誤差 (Bias): {np.mean(errors):.4f}
                            - MAE: {mae:.4f}
                            - RMSE: {rmse:.4f}
                            - 最大誤差: {np.max(abs_errors):.4f}
                            - 誤差の標準偏差: {np.std(errors):.4f}
                            """)
                            
                            # 可視化
                            st.markdown("### 📊 可視化")
                            
                            import matplotlib.pyplot as plt
                            
                            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                            
                            # 1. 散布図 (予測 vs 実測)
                            axes[0].scatter(actual_fds, predicted_fds, alpha=0.6, s=100)
                            axes[0].plot([actual_fds.min(), actual_fds.max()], 
                                        [actual_fds.min(), actual_fds.max()], 
                                        'r--', lw=2, label='理想線 (y=x)')
                            axes[0].set_xlabel('実測FD (高画質)', fontsize=12)
                            axes[0].set_ylabel('予測FD (AI)', fontsize=12)
                            axes[0].set_title(f'予測 vs 実測\n相関係数: {correlation:.4f}', fontsize=14)
                            axes[0].legend()
                            axes[0].grid(True, alpha=0.3)
                            
                            # 2. 誤差分布 (ヒストグラム)
                            axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
                            axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='誤差ゼロ')
                            axes[1].axvline(np.mean(errors), color='green', linestyle='--', 
                                          linewidth=2, label=f'平均誤差: {np.mean(errors):.4f}')
                            axes[1].set_xlabel('誤差 (予測 - 実測)', fontsize=12)
                            axes[1].set_ylabel('頻度', fontsize=12)
                            axes[1].set_title(f'誤差分布\nMAE: {mae:.4f}, RMSE: {rmse:.4f}', fontsize=14)
                            axes[1].legend()
                            axes[1].grid(True, alpha=0.3)
                            
                            # 3. 絶対誤差の分布
                            axes[2].hist(abs_errors, bins=20, edgecolor='black', alpha=0.7, color='orange')
                            axes[2].axvline(mae, color='red', linestyle='--', 
                                          linewidth=2, label=f'MAE: {mae:.4f}')
                            axes[2].set_xlabel('絶対誤差 |予測 - 実測|', fontsize=12)
                            axes[2].set_ylabel('頻度', fontsize=12)
                            axes[2].set_title('絶対誤差分布', fontsize=14)
                            axes[2].legend()
                            axes[2].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # 結果テーブル
                            st.markdown("### 📋 詳細結果テーブル")
                            
                            import pandas as pd
                            result_df = pd.DataFrame({
                                "No.": range(1, len(validation_results) + 1),
                                "ベース名": [r['base_name'] for r in validation_results],
                                "実測FD": [f"{r['actual_fd']:.4f}" for r in validation_results],
                                "予測FD": [f"{r['predicted_fd']:.4f}" for r in validation_results],
                                "誤差": [f"{r['error']:+.4f}" for r in validation_results],
                                "絶対誤差": [f"{r['abs_error']:.4f}" for r in validation_results],
                                "相対誤差%": [f"{r['relative_error']:.2f}%" for r in validation_results]
                            })
                            
                            st.dataframe(result_df, use_container_width=True, hide_index=True)
                            
                            # CSV ダウンロード
                            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 結果をCSVでダウンロード",
                                data=csv,
                                file_name=f"validation_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # 画像ごとの詳細確認
                            st.markdown("### 🖼️ 画像ごとの詳細確認")
                            
                            with st.expander("画像と誤差を表示"):
                                for i, result in enumerate(validation_results):
                                    st.markdown(f"#### {i+1}. {result['base_name']}")
                                    
                                    col1, col2, col3 = st.columns([1, 1, 1])
                                    
                                    with col1:
                                        st.image(cv2.cvtColor(result['high_img'], cv2.COLOR_BGR2RGB), 
                                                caption=f"高画質: {result['high_filename']}", 
                                                use_container_width=True)
                                        st.write(f"**実測FD:** {result['actual_fd']:.4f}")
                                    
                                    with col2:
                                        st.image(cv2.cvtColor(result['low_img'], cv2.COLOR_BGR2RGB), 
                                                caption=f"低画質: {result['low_filename']}", 
                                                use_container_width=True)
                                        st.write(f"**予測FD:** {result['predicted_fd']:.4f}")
                                    
                                    with col3:
                                        # 誤差評価
                                        if result['abs_error'] <= 0.02:
                                            error_level = "🌟 優秀"
                                        elif result['abs_error'] <= 0.05:
                                            error_level = "⭐ 良好"
                                        elif result['abs_error'] <= 0.08:
                                            error_level = "✨ 許容"
                                        else:
                                            error_level = "💫 要改善"
                                        
                                        st.metric("誤差", f"{result['error']:+.4f}", delta=error_level)
                                        st.write(f"**絶対誤差:** {result['abs_error']:.4f}")
                                        st.write(f"**相対誤差:** {result['relative_error']:.2f}%")
                                    
                                    st.markdown("---")
                            
                            # 評価コメント
                            st.markdown("### 💬 評価コメント")
                            
                            if correlation >= 0.95 and mae <= 0.03:
                                st.success("""
                                🌟 **優秀な精度です！**
                                - 予測と実測の相関が非常に高く、MAEも小さいです
                                - このモデルは実用レベルで使用できます
                                - 肌品質評価に十分な精度を持っています
                                """)
                            elif correlation >= 0.90 and mae <= 0.05:
                                st.info("""
                                ⭐ **良好な精度です！**
                                - 予測精度は実用的なレベルです
                                - より多くのデータやデータ拡張で改善の余地があります
                                """)
                            elif correlation >= 0.85 and mae <= 0.08:
                                st.warning("""
                                ✨ **許容範囲の精度です**
                                - 基本的な予測は可能ですが、改善の余地があります
                                - データ数増加、データ拡張、品質レベルの見直しを検討してください
                                """)
                            else:
                                st.error("""
                                💫 **精度改善が必要です**
                                - 予測精度が低い可能性があります
                                - 以下を確認してください:
                                  1. 学習データ数は十分か (最低100組以上推奨)
                                  2. データ拡張は適切か (15種類以上推奨)
                                  3. 品質レベルの選択は適切か (low4-7推奨)
                                  4. 画像の品質は適切か
                                """)
                        
                        else:
                            st.error("検証結果が取得できませんでした")
                    
                    else:
                        st.warning("⚠️ ペアが見つかりませんでした。ファイル名を確認してください。")
                        st.info("""
                        **ペアリングの条件:**
                        - 高画質: `IMG_001.jpg`
                        - 低画質: `IMG_001_low4.jpg` (ベース名が同じで、_low{数字}がつく)
                        """)
        
        return  # 推論モードはここで終了
    
    # ============================================================
    # 📊 研究報告・品質ガイドモード
    # ============================================================
    elif app_mode == "📊 研究報告・品質ガイド":
        show_quality_optimization_report()
        return  # 研究報告モードはここで終了

    # ============================================================
    # 学習モード (既存のコード)
    # ============================================================
    st.header("🎓 学習モード - AIを学習させる")
    
    # 🎯 学習前の診断とアドバイス
    with st.expander("💡 効果的な学習のためのガイド", expanded=False):
        st.markdown("""
        ### 🎯 信頼度10%から75%以上を目指すために
        
        #### ⚠️ 信頼度が低い主な原因
        
        1. **データ拡張が不足** ← 最重要！
           - 元画像30枚だけでは不十分
           - データ拡張で100組以上に増やす必要あり
        
        2. **品質レベルの選択ミス**
           - low1-3: 差が小さすぎて学習困難
           - **推奨: low4-7** (中度劣化、バランス良好)
           - 上級: low8-10 (重度劣化、実用的)
           - **⚠️ 全レベル (low1-10) は非推奨** → 差が大きすぎてMAE悪化
        
        3. **単一レベルの学習**
           - 1つのレベルだけでは汎用性不足
           - **グループ選択で複数レベル同時学習を推奨**
        
        ---
        
        ### ✅ 推奨設定（信頼度75%以上を目指す）
        
        | 項目 | 推奨設定 | 理由 |
        |------|----------|------|
        | **品質レベル** | 🟡 low4-7 (グループ) | バランスが良く、112組のデータ |
        | **データ拡張** | 15種類以上選択 | 最低100組、理想200組以上 |
        | **推奨拡張** | 水平反転、回転、明るさ、ガウシアン | 効果的な変化 |
        
        ---
        
        ### 📊 設定例：信頼度75%達成の実例
        
        **設定:**
        - 元画像: 30組
        - 品質レベル: low4-7 (グループ選択)
        - データ拡張: 20種類
        - 期待データ数: 30 × 4レベル × (1 + 20拡張) = **2,520組**
        
        **結果予測:**
        - 相関係数: 0.85-0.90
        - 誤差(MAE): 0.005-0.010
        - 信頼度: **75-85%** (実用レベル)
        
        ---
        
        ### 🚀 今すぐできる改善策
        
        1. **品質レベル**を「🟡 low4-7 (中度劣化)」に変更
        2. **データ拡張**タブで以下を選択:
           - ✅ 水平反転
           - ✅ 垂直反転  
           - ✅ 90度回転
           - ✅ 180度回転
           - ✅ 明るさ調整（暗く）
           - ✅ 明るさ調整（明るく）
           - ✅ コントラスト調整（低）
           - ✅ コントラスト調整（高）
           - ✅ ガウシアンブラー
           - ✅ ガウシアンノイズ
           - ✅ ソルトペッパーノイズ
           - ✅ JPEG圧縮（品質50）
           - ✅ シャープネス強調
           - ✅ エッジ強調
           - ✅ ヒストグラム平坦化
        
        3. **「すべて選択」ボタン**をクリック（28種類すべて選択）
        
        **期待される結果:**
        - データ数: 30 × 4 × 29 = **3,480組**
        - 信頼度: **80-90%** (プロ～マスターレベル)
        """)
        
        st.success("""
        ### ✨ 簡単な手順
        
        1. 下の「品質レベルグループ」で **「📘 グループ選択」** を選ぶ
        2. **「🟡 low4-7 (中度劣化 - 普通)」** を選択
        3. 「データ拡張」タブで **「すべて選択」** をクリック
        4. **「🚀 学習開始」** ボタンをクリック
        5. 10-15分待つ（GPU使用時は5-8分）
        6. 信頼度が **75%以上** になることを確認！
        """)
    
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
            value=r"E:\画質別頬画像(元画像＋10段階)",
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
            # 品質レベルグループ選択を追加
            quality_group = st.radio(
                "品質レベルグループ",
                ["📗 個別選択", "📘 グループ選択"],
                horizontal=True,
                help="個別で選択するか、グループで一括選択するか"
            )
            
            if quality_group == "📗 個別選択":
                quality_level = st.selectbox(
                    "低画質レベルを選択",
                    ["low1", "low2", "low3", "low4", "low5", "low6", "low7", "low8", "low9", "low10", "カスタム"],
                    index=4,  # デフォルトでlow5を選択
                    help="比較する低画質レベルを選択 (low1が最も高品質、low10が最も低品質)"
                )
                if quality_level == "カスタム":
                    quality_level = st.text_input("カスタムサフィックス", value="low1")
                quality_levels = [quality_level]  # リストに変換
            else:
                # グループ選択
                quality_group_select = st.selectbox(
                    "品質グループを選択",
                    [
                        "🟢 low1-3 (軽度劣化 - 易しい)",
                        "🟡 low4-7 (中度劣化 - 普通) 🌟推奨",
                        "🔴 low8-10 (重度劣化 - 難しい)",
                        "🌈 全レベル (low1-10) ⚠️非推奨"
                    ],
                    index=1,  # デフォルトでlow4-7を選択
                    help="複数の品質レベルを一括で学習に使用します"
                )
                
                if quality_group_select == "🟢 low1-3 (軽度劣化 - 易しい)":
                    quality_levels = ["low1", "low2", "low3"]
                elif quality_group_select == "🟡 low4-7 (中度劣化 - 普通) 🌟推奨":
                    quality_levels = ["low4", "low5", "low6", "low7"]
                elif quality_group_select == "🔴 low8-10 (重度劣化 - 難しい)":
                    quality_levels = ["low8", "low9", "low10"]
                else:  # 全レベル
                    quality_levels = [f"low{i}" for i in range(1, 11)]
                    st.warning("""
                    ⚠️ **全レベル (low1-10) は推奨されません**
                    
                    **理由:**
                    - 品質差が大きすぎてAIが混乱します
                    - MAE（平均絶対誤差）が悪化します
                    - 信頼度が70%以下に留まります
                    
                    **推奨:**
                    - 🟡 low4-7 (中度劣化) を選択してください
                    - 信頼度85%以上を目指せます
                    """)
                
                st.info(f"✅ 選択: {', '.join(quality_levels)} ({len(quality_levels)}レベル)")
                quality_level = quality_levels[0]  # 後方互換性のため最初のレベルを代入
        
        
        if folder_path and os.path.exists(folder_path):
            # フォルダから画像ペアを自動検出
            all_files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
            
            # 高画質画像を検出(_lowがついていないもの)
            # 正規表現を使って_low + 数字のパターンを除外
            high_files = [f for f in all_files if not re.search(r'_low\d+', os.path.basename(f))]
            
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
                # グループ選択の場合は複数レベルを収集
                low_files_all = []
                high_files_all = []
                missing_files = []
                debug_info = []  # デバッグ用
                
                for quality_lv in quality_levels:
                    for hf in high_files:
                        base_name = os.path.splitext(os.path.basename(hf))[0]
                        ext = os.path.splitext(os.path.basename(hf))[1]
                        low_file = os.path.join(folder_path, f"{base_name}_{quality_lv}{ext}")
                        # パスの正規化
                        low_file = os.path.normpath(low_file)
                        
                        # デバッグ情報を記録(最初の品質レベルのみ)
                        if quality_lv == quality_levels[0]:
                            exists = os.path.exists(low_file)
                            debug_info.append({
                                'high': os.path.basename(hf),
                                'base': base_name,
                                'ext': ext,
                                'quality': quality_lv,
                                'expected': os.path.basename(low_file),
                                'full_path': low_file,  # フルパスも追加
                                'exists': exists
                            })
                        
                        if os.path.exists(low_file):
                            low_files_all.append(low_file)
                            high_files_all.append(hf)
                        else:
                            if quality_lv == quality_levels[0]:  # 最初のレベルのみ記録
                                missing_files.append(f"{base_name}_{quality_lv}{ext}")
                
                # 変数名を統一
                low_files = low_files_all
                high_files_for_pairs = high_files_all
                
                # デバッグ: 低画質ファイルパスも表示
                with st.expander("🔍 ペア画像パス (デバッグ情報)", expanded=False):
                    st.write(f"**📁 指定フォルダ:** `{folder_path}`")
                    st.write(f"**📋 ファイルパターン:** `{file_pattern}`")
                    st.write(f"**📋 検出された全画像:** {len(all_files)}枚")
                    st.write(f"**📋 高画質画像:** {len(high_files)}枚")
                    if len(quality_levels) > 1:
                        st.write(f"**🎚️ 選択した品質レベル:** `{', '.join(quality_levels)}` ({len(quality_levels)}レベル)")
                        st.write(f"**📊 元画像数:** {len(high_files)}")
                        st.write(f"**📊 収集ペア数:** {len(low_files)} (期待: {len(high_files) * len(quality_levels)})")
                    else:
                        st.write(f"**🎚️ 選択した品質レベル:** `{quality_level}`")
                        st.write(f"**📊 高画質ファイル数:** {len(high_files)}")
                        st.write(f"**📊 低画質ファイル数:** {len(low_files)}")
                    st.write("")
                    
                    if len(debug_info) > 0:
                        st.write("### 最初の5件の検索結果:")
                        for i, info in enumerate(debug_info[:5]):
                            st.write(f"**{i+1}. {info['high']}**")
                            st.write(f"  - ベース名: `{info['base']}`")
                            st.write(f"  - 拡張子: `{info['ext']}`")
                            st.write(f"  - 品質レベル: `{info['quality']}`")
                            st.write(f"  - 探すファイル: `{info['expected']}`")
                            st.write(f"  - フルパス: `{info['full_path']}`")
                            st.write(f"  - 存在: {'✅ はい' if info['exists'] else '❌ いいえ'}")
                            st.write("---")
                    
                    if len(low_files) > 0:
                        st.success(f"**見つかった低画質ファイル例:** {os.path.basename(low_files[0])}")
                    
                    if missing_files:
                        st.error(f"**見つからないファイル:** {len(missing_files)}件")
                        st.write("最初の5件:")
                        for f in missing_files[:5]:
                            st.write(f"  - `{f}`")
                        
                        # 実際にフォルダ内にあるファイルを表示
                        st.write("### フォルダ内の実際のファイル (最初の20件):")
                        actual_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))[:20]
                        for f in actual_files:
                            fname = os.path.basename(f)
                            # いずれかのquality_levelを含むファイルをハイライト
                            is_target = any(f"_{ql}" in fname for ql in quality_levels)
                            if is_target:
                                st.write(f"  - ✅ `{fname}`")
                            else:
                                st.write(f"  - `{fname}`")
                
                
                # グループ選択の場合は期待値が異なる
                if len(quality_levels) > 1:
                    expected_pairs = len(high_files) * len(quality_levels)
                    if len(low_files) == expected_pairs:
                        st.success(f"✅ {len(low_files)}組の完全なペアを検出しました ({len(high_files)}画像 × {len(quality_levels)}レベル)")
                        uploaded_high = high_files_for_pairs
                        uploaded_low = low_files
                        auto_mode = True
                    else:
                        st.warning(f"⚠️ 一部のペアが見つかりません (期待: {expected_pairs}組, 検出: {len(low_files)}組)")
                        if len(low_files) > 0:
                            if st.checkbox("検出されたペアのみで続行する"):
                                uploaded_high = high_files_for_pairs
                                uploaded_low = low_files
                                auto_mode = True
                                st.info(f"✅ {len(low_files)}組のペアを使用します")
                            else:
                                uploaded_high = None
                                uploaded_low = None
                                auto_mode = False
                        else:
                            uploaded_high = None
                            uploaded_low = None
                            auto_mode = False
                else:
                    # 個別選択の場合(従来通り)
                    if len(low_files) == len(high_files):
                        st.success(f"✅ {len(low_files)}組の完全なペアを検出しました")
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
                                    # パスの正規化
                                    low_file = os.path.normpath(low_file)
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
        
        # augmentation_methodsを初期化（use_augmentationの外側で定義）
        augmentation_methods = []
        
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
                    st.session_state.pop('recommended_preset', None)
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
                    st.session_state.pop('recommended_preset', None)
                    st.rerun()
            with col_btn3:
                # 現在の状態を表示
                select_all_state = st.session_state.get('select_all_augmentation', None)
                if select_all_state == True:
                    st.success("✅ 全選択中 (28種類)")
                elif select_all_state == False:
                    st.warning("全解除中")
            
            # 🚀 クイック推奨設定ボタン（新規追加）
            st.markdown("---")
            st.markdown("### 🚀 クイック設定")
            col_quick1, col_quick2 = st.columns(2)
            
            with col_quick1:
                if st.button("⭐ 推奨設定（15種類）", use_container_width=True, type="secondary",
                            help="信頼度75%達成に最も効果的な15種類を自動選択\n期待データ数: 30×4レベル×16 = 1,920組"):
                    # 推奨設定を適用
                    st.session_state['select_all_augmentation'] = None  # 全選択状態をクリア
                    st.session_state['geo_select_all'] = None
                    st.session_state['bright_select_all'] = None
                    st.session_state['color_select_all'] = None
                    st.session_state['quality_select_all'] = None
                    st.session_state['ai_select_all'] = None
                    
                    # 推奨設定フラグを立てる
                    st.session_state['recommended_preset'] = True
                    st.success("✅ 推奨設定を適用！ 信頼度75%以上を目指します")
                    st.info("📊 期待データ数: 約1,920組（30画像 × 4レベル × 16倍）")
                    st.rerun()
            
            with col_quick2:
                if st.button("🏆 マスター設定（全28種類）", use_container_width=True, type="secondary",
                            help="信頼度90%以上を目指す上級者向け設定\n期待データ数: 30×4レベル×29 = 3,480組"):
                    st.session_state['select_all_augmentation'] = True
                    st.session_state.pop('geo_select_all', None)
                    st.session_state.pop('bright_select_all', None)
                    st.session_state.pop('color_select_all', None)
                    st.session_state.pop('quality_select_all', None)
                    st.session_state.pop('ai_select_all', None)
                    st.session_state.pop('recommended_preset', None)
                    st.success("✅ マスター設定を適用！ 信頼度90%以上を目指します")
                    st.info("📊 期待データ数: 約3,480組（30画像 × 4レベル × 29倍）")
                    st.rerun()
            
            st.markdown("---")
            
            # 全選択/解除の状態を取得
            select_all = st.session_state.get('select_all_augmentation', None)
            recommended = st.session_state.get('recommended_preset', False)
            
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
                
                # 推奨設定の場合の値を設定
                if recommended:
                    rec_flip_h = True
                    rec_flip_v = False
                    rec_rot90 = True
                    rec_rot180 = True
                    rec_rot270 = False
                    rec_rot_small_cw = False
                    rec_rot_small_ccw = False
                else:
                    rec_flip_h = default_geo
                    rec_flip_v = default_geo_off
                    rec_rot90 = default_geo
                    rec_rot180 = default_geo_off
                    rec_rot270 = default_geo_off
                    rec_rot_small_cw = default_geo_off
                    rec_rot_small_ccw = default_geo_off
                
                col1, col2 = st.columns(2)
                with col1:
                    use_flip_h = st.checkbox("🔄 水平反転 (左右反転)", value=rec_flip_h, help="画像を左右反転", key="aug_flip_h")
                    use_flip_v = st.checkbox("🔃 垂直反転 (上下反転)", value=rec_flip_v, help="画像を上下反転", key="aug_flip_v")
                    use_rotate_90 = st.checkbox("↩️ 90度回転", value=rec_rot90, help="時計回りに90度回転", key="aug_rot90")
                    use_rotate_180 = st.checkbox("🔁 180度回転", value=rec_rot180, help="180度回転", key="aug_rot180")
                with col2:
                    use_rotate_270 = st.checkbox("↪️ 270度回転", value=rec_rot270, help="時計回りに270度回転", key="aug_rot270")
                    use_rotate_small_cw = st.checkbox("🔄 微小回転(+5°) 🌟", value=rec_rot_small_cw, help="時計回りに5度回転 - 方向不変性学習に効果的", key="aug_rot_small_cw")
                    use_rotate_small_ccw = st.checkbox("🔄 微小回転(-5°) 🌟", value=rec_rot_small_ccw, help="反時計回りに5度回転 - 方向不変性学習に効果的", key="aug_rot_small_ccw")
            
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
                
                # 推奨設定の場合の値を設定（明るさ調整は重要）
                if recommended:
                    rec_bright = True  # 明るさ系は全て推奨
                else:
                    rec_bright = default_bright
                
                col1, col2 = st.columns(2)
                with col1:
                    use_brightness_up = st.checkbox("☀️ 明るさ増加 (+20%)", value=rec_bright, help="画像を20%明るく", key="aug_br_up")
                    use_brightness_down = st.checkbox("🌙 明るさ減少 (-20%)", value=rec_bright, help="画像を20%暗く", key="aug_br_down")
                    use_contrast_up = st.checkbox("📈 コントラスト増加", value=rec_bright, help="コントラストを強く", key="aug_cont_up")
                with col2:
                    use_contrast_down = st.checkbox("📉 コントラスト減少", value=rec_bright, help="コントラストを弱く", key="aug_cont_down")
                    use_gamma_bright = st.checkbox("✨ ガンマ補正 (明るく)", value=rec_bright if recommended else default_bright, help="ガンマ補正で明るく", key="aug_gamma_br")
                    use_gamma_dark = st.checkbox("🌑 ガンマ補正 (暗く)", value=rec_bright if recommended else default_bright, help="ガンマ補正で暗く", key="aug_gamma_dk")
            
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
                
                # 推奨設定では色調整は控えめに
                rec_color = False if recommended else default_color
                
                col1, col2 = st.columns(2)
                with col1:
                    use_saturation_up = st.checkbox("🌈 彩度増加", value=rec_color, help="色を鮮やかに", key="aug_sat_up")
                    use_saturation_down = st.checkbox("🌫️ 彩度減少", value=rec_color, help="色を淡く", key="aug_sat_down")
                    use_hue_shift = st.checkbox("🎨 色相シフト", value=rec_color, help="色合いを変更", key="aug_hue")
                with col2:
                    use_temp_warm = st.checkbox("🔥 温度調整(暖色) 🌟", value=rec_color, help="照明条件の変化に対応 - AI学習に効果的", key="aug_temp_warm")
                    use_temp_cool = st.checkbox("❄️ 温度調整(寒色) 🌟", value=rec_color, help="照明条件の変化に対応 - AI学習に効果的", key="aug_temp_cool")
            
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
                
                # 推奨設定では重要な画質処理を選択
                if recommended:
                    rec_quality = True
                    rec_quality_off = False
                else:
                    rec_quality = default_quality
                    rec_quality_off = default_quality
                
                col1, col2 = st.columns(2)
                with col1:
                    use_noise = st.checkbox("📡 ノイズ追加", value=rec_quality, help="ガウシアンノイズを追加", key="aug_noise")
                    use_blur = st.checkbox("🌀 ぼかし", value=rec_quality, help="ガウシアンぼかしを適用", key="aug_blur")
                    use_sharpen = st.checkbox("🔪 シャープ化", value=rec_quality, help="エッジを強調", key="aug_sharp")
                with col2:
                    use_equalize = st.checkbox("📊 ヒストグラム均等化", value=rec_quality_off, help="コントラストを自動調整", key="aug_eq")
                    use_median = st.checkbox("🔲 メディアンフィルタ 🌟", value=rec_quality_off, help="ノイズ除去 - フラクタル構造保持に効果的", key="aug_median")
                    use_bilateral = st.checkbox("🎭 バイラテラル 🌟", value=rec_quality_off, help="エッジ保存平滑化 - AI学習に効果的", key="aug_bilateral")
            
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
            
            # 選択された拡張手法を収集（既に初期化済みのリストをクリアして再構築）
            augmentation_methods.clear()
            
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
                
                # ファイルの更新日時を取得
                model_mtime = os.path.getmtime(model_path)
                model_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_mtime))
                
                st.session_state['model_info'] = {
                    'path': model_path,
                    'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'trained_at': model_date,
                    'source': '学習モード（今回の学習）',
                    'file_size': os.path.getsize(model_path)
                }
                
                st.success("""
                ✅ **モデルを永続化しました**
                
                **保存内容:**
                - 🤖 学習済みAIモデル → `trained_fd_model.pkl`
                - 📚 学習履歴 → `training_history.json`
                
                **次回起動時:**
                - ✅ このモデルが自動的に読み込まれます
                - ✅ 学習履歴が引き継がれます
                - ✅ すぐに推論モードで解析できます
                
                💡 アプリを閉じても、あなたのAIの知識は保存されています！
                """)
                
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
                
                # 学習履歴を保存
                training_record = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'num_pairs': len(high_imgs),
                    'quality_level': quality_level,
                    'augmentation_count': len(augmentation_methods),
                    'augmentation_types': augmentation_methods,
                    'total_samples': len(low_imgs),
                    'model_params': {
                        'n_estimators': 400,
                        'max_depth': 8,
                        'learning_rate': 0.05
                    }
                }
                
                # 評価結果を履歴に追加
                if 'metrics' in st.session_state:
                    training_record['metrics'] = st.session_state['metrics']
                
                history_path = save_training_history(training_record)
                if history_path:
                    # 学習履歴統計を更新
                    all_history = load_training_history()
                    st.session_state['history_stats'] = {
                        'total_sessions': len(all_history),
                        'last_trained': training_record['timestamp'],
                        'total_samples': sum(h.get('total_samples', 0) for h in all_history)
                    }
                    
                    st.info(f"""
                    📊 **学習履歴を保存しました**
                    
                    - 📁 ファイル: `{history_path}`
                    - 🎓 今回の学習回数: {len(all_history)}回目
                    - 📈 累計学習データ: {st.session_state['history_stats']['total_samples']:,}組
                    
                    💾 この履歴は次回起動時も保持されます
                    """)
                
                # ⚠️ 全レベル使用時の警告
                if len(quality_levels) >= 10:
                    st.warning("""
                    ⚠️ **全レベル (low1-10) を使用しています**
                    
                    **現在の状況:**
                    - 品質差が大きすぎるため、MAEが悪化している可能性があります
                    - 信頼度が70%以下に留まる可能性があります
                    
                    **改善策:**
                    1. 品質レベルを **「🟡 low4-7 (中度劣化)」** に変更
                    2. データ拡張を **「マスター設定 (28種類)」** に設定
                    3. 再学習を実行
                    
                    **期待される結果:**
                    - MAE: 0.03以下
                    - 信頼度: 85%以上
                    """)
                
                # ============================================================
                # 🎯 AI性能評価を表示
                # ============================================================
                if 'metrics' in st.session_state:
                    metrics = st.session_state['metrics']
                    evaluation = evaluate_ai_performance(
                        metrics.get('correlation_pred', 0),
                        metrics.get('improvement', 0),
                        metrics.get('mae_pred', 0)
                    )
                    
                    st.markdown("---")
                    st.subheader("🎯 AI性能総合評価")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="総合評価",
                            value=evaluation['grade'],
                            help=f"スコア: {evaluation['score']:.1f}/100"
                        )
                    with col2:
                        st.metric(
                            label="相関係数評価",
                            value=evaluation['correlation_grade'],
                            delta=f"{metrics.get('correlation_pred', 0):.3f}"
                        )
                    with col3:
                        st.metric(
                            label="改善率",
                            value=f"{metrics.get('improvement', 0):.1f}%",
                            delta="良好" if metrics.get('improvement', 0) > 50 else "要改善"
                        )
                    
                    st.info(f"{evaluation['emoji']} **{evaluation['comment']}**")
                    
                    # MAEが高い場合の警告
                    if metrics.get('mae_pred', 1.0) > 0.04 and len(quality_levels) >= 10:
                        st.error("""
                        ⚠️ **MAEが高いです (0.04以上)**
                        
                        **原因:**
                        - 全レベル (low1-10) を使用しているため、品質差が大きすぎます
                        - low1 (軽度劣化) と low10 (重度劣化) では復元難易度が異なります
                        - AIが一貫した補正ルールを学習できていません
                        
                        **解決策:**
                        1. 品質レベルを **「🟡 low4-7」** に変更して再学習
                        2. または **「🔴 low8-10」** (重度劣化専門) を選択
                        3. データ拡張を28種類 (マスター設定) に増やす
                        
                        **期待される改善:**
                        - MAE: 0.04 → 0.02-0.03
                        - 信頼度: 70% → 85%以上
                        """)
                    
                    # 詳細分析
                    with st.expander("📊 詳細分析"):
                        st.write("### 各評価項目のポイント")
                        st.write(f"- **相関係数スコア**: {evaluation['details']['corr_points']:.1f}/100")
                        st.write(f"- **改善率スコア**: {evaluation['details']['improve_points']:.1f}/100")
                        st.write(f"- **MAEスコア**: {evaluation['details']['mae_points']:.1f}/100")
                        st.write("")
                        st.write(f"**総合スコア計算式**: 相関50% + 改善率30% + MAE20%")
                        st.write(f"= {evaluation['details']['corr_points']:.1f}×0.5 + {evaluation['details']['improve_points']:.1f}×0.3 + {evaluation['details']['mae_points']:.1f}×0.2")
                        st.write(f"= **{evaluation['score']:.1f}点**")
                    
                    # 成長分析(複数回学習している場合)
                    all_history = load_training_history()
                    if len(all_history) >= 2:
                        growth = analyze_learning_growth(all_history)
                        with st.expander("📈 学習成長トレンド"):
                            st.write(f"### {growth['trend']} {growth['trend_emoji']}")
                            st.write(f"**学習セッション数**: {growth['num_sessions']}回")
                            st.write(f"**前回からの変化**:")
                            st.write(f"  - 相関係数: {growth['correlation_change']:+.3f}")
                            st.write(f"  - 改善率: {growth['improvement_change']:+.1f}%")
                            st.write(f"**歴代最高記録**:")
                            st.write(f"  - 相関係数: {growth['best_correlation']:.3f}")
                            st.write(f"  - 改善率: {growth['best_improvement']:.1f}%")
                            st.write("")
                            st.info(f"💡 **推奨アクション**: {growth['recommendation']}")
                    
                    st.markdown("---")
                
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

    # ============================================================
    # 🌸 顔全体分析モード
    # ============================================================
    elif app_mode == "🌸 顔全体分析モード":
        st.header("🌸 顔全体フラクタル分析 - 部位別肌評価")
        
        if not SKIN_ANALYSIS_AVAILABLE:
            st.error("""
            ❌ **顔分析機能が利用できません**
            
            `skin_analysis.py`モジュールが見つかりません。
            または`mediapipe`がインストールされていません。
            
            以下のコマンドでインストールしてください:
            ```
            pip install mediapipe
            ```
            """)
            return
        
        st.markdown("""
        ### 📸 顔全体を撮影して、各部位を自動分析
        
        **このモードでできること:**
        - 🎯 顔の自動検出と部位分割（額、頬、鼻、口周り、顎など）
        - 📊 各部位のフラクタル次元（肌のキメ細かさ）
        - 🔍 7種類の肌トラブル検出
          * 毛穴の目立ち
          * シワ
          * 色ムラ・くすみ
          * ニキビ・赤み
          * クマ（目の下）
          * テカリ
          * キメの粗さ
        - 🗺️ ヒートマップによる可視化
        - 📋 部位別レポート生成
        
        **撮影のコツ:**
        - 正面から顔全体が写るように撮影
        - 自然光または明るい室内で
        - 距離は約30-50cm
        - 無表情で
        """)
        
        uploaded_file = st.file_uploader(
            "顔写真をアップロード",
            type=['jpg', 'jpeg', 'png'],
            help="顔全体が写った画像をアップロードしてください"
        )
        
        if uploaded_file:
            # 画像読み込み
            image = read_bgr_from_buffer(uploaded_file.read())
            
            if image is None:
                st.error("画像の読み込みに失敗しました")
                return
            
            with st.spinner("🔍 顔を検出中..."):
                landmarks = detect_face_landmarks(image)
            
            if landmarks is None:
                st.error("""
                ❌ **顔が検出できませんでした**
                
                以下を確認してください:
                - 顔全体が写っているか
                - 顔が正面を向いているか
                - 画像が明るいか
                - 顔が大きすぎたり小さすぎたりしないか
                """)
                return
            
            st.success("✅ 顔を検出しました！")
            
            # 各部位を抽出
            with st.spinner("📐 部位を分割中..."):
                regions = extract_face_regions(image, landmarks)
            
            if not regions:
                st.error("部位の抽出に失敗しました")
                return
            
            st.success(f"✅ {len(regions)}つの部位を検出しました")
            
            # タブ表示
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 総合評価",
                "🗺️ 部位表示",
                "🔍 詳細分析",
                "📋 レポート"
            ])
            
            # 部位別にFD計算と肌トラブル検出
            fd_results = {}
            trouble_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (region_name, region_data) in enumerate(regions.items()):
                status_text.text(f"解析中: {REGION_NAMES_JP.get(region_name, region_name)}")
                
                region_img = region_data['image']
                
                if region_img is not None and region_img.size > 0:
                    # FD計算
                    fd_result = calculate_fractal_dimension(region_img)
                    fd_results[region_name] = fd_result['fd']
                    
                    # 肌トラブル検出
                    troubles = detect_skin_troubles(region_img, region_name)
                    # FD値をキメの粗さスコアに反映
                    if 'texture_roughness' in troubles:
                        fd_score = min((fd_result['fd'] - 2.0) / 1.0 * 100, 100)
                        troubles['texture_roughness']['score'] = fd_score
                        troubles['texture_roughness']['level'] = (
                            '高' if fd_score > 65 else '中' if fd_score > 35 else '低'
                        )
                    
                    trouble_results[region_name] = troubles
                
                progress_bar.progress((idx + 1) / len(regions))
            
            progress_bar.empty()
            status_text.empty()
            
            # 総合スコア計算
            if fd_results:
                # FDを100点満点スコアに変換（2.0=100点, 3.0=0点）
                scores = {}
                for region, fd in fd_results.items():
                    score = (3.0 - fd) / 1.0 * 100
                    scores[region] = max(0, min(100, score))
                
                # 重み付き平均（顔の重要部位を重視）
                weights = {
                    'forehead': 0.12,
                    'left_cheek': 0.22,
                    'right_cheek': 0.22,
                    'nose': 0.10,
                    'mouth_area': 0.14,
                    'chin': 0.10,
                    'left_under_eye': 0.05,
                    'right_under_eye': 0.05
                }
                
                overall_score = sum(
                    scores.get(region, 0) * weight
                    for region, weight in weights.items()
                )
                
                # グレード判定
                if overall_score >= 85:
                    grade = 'S (非常に良い)'
                    grade_emoji = '🌟'
                elif overall_score >= 70:
                    grade = 'A (良い)'
                    grade_emoji = '⭐'
                elif overall_score >= 55:
                    grade = 'B (普通)'
                    grade_emoji = '🔵'
                elif overall_score >= 40:
                    grade = 'C (やや粗い)'
                    grade_emoji = '🟡'
                else:
                    grade = 'D (要ケア)'
                    grade_emoji = '🔴'
            
            # Tab 1: 総合評価
            with tab1:
                st.subheader("📊 総合評価")
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    st.markdown("### 元画像")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                            use_container_width=True)
                
                with col2:
                    st.markdown("### スコア")
                    st.metric(
                        "総合スコア",
                        f"{overall_score:.1f}/100",
                        delta=None
                    )
                    st.markdown(f"### {grade_emoji} グレード")
                    st.markdown(f"## {grade}")
                
                with col3:
                    st.markdown("### 部位別スコア")
                    for region, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                        region_jp = REGION_NAMES_JP.get(region, region)
                        st.progress(score / 100)
                        st.caption(f"{region_jp}: {score:.1f}点")
            
            # Tab 2: 部位表示
            with tab2:
                st.subheader("🗺️ 検出された部位")
                
                # 部位を2列で表示
                cols_per_row = 3
                region_items = list(regions.items())
                
                for i in range(0, len(region_items), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(region_items):
                            region_name, region_data = region_items[idx]
                            region_img = region_data['image']
                            
                            with col:
                                region_jp = REGION_NAMES_JP.get(region_name, region_name)
                                if region_img is not None and region_img.size > 0:
                                    st.image(
                                        cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB),
                                        caption=f"{region_jp}",
                                        use_container_width=True
                                    )
                                    if region_name in fd_results:
                                        st.caption(f"FD: {fd_results[region_name]:.4f}")
                                        st.caption(f"スコア: {scores.get(region_name, 0):.1f}点")
            
            # Tab 3: 詳細分析
            with tab3:
                st.subheader("🔍 部位別詳細分析")
                
                for region_name in regions.keys():
                    region_jp = REGION_NAMES_JP.get(region_name, region_name)
                    
                    with st.expander(f"📍 {region_jp}"):
                        if region_name in fd_results:
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.metric("フラクタル次元", f"{fd_results[region_name]:.4f}")
                                st.metric("スコア", f"{scores.get(region_name, 0):.1f}/100")
                            
                            with col2:
                                if region_name in trouble_results:
                                    troubles = trouble_results[region_name]
                                    
                                    st.markdown("**検出された肌トラブル:**")
                                    for trouble_key, trouble_data in troubles.items():
                                        trouble_jp = TROUBLE_NAMES_JP.get(trouble_key, trouble_key)
                                        level = trouble_data.get('level', '不明')
                                        score_val = trouble_data.get('score', 0)
                                        
                                        level_emoji = '🔴' if level == '高' else '🟡' if level == '中' else '🟢'
                                        st.write(f"{level_emoji} **{trouble_jp}**: {level} ({score_val:.1f})")
            
            # Tab 4: レポート
            with tab4:
                st.subheader("📋 総合レポート")
                
                report = create_trouble_report(trouble_results, fd_results)
                st.markdown(report)
                
                # レポートダウンロード
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                report_filename = f"skin_analysis_report_{timestamp}.md"
                
                st.download_button(
                    label="📥 レポートをダウンロード",
                    data=report,
                    file_name=report_filename,
                    mime="text/markdown"
                )
    
    # ============================================================
    # 🔬 実験データ収集モード
    # ============================================================
    elif app_mode == "🔬 実験データ収集":
        st.header("🔬 実験データ収集 - 肌状態とフラクタル次元の相関研究")
        
        if not EXPERIMENT_ANALYSIS_AVAILABLE:
            st.error("""
            ❌ **実験データ収集機能が利用できません**
            
            `experiment_analysis.py`モジュールが見つかりません。
            """)
            return
        
        st.markdown("""
        ### 🎯 研究目的
        フラクタル次元と肌の物理的状態（乾燥、荒れ、水分量など）の定量的関係を明らかにする
        
        **測定項目:**
        - フラクタル次元（客観的指標）
        - 肌状態の主観評価（肌荒れ度、乾燥度）
        - 客観的測定値（水分量、皮脂量など）
        - 環境条件（温度、湿度）
        
        **データの使い道:**
        - 相関分析（FDと肌状態の関係）
        - 論文・レポート作成
        - AI予測モデルの改善
        """)
        
        # データマネージャー初期化
        data_manager = ExperimentDataManager()
        
        # タブで分割
        data_tab, history_tab = st.tabs(["📝 新規データ収集", "📚 履歴表示"])
        
        with data_tab:
            st.subheader("📋 被験者情報")
            
            col1, col2 = st.columns(2)
            with col1:
                subject_id = st.text_input("被験者ID", placeholder="例: S001", help="一意のIDを設定")
                age = st.number_input("年齢", min_value=10, max_value=100, value=25)
                gender = st.selectbox("性別", ["女性", "男性", "その他"])
            
            with col2:
                skin_type = st.selectbox("肌質", [
                    "普通肌", "乾燥肌", "脂性肌", "混合肌", "敏感肌"
                ])
                measurement_date = st.date_input("測定日")
                measurement_time = st.time_input("測定時刻")
            
            st.markdown("---")
            st.subheader("🌡️ 測定条件")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                condition = st.selectbox("肌状態", [
                    "通常状態",
                    "洗顔直後（30分以内）",
                    "保湿クリーム塗布後",
                    "運動後",
                    "睡眠不足後",
                    "その他"
                ])
            with col2:
                temperature = st.number_input("室温 (°C)", min_value=10.0, max_value=40.0, value=22.0, step=0.5)
            with col3:
                humidity = st.number_input("湿度 (%)", min_value=0, max_value=100, value=50)
            
            st.markdown("---")
            st.subheader("👁️ 肌状態評価（目視）")
            
            st.info("""
            💡 **評価のポイント:**
            - 客観的に観察して評価してください
            - 毎回同じ基準で評価することが重要です
            - 迷った場合は中間の値（3）を選択
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                roughness_score = st.slider(
                    "肌荒れ度",
                    min_value=1, max_value=5, value=3,
                    help="1=非常に滑らか, 5=非常に荒れている"
                )
                st.caption("⭐ 非常に滑らか → ⭐⭐⭐⭐⭐ 非常に荒れている")
                
                pore_score = st.slider(
                    "毛穴の目立ち度",
                    min_value=1, max_value=5, value=3,
                    help="1=目立たない, 5=非常に目立つ"
                )
                
                wrinkle_score = st.slider(
                    "シワの目立ち度",
                    min_value=1, max_value=5, value=3,
                    help="1=目立たない, 5=非常に目立つ"
                )
            
            with col2:
                dryness_score = st.slider(
                    "乾燥度",
                    min_value=1, max_value=5, value=3,
                    help="1=非常に潤っている, 5=非常に乾燥している"
                )
                st.caption("💧 非常に潤い → 🔥🔥🔥🔥🔥 非常に乾燥")
                
                redness_score = st.slider(
                    "赤み・炎症",
                    min_value=1, max_value=5, value=3,
                    help="1=なし, 5=強い赤み"
                )
                
                dark_circle_score = st.slider(
                    "クマの目立ち度",
                    min_value=1, max_value=5, value=3,
                    help="1=目立たない, 5=非常に目立つ"
                )
            
            st.markdown("---")
            st.subheader("📊 客観的測定値")
            
            col1, col2 = st.columns(2)
            with col1:
                moisture_level = st.number_input(
                    "肌水分量 (%)",
                    min_value=0.0, max_value=100.0, value=40.0, step=0.1,
                    help="肌水分計での測定値（持っている場合）"
                )
            with col2:
                sebum_level = st.number_input(
                    "皮脂量 (任意)",
                    min_value=0.0, max_value=100.0, value=50.0, step=0.1,
                    help="皮脂測定器での測定値（任意）"
                )
            
            st.markdown("---")
            st.subheader("📸 画像アップロード")
            
            col1, col2 = st.columns(2)
            with col1:
                left_cheek = st.file_uploader("左頬の画像", type=['jpg', 'png'], key='left')
            with col2:
                right_cheek = st.file_uploader("右頬の画像", type=['jpg', 'png'], key='right')
            
            notes = st.text_area("備考・メモ", placeholder="特記事項があれば記入（例：化粧品を変更、体調不良など）")
            
            st.markdown("---")
            
            # データ保存ボタン
            if st.button("💾 データを保存", type="primary", use_container_width=True):
                if not subject_id:
                    st.error("❌ 被験者IDを入力してください")
                else:
                    with st.spinner("🔄 データを処理中..."):
                        # データエントリ作成
                        data_entry = {
                            'subject_id': subject_id,
                            'timestamp': f"{measurement_date} {measurement_time}",
                            'age': age,
                            'gender': gender,
                            'skin_type': skin_type,
                            'condition': condition,
                            'temperature': temperature,
                            'humidity': humidity,
                            'roughness_score': roughness_score,
                            'dryness_score': dryness_score,
                            'pore_score': pore_score,
                            'wrinkle_score': wrinkle_score,
                            'redness_score': redness_score,
                            'dark_circle_score': dark_circle_score,
                            'moisture_level': moisture_level,
                            'sebum_level': sebum_level,
                            'notes': notes
                        }
                        
                        # 左頬のFD計算
                        if left_cheek:
                            left_img = read_bgr_from_buffer(left_cheek.read())
                            if left_img is not None:
                                left_fd_result = calculate_fractal_dimension(left_img)
                                data_entry['left_cheek_fd'] = left_fd_result['fd']
                                data_entry['left_cheek_confidence'] = left_fd_result['confidence']
                        
                        # 右頬のFD計算
                        if right_cheek:
                            right_cheek.seek(0)
                            right_img = read_bgr_from_buffer(right_cheek.read())
                            if right_img is not None:
                                right_fd_result = calculate_fractal_dimension(right_img)
                                data_entry['right_cheek_fd'] = right_fd_result['fd']
                                data_entry['right_cheek_confidence'] = right_fd_result['confidence']
                        
                        # 平均FD
                        if 'left_cheek_fd' in data_entry and 'right_cheek_fd' in data_entry:
                            data_entry['average_fd'] = (data_entry['left_cheek_fd'] + data_entry['right_cheek_fd']) / 2
                        elif 'left_cheek_fd' in data_entry:
                            data_entry['average_fd'] = data_entry['left_cheek_fd']
                        elif 'right_cheek_fd' in data_entry:
                            data_entry['average_fd'] = data_entry['right_cheek_fd']
                        
                        # 保存
                        if data_manager.save_data(data_entry):
                            st.success("✅ データを保存しました！")
                            
                            # 結果表示
                            st.subheader("📊 測定結果")
                            col1, col2, col3 = st.columns(3)
                            
                            if 'left_cheek_fd' in data_entry:
                                with col1:
                                    st.metric("左頬 FD", f"{data_entry['left_cheek_fd']:.4f}")
                            
                            if 'right_cheek_fd' in data_entry:
                                with col2:
                                    st.metric("右頬 FD", f"{data_entry['right_cheek_fd']:.4f}")
                            
                            if 'average_fd' in data_entry:
                                with col3:
                                    st.metric("平均 FD", f"{data_entry['average_fd']:.4f}")
                        else:
                            st.error("❌ データの保存に失敗しました")
        
        with history_tab:
            st.subheader("📚 収集済みデータ")
            
            df = data_manager.load_data()
            
            if df is None or len(df) == 0:
                st.info("まだデータがありません。「新規データ収集」タブでデータを収集してください。")
            else:
                st.success(f"✅ {len(df)}件のデータを読み込みました")
                
                # データ概要
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("総測定回数", f"{len(df)}回")
                with col2:
                    st.metric("被験者数", f"{df['subject_id'].nunique()}人")
                with col3:
                    if 'average_fd' in df.columns:
                        st.metric("FD値範囲", f"{df['average_fd'].min():.3f} - {df['average_fd'].max():.3f}")
                
                # データテーブル
                st.dataframe(df, use_container_width=True, height=400)
                
                # CSVダウンロード
                csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "📥 全データをCSVでダウンロード",
                    data=csv_data,
                    file_name=f"experimental_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # ============================================================
    # 📈 相関分析モード
    # ============================================================
    elif app_mode == "📈 相関分析":
        st.header("📈 相関分析 - フラクタル次元と肌状態の関係")
        
        if not EXPERIMENT_ANALYSIS_AVAILABLE:
            st.error("""
            ❌ **相関分析機能が利用できません**
            
            `experiment_analysis.py`モジュールが見つかりません。
            """)
            return
        
        st.markdown("""
        ### 📊 統計分析
        収集した実験データから、フラクタル次元と肌状態の相関関係を分析します。
        
        **分析内容:**
        - Pearson相関係数の計算
        - 統計的有意性検定（p値）
        - 散布図と回帰直線
        - 相関ヒートマップ
        """)
        
        # データマネージャー初期化
        data_manager = ExperimentDataManager()
        df = data_manager.load_data()
        
        if df is None or len(df) == 0:
            st.warning("""
            ⚠️ **分析するデータがありません**
            
            「🔬 実験データ収集」モードでデータを収集してください。
            最低でも3件以上のデータが必要です。
            """)
            return
        
        st.success(f"✅ {len(df)}件のデータを読み込みました")
        
        if len(df) < 3:
            st.warning("⚠️ データが不足しています。有意な相関分析には最低3件以上のデータが必要です。")
            return
        
        # タブで分割
        summary_tab, correlation_tab, scatter_tab, export_tab = st.tabs([
            "📋 サマリー",
            "🔗 相関分析",
            "📊 散布図",
            "📥 エクスポート"
        ])
        
        with summary_tab:
            st.subheader("📋 データサマリー")
            
            summary = generate_experiment_summary(df)
            st.markdown(summary)
            
            # 基本統計
            if 'average_fd' in df.columns:
                st.subheader("📊 フラクタル次元の分布")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df['average_fd'].dropna(), bins=20, color='steelblue', 
                       edgecolor='darkblue', alpha=0.7)
                ax.set_xlabel('フラクタル次元', fontsize=12, fontweight='bold')
                ax.set_ylabel('頻度', fontsize=12, fontweight='bold')
                ax.set_title('フラクタル次元のヒストグラム', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
        
        with correlation_tab:
            st.subheader("🔗 相関係数分析")
            
            if 'average_fd' not in df.columns:
                st.error("フラクタル次元データがありません")
                return
            
            # 相関計算
            correlations = calculate_correlations(df)
            
            if not correlations:
                st.warning("相関分析に必要なデータが不足しています")
            else:
                # 相関係数表
                st.markdown("### 📋 相関係数一覧")
                
                corr_data = []
                for name, data in correlations.items():
                    significance = "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else ""
                    corr_data.append({
                        '項目': name,
                        '相関係数 (r)': f"{data['r']:.4f}{significance}",
                        'p値': f"{data['p_value']:.6f}",
                        '有意性': '✅ 有意' if data['significant'] else '❌ 非有意',
                        'データ数': data['n']
                    })
                
                corr_df = pd.DataFrame(corr_data)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
                
                st.caption("* p < 0.05, ** p < 0.01")
                
                # 解釈ガイド
                with st.expander("📖 相関係数の解釈ガイド"):
                    st.markdown("""
                    **相関係数 (r) の強さ:**
                    - |r| ≥ 0.7: 強い相関
                    - 0.4 ≤ |r| < 0.7: 中程度の相関
                    - 0.2 ≤ |r| < 0.4: 弱い相関
                    - |r| < 0.2: ほぼ相関なし
                    
                    **符号の意味:**
                    - 正の相関 (r > 0): 一方が増えるともう一方も増える
                    - 負の相関 (r < 0): 一方が増えるともう一方は減る
                    
                    **p値:**
                    - p < 0.05: 統計的に有意（偶然ではない可能性が高い）
                    - p ≥ 0.05: 統計的に非有意（偶然の可能性あり）
                    """)
                
                # ヒートマップ
                st.markdown("### 🔥 相関ヒートマップ")
                fig = create_correlation_heatmap(correlations)
                st.pyplot(fig)
        
        with scatter_tab:
            st.subheader("📊 散布図分析")
            
            if 'average_fd' not in df.columns:
                st.error("フラクタル次元データがありません")
                return
            
            # 散布図作成する項目を選択
            scatter_options = {
                'roughness_score': '肌荒れ度',
                'dryness_score': '乾燥度',
                'pore_score': '毛穴',
                'wrinkle_score': 'シワ',
                'redness_score': '赤み',
                'dark_circle_score': 'クマ',
                'moisture_level': '水分量',
                'sebum_level': '皮脂量',
                'age': '年齢'
            }
            
            available_options = {k: v for k, v in scatter_options.items() if k in df.columns}
            
            if not available_options:
                st.warning("散布図を作成できる項目がありません")
            else:
                selected_var = st.selectbox(
                    "比較する項目を選択",
                    options=list(available_options.keys()),
                    format_func=lambda x: available_options[x]
                )
                
                fig = create_scatter_plot(
                    df, 
                    selected_var, 
                    'average_fd',
                    available_options[selected_var],
                    'フラクタル次元',
                    f'フラクタル次元 vs {available_options[selected_var]}'
                )
                st.pyplot(fig)
                
                # すべての散布図を一括表示
                if st.checkbox("すべての散布図を表示"):
                    cols_per_row = 2
                    var_items = list(available_options.items())
                    
                    for i in range(0, len(var_items), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(var_items):
                                var_key, var_name = var_items[idx]
                                with col:
                                    fig_small = create_scatter_plot(
                                        df,
                                        var_key,
                                        'average_fd',
                                        var_name,
                                        'FD',
                                        f'FD vs {var_name}'
                                    )
                                    st.pyplot(fig_small)
        
        with export_tab:
            st.subheader("📥 分析結果のエクスポート")
            
            # 相関分析レポート生成
            report_lines = ["# 📊 フラクタル次元と肌状態の相関分析レポート\n"]
            report_lines.append(f"**作成日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            report_lines.append(f"**データ数**: {len(df)}件\n")
            
            if correlations:
                report_lines.append("\n## 🔗 相関係数\n")
                for name, data in sorted(correlations.items(), key=lambda x: abs(x[1]['r']), reverse=True):
                    sig = "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else ""
                    report_lines.append(f"- **{name}**: r = {data['r']:.4f}{sig}, p = {data['p_value']:.6f}, n = {data['n']}")
                
                report_lines.append("\n## 📋 解釈\n")
                strong_corr = [name for name, data in correlations.items() if abs(data['r']) >= 0.7 and data['significant']]
                if strong_corr:
                    report_lines.append("**強い相関が見られた項目:**")
                    for name in strong_corr:
                        report_lines.append(f"- {name}")
            
            report_text = '\n'.join(report_lines)
            
            st.markdown(report_text)
            
            # ダウンロードボタン
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 レポートをダウンロード (Markdown)",
                    data=report_text,
                    file_name=f"correlation_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                if 'average_fd' in df.columns:
                    # 分析用データをCSVで出力
                    analysis_df = df[['subject_id', 'timestamp', 'average_fd'] + 
                                    [col for col in df.columns if col.endswith('_score') or col.endswith('_level')]].copy()
                    csv_analysis = analysis_df.to_csv(index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        "📥 分析データをダウンロード (CSV)",
                        data=csv_analysis,
                        file_name=f"analysis_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    app()
