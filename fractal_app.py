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

    gpu_auto = USE_CUPY
    st.sidebar.header("設定")
    st.sidebar.write(f"GPU 利用可能: {USE_CUPY}")
    use_gpu_checkbox = st.sidebar.checkbox("GPU を使う(自動判定)", value=USE_CUPY)
    st.sidebar.write("※ GPU が無い場合は自動的に CPU にフォールバックします。")

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
        else:
            st.warning("⚠️ フォルダパスが無効です")
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

                # Evaluate & show metrics
                st.info("解析・比較を行います...")
                D_high, D_low, D_pred = evaluate_and_plot(high_imgs, low_imgs, model, use_gpu=use_gpu_checkbox)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"❌ **エラーが発生しました:** {str(e)}")
                st.stop()
            
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

            end = time.time()
            st.success(f"✅ 全処理完了! 処理時間: {end - start:.2f} 秒")

    else:
        st.info("📁 フォルダモード: フォルダパスを入力すると自動的に画像ペアを検出します\n📤 手動モード: 高画質と低画質のペア画像を同数アップロードしてください")

if __name__ == "__main__":
    app()
