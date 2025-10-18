import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import io
import base64
from scipy import ndimage

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 肌質評価の基準値
# ----------------------------
SKIN_FD_IDEAL_MIN = 2.4  # 理想的な肌のフラクタル次元下限
SKIN_FD_IDEAL_MAX = 2.8  # 理想的な肌のフラクタル次元上限

# ----------------------------
# データ拡張関数（軽量化版）
# ----------------------------
def augment_image(image):
    """画像を回転・反転して学習データを増やす（3倍に削減）"""
    augmented = [image]
    # 180度回転
    augmented.append(cv2.rotate(image, cv2.ROTATE_180))
    # 水平反転
    augmented.append(cv2.flip(image, 1))
    return augmented

# ----------------------------
# 画像サイズ統一関数
# ----------------------------
def align_image_sizes(low_img, high_img, mode='larger'):
    """
    低画質と高画質の画像サイズを統一する
    mode: 'larger' = 大きい方に合わせる（推奨）
          'smaller' = 小さい方に合わせる
          'high' = 高画質画像に合わせる
          'low' = 低画質画像に合わせる
    """
    low_h, low_w = low_img.shape[:2]
    high_h, high_w = high_img.shape[:2]
    
    # サイズが同じ場合はそのまま返す
    if (low_h, low_w) == (high_h, high_w):
        return low_img, high_img
    
    # ターゲットサイズを決定
    if mode == 'larger':
        target_h = max(low_h, high_h)
        target_w = max(low_w, high_w)
    elif mode == 'smaller':
        target_h = min(low_h, high_h)
        target_w = min(low_w, high_w)
    elif mode == 'high':
        target_h, target_w = high_h, high_w
    elif mode == 'low':
        target_h, target_w = low_h, low_w
    else:
        target_h = max(low_h, high_h)
        target_w = max(low_w, high_w)
    
    # リサイズ
    aligned_low = low_img
    aligned_high = high_img
    
    if (low_h, low_w) != (target_h, target_w):
        aligned_low = cv2.resize(low_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    if (high_h, high_w) != (target_h, target_w):
        aligned_high = cv2.resize(high_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    return aligned_low, aligned_high

# ----------------------------
# AI補完モデルの定義（異なるサイズ対応版）
# ----------------------------
def train_image_enhancer(low_quality_images, high_quality_images, use_augmentation=True, 
                        max_size=384, n_trees=20, max_depth_val=10, align_mode='larger'):
    """
    精度と速度のバランスを取った学習
    異なるサイズの画像も自動的に統一して学習
    max_size: 384px (256→384で精度向上、速度は許容範囲)
    n_trees: 20 (10→20で精度大幅向上、速度は2倍)
    max_depth: 10 (5→10で精度向上、速度への影響は小)
    align_mode: 'larger', 'smaller', 'high', 'low'
    """
    X, y = [], []
    trained_shape = None  # 学習に使用した画像サイズを保存
    
    # 画像サイズの統一と最適化
    resized_low, resized_high = [], []
    for low, high in zip(low_quality_images, high_quality_images):
        # ステップ1: 低画質と高画質のサイズを統一
        low, high = align_image_sizes(low, high, mode=align_mode)
        
        # ステップ2: max_sizeに収まるようにリサイズ
        h, w = low.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            low = cv2.resize(low, new_size, interpolation=cv2.INTER_LANCZOS4)
            high = cv2.resize(high, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 学習に使用する画像サイズを保存
        if trained_shape is None:
            trained_shape = low.shape
        
        resized_low.append(low)
        resized_high.append(high)
    
    # データ拡張の適用
    if use_augmentation:
        aug_low, aug_high = [], []
        for low, high in zip(resized_low, resized_high):
            aug_low.extend(augment_image(low))
            aug_high.extend(augment_image(high))
        resized_low = aug_low
        resized_high = aug_high
    
    # 特徴量抽出（正規化で精度向上）
    for low, high in zip(resized_low, resized_high):
        low_flat = low.flatten() / 255.0
        high_flat = high.flatten() / 255.0
        X.append(low_flat)
        y.append(high_flat)
    
    X = np.array(X)
    y = np.array(y)
    
    # 訓練データとテストデータに分割
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # 最適化: 精度を保ちつつ高速化
    # n_estimators=20 (10→20): 精度↑↑ 速度↓
    # max_depth=10 (5→10): 精度↑ 速度↓わずか
    # min_samples_split=5: 過学習防止で汎化性能向上
    model = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=max_depth_val,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        warm_start=False
    )
    model.fit(X_train, y_train)
    
    # 精度評価
    if len(X_test) > 0:
        score = model.score(X_test, y_test)
    else:
        score = 0.0
    
    # モデルに学習時の画像サイズを属性として保存
    model.trained_shape = trained_shape
    
    # 学習データ数を返す
    training_data_count = len(X)
    
    return model, score, training_data_count

def enhance_image(model, low_quality_image, max_size=384):
    # 元のサイズを保存
    original_shape = low_quality_image.shape
    h, w = original_shape[:2]
    
    # 学習時と同じサイズにリサイズ（重要！）
    if hasattr(model, 'trained_shape') and model.trained_shape is not None:
        # モデルが学習したサイズを使用
        target_shape = model.trained_shape
        resized_image = cv2.resize(low_quality_image, 
                                   (target_shape[1], target_shape[0]), 
                                   interpolation=cv2.INTER_LANCZOS4)
    else:
        # フォールバック: max_sizeベースのリサイズ
        resized_image = low_quality_image.copy()
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            resized_image = cv2.resize(low_quality_image, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    # リサイズ後の画像で予測
    low_flat = resized_image.flatten() / 255.0
    pred = model.predict([low_flat])[0]
    
    # リサイズ後のshapeでreshape（学習時と同じサイズ）
    try:
        enhanced = np.clip(pred * 255, 0, 255).reshape(resized_image.shape).astype(np.uint8)
    except ValueError as e:
        # デバッグ情報を表示
        st.error(f"Reshape エラー: pred size={pred.size}, target shape={resized_image.shape}")
        st.error(f"元画像: {original_shape}, リサイズ後: {resized_image.shape}")
        raise e
    
    # 元のサイズに戻す（高品質補間）
    if enhanced.shape != original_shape:
        enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    return enhanced

# ----------------------------
# AI補正の評価関数
# ----------------------------
def evaluate_ai_correction(fd_low, fd_enhanced, fd_high):
    """
    AI補正の精度を評価
    low: 低画質のフラクタル次元
    enhanced: AI補正後のフラクタル次元
    high: 高画質のフラクタル次元
    """
    if fd_low is None or fd_enhanced is None or fd_high is None:
        return None, "評価不可"
    
    # 改善度: 低画質から補正後への変化
    improvement = abs(fd_enhanced - fd_low)
    
    # 目標との差: 補正後と高画質の差
    target_diff = abs(fd_enhanced - fd_high)
    
    # 低画質と高画質の差
    original_diff = abs(fd_high - fd_low)
    
    # 改善率: どれだけ目標に近づいたか
    if original_diff > 0:
        improvement_rate = (1 - target_diff / original_diff) * 100
    else:
        improvement_rate = 100.0
    
    # 評価ランク
    if improvement_rate >= 90:
        rank = "S (優秀)"
        color = "🟢"
    elif improvement_rate >= 75:
        rank = "A (良好)"
        color = "🔵"
    elif improvement_rate >= 60:
        rank = "B (普通)"
        color = "🟡"
    elif improvement_rate >= 40:
        rank = "C (要改善)"
        color = "🟠"
    else:
        rank = "D (不良)"
        color = "🔴"
    
    evaluation = {
        "improvement_rate": improvement_rate,
        "rank": rank,
        "color": color,
        "improvement": improvement,
        "target_diff": target_diff,
        "original_diff": original_diff
    }
    
    return evaluation, f"{color} ランク: {rank}"

# ----------------------------
# 3D表面凹凸解析（肌質用）
# ----------------------------
def calculate_surface_roughness(image):
    """
    3D表面の凹凸を解析
    肌のテクスチャ（毛穴、シワ、キメ）を定量化
    """
    # グレースケール化
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 高さマップとして扱う（輝度を高さに変換）
    height_map = gray.astype(np.float32)
    
    # 表面粗さの計算
    # 1. 標準偏差（全体的な凹凸）
    roughness_std = np.std(height_map)
    
    # 2. 平均絶対偏差
    roughness_mad = np.mean(np.abs(height_map - np.mean(height_map)))
    
    # 3. 勾配ベースの粗さ（急峻さ）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    roughness_gradient = np.mean(gradient_magnitude)
    
    # 4. ラプラシアン（局所的な変化）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    roughness_laplacian = np.var(laplacian)
    
    return {
        "std": roughness_std,
        "mad": roughness_mad,
        "gradient": roughness_gradient,
        "laplacian": roughness_laplacian
    }

# ----------------------------
# 3D表面フラクタル次元（肌質解析用）
# ----------------------------
def fractal_dimension_3d_surface(image, max_size=256):
    """
    3D表面としてのフラクタル次元を計算
    肌質評価に適した手法（Differential Box Counting法）
    理想的な肌: 2.4~2.8
    """
    # グレースケール化
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # サイズ調整（2の累乗に近いサイズに）
    h, w = gray.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    h, w = gray.shape
    
    # Differential Box Counting (DBC) 法の実装
    # 画像の輝度値を高さとして扱う
    height_map = gray.astype(np.float64)
    
    # スケール（ボックスサイズ）の設定
    min_box_size = 2
    max_box_size = min(h, w) // 4
    box_sizes = []
    box_size = min_box_size
    while box_size <= max_box_size:
        box_sizes.append(box_size)
        box_size *= 2
    
    if len(box_sizes) < 3:
        box_sizes = [2, 4, 8, 16]
    
    counts = []
    
    for r in box_sizes:
        # r x r のグリッドに分割
        n_h = h // r
        n_w = w // r
        
        if n_h < 1 or n_w < 1:
            continue
        
        # このスケールでの総ボックス数
        nr = 0
        
        # 高さ方向のスケール
        # Gは高さ（0-255）をr個のレベルに分割
        G = max(1.0, 255.0 / r)
        
        for i in range(n_h):
            for j in range(n_w):
                # r x r のグリッドセルを取得
                grid = height_map[i*r:(i+1)*r, j*r:(j+1)*r]
                
                if grid.size == 0:
                    continue
                
                # グリッド内の最小・最大輝度値
                min_height = np.min(grid)
                max_height = np.max(grid)
                
                # 高さ方向のボックス位置を計算
                l = int(np.floor(min_height / G))
                k = int(np.floor(max_height / G))
                
                # このグリッドが占める3Dボックスの数
                nr += (k - l + 1)
        
        if nr > 0:
            counts.append(nr)
    
    # box_sizes と counts の長さを揃える
    valid_sizes = box_sizes[:len(counts)]
    valid_counts = counts
    
    # データの妥当性チェック
    if len(valid_sizes) < 3 or len(valid_counts) < 3:
        # データ不足
        return None, np.array([2, 4, 8]), np.array([1, 1, 1])
    
    # すべてのカウントが正であることを確認
    if any(c <= 0 for c in valid_counts):
        return None, np.array(valid_sizes), np.array(valid_counts)
    
    # 対数変換
    log_sizes = np.log(np.array(valid_sizes, dtype=np.float64))
    log_counts = np.log(np.array(valid_counts, dtype=np.float64))
    
    # NaN/Inf チェック
    if np.any(~np.isfinite(log_sizes)) or np.any(~np.isfinite(log_counts)):
        return None, np.array(valid_sizes), np.array(valid_counts)
    
    # 線形回帰でフラクタル次元を計算
    # DBC法: log(Nr) vs log(r)
    # Nr ∝ r^D → log(Nr) = D * log(r) + const
    # 傾きが直接フラクタル次元を表す
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    slope = coeffs[0]
    
    # DBC法では通常、傾きは正（カウント数が増える）
    # 3D表面のフラクタル次元は2.0（平面）から3.0（完全3D）の間
    # 実際の計算では、傾きの符号と値から判断
    
    if abs(slope) < 0.5:
        # 傾きが小さすぎる = ほぼ平面
        fractal_dim_3d = 2.0 + abs(slope)
    elif slope > 0:
        # 正の傾き：通常のDBC法の結果
        # ただし値が大きすぎる場合は調整
        fractal_dim_3d = min(3.0, 2.0 + slope / 2.0)
    else:
        # 負の傾き：通常のボックスカウント法に近い
        fractal_dim_3d = min(3.0, abs(slope))
    
    # 妥当性チェック：2.0～3.0の範囲に収める
    fractal_dim_3d = np.clip(fractal_dim_3d, 2.0, 3.0)
    
    return fractal_dim_3d, np.array(valid_sizes), np.array(valid_counts)

# ----------------------------
# 3D表面フラクタル次元グラフ
# ----------------------------
def plot_3d_fractal_analysis(box_sizes, counts, fd_3d):
    """3D表面フラクタル次元の解析グラフ"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # FDがNoneの場合のエラー表示
    if fd_3d is None or fd_3d == 0:
        ax1.text(0.5, 0.5, '計算エラー\n有効なデータがありません', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14, color='red')
        ax2.text(0.5, 0.5, '計算エラー', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='red')
        return fig
    
    # 左: 対数プロット
    # マスクされた値を除外
    try:
        valid_mask = ~np.ma.getmaskarray(counts)
    except:
        # マスクがない場合
        valid_mask = np.ones(len(counts), dtype=bool)
    
    if not np.any(valid_mask):
        # すべてマスクされている場合
        ax1.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
        log_sizes = np.array([])
        log_counts = np.array([])
    else:
        valid_sizes = np.asarray(box_sizes)[valid_mask]
        valid_counts = np.asarray(counts)[valid_mask]
        log_sizes = np.log(valid_sizes)
        log_counts = np.log(valid_counts)
    
    # データがある場合のみ描画
    if len(log_sizes) > 0 and len(log_counts) > 0:
        ax1.scatter(log_sizes, log_counts, s=100, color='#e74c3c', zorder=5, 
                   edgecolors='white', linewidth=2, label='実測値')
        
        # 回帰直線
        if len(log_sizes) >= 2:
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fit_line = coeffs[0] * log_sizes + coeffs[1]
            ax1.plot(log_sizes, fit_line, '--', color='#3498db', linewidth=2, label='回帰直線')
    
    ax1.set_xlabel('log(ボックスサイズ)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('log(カウント数)', fontsize=11, fontweight='bold')
    ax1.set_title(f'3D表面フラクタル次元解析\nFD = {fd_3d:.4f}', 
                 fontsize=12, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    if len(log_sizes) > 0:
        ax1.legend(fontsize=9)
    
    # 右: 理想範囲との比較
    categories = ['現在の値', '理想範囲\n(下限)', '理想範囲\n(上限)']
    values = [fd_3d, SKIN_FD_IDEAL_MIN, SKIN_FD_IDEAL_MAX]
    colors_bar = ['#3498db', '#2ecc71', '#2ecc71']
    
    bars = ax2.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='white', linewidth=2)
    
    # 理想範囲を強調
    ax2.axhspan(SKIN_FD_IDEAL_MIN, SKIN_FD_IDEAL_MAX, alpha=0.2, color='green', 
               label='理想範囲 (2.4-2.8)')
    
    # 値をバーの上に表示
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('フラクタル次元', fontsize=11, fontweight='bold')
    ax2.set_title('肌質基準との比較', fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylim(min(values) - 0.3, max(values) + 0.3)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    return fig

# ----------------------------
# 表面粗さ可視化
# ----------------------------
def plot_surface_roughness(image, roughness):
    """表面の凹凸を可視化"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 元画像
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('元画像（グレースケール）', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 勾配マップ（凹凸の急峻さ）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    im1 = axes[0, 1].imshow(gradient, cmap='hot')
    axes[0, 1].set_title(f'勾配マップ (凹凸の急峻さ)\n平均: {roughness["gradient"]:.2f}', 
                        fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # ラプラシアン（局所的な変化）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    im2 = axes[1, 0].imshow(np.abs(laplacian), cmap='viridis')
    axes[1, 0].set_title(f'ラプラシアン (局所変化)\n分散: {roughness["laplacian"]:.2f}', 
                        fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # ヒストグラム
    axes[1, 1].hist(gray.ravel(), bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(gray), color='red', linestyle='--', linewidth=2, label='平均')
    axes[1, 1].set_xlabel('輝度値', fontsize=10)
    axes[1, 1].set_ylabel('頻度', fontsize=10)
    axes[1, 1].set_title(f'輝度分布\n標準偏差: {roughness["std"]:.2f}', 
                        fontsize=10, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ----------------------------
# 肌質評価関数
# ----------------------------
def evaluate_skin_quality(fd_3d, roughness):
    """
    3Dフラクタル次元と表面粗さから肌質を評価
    理想的な肌: FD 2.4~2.8
    """
    if fd_3d is None:
        return None, "評価不可"
    
    # フラクタル次元による評価
    if SKIN_FD_IDEAL_MIN <= fd_3d <= SKIN_FD_IDEAL_MAX:
        fd_score = 100
        fd_comment = "理想的"
        fd_color = "🟢"
    elif fd_3d < SKIN_FD_IDEAL_MIN:
        # 低すぎる：滑らかすぎる（不自然）
        diff = SKIN_FD_IDEAL_MIN - fd_3d
        fd_score = max(0, 100 - diff * 100)
        fd_comment = "滑らかすぎる"
        fd_color = "🟡"
    else:
        # 高すぎる：粗い（肌荒れ）
        diff = fd_3d - SKIN_FD_IDEAL_MAX
        fd_score = max(0, 100 - diff * 50)
        if fd_3d > 3.0:
            fd_comment = "粗い（肌荒れ）"
            fd_color = "🔴"
        else:
            fd_comment = "やや粗い"
            fd_color = "🟠"
    
    # 表面粗さによる評価
    roughness_score = 100 - min(100, roughness['std'] / 2.55)
    
    # 総合評価
    total_score = (fd_score * 0.7 + roughness_score * 0.3)
    
    # ランク付け
    if total_score >= 90:
        rank = "S (非常に良い)"
        rank_color = "🟢"
    elif total_score >= 75:
        rank = "A (良い)"
        rank_color = "🔵"
    elif total_score >= 60:
        rank = "B (普通)"
        rank_color = "🟡"
    elif total_score >= 40:
        rank = "C (やや悪い)"
        rank_color = "🟠"
    else:
        rank = "D (悪い)"
        rank_color = "🔴"
    
    evaluation = {
        "fd_3d": fd_3d,
        "fd_score": fd_score,
        "fd_comment": fd_comment,
        "fd_color": fd_color,
        "roughness_score": roughness_score,
        "total_score": total_score,
        "rank": rank,
        "rank_color": rank_color,
        "in_ideal_range": SKIN_FD_IDEAL_MIN <= fd_3d <= SKIN_FD_IDEAL_MAX
    }
    
    return evaluation, f"{rank_color} {rank}"

# ----------------------------
# フラクタル次元(ボックスカウント法・閾値調整対応・高速化版)
# ----------------------------
def fractal_dimension(image, threshold_value=128, use_otsu=False, max_size=512):
    # 画像サイズを制限（高速化）
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size)
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 閾値処理
    if use_otsu:
        threshold_value, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # ボックスサイズを削減（8段階→6段階）
    sizes = 2 ** np.arange(1, 7)
    counts = []
    for size in sizes:
        resized = cv2.resize(binary, (binary.shape[1] // size, binary.shape[0] // size))
        count = np.sum(resized > 0)
        counts.append(count)

    # 異常検出
    if all(c == 0 for c in counts) or all(c == counts[0] for c in counts):
        return None, sizes, counts, binary, threshold_value  # 無効な結果
    
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]
    
    # フラクタル次元の妥当性チェック
    if fractal_dim < 0 or fractal_dim > 3:
        return None, sizes, counts, binary, threshold_value  # 無効な結果
    
    return fractal_dim, sizes, counts, binary, threshold_value

# ----------------------------
# フラクタル次元比較グラフ（線グラフ）
# ----------------------------
def plot_fractal_comparison(fd_low, fd_enhanced, fd_high):
    """3つの画像のフラクタル次元を線グラフで比較"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # None値のチェック
    if None in [fd_low, fd_enhanced, fd_high]:
        ax.text(0.5, 0.5, '計算エラー\nフラクタル次元の計算に失敗しました', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        return fig
    
    categories = ['低画質', 'AI補正後', '高画質\n(目標)']
    values = [fd_low, fd_enhanced, fd_high]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # NaN/Infチェック
    if any(not np.isfinite(v) for v in values):
        ax.text(0.5, 0.5, '計算エラー\n無効な値が含まれています', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        return fig
    
    # 線グラフ
    ax.plot(categories, values, marker='o', linewidth=3, markersize=12, color='#34495e')
    
    # 各点に色をつける
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        ax.scatter(i, val, s=200, color=color, zorder=5, edgecolors='white', linewidth=2)
        ax.text(i, val + 0.05, f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 改善の矢印
    if fd_enhanced > fd_low:
        ax.annotate('', xy=(1, fd_enhanced), xytext=(0, fd_low),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.5))
        ax.text(0.5, (fd_low + fd_enhanced) / 2, '改善↑', ha='center', color='green', fontweight='bold')
    elif fd_enhanced < fd_low:
        ax.annotate('', xy=(1, fd_enhanced), xytext=(0, fd_low),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5))
        ax.text(0.5, (fd_low + fd_enhanced) / 2, '低下↓', ha='center', color='red', fontweight='bold')
    
    ax.set_ylabel('フラクタル次元', fontsize=12, fontweight='bold')
    ax.set_title('画像品質のフラクタル次元比較', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(min(values) - 0.2, max(values) + 0.2)
    
    return fig

# ----------------------------
# 3Dグラフ生成（図を返す・高速化版・サイズ調整）
# ----------------------------
def generate_3d_surface(binary_image, max_resolution=128):
    h, w = binary_image.shape
    
    # 解像度を制限（高速化）
    if max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        binary_image = cv2.resize(binary_image, new_size)
        h, w = binary_image.shape
    
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = binary_image.astype(np.float32) / 255.0 * 10  # 明度を高さに変換
    fig = plt.figure(figsize=(7, 5))  # サイズ縮小: 8,6 → 7,5
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title("3D フラクタル表面 (明度ベース)", fontsize=12, pad=10)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('明度', fontsize=9)
    return fig

# ----------------------------
# 空間占有率の計算（黒・白）
# ----------------------------
def calculate_occupancy(binary_image):
    total = binary_image.size
    white = np.sum(binary_image == 255)
    black = total - white
    return black / total * 100, white / total * 100

# ----------------------------
# 画像保存用のヘルパー関数
# ----------------------------
def save_image_to_bytes(image):
    """OpenCV画像をバイト列に変換"""
    is_success, buffer = cv2.imencode(".png", image)
    return buffer.tobytes() if is_success else None

def fig_to_bytes(fig):
    """matplotlibのfigureをバイト列に変換"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# CSV出力関数
# ----------------------------
def create_results_csv(results_data):
    """解析結果をCSV形式で出力"""
    df = pd.DataFrame([results_data])
    return df.to_csv(index=False).encode('utf-8-sig')

# ----------------------------
# Streamlitアプリ本体
# ----------------------------
st.title("🧠 フラクタル次元解析＋AI画像補完システム")
st.markdown("---")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # AI学習の精度・速度設定
    st.subheader("🤖 AI学習設定")
    quality_mode = st.radio(
        "精度・速度バランス",
        ["⚡ 高速（低精度）", "⚖️ バランス（推奨）", "🎯 高精度（低速）"],
        index=1,
        help="学習時間と精度のトレードオフを選択"
    )
    
    # モード別パラメータ設定
    if quality_mode == "⚡ 高速（低精度）":
        max_size, n_trees, max_depth_val = 256, 10, 5
        speed_text = "約15秒 | 精度: 中"
    elif quality_mode == "⚖️ バランス（推奨）":
        max_size, n_trees, max_depth_val = 384, 20, 10
        speed_text = "約30秒 | 精度: 高"
    else:  # 高精度
        max_size, n_trees, max_depth_val = 512, 50, 15
        speed_text = "約90秒 | 精度: 最高"
    
    st.caption(f"⏱️ 予想処理時間: {speed_text}")
    
    # データ拡張オプション
    use_augmentation = st.checkbox("データ拡張を使用", value=True, 
                                   help="学習データを回転・反転して3倍に増やします")
    
    # 画像サイズ統一設定
    st.caption("📐 画像サイズ統一")
    size_align_mode = st.selectbox(
        "サイズが異なる場合",
        ["🔼 大きい方に合わせる（推奨）", "🔽 小さい方に合わせる", "📷 高画質に合わせる", "📱 低画質に合わせる"],
        index=0,
        help="低画質と高画質の画像サイズが異なる場合の処理方法"
    )
    
    # モード変換
    if "大きい方" in size_align_mode:
        align_mode = "larger"
    elif "小さい方" in size_align_mode:
        align_mode = "smaller"
    elif "高画質" in size_align_mode:
        align_mode = "high"
    else:
        align_mode = "low"
    
    st.markdown("---")
    
    # 解析モード選択
    st.subheader("🔬 解析モード")
    analysis_mode = st.radio(
        "解析手法を選択",
        ["🔲 2D解析（標準）", "🌐 3D表面解析（肌質用）", "📊 両方実行"],
        index=0,
        help="2D: 通常のフラクタル解析 | 3D: 表面凹凸を考慮した肌質解析"
    )
    
    st.markdown("---")
    
    # 閾値設定（2D解析用）
    if "2D" in analysis_mode or "両方" in analysis_mode:
        st.subheader("二値化設定（2D解析用）")
        use_otsu = st.checkbox("大津の二値化を使用", value=False,
                              help="自動で最適な閾値を計算します")
        
        threshold_value = 128
        if not use_otsu:
            threshold_value = st.slider("手動閾値", 0, 255, 128,
                                       help="二値化の閾値を手動で設定します")
    else:
        use_otsu = False
        threshold_value = 128

# ファイルアップロード
col1, col2 = st.columns(2)
with col1:
    uploaded_low = st.file_uploader("📁 低画質画像をアップロード", type=["jpg", "png", "bmp"])
with col2:
    uploaded_high = st.file_uploader("📁 高画質画像(学習用)をアップロード", type=["jpg", "png", "bmp"])

if uploaded_low is not None:
    low_img = cv2.imdecode(np.frombuffer(uploaded_low.read(), np.uint8), cv2.IMREAD_COLOR)
    
    st.markdown("---")
    
    # 画像サイズ情報を表示
    st.info(f"📏 低画質画像サイズ: {low_img.shape[1]} × {low_img.shape[0]} px")
    
    # AI画像補完
    enhanced_img = None
    model_score = None
    
    if uploaded_high is not None:
        high_img = cv2.imdecode(np.frombuffer(uploaded_high.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # 画像サイズの比較表示
        low_size = f"{low_img.shape[1]}×{low_img.shape[0]}"
        high_size = f"{high_img.shape[1]}×{high_img.shape[0]}"
        
        size_col1, size_col2, size_col3 = st.columns(3)
        with size_col1:
            st.metric("📱 低画質", low_size)
        with size_col2:
            if low_img.shape[:2] != high_img.shape[:2]:
                st.warning("⚠️ サイズ不一致")
                st.caption(f"→ {align_mode}モードで統一")
            else:
                st.success("✅ サイズ一致")
        with size_col3:
            st.metric("📷 高画質", high_size)
        
        # プログレスバー表示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f'🤖 学習準備中... ({quality_mode})')
        progress_bar.progress(20)
        
        # 選択されたモードのパラメータで学習
        model, model_score, training_count = train_image_enhancer(
            [low_img], [high_img], 
            use_augmentation=use_augmentation,
            max_size=max_size,
            n_trees=n_trees,
            max_depth_val=max_depth_val,
            align_mode=align_mode
        )
        progress_bar.progress(70)
        
        status_text.text('🖼️ 画像補完中...')
        enhanced_img = enhance_image(model, low_img, max_size=max_size)
        progress_bar.progress(100)
        
        status_text.empty()
        progress_bar.empty()
        
        # 学習データ数と精度を表示
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("📚 学習データ数", f"{training_count} 枚")
        with metric_cols[1]:
            st.metric("🎯 モデル精度", f"{model_score:.3f}")
        with metric_cols[2]:
            st.metric("⚙️ 学習モード", quality_mode)
        
        st.success(f"✅ 学習完了!")
        
        # 画像比較表示
        st.subheader("📊 画像比較")
        img_cols = st.columns(3)
        with img_cols[0]:
            st.image(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB), caption="低画質", use_container_width=True)
        with img_cols[1]:
            st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), caption="AI補完後", use_container_width=True)
        with img_cols[2]:
            st.image(cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB), caption="高画質(正解)", use_container_width=True)
        
        target_img = enhanced_img
        
        # 解析モードに応じた処理
        st.markdown("---")
        
        # 3D表面解析（肌質用）
        if "3D" in analysis_mode or "両方" in analysis_mode:
            st.subheader("🌐 3D表面解析（肌質評価）")
            
            with st.spinner('🔍 3D表面凹凸を解析中...'):
                # 表面粗さ計算
                roughness = calculate_surface_roughness(target_img)
                
                # 3Dフラクタル次元計算
                fd_3d, box_sizes_3d, counts_3d = fractal_dimension_3d_surface(target_img)
            
            if fd_3d is not None:
                # 肌質評価
                skin_eval, skin_eval_text = evaluate_skin_quality(fd_3d, roughness)
                
                # 評価結果表示
                st.success(f"✅ 3D表面フラクタル次元: **{fd_3d:.4f}**")
                
                eval_cols = st.columns(4)
                with eval_cols[0]:
                    st.metric("🎯 総合スコア", f"{skin_eval['total_score']:.1f}点")
                with eval_cols[1]:
                    st.metric("📊 評価ランク", skin_eval_text)
                with eval_cols[2]:
                    if skin_eval['in_ideal_range']:
                        st.metric("✅ 理想範囲", "範囲内", delta="良好")
                    else:
                        st.metric("⚠️ 理想範囲", "範囲外", delta=skin_eval['fd_comment'])
                with eval_cols[3]:
                    st.metric("📏 FD評価", f"{skin_eval['fd_score']:.1f}点")
                
                # 3D解析グラフ
                st.subheader("📈 3D表面フラクタル次元解析")
                fig_3d_analysis = plot_3d_fractal_analysis(box_sizes_3d, counts_3d, fd_3d)
                st.pyplot(fig_3d_analysis)
                plt.close(fig_3d_analysis)
                
                # 表面粗さ可視化
                st.subheader("🔬 表面凹凸の詳細解析")
                fig_roughness = plot_surface_roughness(target_img, roughness)
                st.pyplot(fig_roughness)
                plt.close(fig_roughness)
                
                # 詳細情報
                with st.expander("📖 3D解析の詳細データ"):
                    detail_cols = st.columns(2)
                    with detail_cols[0]:
                        st.markdown("**表面粗さ指標:**")
                        st.write(f"- 標準偏差: {roughness['std']:.2f}")
                        st.write(f"- 平均絶対偏差: {roughness['mad']:.2f}")
                        st.write(f"- 勾配平均: {roughness['gradient']:.2f}")
                        st.write(f"- ラプラシアン分散: {roughness['laplacian']:.2f}")
                    with detail_cols[1]:
                        st.markdown("**理想的な肌の基準:**")
                        st.write(f"- フラクタル次元: {SKIN_FD_IDEAL_MIN}~{SKIN_FD_IDEAL_MAX}")
                        st.write(f"- 現在の値: {fd_3d:.4f}")
                        if fd_3d < SKIN_FD_IDEAL_MIN:
                            st.write(f"- 差分: -{(SKIN_FD_IDEAL_MIN - fd_3d):.4f} (滑らかすぎ)")
                        elif fd_3d > SKIN_FD_IDEAL_MAX:
                            st.write(f"- 差分: +{(fd_3d - SKIN_FD_IDEAL_MAX):.4f} (粗すぎ)")
                        else:
                            st.write(f"- ✅ 理想範囲内")
            else:
                st.error("❌ 3D表面フラクタル次元の計算に失敗しました。")
            
            st.markdown("---")
        
        # 2D解析（標準）
        if "2D" in analysis_mode or "両方" in analysis_mode:
            st.subheader("📊 2Dフラクタル次元比較分析")
            
            with st.spinner('🔍 3つの画像を解析中...'):
                fd_low, _, _, _, _ = fractal_dimension(low_img, threshold_value, use_otsu)
                fd_enhanced, sizes, counts, binary, used_threshold = fractal_dimension(enhanced_img, threshold_value, use_otsu)
                fd_high, _, _, _, _ = fractal_dimension(high_img, threshold_value, use_otsu)
            
            # エラーチェック
            if fd_low is None or fd_enhanced is None or fd_high is None:
                st.error("❌ エラー: フラクタル次元の計算に失敗しました。")
            else:
                # フラクタル次元の比較表示
                fd_compare_cols = st.columns(3)
                with fd_compare_cols[0]:
                    st.metric("📱 低画質", f"{fd_low:.4f}")
                with fd_compare_cols[1]:
                    delta = fd_enhanced - fd_low
                    st.metric("🤖 AI補正後", f"{fd_enhanced:.4f}", delta=f"{delta:+.4f}")
                with fd_compare_cols[2]:
                    st.metric("📷 高画質(目標)", f"{fd_high:.4f}")
                
                # フラクタル次元比較グラフ
                st.subheader("📈 フラクタル次元の推移")
                fig_comparison = plot_fractal_comparison(fd_low, fd_enhanced, fd_high)
                st.pyplot(fig_comparison)
                plt.close(fig_comparison)
                
                # AI補正の評価
                evaluation, eval_text = evaluate_ai_correction(fd_low, fd_enhanced, fd_high)
                
                if evaluation:
                    st.subheader("🎯 AI補正精度評価")
                    
                    eval_cols = st.columns(4)
                    with eval_cols[0]:
                        st.metric("改善率", f"{evaluation['improvement_rate']:.1f}%")
                    with eval_cols[1]:
                        st.metric("評価ランク", eval_text)
                    with eval_cols[2]:
                        st.metric("目標との差", f"{evaluation['target_diff']:.4f}")
                    with eval_cols[3]:
                        st.metric("改善度", f"{evaluation['improvement']:.4f}")
                    
                    # 評価の詳細説明
                    with st.expander("📖 評価基準の詳細"):
                        st.markdown("""
                        **評価ランク:**
                        - 🟢 **S (90%以上)**: 優秀 - 高画質にほぼ完全に近づいています
                        - 🔵 **A (75-90%)**: 良好 - 高い補正精度を達成しています
                        - 🟡 **B (60-75%)**: 普通 - 一定の補正効果が見られます
                        - 🟠 **C (40-60%)**: 要改善 - 補正効果が限定的です
                        - 🔴 **D (40%未満)**: 不良 - 補正が不十分です
                        
                        **改善率**: 低画質から高画質への距離のうち、どれだけ近づいたかを示します
                        """)
    else:
        st.warning("⚠️ 高画質画像がアップロードされていません。補完をスキップします。")
        target_img = low_img
        
        st.markdown("---")
        
        # 3D表面解析（単独画像）
        if "3D" in analysis_mode or "両方" in analysis_mode:
            st.subheader("🌐 3D表面解析（肌質評価）")
            
            with st.spinner('🔍 3D表面凹凸を解析中...'):
                roughness = calculate_surface_roughness(target_img)
                fd_3d, box_sizes_3d, counts_3d = fractal_dimension_3d_surface(target_img)
            
            if fd_3d is not None:
                skin_eval, skin_eval_text = evaluate_skin_quality(fd_3d, roughness)
                
                st.success(f"✅ 3D表面フラクタル次元: **{fd_3d:.4f}**")
                
                eval_cols = st.columns(4)
                with eval_cols[0]:
                    st.metric("🎯 総合スコア", f"{skin_eval['total_score']:.1f}点")
                with eval_cols[1]:
                    st.metric("📊 評価ランク", skin_eval_text)
                with eval_cols[2]:
                    if skin_eval['in_ideal_range']:
                        st.metric("✅ 理想範囲", "範囲内", delta="良好")
                    else:
                        st.metric("⚠️ 理想範囲", "範囲外", delta=skin_eval['fd_comment'])
                with eval_cols[3]:
                    st.metric("📏 FD評価", f"{skin_eval['fd_score']:.1f}点")
                
                fig_3d_analysis = plot_3d_fractal_analysis(box_sizes_3d, counts_3d, fd_3d)
                st.pyplot(fig_3d_analysis)
                plt.close(fig_3d_analysis)
                
                fig_roughness = plot_surface_roughness(target_img, roughness)
                st.pyplot(fig_roughness)
                plt.close(fig_roughness)
            else:
                st.error("❌ 3D表面フラクタル次元の計算に失敗しました。")
            
            st.markdown("---")
        
        # 2D解析（単独画像）
        if "2D" in analysis_mode or "両方" in analysis_mode:
            st.subheader("📈 2Dフラクタル次元解析")
            
            with st.spinner('🔍 解析中...'):
                fd_enhanced, sizes, counts, binary, used_threshold = fractal_dimension(target_img, threshold_value, use_otsu)
            
            if fd_enhanced is None:
                st.error("❌ エラー: フラクタル次元の計算に失敗しました。")
            else:
                # フラクタル次元表示
                col_fd1, col_fd2 = st.columns(2)
                with col_fd1:
                    st.metric("フラクタル次元", f"{fd_enhanced:.4f}")
                with col_fd2:
                    st.metric("使用した閾値", f"{used_threshold}")
                
                # ボックスカウントグラフ（サイズ縮小）
                st.subheader("📉 ボックスカウント解析")
                fig_boxcount, ax = plt.subplots(figsize=(7, 4))  # 8,5 → 7,4
                ax.plot(np.log(sizes), np.log(counts), marker="o", linewidth=2, markersize=8, color='#3498db')
                ax.set_xlabel("log(ボックスサイズ)", fontsize=10)
                ax.set_ylabel("log(カウント数)", fontsize=10)
                ax.set_title("ボックスカウント法によるフラクタル次元", fontsize=11, pad=10)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_boxcount)
                plt.close(fig_boxcount)
                
                # 二値化画像表示
                st.subheader("🖼️ 二値化画像")
                st.image(binary, caption="二値化結果", use_container_width=True, clamp=True)
                
                # 3Dグラフ出力
                st.subheader("🌐 3D フラクタル表面")
                fig_3d = generate_3d_surface(binary)
                st.pyplot(fig_3d)
                plt.close(fig_3d)
                
                # 空間占有率
                black_rate, white_rate = calculate_occupancy(binary)
                
                st.subheader("📊 空間占有率")
                col_occ1, col_occ2 = st.columns(2)
                with col_occ1:
                    st.metric("黒ピクセル", f"{black_rate:.2f}%")
                with col_occ2:
                    st.metric("白ピクセル", f"{white_rate:.2f}%")
                
                # 円グラフ（サイズ縮小）
                fig_pie, ax_pie = plt.subplots(figsize=(5, 5))  # 6,6 → 5,5
                ax_pie.pie([black_rate, white_rate], labels=["黒", "白"], autopct="%.1f%%", 
                           startangle=90, colors=['#2c3e50', '#ecf0f1'], textprops={'fontsize': 10})
                ax_pie.set_title("空間占有率の分布", fontsize=11, pad=10)
                st.pyplot(fig_pie)
                plt.close(fig_pie)
                
                st.markdown("---")
                
                # 結果の保存セクション
                st.subheader("💾 結果の保存")
                
                # CSVデータ作成
                results_data = {
                    "解析日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "フラクタル次元(AI補正後)": fd_enhanced,
                    "閾値": used_threshold,
                    "大津法使用": use_otsu,
                    "黒ピクセル率(%)": black_rate,
                    "白ピクセル率(%)": white_rate,
                    "データ拡張": use_augmentation,
                    "モデル精度": model_score if model_score else "N/A"
                }
                
                # 高画質画像がある場合は比較データも追加（変数が定義されている場合のみ）
                # これは比較画像解析時のみ有効
                # results_data["フラクタル次元(低画質)"] = fd_low (単独画像では未定義)
                # results_data["フラクタル次元(高画質)"] = fd_high (単独画像では未定義)
                
                csv_data = create_results_csv(results_data)
                
                # ダウンロードボタン
                download_cols = st.columns(4)
                
                with download_cols[0]:
                    st.download_button(
                        label="📄 CSV出力",
                        data=csv_data,
                        file_name=f"fractal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with download_cols[1]:
                    img_bytes = save_image_to_bytes(target_img)
                    if img_bytes:
                        st.download_button(
                            label="🖼️ 画像",
                            data=img_bytes,
                            file_name=f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                
                with download_cols[2]:
                    graph_bytes = fig_to_bytes(fig_boxcount)
                    st.download_button(
                        label="📊 ボックスカウント",
                        data=graph_bytes,
                        file_name=f"boxcount_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                
                with download_cols[3]:
                    graph_3d_bytes = fig_to_bytes(fig_3d)
                    st.download_button(
                        label="🌐 3Dグラフ",
                        data=graph_3d_bytes,
                        file_name=f"3d_surface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )

else:
    st.info("👆 低画質画像をアップロードして解析を開始してください")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>🔬 Fractal Analyzer V2 with AI Enhancement | Powered by Streamlit & OpenCV</p>
</div>
""", unsafe_allow_html=True)
