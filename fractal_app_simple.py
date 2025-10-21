# ============================================================
# シンプル フラクタル次元解析アプリ
# 画像のフラクタル次元を計算します
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# GPU対応（オプション）
USE_CUPY = False
xp = np
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

# ============================================================
# フラクタル次元計算（Box-counting法）
# ============================================================
def calculate_fractal_dimension(img_bgr, scales=(2,4,8,16,32,64)):
    """
    画像のフラクタル次元を計算
    
    Args:
        img_bgr: BGR画像（OpenCV形式）
        scales: ボックスサイズのリスト
    
    Returns:
        D: フラクタル次元
        scales_used: 使用したスケール
        Nh_values: 各スケールでのボックス数
    """
    # グレースケール変換
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = img_gray.shape

    Nh_vals = []
    valid_scales = []
    
    for h in scales:
        # スケールに合わせて画像をクロップ
        Hc = (H // h) * h
        Wc = (W // h) * h
        if Hc < h or Wc < h:
            continue

        gray_crop = img_gray[:Hc, :Wc]
        arr = to_xp(gray_crop)

        # ブロックに分割
        new_shape = (Hc//h, h, Wc//h, h)
        blocks = arr.reshape(new_shape).transpose(0,2,1,3)

        # 各ブロックの標準偏差を計算
        mean_blk = blocks.mean(axis=(2,3))
        sq_mean = (blocks**2).mean(axis=(2,3))
        std_blk = xp.sqrt(xp.maximum(0, sq_mean - mean_blk**2))

        # 標準偏差が0より大きいブロック数をカウント
        count = int((std_blk > 0).sum())
        
        Nh_vals.append(count)
        valid_scales.append(h)

    if len(valid_scales) < 2:
        return 0.0, [], []

    # log-log回帰でフラクタル次元を計算
    valid_scales = to_host(np.array(valid_scales))
    Nh_vals = to_host(np.array(Nh_vals))
    
    log_h = np.log(valid_scales)
    log_Nh = np.log(Nh_vals + 1)
    
    # 線形回帰
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    slope, intercept = np.linalg.lstsq(A, log_Nh, rcond=None)[0]
    
    D = -slope  # フラクタル次元

    return D, valid_scales, Nh_vals

# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(
        page_title="フラクタル次元解析",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 フラクタル次元解析アプリ")
    st.markdown("画像のフラクタル次元を計算します")

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # GPU使用状況
        if USE_CUPY:
            st.success("✅ GPU使用中（CuPy）")
        else:
            st.info("💻 CPU使用中")
        
        st.markdown("---")
        
        # スケール設定
        st.subheader("📏 スケール設定")
        scale_min = st.slider("最小スケール", 2, 16, 2)
        scale_max = st.slider("最大スケール", 32, 128, 64)
        
        # スケールを生成（2の累乗）
        scales = []
        scale = scale_min
        while scale <= scale_max:
            scales.append(scale)
            scale *= 2
        
        st.write(f"使用スケール: {scales}")

    # メインエリア
    st.markdown("---")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "📁 画像をアップロード",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="PNG、JPG、BMPファイルに対応"
    )

    if uploaded_file is not None:
        # 画像を読み込み
        file_bytes = uploaded_file.read()
        arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            st.error("❌ 画像の読み込みに失敗しました")
            return
        
        # 画像情報
        st.subheader("📷 画像情報")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 
                    caption="アップロード画像",
                    use_container_width=True)
        
        with col2:
            st.write(f"- **サイズ**: {img_bgr.shape[1]} × {img_bgr.shape[0]} ピクセル")
            st.write(f"- **チャンネル**: {img_bgr.shape[2]}")
            st.write(f"- **ファイル名**: {uploaded_file.name}")
        
        st.markdown("---")
        
        # 解析実行
        if st.button("🔬 フラクタル次元を計算", type="primary", use_container_width=True):
            with st.spinner("計算中..."):
                start_time = st.empty()
                t0 = st.session_state.get('t0', 0)
                
                # フラクタル次元計算
                import time
                t_start = time.time()
                D, scales_used, Nh_values = calculate_fractal_dimension(img_bgr, tuple(scales))
                t_end = time.time()
                
                elapsed = t_end - t_start
            
            # 結果表示
            st.success("✅ 計算完了！")
            
            st.markdown("---")
            st.subheader("📊 解析結果")
            
            # メトリクス表示
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("フラクタル次元", f"{D:.4f}")
            
            with col2:
                st.metric("計算時間", f"{elapsed:.3f}秒")
            
            with col3:
                st.metric("使用スケール数", f"{len(scales_used)}個")
            
            st.markdown("---")
            
            # グラフ表示
            if len(scales_used) >= 2:
                st.subheader("📈 Log-Logプロット")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                log_h = np.log(scales_used)
                log_Nh = np.log(Nh_values + 1)
                
                # データプロット
                ax.scatter(log_h, log_Nh, s=100, alpha=0.6, label='データ')
                
                # 回帰直線
                A = np.vstack([log_h, np.ones(len(log_h))]).T
                slope, intercept = np.linalg.lstsq(A, log_Nh, rcond=None)[0]
                
                x_line = np.array([log_h.min(), log_h.max()])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r--', linewidth=2, 
                       label=f'回帰直線 (傾き={slope:.3f})')
                
                ax.set_xlabel('log(スケール h)', fontsize=12)
                ax.set_ylabel('log(ボックス数 N(h))', fontsize=12)
                ax.set_title(f'フラクタル次元: D = {D:.4f}', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # 詳細データ
                with st.expander("📋 詳細データを表示"):
                    import pandas as pd
                    df = pd.DataFrame({
                        'スケール h': scales_used,
                        'ボックス数 N(h)': Nh_values,
                        'log(h)': log_h,
                        'log(N(h))': log_Nh
                    })
                    st.dataframe(df, use_container_width=True)
    
    else:
        st.info("👆 画像をアップロードして解析を開始してください")

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🔬 フラクタル次元解析アプリ | Box-counting法</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
