"""
ほこり画像フラクタル解析アプリ
Box-counting法によるフラクタル次元の計算と可視化
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io
from PIL import Image

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False

# ============================================================
# Box-counting法によるフラクタル次元計算
# ============================================================

def preprocess_dust_image(image):
    """
    ほこり画像の前処理
    
    Args:
        image: BGR画像
    
    Returns:
        二値化画像
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # ノイズ除去
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 適応的二値化（ほこりのコントラストを強調）
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # モルフォロジー処理（ノイズ除去）
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary


def box_counting_fractal_dimension(binary_image, min_box_size=2, max_box_size=None):
    """
    Box-counting法によるフラクタル次元の計算
    
    Args:
        binary_image: 二値化画像（0 or 255）
        min_box_size: 最小ボックスサイズ
        max_box_size: 最大ボックスサイズ（Noneの場合は画像サイズの1/2）
    
    Returns:
        dict: {
            'fractal_dimension': フラクタル次元,
            'box_sizes': ボックスサイズのリスト,
            'box_counts': カウント数のリスト,
            'log_box_sizes': log(1/ボックスサイズ),
            'log_box_counts': log(カウント数)
        }
    """
    height, width = binary_image.shape
    
    # 最大ボックスサイズの設定
    if max_box_size is None:
        max_box_size = min(height, width) // 2
    
    # ボックスサイズのリスト（2のべき乗）
    box_sizes = []
    size = min_box_size
    while size <= max_box_size:
        box_sizes.append(size)
        size *= 2
    
    # 各ボックスサイズでカウント
    box_counts = []
    
    for box_size in box_sizes:
        count = 0
        
        # 画像をボックスで分割してカウント
        for i in range(0, height, box_size):
            for j in range(0, width, box_size):
                # ボックス内の領域を取得
                box = binary_image[i:i+box_size, j:j+box_size]
                
                # ボックス内に白ピクセル（ほこり）があればカウント
                if np.any(box > 0):
                    count += 1
        
        box_counts.append(count)
    
    # 対数変換
    log_box_sizes = np.log(1.0 / np.array(box_sizes))
    log_box_counts = np.log(np.array(box_counts))
    
    # 線形回帰でフラクタル次元を計算
    # log(N) = D * log(1/r) + C
    # D = フラクタル次元
    coefficients = np.polyfit(log_box_sizes, log_box_counts, 1)
    fractal_dimension = coefficients[0]
    
    # 相関係数を計算
    correlation = np.corrcoef(log_box_sizes, log_box_counts)[0, 1]
    
    # R²値を計算
    residuals = log_box_counts - (coefficients[0] * log_box_sizes + coefficients[1])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_box_counts - np.mean(log_box_counts))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'fractal_dimension': fractal_dimension,
        'box_sizes': box_sizes,
        'box_counts': box_counts,
        'log_box_sizes': log_box_sizes,
        'log_box_counts': log_box_counts,
        'regression_line': coefficients,
        'correlation': correlation,
        'r_squared': r_squared
    }


def create_fractal_plot(result):
    """
    フラクタル次元の可視化グラフを作成
    
    Args:
        result: box_counting_fractal_dimensionの結果
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # グラフ1: Box-counting プロット（対数グラフ）
    ax1.plot(
        result['log_box_sizes'],
        result['log_box_counts'],
        'o-',
        markersize=8,
        linewidth=2,
        color='#2E86AB',
        label='測定データ'
    )
    
    # 回帰直線
    x_line = np.array(result['log_box_sizes'])
    y_line = result['regression_line'][0] * x_line + result['regression_line'][1]
    ax1.plot(
        x_line,
        y_line,
        '--',
        linewidth=2,
        color='#E63946',
        label=f'回帰直線 (傾き = {result["fractal_dimension"]:.4f})'
    )
    
    ax1.set_xlabel('log(1/ボックスサイズ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('log(カウント数)', fontsize=12, fontweight='bold')
    ax1.set_title('Box-counting法 フラクタル解析', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # グラフ2: 実際のボックスサイズとカウント数
    ax2.plot(
        result['box_sizes'],
        result['box_counts'],
        'o-',
        markersize=8,
        linewidth=2,
        color='#06A77D',
        label='カウント数'
    )
    
    ax2.set_xlabel('ボックスサイズ (pixels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('カウント数', fontsize=12, fontweight='bold')
    ax2.set_title('ボックスサイズ vs カウント数', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    # X軸を対数スケールに
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    return fig


def create_comprehensive_result_image(original_image, binary_image, result, interpretation):
    """
    解析結果を1枚の画像にまとめる
    
    Args:
        original_image: 元画像 (BGR)
        binary_image: 二値化画像
        result: box_counting_fractal_dimensionの結果
        interpretation: フラクタル次元の解釈
    
    Returns:
        PIL Image: 統合結果画像
    """
    # 大きなキャンバスを作成 (2000x1500)
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5], hspace=0.3, wspace=0.25)
    
    # 1. 元画像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('元画像', fontsize=16, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # 2. 二値化画像
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(binary_image, cmap='gray')
    ax2.set_title('二値化画像（白：ほこり）', fontsize=16, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # 3. Box-counting プロット（対数グラフ）
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        result['log_box_sizes'],
        result['log_box_counts'],
        'o-',
        markersize=10,
        linewidth=3,
        color='#2E86AB',
        label='測定データ'
    )
    
    # 回帰直線
    x_line = np.array(result['log_box_sizes'])
    y_line = result['regression_line'][0] * x_line + result['regression_line'][1]
    ax3.plot(
        x_line,
        y_line,
        '--',
        linewidth=3,
        color='#E63946',
        label=f'回帰直線 (傾き = {result["fractal_dimension"]:.4f})'
    )
    
    ax3.set_xlabel('log(1/ボックスサイズ)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('log(カウント数)', fontsize=14, fontweight='bold')
    ax3.set_title('Box-counting法 フラクタル解析', fontsize=16, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=12)
    
    # 4. ボックスサイズ vs カウント数
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        result['box_sizes'],
        result['box_counts'],
        'o-',
        markersize=10,
        linewidth=3,
        color='#06A77D',
        label='カウント数'
    )
    
    ax4.set_xlabel('ボックスサイズ (pixels)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('カウント数', fontsize=14, fontweight='bold')
    ax4.set_title('ボックスサイズ vs カウント数', fontsize=16, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=12)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # 5. フラクタル次元の大きな表示（下部全体）
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # 背景色
    ax5.add_patch(plt.Rectangle(
        (0.05, 0.2), 0.9, 0.6,
        facecolor=interpretation['color'],
        alpha=0.2,
        transform=ax5.transAxes,
        zorder=1
    ))
    
    # 左側の枠
    ax5.add_patch(plt.Rectangle(
        (0.05, 0.2), 0.05, 0.6,
        facecolor=interpretation['color'],
        transform=ax5.transAxes,
        zorder=2
    ))
    
    # フラクタル次元
    ax5.text(
        0.15, 0.7,
        'フラクタル次元',
        fontsize=20,
        fontweight='bold',
        transform=ax5.transAxes,
        va='center'
    )
    
    ax5.text(
        0.4, 0.7,
        f'{result["fractal_dimension"]:.4f}',
        fontsize=48,
        fontweight='bold',
        color=interpretation['color'],
        transform=ax5.transAxes,
        va='center'
    )
    
    # 解釈情報
    ax5.text(
        0.15, 0.4,
        f'パターン: {interpretation["pattern"]}  |  分布: {interpretation["distribution"]}  |  複雑度: {interpretation["complexity"]}',
        fontsize=16,
        transform=ax5.transAxes,
        va='center'
    )
    
    # 統計情報
    ax5.text(
        0.7, 0.7,
        f'相関係数: {result["correlation"]:.4f}',
        fontsize=14,
        transform=ax5.transAxes,
        va='center'
    )
    
    ax5.text(
        0.7, 0.5,
        f'R²値: {result["r_squared"]:.4f}',
        fontsize=14,
        transform=ax5.transAxes,
        va='center'
    )
    
    ax5.text(
        0.7, 0.3,
        f'測定点: {len(result["box_sizes"])}点',
        fontsize=14,
        transform=ax5.transAxes,
        va='center'
    )
    
    # タイトル
    fig.suptitle(
        'ほこり画像フラクタル解析 - 総合結果',
        fontsize=24,
        fontweight='bold',
        y=0.98
    )
    
    # 画像バッファに保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    # PIL Imageに変換
    result_image = Image.open(buf)
    
    plt.close(fig)
    
    return result_image


def interpret_dust_fractal_dimension(fd):
    """
    ほこりのフラクタル次元を解釈
    
    Args:
        fd: フラクタル次元
    
    Returns:
        dict: 解釈結果
    """
    if fd < 1.3:
        pattern = "非常に単純"
        distribution = "まばら・均一"
        complexity = "低"
        color = "#06A77D"
    elif fd < 1.5:
        pattern = "やや単純"
        distribution = "やや均一"
        complexity = "中低"
        color = "#52B788"
    elif fd < 1.7:
        pattern = "標準的"
        distribution = "標準的"
        complexity = "中"
        color = "#FFB703"
    elif fd < 1.9:
        pattern = "やや複雑"
        distribution = "やや不均一"
        complexity = "中高"
        color = "#FB8500"
    else:
        pattern = "非常に複雑"
        distribution = "不均一・集中"
        complexity = "高"
        color = "#E63946"
    
    return {
        'pattern': pattern,
        'distribution': distribution,
        'complexity': complexity,
        'color': color
    }


# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(
        page_title="ほこりフラクタル解析",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 ほこり画像フラクタル解析アプリ")
    st.markdown("**Box-counting法によるフラクタル次元の計算と可視化**")
    
    st.markdown("---")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        st.subheader("前処理パラメータ")
        use_preprocessing = st.checkbox("前処理を使用", value=True)
        
        if use_preprocessing:
            blur_kernel = st.slider("ぼかしカーネルサイズ", 3, 15, 5, step=2)
            adaptive_block = st.slider("適応的二値化ブロックサイズ", 3, 31, 11, step=2)
            adaptive_c = st.slider("適応的二値化定数", 0, 10, 2)
        
        st.subheader("Box-counting パラメータ")
        min_box_size = st.slider("最小ボックスサイズ", 2, 16, 2)
        max_box_size_ratio = st.slider("最大ボックスサイズ（画像サイズに対する比率）", 0.1, 0.5, 0.25)
        
        st.markdown("---")
        st.markdown("### 📊 フラクタル次元について")
        st.markdown("""
        **フラクタル次元 (FD)** は、ほこりの分布の複雑さを表します：
        
        - **FD < 1.5**: 単純な分布（まばら）
        - **FD ≈ 1.5-1.7**: 標準的な分布
        - **FD > 1.7**: 複雑な分布（集中・不均一）
        
        値が大きいほど、ほこりが不均一に集中しています。
        """)
    
    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 画像アップロード")
        uploaded_file = st.file_uploader(
            "ほこりの画像をアップロードしてください",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="JPG、PNG、BMP形式の画像ファイルをアップロードできます"
        )
    
    if uploaded_file is not None:
        # 画像読み込み
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        with col1:
            st.subheader("🖼️ 元画像")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # 画像情報
            height, width = image.shape[:2]
            st.info(f"**画像サイズ**: {width} × {height} pixels")
        
        # 前処理
        if use_preprocessing:
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # ぼかし
            denoised = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            
            # 適応的二値化
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                adaptive_block,
                adaptive_c
            )
            
            # モルフォロジー処理
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            # 単純な二値化
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        with col2:
            st.subheader("🔲 二値化画像")
            st.image(binary, use_container_width=True, caption="ほこり検出（白：ほこり、黒：背景）")
            
            # ほこりの割合
            dust_ratio = (np.sum(binary > 0) / binary.size) * 100
            st.info(f"**ほこり検出割合**: {dust_ratio:.2f}%")
        
        st.markdown("---")
        
        # フラクタル次元計算
        with st.spinner('🔬 フラクタル次元を計算中...'):
            max_box_size = int(min(height, width) * max_box_size_ratio)
            result = box_counting_fractal_dimension(
                binary,
                min_box_size=min_box_size,
                max_box_size=max_box_size
            )
        
        # 結果表示
        st.header("📊 解析結果")
        
        # フラクタル次元の大きな表示
        interpretation = interpret_dust_fractal_dimension(result['fractal_dimension'])
        
        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
        
        with result_col1:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {interpretation['color']}22 0%, {interpretation['color']}44 100%);
                    border-left: 5px solid {interpretation['color']};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <h2 style="margin: 0; color: #1a1a1a;">フラクタル次元</h2>
                    <h1 style="margin: 10px 0; color: {interpretation['color']}; font-size: 3em;">
                        {result['fractal_dimension']:.4f}
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with result_col2:
            st.markdown(
                f"""
                <div style="
                    background: #f8f9fa;
                    border-left: 5px solid #6c757d;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <h3 style="margin: 0; color: #1a1a1a;">相関係数</h3>
                    <h2 style="margin: 10px 0; color: #2E86AB; font-size: 2em;">
                        {result['correlation']:.4f}
                    </h2>
                    <p style="margin: 0; color: #6c757d;">R² = {result['r_squared']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with result_col3:
            st.markdown(
                f"""
                <div style="
                    background: #f8f9fa;
                    border-left: 5px solid #6c757d;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <h3 style="margin: 0; color: #1a1a1a;">測定点数</h3>
                    <h2 style="margin: 10px 0; color: #06A77D; font-size: 2em;">
                        {len(result['box_sizes'])} 点
                    </h2>
                    <p style="margin: 0; color: #6c757d;">
                        {min_box_size}～{result['box_sizes'][-1]} pixels
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # 解釈
        st.subheader("📋 フラクタル次元の解釈")
        
        interp_col1, interp_col2, interp_col3 = st.columns(3)
        
        with interp_col1:
            st.metric("パターン", interpretation['pattern'])
        
        with interp_col2:
            st.metric("分布状態", interpretation['distribution'])
        
        with interp_col3:
            st.metric("複雑度", interpretation['complexity'])
        
        st.markdown("---")
        
        # グラフ表示
        st.subheader("📈 フラクタル解析グラフ")
        
        fig = create_fractal_plot(result)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # 統合結果画像の作成と表示
        st.subheader("🖼️ 統合結果画像")
        
        with st.spinner('📸 統合結果画像を作成中...'):
            comprehensive_image = create_comprehensive_result_image(
                image,
                binary,
                result,
                interpretation
            )
        
        # 画像表示
        st.image(comprehensive_image, use_container_width=True, caption="解析結果の統合画像")
        
        # ダウンロードボタン
        buf = io.BytesIO()
        comprehensive_image.save(buf, format='PNG')
        buf.seek(0)
        
        st.download_button(
            label="📥 統合結果画像をダウンロード",
            data=buf,
            file_name=f"dust_fractal_analysis_{result['fractal_dimension']:.4f}.png",
            mime="image/png",
            help="元画像、二値化画像、グラフ、フラクタル次元を含む統合画像をダウンロード"
        )
        
        st.markdown("---")
        with st.expander("📊 詳細データを表示"):
            data_col1, data_col2 = st.columns(2)
            
            with data_col1:
                st.markdown("**ボックスサイズとカウント数**")
                import pandas as pd
                df = pd.DataFrame({
                    'ボックスサイズ (pixels)': result['box_sizes'],
                    'カウント数': result['box_counts'],
                    'log(1/r)': result['log_box_sizes'],
                    'log(N)': result['log_box_counts']
                })
                st.dataframe(df, use_container_width=True)
            
            with data_col2:
                st.markdown("**統計情報**")
                st.write(f"- フラクタル次元: **{result['fractal_dimension']:.6f}**")
                st.write(f"- 相関係数: **{result['correlation']:.6f}**")
                st.write(f"- R²値: **{result['r_squared']:.6f}**")
                st.write(f"- 回帰直線の傾き: **{result['regression_line'][0]:.6f}**")
                st.write(f"- 回帰直線の切片: **{result['regression_line'][1]:.6f}**")
                st.write(f"- 測定点数: **{len(result['box_sizes'])}**")
                st.write(f"- ボックスサイズ範囲: **{min_box_size} ～ {result['box_sizes'][-1]} pixels**")
    
    else:
        st.info("👆 左側のエリアから画像をアップロードしてください")
        
        # 使い方ガイド
        st.markdown("---")
        st.subheader("📖 使い方")
        
        st.markdown("""
        1. **画像をアップロード**: ほこりを撮影した画像をアップロードします
        2. **パラメータ調整**: 左サイドバーで前処理やBox-countingのパラメータを調整できます
        3. **結果確認**: フラクタル次元とグラフが自動的に表示されます
        
        ### Box-counting法について
        
        Box-counting法は、画像を様々なサイズのボックスで分割し、
        ほこりを含むボックスの数をカウントすることで、
        ほこりの分布の複雑さ（フラクタル次元）を計算します。
        
        **フラクタル次元の意味**:
        - 値が小さい（<1.5）: ほこりがまばらで単純な分布
        - 値が中程度（1.5-1.7）: 標準的な分布
        - 値が大きい（>1.7）: ほこりが集中的で複雑な分布
        """)


if __name__ == "__main__":
    main()
