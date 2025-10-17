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

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# データ拡張関数
# ----------------------------
def augment_image(image):
    """画像を回転・反転して学習データを増やす"""
    augmented = [image]
    # 90度回転
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    # 180度回転
    augmented.append(cv2.rotate(image, cv2.ROTATE_180))
    # 270度回転
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    # 水平反転
    augmented.append(cv2.flip(image, 1))
    # 垂直反転
    augmented.append(cv2.flip(image, 0))
    return augmented

# ----------------------------
# AI補完モデルの定義（データ拡張対応）
# ----------------------------
def train_image_enhancer(low_quality_images, high_quality_images, use_augmentation=True):
    X, y = [], []
    
    # データ拡張の適用
    if use_augmentation:
        aug_low, aug_high = [], []
        for low, high in zip(low_quality_images, high_quality_images):
            aug_low.extend(augment_image(low))
            aug_high.extend(augment_image(high))
        low_quality_images = aug_low
        high_quality_images = aug_high
    
    for low, high in zip(low_quality_images, high_quality_images):
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
    
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # 精度評価
    if len(X_test) > 0:
        score = model.score(X_test, y_test)
    else:
        score = 0.0
    
    return model, score

def enhance_image(model, low_quality_image):
    low_flat = low_quality_image.flatten() / 255.0
    pred = model.predict([low_flat])[0]
    enhanced = np.clip(pred * 255, 0, 255).reshape(low_quality_image.shape).astype(np.uint8)
    return enhanced

# ----------------------------
# フラクタル次元(ボックスカウント法・閾値調整対応)
# ----------------------------
def fractal_dimension(image, threshold_value=128, use_otsu=False):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 閾値処理
    if use_otsu:
        threshold_value, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

    sizes = 2 ** np.arange(1, 8)
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
# 3Dグラフ生成（図を返す）
# ----------------------------
def generate_3d_surface(binary_image):
    h, w = binary_image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = binary_image.astype(np.float32) / 255.0 * 10  # 明度を高さに変換
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title("3D フラクタル表面 (明度ベース)")
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
    
    # データ拡張オプション
    use_augmentation = st.checkbox("データ拡張を使用", value=True, 
                                   help="学習データを回転・反転して6倍に増やします")
    
    # 閾値設定
    st.subheader("二値化設定")
    use_otsu = st.checkbox("大津の二値化を使用", value=False,
                          help="自動で最適な閾値を計算します")
    
    threshold_value = 128
    if not use_otsu:
        threshold_value = st.slider("手動閾値", 0, 255, 128,
                                   help="二値化の閾値を手動で設定します")

# ファイルアップロード
col1, col2 = st.columns(2)
with col1:
    uploaded_low = st.file_uploader("📁 低画質画像をアップロード", type=["jpg", "png", "bmp"])
with col2:
    uploaded_high = st.file_uploader("📁 高画質画像(学習用)をアップロード", type=["jpg", "png", "bmp"])

if uploaded_low is not None:
    low_img = cv2.imdecode(np.frombuffer(uploaded_low.read(), np.uint8), cv2.IMREAD_COLOR)
    
    st.markdown("---")
    
    # AI画像補完
    enhanced_img = None
    model_score = None
    
    if uploaded_high is not None:
        high_img = cv2.imdecode(np.frombuffer(uploaded_high.read(), np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner('🤖 AI学習中...'):
            model, model_score = train_image_enhancer([low_img], [high_img], use_augmentation)
            enhanced_img = enhance_image(model, low_img)
        
        st.success(f"✅ 学習完了！ モデル精度: {model_score:.3f}")
        
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
    else:
        st.warning("⚠️ 高画質画像がアップロードされていません。補完をスキップします。")
        target_img = low_img
    
    st.markdown("---")
    
    # フラクタル解析
    st.subheader("📈 フラクタル次元解析")
    
    with st.spinner('🔍 解析中...'):
        fd, sizes, counts, binary, used_threshold = fractal_dimension(target_img, threshold_value, use_otsu)
    
    # 異常検出
    if fd is None:
        st.error("❌ エラー: フラクタル次元の計算に失敗しました。画像や閾値を確認してください。")
        st.stop()
    
    # フラクタル次元表示
    col_fd1, col_fd2 = st.columns(2)
    with col_fd1:
        st.metric("フラクタル次元", f"{fd:.4f}")
    with col_fd2:
        st.metric("使用した閾値", f"{used_threshold}")
    
    # ボックスカウントグラフ
    fig_boxcount, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.log(sizes), np.log(counts), marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("log(ボックスサイズ)")
    ax.set_ylabel("log(カウント数)")
    ax.set_title("ボックスカウント法によるフラクタル次元")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_boxcount)
    
    # 二値化画像表示
    st.subheader("🖼️ 二値化画像")
    st.image(binary, caption="二値化結果", use_container_width=True, clamp=True)
    
    # 3Dグラフ出力
    st.subheader("🌐 3D フラクタル表面")
    fig_3d = generate_3d_surface(binary)
    st.pyplot(fig_3d)
    
    # 空間占有率
    black_rate, white_rate = calculate_occupancy(binary)
    
    st.subheader("📊 空間占有率")
    col_occ1, col_occ2 = st.columns(2)
    with col_occ1:
        st.metric("黒ピクセル", f"{black_rate:.2f}%")
    with col_occ2:
        st.metric("白ピクセル", f"{white_rate:.2f}%")
    
    # 円グラフ
    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
    ax_pie.pie([black_rate, white_rate], labels=["黒", "白"], autopct="%.1f%%", 
               startangle=90, colors=['#2c3e50', '#ecf0f1'])
    ax_pie.set_title("空間占有率の分布")
    st.pyplot(fig_pie)
    
    st.markdown("---")
    
    # 結果の保存セクション
    st.subheader("💾 結果の保存")
    
    # CSVデータ作成
    results_data = {
        "解析日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "フラクタル次元": fd,
        "閾値": used_threshold,
        "大津法使用": use_otsu,
        "黒ピクセル率(%)": black_rate,
        "白ピクセル率(%)": white_rate,
        "データ拡張": use_augmentation,
        "モデル精度": model_score if model_score else "N/A"
    }
    
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
        if enhanced_img is not None:
            img_bytes = save_image_to_bytes(enhanced_img)
            if img_bytes:
                st.download_button(
                    label="🖼️ 補完画像",
                    data=img_bytes,
                    file_name=f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
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
