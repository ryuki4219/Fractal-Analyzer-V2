import streamlit as st
import cv2
<<<<<<< Updated upstream
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
=======
from PIL import Image
import io
import os
import base64
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
from skimage import filters, color
from skimage.feature import canny

# matplotlibで日本語フォントを設定（文字化け対策）
import matplotlib
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
@st.cache_data
def load_image_from_bytes(bytes_data: bytes) -> np.ndarray:
    # バイトデータから BGR(OpenCV) 画像を返す（キャッシュ対応）
    img = Image.open(io.BytesIO(bytes_data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1].copy()  # RGB->BGR
    return arr

def load_image_bytes(file) -> np.ndarray:
    # Streamlit の UploadedFile から BGR(OpenCV) 画像を返す
    bytes_data = file.read()
    return load_image_from_bytes(bytes_data)


@st.cache_data
def resize_image(img: np.ndarray, max_side: float):
    # 最長辺が max_side を超える場合リサイズする
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side and max_side > 0:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


@st.cache_data
def binarize_image_gray(gray: np.ndarray, thresh: float):
    # thresh は 0..255 の実数値。ここでは固定閾値による二値化
    _, bw = cv2.threshold(gray.astype('uint8'), thresh, 255, cv2.THRESH_BINARY)
    return bw


def adaptive_binarize(gray: np.ndarray, block_size: int = 11, c: int = 2):
    # ガウシアン適応閾値（ブロックサイズは奇数）
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    bw = cv2.adaptiveThreshold(gray.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block_size, c)
    return bw


def apply_gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(img, table)


def apply_brightness_offset(img: np.ndarray, beta: float) -> np.ndarray:
    if beta == 0:
        return img
    adjusted = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
    return adjusted


def apply_saturation_adjustment(img: np.ndarray, factor: float) -> np.ndarray:
    if np.isclose(factor, 1.0):
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted


def image_to_base64_png(img: np.ndarray) -> str:
    success, buffer = cv2.imencode('.png', img)
    if not success:
        return ''
    return base64.b64encode(buffer).decode('utf-8')


def build_html_report(run_timestamp: datetime, run_settings: dict, summary_df: pd.DataFrame, detail_records: list[dict]) -> str:
    timestamp_str = run_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    summary_html = summary_df.to_html(index=False, classes='summary-table', float_format=lambda x: f'{x:.4f}' if isinstance(x, (int, float, np.floating)) else x)

    styles = """
    <style>
    body { font-family: 'Segoe UI', sans-serif; margin: 2rem; background-color: #f9fafb; color: #1f2933; }
    h1 { color: #0f4c81; }
    h2 { color: #1f2933; margin-top: 2rem; }
    .meta, .summary { background: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(15,76,129,0.08); margin-bottom: 2rem; }
    .meta ul { list-style: none; padding-left: 0; }
    .meta li { margin-bottom: 0.4rem; }
    .images { display: flex; flex-wrap: wrap; gap: 1rem; }
    .images figure { flex: 1 1 200px; text-align: center; background: #f1f5f9; padding: 1rem; border-radius: 12px; }
    .images img { max-width: 100%; border-radius: 8px; box-shadow: 0 10px 20px rgba(15,76,129,0.15); }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.85rem; margin-right: 0.4rem; }
    .badge-ok { background: #d1fae5; color: #047857; }
    .badge-ng { background: #fee2e2; color: #b91c1c; }
    .summary-table { width: 100%; border-collapse: collapse; }
    .summary-table th, .summary-table td { border: 1px solid #d1d5db; padding: 0.6rem; text-align: center; }
    .summary-table th { background: #e5f0ff; }
    .card { background: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 12px 30px rgba(15,76,129,0.10); margin-bottom: 2rem; }
    .metrics { margin-top: 1rem; }
    .metrics table { width: 100%; border-collapse: collapse; }
    .metrics th, .metrics td { border: 1px solid #e5e7eb; padding: 0.5rem; }
    .metrics th { background: #f3f4f6; text-align: left; }
    footer { text-align: center; color: #6b7280; margin-top: 3rem; font-size: 0.85rem; }
    </style>
    """

    meta_items = ''.join(
        f"<li><strong>{key}</strong>: {value}</li>" for key, value in run_settings.items()
    )

    detail_sections = []
    for record in detail_records:
        status_badge = '<span class="badge badge-ok">正常</span>' if not record['fail_flag'] else '<span class="badge badge-ng">異常</span>'
        reasons = record['fail_reasons'] if record['fail_reasons'] else '特記事項なし'
        pred_fractal = record['pred_fractal'] if record['pred_fractal'] is not None else 'N/A'
        pred_occupancy = record['pred_occupancy'] if record['pred_occupancy'] is not None else 'N/A'
        detail_sections.append(f"""
        <section class="card">
            <h2>{record['filename']} {status_badge}</h2>
            <div class="images">
                <figure>
                    <img src="data:image/png;base64,{record['original_b64']}" alt="Original">
                    <figcaption>元画像（リサイズ後）</figcaption>
                </figure>
                <figure>
                    <img src="data:image/png;base64,{record['processed_b64']}" alt="Preprocessed">
                    <figcaption>前処理後画像</figcaption>
                </figure>
                <figure>
                    <img src="data:image/png;base64,{record['binary_b64']}" alt="Binary">
                    <figcaption>二値化画像</figcaption>
                </figure>
            </div>
            <div class="metrics">
                <table>
                    <tr><th>フラクタル次元</th><td>{record['fractal']:.4f}</td></tr>
                    <tr><th>空間占有率</th><td>{record['occupancy']*100:.2f}%</td></tr>
                    <tr><th>予測フラクタル次元</th><td>{pred_fractal}</td></tr>
                    <tr><th>予測空間占有率</th><td>{pred_occupancy}</td></tr>
                    <tr><th>閾値方式</th><td>{record['threshold_mode']}</td></tr>
                    <tr><th>閾値値</th><td>{record['threshold_value'] if record['threshold_value'] is not None else 'N/A'}</td></tr>
                    <tr><th>適応閾値ブロックサイズ</th><td>{record['adaptive_block_size'] if record['adaptive_block_size'] is not None else 'N/A'}</td></tr>
                    <tr><th>適応閾値C値</th><td>{record['adaptive_c'] if record['adaptive_c'] is not None else 'N/A'}</td></tr>
                    <tr><th>輝度補正</th><td>{'ON' if record['gamma_applied'] else 'OFF'} / γ={record['gamma_value'] if record['gamma_value'] is not None else '1.0'} / β={record['brightness_offset'] if record['brightness_offset'] is not None else 0}</td></tr>
                    <tr><th>彩度補正</th><td>{'ON' if record['saturation_applied'] else 'OFF'} / 倍率={record['saturation_factor'] if record['saturation_factor'] is not None else 1.0}</td></tr>
                    <tr><th>メモ</th><td>{reasons}</td></tr>
                </table>
            </div>
        </section>
        """)

    html = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>フラクタル解析レポート</title>
        {styles}
    </head>
    <body>
        <h1>フラクタル解析レポート</h1>
        <section class="meta">
            <h2>解析メタ情報</h2>
            <ul>
                <li><strong>生成日時</strong>: {timestamp_str}</li>
                {meta_items}
            </ul>
        </section>
        <section class="summary">
            <h2>解析結果サマリ</h2>
            {summary_html}
        </section>
        {''.join(detail_sections)}
        <footer>生成システム: フラクタル画像解析アプリ</footer>
    </body>
    </html>
    """
    return html


@st.cache_data
def boxcount_fractal_dim(bw: np.ndarray, sizes=None):
    # 白(255) を対象に箱ひき（box-counting法）でフラクタル次元を推定
    # bw: 二値画像（0 or 255）
    # sizes: list of box sizes to use (pixels)
    S = bw.shape
    if sizes is None:
        max_dim = max(S)
        min_dim = min(S)
        # 箱サイズは 2^k 系列で生成（最小3ポイント確保）
        if min_dim >= 8:
            sizes = np.array([2 ** i for i in range(1, int(np.log2(min_dim)) + 1)])
        else:
            sizes = np.array([1, 2, 4])
        sizes = sizes[sizes <= min_dim]
        if len(sizes) < 2:
            sizes = np.array([1, 2, 4, 8])
    counts = []
    for size in sizes:
        # 画像を size x size のブロックに分割して、白が含まれるブロックを数える
        nx = int(np.ceil(S[1] / size))
        ny = int(np.ceil(S[0] / size))
        count = 0
        for i in range(ny):
            for j in range(nx):
                y0 = i * size
                x0 = j * size
                block = bw[y0:y0 + size, x0:x0 + size]
                if np.any(block > 0):
                    count += 1
        counts.append(count)
    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    # fractal dimension D is slope of log(count) vs log(1/size)
    # linear regression via least squares
    # ゼロや負の値を除外
    valid_mask = (counts > 0) & (sizes > 0)
    if np.sum(valid_mask) < 2:
        # 有効なデータポイントが2つ未満の場合はフラクタル次元を計算できない
        return 0.0, sizes, counts
    
    sizes_valid = sizes[valid_mask]
    counts_valid = counts[valid_mask]
    
    logs = np.log(counts_valid)
    loginv = np.log(1.0 / sizes_valid)
    
    # 単純な線形回帰
    A = np.vstack([loginv, np.ones_like(loginv)]).T
    try:
        m, c = np.linalg.lstsq(A, logs, rcond=None)[0]
    except Exception:
        m = 0.0
    return float(m), sizes, counts


@st.cache_data
def compute_spatial_occupancy(bw: np.ndarray):
    # 白（255）が占める割合
    total = bw.size
    white = np.count_nonzero(bw > 0)
    return float(white / total)


@st.cache_data
def extract_features_from_image(img_bgr: np.ndarray, bw: np.ndarray, fractal_dim: float):
    # シンプルな特徴量ベクトル
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_int = float(np.mean(gray))
    std_int = float(np.std(gray))
    edge = canny(gray / 255.0)
    edge_density = float(np.count_nonzero(edge) / edge.size)
    occupancy = compute_spatial_occupancy(bw)
    # フラクタル次元自身も特徴として含める
    return np.array([mean_int, std_int, edge_density, occupancy, fractal_dim], dtype=float)

# --- 永続化ファイル & モデル初期化 ----------------------------------------
MODEL_PATH = 'model_joblib.pkl'
SCALER_PATH = 'scaler_joblib.pkl'
CLASS_PATH = 'classifier_joblib.pkl'
EXCEL_PATH = 'results.xlsx'
TRAIN_CSV = 'train_data.csv'

# モデルロード関数
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            models['reg'] = joblib.load(MODEL_PATH)
            models['scaler'] = joblib.load(SCALER_PATH)
        except Exception:
            models = {}
    if os.path.exists(CLASS_PATH):
        try:
            models['clf'] = joblib.load(CLASS_PATH)
        except Exception:
            pass
    return models


def save_models(reg, scaler, clf=None):
    joblib.dump(reg, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    if clf is not None:
        joblib.dump(clf, CLASS_PATH)

# --- トレーニングデータの取り扱い --------------------------------------

def append_to_train_csv(features, y_reg, is_valid):
    # features: 1d array, y_reg: dict {'fractal':..., 'occupancy':...}
    cols = ['mean_int', 'std_int', 'edge_density', 'occupancy', 'fractal_dim_feature',
            'target_fractal', 'target_occupancy', 'is_valid']
    row = list(features) + [y_reg['fractal'], y_reg['occupancy'], int(is_valid)]
    df = pd.DataFrame([row], columns=cols)
    if os.path.exists(TRAIN_CSV):
        df.to_csv(TRAIN_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(TRAIN_CSV, index=False)


@st.cache_data(ttl=1)  # 1秒間キャッシュ（新しいデータが追加される可能性があるため短め）
def load_train_data():
    if os.path.exists(TRAIN_CSV):
        return pd.read_csv(TRAIN_CSV)
    else:
        return None

# --- Streamlit UI -------------------------------------------------------

st.set_page_config(layout='wide', page_title='フラクタル画像解析アプリ')
st.title('フラクタルを用いた画像解析アプリ')

st.sidebar.header('入力と解析条件')
uploaded_files = st.sidebar.file_uploader('画像ファイルを選択（複数可）', type=['png','jpg','jpeg','bmp','tif','tiff'], accept_multiple_files=True)

st.sidebar.markdown('---')
st.sidebar.subheader('前処理（任意）')
enable_brightness = st.sidebar.checkbox('輝度補正を有効化', value=False)
if enable_brightness:
    gamma_value = st.sidebar.slider('ガンマ値', min_value=0.10, max_value=3.0, value=1.0, step=0.05)
    brightness_offset = st.sidebar.slider('明るさ調整 (β)', min_value=-100, max_value=100, value=0, step=1)
else:
    gamma_value = 1.0
    brightness_offset = 0

enable_saturation = st.sidebar.checkbox('彩度補正を有効化', value=False)
if enable_saturation:
    saturation_factor = st.sidebar.slider('彩度倍率', min_value=0.5, max_value=2.0, value=1.0, step=0.05)
else:
    saturation_factor = 1.0

st.sidebar.markdown('---')
st.sidebar.subheader('二値化・解析条件')

if 'threshold_value' not in st.session_state:
    st.session_state['threshold_value'] = 128.0
if 'threshold_slider' not in st.session_state:
    st.session_state['threshold_slider'] = st.session_state['threshold_value']
if 'threshold_number' not in st.session_state:
    st.session_state['threshold_number'] = st.session_state['threshold_value']
if 'max_side_value' not in st.session_state:
    st.session_state['max_side_value'] = 1024.0
if 'max_side_slider' not in st.session_state:
    st.session_state['max_side_slider'] = st.session_state['max_side_value']
if 'max_side_number' not in st.session_state:
    st.session_state['max_side_number'] = st.session_state['max_side_value']
if 'adaptive_block_size' not in st.session_state:
    st.session_state['adaptive_block_size'] = 11
if 'adaptive_c' not in st.session_state:
    st.session_state['adaptive_c'] = 2


def _sync_threshold_from_slider():
    st.session_state['threshold_value'] = float(st.session_state['threshold_slider'])
    st.session_state['threshold_number'] = st.session_state['threshold_value']


def _sync_threshold_from_number():
    st.session_state['threshold_value'] = float(st.session_state['threshold_number'])
    st.session_state['threshold_slider'] = st.session_state['threshold_value']


def _sync_max_side_from_slider():
    st.session_state['max_side_value'] = float(st.session_state['max_side_slider'])
    st.session_state['max_side_number'] = st.session_state['max_side_value']


def _sync_max_side_from_number():
    st.session_state['max_side_value'] = float(st.session_state['max_side_number'])
    st.session_state['max_side_slider'] = st.session_state['max_side_value']


binarize_mode = st.sidebar.radio('二値化方式', ['固定閾値', '適応閾値'], index=0)

if binarize_mode == '固定閾値':
    st.sidebar.slider('二値化閾値 (スライダー)', min_value=0.0, max_value=255.0, key='threshold_slider', value=float(st.session_state['threshold_value']), step=1.0, on_change=_sync_threshold_from_slider)
    st.sidebar.number_input('二値化閾値 (数値入力)', min_value=0.0, max_value=255.0, key='threshold_number', step=0.1, value=st.session_state['threshold_value'], on_change=_sync_threshold_from_number)
    thresh_value = float(st.session_state['threshold_value'])
    adaptive_block_size = None
    adaptive_c = None
else:
    adaptive_block_size = st.sidebar.slider('適応閾値ブロックサイズ (奇数)', min_value=3, max_value=51, step=2, value=int(st.session_state['adaptive_block_size']))
    st.session_state['adaptive_block_size'] = adaptive_block_size
    adaptive_c = st.sidebar.slider('適応閾値 C 値', min_value=-20, max_value=20, value=int(st.session_state['adaptive_c']))
    st.session_state['adaptive_c'] = adaptive_c
    thresh_value = None

st.sidebar.slider('リサイズ最大辺 (スライダー)', min_value=0.0, max_value=6000.0, key='max_side_slider', value=float(st.session_state['max_side_value']), step=10.0, on_change=_sync_max_side_from_slider)
st.sidebar.number_input('リサイズ最大辺 (数値入力)', min_value=0.0, max_value=10000.0, key='max_side_number', step=10.0, value=st.session_state['max_side_value'], on_change=_sync_max_side_from_number)
max_side = float(st.session_state['max_side_value'])

st.sidebar.markdown('---')
# 学習ボタン
do_train_now = st.sidebar.button('学習を実行（保存済みデータで再学習）')
# モデルロード
models = load_models()

# 解析/学習用の表示領域
col1, col2 = st.columns([2,1])

with col1:
    st.header('解析結果')
    if uploaded_files is not None and len(uploaded_files) > 0:
        results_list = []  # Excel/学習データ用
        summary_records = []  # 表示・保存用
        detail_records = []  # HTMLレポート用
        run_timestamp = datetime.now()

        for file in uploaded_files:
            st.subheader(f'ファイル: {file.name}')
            file_bytes = file.getvalue()
            if file_bytes is None or len(file_bytes) == 0:
                st.error('ファイルを読み込めませんでした。')
                continue

            original_bgr = load_image_from_bytes(file_bytes)
            resized_bgr, scale = resize_image(original_bgr, max_side)

            processed_bgr = resized_bgr.copy()
            preprocessing_steps = []
            if enable_brightness:
                processed_bgr = apply_gamma_correction(processed_bgr, gamma_value)
                processed_bgr = apply_brightness_offset(processed_bgr, brightness_offset)
                preprocessing_steps.append(f'輝度補正 (γ={gamma_value:.2f}, β={brightness_offset})')
            if enable_saturation:
                processed_bgr = apply_saturation_adjustment(processed_bgr, saturation_factor)
                preprocessing_steps.append(f'彩度補正 (×{saturation_factor:.2f})')
            if not preprocessing_steps:
                preprocessing_steps.append('前処理なし')

            gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)

            if binarize_mode == '固定閾値':
                bw = binarize_image_gray(gray, thresh_value)
                threshold_info = f'固定閾値: {thresh_value:.2f}'
                adaptive_bs = None
                adaptive_c_value = None
            else:
                adaptive_bs = adaptive_block_size
                adaptive_c_value = adaptive_c
                bw = adaptive_binarize(gray, adaptive_bs, adaptive_c_value)
                threshold_info = f'適応閾値: block={adaptive_bs}, C={adaptive_c_value}'

            fractal_d, sizes, counts = boxcount_fractal_dim(bw)
            occupancy = compute_spatial_occupancy(bw)

            fail_flag = False
            fail_reasons: list[str] = []
            if occupancy < 0.01:
                fail_flag = True
                fail_reasons.append('ほとんど白が無い(占有率 <1%)')
            if occupancy > 0.99:
                fail_flag = True
                fail_reasons.append('ほとんど白で埋まっている(占有率 >99%)')
            if not (-5.0 < fractal_d < 5.0):
                fail_flag = True
                fail_reasons.append(f'フラクタル次元が異常値:{fractal_d:.3f}')

            feat = extract_features_from_image(processed_bgr, bw, fractal_d)

            pred = None
            if 'reg' in models and 'scaler' in models:
                try:
                    Xs = models['scaler'].transform(feat.reshape(1, -1))
                    ypred = models['reg'].predict(Xs)[0]
                    if isinstance(ypred, (list, tuple, np.ndarray)) and len(ypred) >= 2:
                        pred = {'fractal': float(ypred[0]), 'occupancy': float(ypred[1])}
                    else:
                        pred = {'fractal': float(ypred), 'occupancy': None}
                except Exception as e:
                    st.warning(f'予測中にエラーが発生しました: {e}')

            st.caption('処理フロー: 画像入力 → 前処理 → 二値化 → フラクタル解析 → 可視化・保存')

            st.subheader('📊 解析結果')
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    label='フラクタル次元',
                    value=f'{fractal_d:.4f}',
                    help='フラクタル次元は画像パターンの複雑さを表す指標です'
                )
            with metric_col2:
                st.metric(
                    label='空間占有率（白ピクセル）',
                    value=f'{occupancy*100:.2f}%',
                    help='白ピクセルが占める割合を示します'
                )

            if pred is not None:
                st.write('**🤖 学習モデルの予測値**')
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    delta_fractal = None if pred['fractal'] is None else fractal_d - pred['fractal']
                    st.metric(
                        label='予測フラクタル次元',
                        value=f"{pred['fractal']:.4f}" if pred['fractal'] is not None else 'N/A',
                        delta=f'{delta_fractal:.4f}' if delta_fractal is not None else None,
                        delta_color='off'
                    )
                with pred_col2:
                    if pred['occupancy'] is not None:
                        delta_occupancy = (occupancy - pred['occupancy']) * 100
                        st.metric(
                            label='予測空間占有率',
                            value=f"{pred['occupancy']*100:.2f}%",
                            delta=f'{delta_occupancy:.2f}%' if delta_occupancy is not None else None,
                            delta_color='off'
                        )

            preprocessing_text = '\n'.join([f'- {step}' for step in preprocessing_steps])
            st.markdown(f'**前処理ステップ**\n{preprocessing_text}')
            st.markdown(f'**二値化条件**: {threshold_info}')

            if fail_flag:
                st.warning('⚠️ 自動検知: 失敗と判定されました。理由: ' + '; '.join(fail_reasons))
            else:
                st.success('✅ 自動検知: 正常と判定')

            st.divider()

            col_img1, col_img2, col_img3 = st.columns(3)
            with col_img1:
                st.subheader('元画像（リサイズ後）')
                st.image(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_img2:
                st.subheader('前処理後画像')
                st.image(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_img3:
                st.subheader('二値化画像')
                st.image(bw, use_container_width=True, clamp=True)

            st.subheader('📈 グラフ表示')
            graph_col1, graph_col2 = st.columns(2)
            with graph_col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(np.log(1.0 / sizes), np.log(counts), marker='o', linewidth=2, markersize=8)
                ax1.set_xlabel('log(1/size)')
                ax1.set_ylabel('log(count)')
                ax1.set_title('フラクタル次元解析グラフ')
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                plt.close(fig1)
            with graph_col2:
                fig2, ax2 = plt.subplots()
                colors = ['white', 'black']
                wedgeprops = {'edgecolor': 'gray', 'linewidth': 1}
                ax2.pie([occupancy, 1 - occupancy], labels=['白ピクセル', '黒ピクセル'], autopct='%1.1f%%', colors=colors, wedgeprops=wedgeprops, startangle=90)
                ax2.set_title('空間占有率（白ピクセル vs 黒ピクセル）')
                st.pyplot(fig2)
                plt.close(fig2)

            if pred is not None:
                st.subheader('🔍 予測 vs 実測の比較')
                compare_col1, compare_col2 = st.columns(2)
                with compare_col1:
                    fig3, ax3 = plt.subplots()
                    ax3.plot([0, 1], [fractal_d, pred['fractal']], marker='o', linewidth=2, markersize=10)
                    ax3.set_xticks([0, 1])
                    ax3.set_xticklabels(['実測', '予測'])
                    ax3.set_ylabel('フラクタル次元')
                    ax3.set_title('フラクタル次元の比較')
                    ax3.grid(True, alpha=0.3)
                    st.pyplot(fig3)
                    plt.close(fig3)
                with compare_col2:
                    if pred['occupancy'] is not None:
                        fig4, ax4 = plt.subplots()
                        ax4.plot([0, 1], [occupancy, pred['occupancy']], marker='o', linewidth=2, markersize=10)
                        ax4.set_xticks([0, 1])
                        ax4.set_xticklabels(['実測', '予測'])
                        ax4.set_ylabel('占有率')
                        ax4.set_title('空間占有率の比較')
                        ax4.grid(True, alpha=0.3)
                        st.pyplot(fig4)
                        plt.close(fig4)

            rec = {
                'filename': file.name,
                'fractal': fractal_d,
                'occupancy': occupancy,
                'pred_fractal': pred['fractal'] if pred is not None else None,
                'pred_occupancy': pred['occupancy'] if (pred is not None and pred['occupancy'] is not None) else None,
                'is_valid': int(not fail_flag)
            }
            results_list.append(rec)

            summary_records.append({
                'ファイル名': file.name,
                'フラクタル次元': fractal_d,
                '空間占有率(%)': occupancy * 100,
                '予測フラクタル次元': rec['pred_fractal'],
                '予測空間占有率(%)': rec['pred_occupancy'] * 100 if rec['pred_occupancy'] is not None else None,
                '閾値方式': binarize_mode,
                '閾値値': thresh_value if thresh_value is not None else None,
                '適応ブロックサイズ': adaptive_bs,
                '適応C値': adaptive_c_value,
                'リサイズ最大辺': max_side,
                '輝度補正': 'ON' if enable_brightness else 'OFF',
                '彩度補正': 'ON' if enable_saturation else 'OFF',
                '異常判定': '正常' if not fail_flag else '失敗',
                '異常理由': '; '.join(fail_reasons) if fail_reasons else ''
            })

            detail_records.append({
                'filename': file.name,
                'fractal': fractal_d,
                'occupancy': occupancy,
                'pred_fractal': rec['pred_fractal'],
                'pred_occupancy': rec['pred_occupancy'],
                'fail_flag': fail_flag,
                'fail_reasons': '; '.join(fail_reasons) if fail_reasons else '',
                'threshold_mode': binarize_mode,
                'threshold_value': thresh_value,
                'adaptive_block_size': adaptive_bs,
                'adaptive_c': adaptive_c_value,
                'gamma_applied': enable_brightness,
                'gamma_value': gamma_value if enable_brightness else None,
                'brightness_offset': brightness_offset if enable_brightness else None,
                'saturation_applied': enable_saturation,
                'saturation_factor': saturation_factor if enable_saturation else None,
                'original_b64': image_to_base64_png(resized_bgr),
                'processed_b64': image_to_base64_png(processed_bgr),
                'binary_b64': image_to_base64_png(bw)
            })

            append_to_train_csv(feat, {'fractal': fractal_d, 'occupancy': occupancy}, not fail_flag)

            st.markdown('---')

        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            st.subheader('📋 解析サマリ（数値一覧）')
            st.dataframe(summary_df, use_container_width=True)

            csv_data = summary_df.to_csv(index=False).encode('utf-8-sig')
            csv_filename = f'fractal_results_{run_timestamp.strftime("%Y%m%d_%H%M%S")}.csv'
            st.download_button('CSVとして保存', data=csv_data, file_name=csv_filename, mime='text/csv')

            run_settings = {
                '二値化方式': binarize_mode,
                '固定閾値': f'{thresh_value:.2f}' if thresh_value is not None else 'N/A',
                '適応ブロックサイズ': adaptive_block_size if binarize_mode == '適応閾値' else 'N/A',
                '適応C値': adaptive_c if binarize_mode == '適応閾値' else 'N/A',
                'リサイズ最大辺(px)': max_side,
                '輝度補正': 'ON' if enable_brightness else 'OFF',
                '彩度補正': 'ON' if enable_saturation else 'OFF'
            }
            html_report = build_html_report(run_timestamp, run_settings, summary_df, detail_records)
            html_filename = f'fractal_report_{run_timestamp.strftime("%Y%m%d_%H%M%S")}.html'
            st.download_button('HTMLレポートを作成', data=html_report.encode('utf-8'), file_name=html_filename, mime='text/html')

        if len(results_list) >= 2:
            df_results = pd.DataFrame(results_list)
            if os.path.exists(EXCEL_PATH):
                with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    sheet_name = pd.Timestamp.now().strftime('run_%Y%m%d_%H%M%S')
                    df_results.to_excel(writer, sheet_name=sheet_name, index=False)
                st.info(f'解析結果を既存Excel ({EXCEL_PATH}) に追記しました。')
            else:
                df_results.to_excel(EXCEL_PATH, sheet_name='run', index=False)
                st.info(f'解析結果を新規Excel ({EXCEL_PATH}) に保存しました。')

        train_df = load_train_data()
        if train_df is not None:
            st.sidebar.write(f'学習データ件数: {len(train_df)}')
        else:
            st.sidebar.write('学習データはまだありません。')

>>>>>>> Stashed changes
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
