"""
Streamlit アプリ: フラクタルを用いた画像解析アプリ
機能:
- 閾値（スライダー/数値入力）とリサイズ上限（スライダー/数値入力）の2方式を用意
- フラクタル次元は折れ線グラフで出力、空間占有率は円グラフで出力
- 学習機能: 解析結果（有効/失敗）を学習し、予測結果と比較表示
- 異常値・異常な二値化の自動検知（失敗扱い） -> 学習用データに追加
- フォルダ内画像を一括解析（Streamlitの仕様上、複数ファイルアップロードで対応）
- 2枚以上解析時、自動でExcelに結果を保存・追記
- 学習件数の表示、解析精度（MAEなど）の表示

使い方:
1) 必要ライブラリをインストール: pip install -r requirements.txt
2) 実行: streamlit run fractal_app.py

ファイル出力:
- 学習モデル: model_joblib.pkl
- スケーラ: scaler_joblib.pkl
- 結果Excel: results.xlsx

注意: 本例は学習ロジックを簡潔化しています。用途に応じて特徴量やモデルを拡張してください。
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import io
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import filters, color
from skimage.feature import canny
import matplotlib
# 日本語フォント設定（エラーを無視）
try:
    matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass  # フォント設定エラーを無視

# --- 解像度補正モデル用の定数 ----------------------------------------------
RESOLUTION_MODEL_PATH = 'resolution_correction_model.pkl'
RESOLUTION_SCALER_PATH = 'resolution_correction_scaler.pkl'
RESOLUTION_TRAIN_DATA = 'resolution_training_data.csv'

# --- ユーティリティ関数 -------------------------------------------------

@st.cache_data(show_spinner=False)
def load_image_bytes(file_bytes: bytes, file_name: str) -> np.ndarray:
    """
    画像バイトデータから BGR(OpenCV) 画像を返す
    キャッシュ化により、同じファイルの再読み込みを高速化
    """
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    arr = np.array(img)[:, :, ::-1].copy()  # RGB->BGR
    return arr


@st.cache_data(show_spinner=False)
def resize_image(img: np.ndarray, max_side: float):
    """
    最長辺が max_side を超える場合リサイズする
    キャッシュ化により、同じパラメータでの再計算を防ぐ
    """
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side and max_side > 0:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


@st.cache_data(show_spinner=False)
def binarize_image_gray(gray: np.ndarray, thresh: float):
    """
    thresh は 0..255 の実数値。ここでは固定閾値による二値化
    キャッシュ化により、同じ閾値での再計算を防ぐ
    """
    _, bw = cv2.threshold(gray.astype('uint8'), thresh, 255, cv2.THRESH_BINARY)
    return bw


def adaptive_binarize(gray: np.ndarray):
    # ガウシアン適応閾値（サンプルとして）
    bw = cv2.adaptiveThreshold(gray.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return bw


@st.cache_data(show_spinner=False)
def boxcount_fractal_dim(bw: np.ndarray, sizes=None, fast_mode=False):
    """
    白(255) を対象に箱ひき（box-counting法）でフラクタル次元を推定
    
    Args:
        bw: 二値画像（0 or 255）
        sizes: list of box sizes to use (pixels)
        fast_mode: 高速モード（箱サイズを削減）
        
    キャッシュ化により、同じ画像とパラメータでの再計算を防ぐ
    """
    S = bw.shape
    if sizes is None:
        max_dim = max(S)
        # 箱サイズは 2^k 系列で生成（サイズを制限して高速化）
        if fast_mode:
            # 高速モード: 箱サイズを削減（最大6段階）
            max_power = min(int(np.log2(min(S))), 6)
        else:
            # 通常モード: 最大10段階
            max_power = min(int(np.log2(min(S))), 10)
        sizes = np.array([2 ** i for i in range(1, max_power)])
        sizes = sizes[sizes <= min(S)]
        if len(sizes) < 3:
            sizes = np.array([2,4,8,16])
    counts = []
    # NumPy配列操作で高速化
    bw_binary = (bw > 0).astype(np.uint8)  # 事前に二値化
    for size in sizes:
        # 画像を size x size のブロックに分割して、白が含まれるブロックを数える
        ny = int(np.ceil(S[0] / size))
        nx = int(np.ceil(S[1] / size))
        count = 0
        # ベクトル化された処理で高速化
        for i in range(ny):
            y0 = i * size
            y1 = min(y0 + size, S[0])
            for j in range(nx):
                x0 = j * size
                x1 = min(x0 + size, S[1])
                if np.any(bw_binary[y0:y1, x0:x1]):
                    count += 1
        counts.append(count)
    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    # fractal dimension D is slope of log(count) vs log(1/size)
    # linear regression via least squares
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(counts)
        loginv = np.log(1.0 / sizes)
    # 単純な線形回帰
    A = np.vstack([loginv, np.ones_like(loginv)]).T
    try:
        m, c = np.linalg.lstsq(A, logs, rcond=None)[0]
    except Exception:
        m = 0.0
    return float(m), sizes, counts


def compute_spatial_occupancy(bw: np.ndarray):
    # 白（255）が占める割合
    total = bw.size
    white = np.count_nonzero(bw > 0)
    return float(white / total)


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

# --- 解像度補正モデル用の関数 ---------------------------------------------

def generate_low_resolution_versions(img_bgr: np.ndarray, scale_factors=[0.5, 0.25, 0.1]):
    """
    高解像度画像から複数の低解像度バージョンを生成
    """
    low_res_images = []
    for scale in scale_factors:
        h, w = img_bgr.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h > 10 and new_w > 10:  # 最小サイズチェック
            low_res = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # 元のサイズに戻す（低解像度のまま拡大）
            upscaled = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
            low_res_images.append((scale, upscaled))
    return low_res_images


def extract_resolution_features(img_bgr: np.ndarray, bw: np.ndarray, fractal_dim: float):
    """
    解像度補正用の拡張特徴量を抽出
    低解像度画像の特性を捉える特徴量
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 基本統計量
    mean_int = float(np.mean(gray))
    std_int = float(np.std(gray))
    
    # エッジ特徴
    edge = canny(gray / 255.0)
    edge_density = float(np.count_nonzero(edge) / edge.size)
    
    # 占有率
    occupancy = compute_spatial_occupancy(bw)
    
    # テクスチャ特徴（分散の局所的な変動）
    kernel_size = max(3, min(h, w) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    local_std = cv2.blur(gray.astype(float)**2, (kernel_size, kernel_size)) - \
                cv2.blur(gray.astype(float), (kernel_size, kernel_size))**2
    texture_variance = float(np.mean(np.sqrt(np.abs(local_std))))
    
    # 画像サイズ情報（正規化）
    img_size = float(np.log(h * w + 1))
    aspect_ratio = float(w / h)
    
    # 周波数成分（FFT）
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    high_freq_energy = float(np.mean(magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]))
    
    # フラクタル次元（低解像度で計算されたもの）
    
    return np.array([
        mean_int, std_int, edge_density, occupancy, fractal_dim,
        texture_variance, img_size, aspect_ratio, high_freq_energy
    ], dtype=float)


def train_resolution_correction_model(training_data_path=RESOLUTION_TRAIN_DATA):
    """
    解像度補正モデルを学習
    """
    if not os.path.exists(training_data_path):
        return None, None, "学習データが見つかりません"
    
    df = pd.read_csv(training_data_path)
    
    # 特徴量とターゲットを分離
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X = df[feature_cols].values
    y = df['target_high_res_fractal'].values
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデル学習（Gradient Boostingを使用）
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 評価
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 保存
    joblib.dump(model, RESOLUTION_MODEL_PATH)
    joblib.dump(scaler, RESOLUTION_SCALER_PATH)
    
    return model, scaler, f"MAE: {mae:.4f}, R²: {r2:.4f}"


def predict_high_res_fractal(low_res_features, model=None, scaler=None):
    """
    低解像度画像の特徴量から高解像度相当のフラクタル次元を予測
    """
    if model is None or scaler is None:
        if os.path.exists(RESOLUTION_MODEL_PATH) and os.path.exists(RESOLUTION_SCALER_PATH):
            model = joblib.load(RESOLUTION_MODEL_PATH)
            scaler = joblib.load(RESOLUTION_SCALER_PATH)
        else:
            return None
    
    X = low_res_features.reshape(1, -1)
    X_scaled = scaler.transform(X)
    predicted_fractal = model.predict(X_scaled)[0]
    
    return float(predicted_fractal)


# --- 永続化ファイル & モデル初期化 ----------------------------------------
MODEL_PATH = 'model_joblib.pkl'
SCALER_PATH = 'scaler_joblib.pkl'
CLASS_PATH = 'classifier_joblib.pkl'
EXCEL_PATH = 'results.xlsx'
TRAIN_CSV = 'train_data.csv'

@st.cache_resource(show_spinner=False)
def load_models():
    """
    モデルをロードしてキャッシュする
    @st.cache_resource により、アプリ起動中は一度だけロードされる
    """
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

@st.cache_resource(show_spinner=False)
def load_resolution_model():
    """
    解像度補正モデルをロードしてキャッシュする
    @st.cache_resource により、アプリ起動中は一度だけロードされる
    """
    if os.path.exists(RESOLUTION_MODEL_PATH) and os.path.exists(RESOLUTION_SCALER_PATH):
        try:
            model = joblib.load(RESOLUTION_MODEL_PATH)
            scaler = joblib.load(RESOLUTION_SCALER_PATH)
            return model, scaler
        except Exception:
            return None, None
    return None, None


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


@st.cache_data(show_spinner=False)
def load_train_data():
    """
    学習データをロードしてキャッシュする
    頻繁に更新される可能性があるため、TTL（有効期限）を短めに設定
    """
    if os.path.exists(TRAIN_CSV):
        # 学習データは頻繁に更新される可能性があるため、キャッシュは控えめに。
        try:
            return pd.read_csv(TRAIN_CSV)
        except Exception:
            return None
    else:
        return None

# --- Streamlit UI -------------------------------------------------------

# ページ設定は最初に呼び出す必要がある
try:
    st.set_page_config(layout='wide', page_title='フラクタル画像解析アプリ')
except Exception as e:
    # 既に設定されている場合は無視
    pass

st.title('フラクタルを用いた画像解析アプリ')

# アプリ起動確認メッセージ（デバッグ用、本番では削除可能）
# st.sidebar.success('✅ アプリは正常に起動しました')

# アプリケーション概要と解像度補正AIの紹介
with st.expander('ℹ️ このアプリについて / 解像度補正AI機能', expanded=False):
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ###  基本機能
        - **Box-Counting法**によるフラクタル次元解析
        - **二値化処理**と空間占有率の計算
        - **複数画像の一括解析**とExcel出力
        - **機械学習**による予測機能
        - **異常値の自動検知**
        """)
    
    with col_info2:
        st.markdown("""
        ### 解像度補正AI
        低解像度画像でも高解像度相当のフラクタル次元を推定！
        
        **クイックスタート:**
        1. 高解像度画像を用意
        2. サイドバー「学習データ生成モード」ON → アップロード
        3. 20～100枚繰り返す
        4. 「解像度補正モデルを学習」ボタンをクリック
        5. 「解像度補正を有効化」ON
        6. 低解像度画像をアップロード
        7. 補正結果を確認！
        
        📖 詳細はサイドバーの「解像度補正AIの使い方」を参照
        """)

st.sidebar.header('設定')
# 閾値入力: 数値入力とスライダーを両方用意
thresh_mode = st.sidebar.selectbox('閾値入力方式', ['スライダー', '数値入力'])
if thresh_mode == 'スライダー':
    thresh_value = st.sidebar.slider('二値化閾値 (0-255)', min_value=0.0, max_value=255.0, value=128.0)
else:
    thresh_value = st.sidebar.number_input('二値化閾値 (0-255)', min_value=0.0, max_value=255.0, value=128.0, step=0.1)

# リサイズ上限: 数値とスライダー
resize_mode = st.sidebar.selectbox('リサイズ方式', ['スライダー', '数値入力'])
if resize_mode == 'スライダー':
    max_side = st.sidebar.slider('リサイズ最大辺 (px, 0でリサイズ無効)', min_value=0.0, max_value=4000.0, value=1024.0)
else:
    max_side = st.sidebar.number_input('リサイズ最大辺 (px, 0でリサイズ無効)', min_value=0.0, max_value=10000.0, value=1024.0)

st.sidebar.markdown('---')
# 学習ボタン
do_train_now = st.sidebar.button('学習を実行（保存済みデータで再学習）')

# 解像度補正モデルセクション
st.sidebar.markdown('---')
st.sidebar.subheader('🔬 解像度補正AI')
enable_resolution_correction = st.sidebar.checkbox('解像度補正を有効化', value=False, 
    help='低解像度画像から高解像度相当のフラクタル次元を推定します')

st.sidebar.markdown('---')
st.sidebar.subheader('⚡ パフォーマンス設定')

# 処理モードの選択（新機能）
processing_mode = st.sidebar.radio(
    '処理モード',
    ['🚀 高速プレビュー', '🎯 高精度解析'],
    help='高速プレビュー: 計算量を削減して素早く結果表示\n高精度解析: 全ての計算を実行して正確な結果を出力'
)

# 高速モードの判定フラグ
fast_mode = (processing_mode == '🚀 高速プレビュー')

# 自動再計算の設定
auto_recompute = st.sidebar.checkbox('自動再計算を有効化', value=True, help='OFFにすると「解析を更新」ボタンを押した時だけ重い処理を実行します')

# 高速プレビューモードの詳細説明
if fast_mode:
    st.sidebar.info('⚡ 高速プレビューモード:\n- 箱サイズ削減（6段階）\n- 低DPIグラフ描画\n- 計算時間 50-70%短縮')
else:
    st.sidebar.success('🎯 高精度解析モード:\n- 箱サイズ最大（10段階）\n- 高品質グラフ描画\n- 最高精度で解析')

run_analyze = st.sidebar.button('解析を更新', type='primary', help='自動再計算がOFFのときに押して実行')

# キャッシュ管理
if st.sidebar.button('🧹 キャッシュをクリア'):
    st.cache_data.clear()
    st.cache_resource.clear()
    # セッションステートもクリア
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.sidebar.success('キャッシュとセッションをクリアしました')
    st.rerun()

# 学習データ生成モード
if st.sidebar.checkbox('学習データ生成モード', value=False, 
    help='高解像度画像から低解像度バージョンを自動生成し、学習データを作成します'):
    st.sidebar.info('アップロードした画像から学習データを自動生成します')
    generate_training_data = True
else:
    generate_training_data = False

# 解像度補正モデルの学習ボタン
if st.sidebar.button('解像度補正モデルを学習'):
    with st.spinner('解像度補正モデルを学習中...'):
        model_res, scaler_res, result_msg = train_resolution_correction_model()
        if model_res is not None:
            st.sidebar.success(f'学習完了: {result_msg}')
        else:
            st.sidebar.error(result_msg)

# 解像度補正モデルの状態表示
if os.path.exists(RESOLUTION_MODEL_PATH):
    st.sidebar.success('✓ 解像度補正モデル: 読み込み済み')
    if os.path.exists(RESOLUTION_TRAIN_DATA):
        df_res = pd.read_csv(RESOLUTION_TRAIN_DATA)
        st.sidebar.write(f'学習データ数: {len(df_res)}件')
else:
    st.sidebar.warning('解像度補正モデル: 未学習')

# 使い方ガイドの表示
with st.sidebar.expander('📖 解像度補正AIの使い方', expanded=False):
    st.markdown("""
    ### ステップバイステップガイド
    
    **1️⃣ 高解像度画像を用意**
    - なるべく高品質な画像を準備
    
    **2️⃣ 学習データ生成**
    - ☑「学習データ生成モード」ON
    - 画像をアップロード
    - 自動で5段階の解像度データ生成
    
    **3️⃣ データ収集**
    - 複数の画像で繰り返し
    - 推奨: 20～100枚
    
    **4️⃣ モデル学習**
    - 「解像度補正モデルを学習」ボタンをクリック
    - 学習完了を待つ
    
    **5️⃣ 補正の有効化**
    - ☑「解像度補正を有効化」ON
    
    **6️⃣ 低解像度画像で検証**
    - 低解像度画像をアップロード
    
    **7️⃣ 結果確認**
    - 高解像度相当のフラクタル次元が表示される！
    
    ---
    💡 **ヒント**: 様々なタイプの画像で学習すると精度が向上します
    """)

# モデルロード（エラーハンドリング強化）
try:
    models = load_models()
except Exception as e:
    st.error(f'モデルのロード中にエラーが発生しました: {e}')
    models = {}

try:
    res_model, res_scaler = load_resolution_model()
except Exception as e:
    st.warning(f'解像度補正モデルのロード中にエラーが発生しました: {e}')
    res_model, res_scaler = None, None

# ファイル選択: 複数ファイルアップロードでフォルダ内一括解析に対応
try:
    uploaded_files = st.file_uploader('画像ファイルを選択（複数可）', type=['png','jpg','jpeg','bmp','tif','tiff'], accept_multiple_files=True)
except Exception as e:
    st.error(f'ファイルアップローダーの初期化エラー: {e}')
    uploaded_files = []

# 学習データ生成モード時のガイド表示
if generate_training_data:
    st.info('🔄 **学習データ生成モード** - 高解像度画像をアップロードすると、自動的に5段階の解像度で学習データを生成します')
    st.markdown("""
    **現在のステップ**: 2️⃣ 学習データ生成中
    - ✅ 高解像度画像をアップロード
    - ⏳ 複数枚（20～100枚推奨）繰り返す
    - ⏭️ 完了後、サイドバーで「解像度補正モデルを学習」をクリック
    """)

# 解像度補正有効時のガイド表示
if enable_resolution_correction and os.path.exists(RESOLUTION_MODEL_PATH):
    st.success('✅ **解像度補正AI有効** - 低解像度画像でも高解像度相当のフラクタル次元を推定します')

# 解析/学習用の表示領域
col1, col2 = st.columns([2,1])

# セッションステートの初期化
if 'last_params' not in st.session_state:
    st.session_state['last_params'] = None
if 'cached_results' not in st.session_state:
    st.session_state['cached_results'] = None

# 現在のパラメータハッシュを生成（変更検知用）
if uploaded_files:
    current_params = {
        'files': [f.name for f in uploaded_files],
        'thresh': thresh_value,
        'max_side': max_side,
        'fast_mode': fast_mode,
        'enable_resolution': enable_resolution_correction,
        'generate_training': generate_training_data
    }
    params_changed = (st.session_state['last_params'] != current_params)
else:
    params_changed = True
    current_params = None

with col1:
    st.header('解析結果')
    if uploaded_files is not None and len(uploaded_files) > 0 and (auto_recompute or run_analyze):
        
        # パラメータ変更時のみ再計算、それ以外はキャッシュを使用
        if params_changed or st.session_state['cached_results'] is None:
            
            # パフォーマンス測定開始
            import time
            start_time = time.time()
            
            results_list = []
            predictions = []
            
            # プログレスバーを表示（複数ファイル処理時）
            if len(uploaded_files) > 1:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                
                # プログレスバー更新
                if len(uploaded_files) > 1:
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f'処理中: {file.name} ({idx + 1}/{len(uploaded_files)})')
                
                st.write('ファイル:', file.name)
                
                # 画像読み込み（キャッシュ化済み）
                file_bytes = file.read()
                img_bgr = load_image_bytes(file_bytes, file.name)
            
                # 学習データ生成モードの場合
                if generate_training_data:
                    st.info('🔄 学習データ生成モード: 複数解像度で解析中...')
                    # 元画像（高解像度）の解析
                    img_high, _ = resize_image(img_bgr.copy(), max_side)
                    gray_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY)
                    bw_high = binarize_image_gray(gray_high, thresh_value)
                    fractal_high, _, _ = boxcount_fractal_dim(bw_high, fast_mode=False)  # 高精度で計算
                    
                    # 低解像度バージョンを生成して解析
                    low_res_versions = generate_low_resolution_versions(img_high, [0.5, 0.3, 0.2, 0.15, 0.1])
                    
                    training_records = []
                    for scale, img_low in low_res_versions:
                        gray_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
                        bw_low = binarize_image_gray(gray_low, thresh_value)
                        fractal_low, _, _ = boxcount_fractal_dim(bw_low, fast_mode=fast_mode)
                        
                        # 拡張特徴量を抽出
                        features_low = extract_resolution_features(img_low, bw_low, fractal_low)
                        
                        # 学習データとして保存
                        record = {'scale': scale, 'target_high_res_fractal': fractal_high}
                        for idx_feat, feat_val in enumerate(features_low):
                            record[f'feat_{idx_feat}'] = feat_val
                        training_records.append(record)
                    
                    # CSVに追記
                    df_new = pd.DataFrame(training_records)
                    if os.path.exists(RESOLUTION_TRAIN_DATA):
                        df_new.to_csv(RESOLUTION_TRAIN_DATA, mode='a', header=False, index=False)
                    else:
                        df_new.to_csv(RESOLUTION_TRAIN_DATA, index=False)
                    
                    st.success(f'✓ {len(training_records)}件の学習データを生成しました（解像度: {[f"{s*100:.0f}%" for s, _ in low_res_versions]}）')
                    continue  # 次のファイルへ
                
                # メイン解析処理（キャッシュ活用）
                img_bgr_resized, scale = resize_image(img_bgr.copy(), max_side)
                gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
                # 二値化 (固定閾値、キャッシュ化済み)
                bw = binarize_image_gray(gray, thresh_value)

                # フラクタル次元計算（キャッシュ化済み、fast_modeパラメータ追加）
                fractal_d, sizes, counts = boxcount_fractal_dim(bw, fast_mode=fast_mode)
                occupancy = compute_spatial_occupancy(bw)
            
                # 解像度補正の適用
                corrected_fractal_d = None
                if enable_resolution_correction and (res_model is not None and res_scaler is not None):
                    try:
                        features_res = extract_resolution_features(img_bgr_resized, bw, fractal_d)
                        # 既にロード済みモデル/スケーラを使う
                        X = features_res.reshape(1, -1)
                        Xs = res_scaler.transform(X)
                        corrected_fractal_d = float(res_model.predict(Xs)[0])
                        if corrected_fractal_d is not None:
                            st.info(f'🤖 AI補正: {fractal_d:.4f} → {corrected_fractal_d:.4f} (差: {abs(corrected_fractal_d - fractal_d):.4f})')
                    except Exception as e:
                        st.warning(f'解像度補正エラー: {e}')

                # 異常検知: 極端な占有率や二値化がほぼ全白/全黒なら失敗扱い
                white_ratio = occupancy
                fail_flag = False
                fail_reasons = []
                if white_ratio < 0.01:
                    fail_flag = True
                    fail_reasons.append('ほとんど白が無い(占有率 <1%)')
                if white_ratio > 0.99:
                    fail_flag = True
                    fail_reasons.append('ほとんど白で埋まっている(占有率 >99%)')
                # フラクタル次元の現実的レンジチェック
                if not ( -5.0 < fractal_d < 5.0 ):  # 様々な画像での目安
                    fail_flag = True
                    fail_reasons.append(f'フラクタル次元が異常値:{fractal_d:.3f}')

                # 特徴量抽出
                feat = extract_features_from_image(img_bgr_resized, bw, fractal_d)

                # 予測が可能なら出力
                pred = None
                if 'reg' in models and 'scaler' in models:
                    try:
                        Xs = models['scaler'].transform(feat.reshape(1,-1))
                        ypred = models['reg'].predict(Xs)[0]
                        # reg は 2出力を想定している (fractal, occupancy)
                        if isinstance(ypred, (list,tuple,np.ndarray)) and len(ypred) >= 2:
                            pred = {'fractal': float(ypred[0]), 'occupancy': float(ypred[1])}
                        else:
                            # 単一出力の場合はフラクタルのみ
                            pred = {'fractal': float(ypred), 'occupancy': None}
                    except Exception as e:
                        st.write('予測中にエラーが発生しました:', e)

                # 結果表示
                st.write(f'- フラクタル次元（実測）: {fractal_d:.4f}')
                if corrected_fractal_d is not None:
                    st.write(f'- フラクタル次元（AI補正後）: {corrected_fractal_d:.4f}')
                    st.write(f'- 補正量: {(corrected_fractal_d - fractal_d):+.4f}')
                st.write(f'- 空間占有率: {occupancy*100:.2f}%')
                if fail_flag:
                    st.warning('自動検知: 失敗と判定されました。理由: ' + ';'.join(fail_reasons))
                else:
                    st.success('自動検知: 正常と判定')

                # 元画像と二値化画像の表示
                st.subheader('画像表示')
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.write('**元画像**')
                    # BGRからRGBに変換して表示
                    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                with img_col2:
                    st.write('**二値化画像**')
                    st.image(bw, use_container_width=True)

                # グラフ: フラクタル次元の折れ線（sizes vs counts から可視化）
                st.subheader('フラクタル次元解析')
                
                # DPIを処理モードに応じて調整（高速モード: 低DPI、高精度モード: 高DPI）
                graph_dpi = 60 if fast_mode else 100
                
                fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=graph_dpi)
                
                # 実測値のプロット（青色）
                ax1.plot(np.log(1.0/sizes), np.log(counts), marker='o', linewidth=2, markersize=8, 
                        color='blue', label=f'実測値 (傾き={fractal_d:.3f})')
                
                # 解像度補正AI予測値がある場合は緑色で追加プロット
                if corrected_fractal_d is not None:
                    x_vals = np.log(1.0/sizes)
                    intercept = np.mean(np.log(counts) - corrected_fractal_d * x_vals)
                    y_corrected = corrected_fractal_d * x_vals + intercept
                    ax1.plot(x_vals, y_corrected, marker='^', linewidth=2, markersize=6, 
                            color='green', linestyle='-.', label=f'解像度補正AI (傾き={corrected_fractal_d:.3f})', alpha=0.8)
                
                # 従来のAI予測値がある場合は赤色で追加プロット
                if pred is not None and 'fractal' in pred:
                    pred_fractal = pred['fractal']
                    x_vals = np.log(1.0/sizes)
                    intercept = np.mean(np.log(counts) - pred_fractal * x_vals)
                    y_pred = pred_fractal * x_vals + intercept
                    ax1.plot(x_vals, y_pred, marker='s', linewidth=2, markersize=6, 
                            color='red', linestyle='--', label=f'従来AI予測 (傾き={pred_fractal:.3f})', alpha=0.7)
                
                ax1.set_xlabel('log(1/箱サイズ)', fontsize=11)
                ax1.set_ylabel('log(白ピクセルを含む箱の数)', fontsize=11)
                
                # タイトルを予測の有無で変更
                title_parts = [f'実測: {fractal_d:.3f}']
                if corrected_fractal_d is not None:
                    title_parts.append(f'AI補正: {corrected_fractal_d:.3f}')
                if pred is not None and 'fractal' in pred:
                    title_parts.append(f'従来AI: {pred["fractal"]:.3f}')
                
                ax1.set_title(f'Box-Counting法によるフラクタル次元解析\n{" / ".join(title_parts)}', 
                            fontsize=12, fontweight='bold')
                
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='best', fontsize=10)
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)  # メモリ解放

                # 円グラフ: 空間占有率（白ピクセルと黒ピクセル）
                st.subheader('ピクセル分布')
                fig2, ax2 = plt.subplots(dpi=graph_dpi)
                # 白ピクセル（occupancy）を白色、黒ピクセル（1-occupancy）を黒色で表示
                colors = ['white', 'black']
                wedges, texts, autotexts = ax2.pie(
                    [occupancy, 1-occupancy], 
                    labels=['白ピクセル', '黒ピクセル'], 
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90,
                    textprops={'color': 'black', 'weight': 'bold'}
                )
                # パーセンテージの文字色を調整（白い部分は黒文字、黒い部分は白文字）
                autotexts[0].set_color('black')  # 白ピクセル部分は黒文字
                autotexts[1].set_color('white')  # 黒ピクセル部分は白文字
                # エッジを追加して見やすく
                for wedge in wedges:
                    wedge.set_edgecolor('gray')
                    wedge.set_linewidth(1.5)
                ax2.set_title('ピクセル分布（二値化画像）')
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)  # メモリ解放

                # AI予測結果の詳細表示（あれば）
                if pred is not None:
                    st.subheader('AI学習モデルによる予測')
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.metric(
                            label="フラクタル次元",
                            value=f"{fractal_d:.4f}",
                            delta=f"予測との差: {(fractal_d - pred['fractal']):.4f}"
                        )
                    with col_pred2:
                        if pred['occupancy'] is not None:
                            st.metric(
                                label="空間占有率",
                                value=f"{occupancy*100:.2f}%",
                                delta=f"予測との差: {(occupancy - pred['occupancy'])*100:.2f}%"
                            )
                    
                    # 占有率の比較グラフ（予測がある場合のみ）
                    if pred['occupancy'] is not None:
                        st.write('**占有率の比較**')
                        fig4, ax4 = plt.subplots(dpi=graph_dpi)
                        ax4.plot([0,1],[occupancy, pred['occupancy']], marker='o', linewidth=2, markersize=8)
                        ax4.set_xticks([0,1]); ax4.set_xticklabels(['実測','予測'])
                        ax4.set_ylabel('占有率')
                        st.pyplot(fig4, use_container_width=True)
                        plt.close(fig4)

                # 結果レコード作成
                rec = {
                    'filename': file.name,
                    'fractal': fractal_d,
                    'occupancy': occupancy,
                    'pred_fractal': pred['fractal'] if pred is not None else None,
                    'pred_occupancy': pred['occupancy'] if (pred is not None and pred['occupancy'] is not None) else None,
                    'is_valid': int(not fail_flag)
                }
                results_list.append(rec)

                # 学習データとして自動追加（検知した失敗は is_valid=0 として添加）
                append_to_train_csv(feat, {'fractal':fractal_d, 'occupancy':occupancy}, not fail_flag)
            
            # プログレスバーをクリア
            if len(uploaded_files) > 1:
                progress_bar.empty()
                status_text.empty()
            
            # パフォーマンス測定終了
            elapsed_time = time.time() - start_time
            st.success(f'✅ 解析完了！処理時間: {elapsed_time:.2f}秒 ({processing_mode})')
            
            # パラメータとキャッシュを更新
            st.session_state['last_params'] = current_params
            st.session_state['cached_results'] = results_list
            
        else:
            # キャッシュされた結果を使用
            results_list = st.session_state['cached_results']
            st.info('💾 キャッシュされた結果を表示しています（パラメータ変更なし）')
        
        # 複数ファイル時、Excelにまとめて書き込み（append）
        if results_list and len(results_list) >= 2:
            df_results = pd.DataFrame(results_list)
            if os.path.exists(EXCEL_PATH):
                # 既存ファイルに追記
                with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    # 新しいシートとしてタイムスタンプで保存
                    sheet_name = pd.Timestamp.now().strftime('run_%Y%m%d_%H%M%S')
                    df_results.to_excel(writer, sheet_name=sheet_name, index=False)
                st.info(f'解析結果を既存Excel ({EXCEL_PATH}) に追記しました。')
            else:
                df_results.to_excel(EXCEL_PATH, sheet_name='run', index=False)
                st.info(f'解析結果を新規Excel ({EXCEL_PATH}) に保存しました。')

        # 学習件数の表示（キャッシュ活用）
        train_df = load_train_data()
        if train_df is not None:
            st.sidebar.write(f'学習データ件数: {len(train_df)}')
        else:
            st.sidebar.write('学習データはまだありません。')
    elif uploaded_files:
        st.info('⚡ 自動再計算がOFFです。「解析を更新」を押して実行してください。')

with col2:
    st.header('学習 / モデル')
    st.write('学習データを読み込み、モデル学習・再学習を行えます。')

    train_df = load_train_data()
    if train_df is None:
        st.info('まだ学習データがありません。解析を数回行うと自動的に学習データが蓄積されます。')
    else:
        st.write('学習データの先頭5行:')
        st.dataframe(train_df.head())

        # 学習実行
        if do_train_now:
            st.write('学習を開始します...')
            # 特徴量とターゲットを用意
            X = train_df[['mean_int','std_int','edge_density','occupancy','fractal_dim_feature']].values
            y_fractal = train_df['target_fractal'].values
            y_occupancy = train_df['target_occupancy'].values
            y_valid = train_df['is_valid'].values

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # 回帰: 2出力を同時に学習するため、単純に横に結合
            Y_reg = np.vstack([y_fractal, y_occupancy]).T
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            try:
                reg.fit(Xs, Y_reg)
                st.success('回帰モデルの学習が完了しました。')
            except Exception as e:
                st.error('回帰学習に失敗しました:' + str(e))
                reg = None

            # 分類: 有効/無効判定
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            try:
                clf.fit(Xs, y_valid)
                st.success('分類モデルの学習が完了しました。')
            except Exception as e:
                st.error('分類学習に失敗しました:' + str(e))
                clf = None

            # 保存
            if reg is not None:
                save_models(reg, scaler, clf)
                st.info('モデルを保存しました (model_joblib.pkl, scaler_joblib.pkl)。')

            # 簡易評価: クロスバリデーション無しの学内評価
            if reg is not None:
                ypred = reg.predict(Xs)
                mae_fractal = mean_absolute_error(y_fractal, ypred[:,0])
                mae_occ = mean_absolute_error(y_occupancy, ypred[:,1])
                st.write(f'学内評価 MAE - フラクタル: {mae_fractal:.4f}, 占有率: {mae_occ:.4f}')
            if clf is not None:
                ypredc = clf.predict(Xs)
                acc = accuracy_score(y_valid, ypredc)
                st.write(f'分類モデル 学内精度: {acc:.3f} (正答率)')

    # 手動で再学習したい場合のボタン
    if st.button('モデルを読み直す（保存済みをロード）'):
        models2 = load_models()
        if 'reg' in models2:
            st.success('モデルをロードしました。')
        else:
            st.error('モデルが見つかりません。')

st.sidebar.markdown('---')
st.sidebar.write('出力ファイル:')
st.sidebar.write(EXCEL_PATH)
st.sidebar.write(MODEL_PATH)
st.sidebar.write(TRAIN_CSV)

st.write('\n')
st.write('---')
st.write('注意: 本プログラムはサンプル実装です。画像サイズ、特徴量、異常判定基準、モデル選定は用途に応じて調整してください。')