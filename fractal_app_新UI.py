"""
Streamlit アプリ: フラクタルを用いた画像解析アプリ
【新UIバージョン】コンパクト3カラムレイアウト + 色覚バリアフリー配色

機能:
- 閾値（スライダー/数値入力）とリサイズ上限（スライダー/数値入力）の2方式を用意
- フラクタル次元は折れ線グラフで出力、空間占有率は円グラフで出力
- 学習機能: 解析結果（有効/失敗）を学習し、予測結果と比較表示
- 異常値・異常な二値化の自動検知（失敗扱い） -> 学習用データに追加
- フォルダ内画像を一括解析（Streamlitの仕様上、複数ファイルアップロードで対応）
- 2枚以上解析時、自動でExcelに結果を保存・追記
- 学習件数の表示、解析精度（MAEなど）の表示
- 解像度補正AI: 低解像度画像から高解像度相当のフラクタル次元を推定

使い方:
1) 必要ライブラリをインストール: pip install -r requirements.txt
2) 実行: streamlit run fractal_app_新UI.py

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
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# === 色覚バリアフリー配色定数 (Okabe-Ito/Wong palette) ===
COLOR_BLUE = '#0173B2'        # 青 (実測データ)
COLOR_GREEN = '#029E73'       # 緑 (AI補正)
COLOR_ORANGE = '#DE8F05'      # オレンジ (従来AI)
COLOR_SKY = '#56B4E9'         # 水色 (円グラフ・白)
COLOR_YELLOW = '#ECE133'      # 黄色 (円グラフ・黒)
COLOR_PINK = '#CC78BC'        # ピンク (警告)
COLOR_VERMILION = '#D55E00'   # 朱色 (エラー)

# --- 解像度補正モデル用の定数 ----------------------------------------------
RESOLUTION_MODEL_PATH = 'resolution_correction_model.pkl'
RESOLUTION_SCALER_PATH = 'resolution_correction_scaler.pkl'
RESOLUTION_TRAIN_DATA = 'resolution_training_data.csv'

# --- ユーティリティ関数 -------------------------------------------------

def load_image_bytes(file) -> np.ndarray:
    bytes_data = file.read()
    img = Image.open(io.BytesIO(bytes_data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1].copy()  # RGB->BGR
    return arr


def resize_image(img: np.ndarray, max_side: float):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side and max_side > 0:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


def binarize_image_gray(gray: np.ndarray, thresh: float):
    _, bw = cv2.threshold(gray.astype('uint8'), thresh, 255, cv2.THRESH_BINARY)
    return bw


def adaptive_binarize(gray: np.ndarray):
    bw = cv2.adaptiveThreshold(gray.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return bw


def boxcount_fractal_dim(bw: np.ndarray, sizes=None):
    S = bw.shape
    if sizes is None:
        max_power = min(int(np.log2(min(S))), 10)
        sizes = np.array([2 ** i for i in range(1, max_power)])
        sizes = sizes[sizes <= min(S)]
        if len(sizes) < 3:
            sizes = np.array([2,4,8,16])
    counts = []
    bw_binary = (bw > 0).astype(np.uint8)
    for size in sizes:
        ny = int(np.ceil(S[0] / size))
        nx = int(np.ceil(S[1] / size))
        count = 0
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
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(counts)
        loginv = np.log(1.0 / sizes)
    A = np.vstack([loginv, np.ones_like(loginv)]).T
    try:
        m, c = np.linalg.lstsq(A, logs, rcond=None)[0]
    except Exception:
        m = 0.0
    return float(m), sizes, counts


def compute_spatial_occupancy(bw: np.ndarray):
    total = bw.size
    white = np.count_nonzero(bw > 0)
    return float(white / total)


def extract_features_from_image(img_bgr: np.ndarray, bw: np.ndarray, fractal_dim: float):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_int = float(np.mean(gray))
    std_int = float(np.std(gray))
    edge = canny(gray / 255.0)
    edge_density = float(np.count_nonzero(edge) / edge.size)
    occupancy = compute_spatial_occupancy(bw)
    return np.array([mean_int, std_int, edge_density, occupancy, fractal_dim], dtype=float)

# --- 解像度補正モデル用の関数 ---------------------------------------------

def generate_low_resolution_versions(img_bgr: np.ndarray, scale_factors=[0.5, 0.25, 0.1]):
    low_res_images = []
    for scale in scale_factors:
        h, w = img_bgr.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h > 10 and new_w > 10:
            low_res = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            upscaled = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
            low_res_images.append((scale, upscaled))
    return low_res_images


def extract_resolution_features(img_bgr: np.ndarray, bw: np.ndarray, fractal_dim: float):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    mean_int = float(np.mean(gray))
    std_int = float(np.std(gray))
    
    edge = canny(gray / 255.0)
    edge_density = float(np.count_nonzero(edge) / edge.size)
    
    occupancy = compute_spatial_occupancy(bw)
    
    kernel_size = max(3, min(h, w) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    local_std = cv2.blur(gray.astype(float)**2, (kernel_size, kernel_size)) - \
                cv2.blur(gray.astype(float), (kernel_size, kernel_size))**2
    texture_variance = float(np.mean(np.sqrt(np.abs(local_std))))
    
    img_size = float(np.log(h * w + 1))
    aspect_ratio = float(w / h)
    
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    high_freq_energy = float(np.mean(magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]))
    
    return np.array([
        mean_int, std_int, edge_density, occupancy, fractal_dim,
        texture_variance, img_size, aspect_ratio, high_freq_energy
    ], dtype=float)


def train_resolution_correction_model(training_data_path=RESOLUTION_TRAIN_DATA):
    if not os.path.exists(training_data_path):
        return None, None, "学習データが見つかりません"
    
    df = pd.read_csv(training_data_path)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X = df[feature_cols].values
    y = df['target_high_res_fractal'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    joblib.dump(model, RESOLUTION_MODEL_PATH)
    joblib.dump(scaler, RESOLUTION_SCALER_PATH)
    
    return model, scaler, f"MAE: {mae:.4f}, R²: {r2:.4f}"


def predict_high_res_fractal(low_res_features, model=None, scaler=None):
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

def append_to_train_csv(features, y_reg, is_valid):
    cols = ['mean_int', 'std_int', 'edge_density', 'occupancy', 'fractal_dim_feature',
            'target_fractal', 'target_occupancy', 'is_valid']
    row = list(features) + [y_reg['fractal'], y_reg['occupancy'], int(is_valid)]
    df = pd.DataFrame([row], columns=cols)
    if os.path.exists(TRAIN_CSV):
        df.to_csv(TRAIN_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(TRAIN_CSV, index=False)


def load_train_data():
    if os.path.exists(TRAIN_CSV):
        return pd.read_csv(TRAIN_CSV)
    else:
        return None

# ============================================================================
# Streamlit UI - コンパクト3カラムレイアウト
# ============================================================================

st.set_page_config(layout='wide', page_title='フラクタル画像解析アプリ (新UI)')
st.title('🔬 フラクタルを用いた画像解析アプリ')
st.caption('**新UI**: コンパクト3カラムレイアウト + 色覚バリアフリー配色')

# アプリケーション概要
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
        """)

# === サイドバー設定 ===
st.sidebar.header('⚙️ 設定')

# 設定をポップオーバー内に集約
with st.sidebar.popover('📐 解析パラメータ設定', help='閾値とリサイズの詳細設定'):
    thresh_mode = st.selectbox('閾値入力方式', ['スライダー', '数値入力'])
    if thresh_mode == 'スライダー':
        thresh_value = st.slider('二値化閾値 (0-255)', min_value=0.0, max_value=255.0, value=128.0)
    else:
        thresh_value = st.number_input('二値化閾値 (0-255)', min_value=0.0, max_value=255.0, value=128.0, step=0.1)
    
    resize_mode = st.selectbox('リサイズ方式', ['スライダー', '数値入力'])
    if resize_mode == 'スライダー':
        max_side = st.slider('リサイズ最大辺 (px, 0でリサイズ無効)', min_value=0.0, max_value=4000.0, value=1024.0)
    else:
        max_side = st.number_input('リサイズ最大辺 (px, 0でリサイズ無効)', min_value=0.0, max_value=10000.0, value=1024.0)

st.sidebar.markdown('---')
st.sidebar.subheader('🔬 解像度補正AI')
enable_resolution_correction = st.sidebar.checkbox('解像度補正を有効化', value=False, 
    help='低解像度画像から高解像度相当のフラクタル次元を推定します')

if st.sidebar.checkbox('学習データ生成モード', value=False, 
    help='高解像度画像から低解像度バージョンを自動生成し、学習データを作成します'):
    st.sidebar.info('アップロードした画像から学習データを自動生成します')
    generate_training_data = True
else:
    generate_training_data = False

if st.sidebar.button('解像度補正モデルを学習'):
    with st.spinner('解像度補正モデルを学習中...'):
        model_res, scaler_res, result_msg = train_resolution_correction_model()
        if model_res is not None:
            st.sidebar.success(f'学習完了: {result_msg}')
        else:
            st.sidebar.error(result_msg)

if os.path.exists(RESOLUTION_MODEL_PATH):
    st.sidebar.success('✓ 解像度補正モデル: 読み込み済み')
    if os.path.exists(RESOLUTION_TRAIN_DATA):
        df_res = pd.read_csv(RESOLUTION_TRAIN_DATA)
        st.sidebar.write(f'学習データ数: {len(df_res)}件')
else:
    st.sidebar.warning('解像度補正モデル: 未学習')

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
    """)

st.sidebar.markdown('---')
do_train_now = st.sidebar.button('🎯 学習を実行（保存済みデータで再学習）')

# モデルロード
models = load_models()

# ファイル選択
st.markdown('## 📁 画像アップロード')
uploaded_files = st.file_uploader('画像ファイルを選択（複数可）', type=['png','jpg','jpeg','bmp','tif','tiff'], accept_multiple_files=True)

# 学習データ生成モード時のガイド表示
if generate_training_data:
    st.info('🔄 **学習データ生成モード** - 高解像度画像をアップロードすると、自動的に5段階の解像度で学習データを生成します')

# 解像度補正有効時のガイド表示
if enable_resolution_correction and os.path.exists(RESOLUTION_MODEL_PATH):
    st.success('✅ **解像度補正AI有効** - 低解像度画像でも高解像度相当のフラクタル次元を推定します')

st.divider()

# === メイン解析エリア ===
if uploaded_files is not None and len(uploaded_files) > 0:
    results_list = []
    
    for idx, file in enumerate(uploaded_files):
        st.markdown(f"### 📁 {file.name}")
        
        img_bgr = load_image_bytes(file)
        
        # 学習データ生成モードの処理
        if generate_training_data:
            with st.spinner('🔄 学習データ生成中...'):
                img_high, _ = resize_image(img_bgr, max_side)
                gray_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY)
                bw_high = binarize_image_gray(gray_high, thresh_value)
                fractal_high, _, _ = boxcount_fractal_dim(bw_high)
                
                low_res_versions = generate_low_resolution_versions(img_high, [0.5, 0.3, 0.2, 0.15, 0.1])
                training_records = []
                for scale, img_low in low_res_versions:
                    gray_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
                    bw_low = binarize_image_gray(gray_low, thresh_value)
                    fractal_low, _, _ = boxcount_fractal_dim(bw_low)
                    features_low = extract_resolution_features(img_low, bw_low, fractal_low)
                    record = {'scale': scale, 'target_high_res_fractal': fractal_high}
                    for feat_idx, feat_val in enumerate(features_low):
                        record[f'feat_{feat_idx}'] = feat_val
                    training_records.append(record)
                
                df_new = pd.DataFrame(training_records)
                if os.path.exists(RESOLUTION_TRAIN_DATA):
                    df_new.to_csv(RESOLUTION_TRAIN_DATA, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(RESOLUTION_TRAIN_DATA, index=False)
            
            st.success(f'✓ {len(training_records)}件の学習データを生成しました（解像度: {[f"{s*100:.0f}%" for s, _ in low_res_versions]}）')
            continue
        
        # 通常解析処理
        img_bgr, scale = resize_image(img_bgr, max_side)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bw = binarize_image_gray(gray, thresh_value)
        
        fractal_d, sizes, counts = boxcount_fractal_dim(bw)
        occupancy = compute_spatial_occupancy(bw)
        
        # 解像度補正AI
        corrected_fractal_d = None
        if enable_resolution_correction and os.path.exists(RESOLUTION_MODEL_PATH):
            try:
                features_res = extract_resolution_features(img_bgr, bw, fractal_d)
                corrected_fractal_d = predict_high_res_fractal(features_res)
            except Exception as e:
                st.warning(f'解像度補正エラー: {e}')
        
        # 品質チェック
        fail_flag = False
        fail_reasons = []
        if occupancy < 0.01:
            fail_flag = True
            fail_reasons.append('ほとんど白が無い(占有率 <1%)')
        if occupancy > 0.99:
            fail_flag = True
            fail_reasons.append('ほとんど白で埋まっている(占有率 >99%)')
        if not (-5.0 < fractal_d < 5.0):
            fail_flag = True
            fail_reasons.append(f'フラクタル次元が異常値:{fractal_d:.3f}')
        
        feat = extract_features_from_image(img_bgr, bw, fractal_d)
        
        # 従来AI予測
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
                pass
        
        # === 3カラムレイアウト ===
        col_left, col_center, col_right = st.columns([1.0, 1.6, 1.0])
        
        # 【左カラム】画像プレビュー & 設定
        with col_left:
            st.markdown('#### 🖼️ 画像')
            preview_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption='元画像', use_container_width=True)
            st.image(bw, caption='二値化画像', use_container_width=True)
            
            with st.popover('⚙️ 解析パラメータ', help='現在の設定値を確認'):
                st.caption(f'**閾値**: {thresh_value:.1f}')
                st.caption(f'**リサイズ上限**: {max_side:.0f} px' if max_side > 0 else '**リサイズ上限**: なし')
                st.caption(f'**画像サイズ**: {img_bgr.shape[1]} × {img_bgr.shape[0]} px')
                st.caption(f'**リサイズ率**: {scale*100:.1f}%')
                st.caption(f'**解像度補正AI**: {"✓ 有効" if enable_resolution_correction else "無効"}')
        
        # 【中央カラム】メトリクス & グラフ
        with col_center:
            st.markdown('#### 📊 解析結果')
            
            # メトリクス表示
            metric_row = st.columns(3)
            metric_row[0].metric(
                '実測フラクタル次元',
                f'{fractal_d:.4f}',
                help='Box-Counting法で計測'
            )
            
            if corrected_fractal_d is not None:
                delta_val = corrected_fractal_d - fractal_d
                metric_row[1].metric(
                    'AI補正フラクタル次元',
                    f'{corrected_fractal_d:.4f}',
                    delta=f'{delta_val:+.4f}',
                    help='解像度補正AIによる推定値'
                )
            else:
                metric_row[1].metric(
                    'AI補正フラクタル次元',
                    '－',
                    help='解像度補正モデル未学習'
                )
            
            metric_row[2].metric(
                '空間占有率',
                f'{occupancy*100:.2f}%',
                help='白ピクセルの割合'
            )
            
            # グラフ表示
            st.markdown('##### 📈 Box-Counting解析')
            x_vals = np.log(1.0 / sizes)
            y_vals = np.log(counts)
            
            fig_main, (ax_line, ax_pie) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=90, 
                                                        gridspec_kw={'width_ratios': [2.3, 1]})
            
            # 折れ線グラフ (色覚バリアフリー)
            ax_line.plot(x_vals, y_vals, marker='o', linewidth=2.5, markersize=8, 
                        color=COLOR_BLUE, label=f'実測 (D={fractal_d:.3f})', zorder=3)
            
            if corrected_fractal_d is not None:
                intercept = np.mean(y_vals - corrected_fractal_d * x_vals)
                y_corrected = corrected_fractal_d * x_vals + intercept
                ax_line.plot(x_vals, y_corrected, marker='^', linewidth=2.2, markersize=7, 
                            color=COLOR_GREEN, linestyle='-.', alpha=0.9,
                            label=f'AI補正 (D={corrected_fractal_d:.3f})', zorder=2)
            
            if pred is not None and 'fractal' in pred:
                intercept_pred = np.mean(y_vals - pred['fractal'] * x_vals)
                y_pred_line = pred['fractal'] * x_vals + intercept_pred
                ax_line.plot(x_vals, y_pred_line, marker='s', linewidth=2, markersize=6, 
                            color=COLOR_ORANGE, linestyle='--', alpha=0.85,
                            label=f'従来AI (D={pred["fractal"]:.3f})', zorder=1)
            
            ax_line.set_xlabel('log(1/箱サイズ)', fontsize=10, fontweight='bold')
            ax_line.set_ylabel('log(カウント数)', fontsize=10, fontweight='bold')
            ax_line.grid(True, alpha=0.3, linestyle=':', linewidth=1)
            ax_line.legend(loc='best', fontsize=9, framealpha=0.95)
            ax_line.set_title('フラクタル次元の比較', fontsize=11, fontweight='bold', pad=10)
            
            # 円グラフ (色覚バリアフリー)
            wedges, texts, autotexts = ax_pie.pie(
                [occupancy, 1 - occupancy],
                labels=['白', '黒'],
                autopct='%1.1f%%',
                colors=[COLOR_SKY, COLOR_YELLOW],
                startangle=90,
                textprops={'fontsize': 10, 'weight': 'bold'},
                wedgeprops={'edgecolor': '#333', 'linewidth': 1.5}
            )
            for autotext in autotexts:
                autotext.set_color('#000')
                autotext.set_fontsize(11)
            ax_pie.set_title('ピクセル分布', fontsize=11, fontweight='bold', pad=10)
            
            plt.tight_layout()
            st.pyplot(fig_main, use_container_width=True)
            plt.close(fig_main)
        
        # 【右カラム】詳細診断 & AI比較
        with col_right:
            st.markdown('#### 🔍 詳細情報')
            
            # 品質判定
            if fail_flag:
                st.error('⚠️ 品質: 要確認')
                with st.expander('異常検出の詳細', expanded=True):
                    for reason in fail_reasons:
                        st.warning(f'• {reason}')
            else:
                st.success('✓ 品質: 正常')
            
            # 従来AIとの比較
            if pred is not None:
                with st.expander('🤖 従来AI予測', expanded=False):
                    st.metric(
                        '予測フラクタル次元',
                        f"{pred['fractal']:.4f}",
                        delta=f"{fractal_d - pred['fractal']:+.4f}",
                        delta_color="off"
                    )
                    if pred['occupancy'] is not None:
                        st.metric(
                            '予測占有率',
                            f"{pred['occupancy']*100:.2f}%",
                            delta=f"{(occupancy - pred['occupancy'])*100:+.2f}%",
                            delta_color="off"
                        )
            
            # AI補正情報
            if corrected_fractal_d is not None:
                with st.expander('🔬 AI補正の詳細', expanded=False):
                    st.write(f'**補正前**: {fractal_d:.4f}')
                    st.write(f'**補正後**: {corrected_fractal_d:.4f}')
                    st.write(f'**差分**: {corrected_fractal_d - fractal_d:+.4f}')
                    improvement = abs(corrected_fractal_d - fractal_d) / max(abs(fractal_d), 0.0001) * 100
                    st.write(f'**変化率**: {improvement:.2f}%')
            
            # 画像統計情報
            with st.expander('📐 画像統計', expanded=False):
                st.write(f'**平均輝度**: {np.mean(gray):.2f}')
                st.write(f'**標準偏差**: {np.std(gray):.2f}')
                st.write(f'**元サイズ**: {img_bgr.shape[1]} × {img_bgr.shape[0]} px')
                st.write(f'**白ピクセル数**: {int(occupancy * bw.size):,}')
                st.write(f'**黒ピクセル数**: {int((1-occupancy) * bw.size):,}')
        
        # データ保存
        rec = {
            'filename': file.name,
            'fractal': fractal_d,
            'occupancy': occupancy,
            'corrected_fractal': corrected_fractal_d,
            'pred_fractal': pred['fractal'] if pred is not None else None,
            'pred_occupancy': pred['occupancy'] if (pred is not None and pred['occupancy'] is not None) else None,
            'is_valid': int(not fail_flag)
        }
        results_list.append(rec)
        append_to_train_csv(feat, {'fractal': fractal_d, 'occupancy': occupancy}, not fail_flag)
        
        st.divider()
    
    # 複数ファイル時、Excelにまとめて書き込み
    if len(results_list) >= 2:
        df_results = pd.DataFrame(results_list)
        if os.path.exists(EXCEL_PATH):
            with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                sheet_name = pd.Timestamp.now().strftime('run_%Y%m%d_%H%M%S')
                df_results.to_excel(writer, sheet_name=sheet_name, index=False)
            st.info(f'✅ 解析結果を既存Excel ({EXCEL_PATH}) に追記しました。')
        else:
            df_results.to_excel(EXCEL_PATH, sheet_name='run', index=False)
            st.info(f'✅ 解析結果を新規Excel ({EXCEL_PATH}) に保存しました。')

else:
    st.info('👆 上部のアップローダーから解析したい画像を選択してください。')

# === 学習セクション (Expander内に格納) ===
st.divider()
with st.expander('🤖 学習 / モデル管理', expanded=False):
    st.markdown('### モデル学習ワークフロー')
    train_df = load_train_data()
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        st.subheader('📚 学習データ')
        if train_df is None:
            st.info('まだ学習データがありません。解析を数回行うと自動的に蓄積されます。')
        else:
            st.metric('学習データ件数', f'{len(train_df)}件')
            with st.expander('データプレビュー (先頭10件)', expanded=False):
                st.dataframe(train_df.head(10), use_container_width=True)
    
    with model_col2:
        st.subheader('⚙️ モデル設定')
        if train_df is not None and len(train_df) > 0:
            st.success(f'✓ 学習可能なデータがあります（{len(train_df)}件）')
        else:
            st.warning('学習データが不足しています')
    
    st.markdown('---')
    if train_df is not None and len(train_df) > 0:
        if do_train_now:
            st.write('学習を開始します...')
            X = train_df[['mean_int','std_int','edge_density','occupancy','fractal_dim_feature']].values
            y_fractal = train_df['target_fractal'].values
            y_occupancy = train_df['target_occupancy'].values
            y_valid = train_df['is_valid'].values
            
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            
            Y_reg = np.vstack([y_fractal, y_occupancy]).T
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            try:
                reg.fit(Xs, Y_reg)
                st.success('回帰モデルの学習が完了しました。')
            except Exception as e:
                st.error('回帰学習に失敗しました:' + str(e))
                reg = None
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            try:
                clf.fit(Xs, y_valid)
                st.success('分類モデルの学習が完了しました。')
            except Exception as e:
                st.error('分類学習に失敗しました:' + str(e))
                clf = None
            
            if reg is not None:
                save_models(reg, scaler, clf)
                st.info('モデルを保存しました (model_joblib.pkl, scaler_joblib.pkl)。')
            
            if reg is not None:
                ypred = reg.predict(Xs)
                mae_fractal = mean_absolute_error(y_fractal, ypred[:,0])
                mae_occ = mean_absolute_error(y_occupancy, ypred[:,1])
                st.write(f'学内評価 MAE - フラクタル: {mae_fractal:.4f}, 占有率: {mae_occ:.4f}')
            if clf is not None:
                ypredc = clf.predict(Xs)
                acc = accuracy_score(y_valid, ypredc)
                st.write(f'分類モデル 学内精度: {acc:.3f} (正答率)')
    
    if st.button('モデルを読み直す(保存済みをロード)', key='reload_model'):
        models2 = load_models()
        if 'reg' in models2:
            st.success('モデルをロードしました。')
        else:
            st.error('モデルが見つかりません。')

# サイドバー学習データ件数表示
train_df_sidebar = load_train_data()
if train_df_sidebar is not None:
    st.sidebar.markdown('---')
    st.sidebar.write(f'📊 学習データ件数: **{len(train_df_sidebar)}**件')

st.sidebar.markdown('---')
st.sidebar.write('**出力ファイル:**')
st.sidebar.caption(f'📁 {EXCEL_PATH}')
st.sidebar.caption(f'📁 {MODEL_PATH}')
st.sidebar.caption(f'📁 {TRAIN_CSV}')

st.markdown('---')
st.caption('💡 **注意**: 本プログラムはサンプル実装です。画像サイズ、特徴量、異常判定基準、モデル選定は用途に応じて調整してください。')
st.caption('🎨 **色覚バリアフリー配色**: Okabe-Ito/Wong paletteを採用しています。')
