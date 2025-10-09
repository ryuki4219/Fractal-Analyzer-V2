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

# --- ユーティリティ関数 -------------------------------------------------

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


def adaptive_binarize(gray: np.ndarray):
    # ガウシアン適応閾値（サンプルとして）
    bw = cv2.adaptiveThreshold(gray.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return bw


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
# モデルロード
models = load_models()

# ファイル選択: 複数ファイルアップロードでフォルダ内一括解析に対応
uploaded_files = st.file_uploader('画像ファイルを選択（複数可）', type=['png','jpg','jpeg','bmp','tif','tiff'], accept_multiple_files=True)

# 解析/学習用の表示領域
col1, col2 = st.columns([2,1])

with col1:
    st.header('解析結果')
    if uploaded_files is not None and len(uploaded_files) > 0:
        results_list = []
        predictions = []
        for file in uploaded_files:
            st.write('ファイル:', file.name)
            img_bgr = load_image_bytes(file)
            img_bgr, scale = resize_image(img_bgr, max_side)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 二値化 (固定閾値)
            bw = binarize_image_gray(gray, thresh_value)

            # フラクタル次元計算
            fractal_d, sizes, counts = boxcount_fractal_dim(bw)
            occupancy = compute_spatial_occupancy(bw)

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
            feat = extract_features_from_image(img_bgr, bw, fractal_d)

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

            # 結果表示 - 見やすいメトリクス形式で表示
            st.subheader('📊 解析結果')
            
            # メトリクスを2列で表示
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    label="フラクタル次元",
                    value=f"{fractal_d:.4f}",
                    help="フラクタル次元は画像の複雑さを表す指標です"
                )
            with metric_col2:
                st.metric(
                    label="空間占有率（白ピクセル）",
                    value=f"{occupancy*100:.2f}%",
                    help="画像全体における白ピクセルの割合です"
                )
            
            # 予測値がある場合は追加表示
            if pred is not None:
                st.write("**🤖 機械学習モデルによる予測値:**")
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    delta_fractal = fractal_d - pred['fractal'] if pred['fractal'] is not None else None
                    st.metric(
                        label="予測フラクタル次元",
                        value=f"{pred['fractal']:.4f}" if pred['fractal'] is not None else "N/A",
                        delta=f"{delta_fractal:.4f}" if delta_fractal is not None else None,
                        delta_color="off"
                    )
                with pred_col2:
                    if pred['occupancy'] is not None:
                        delta_occupancy = (occupancy - pred['occupancy']) * 100
                        st.metric(
                            label="予測空間占有率",
                            value=f"{pred['occupancy']*100:.2f}%",
                            delta=f"{delta_occupancy:.2f}%" if delta_occupancy is not None else None,
                            delta_color="off"
                        )
            
            # 異常検知結果
            if fail_flag:
                st.warning('⚠️ 自動検知: 失敗と判定されました。理由: ' + '; '.join(fail_reasons))
            else:
                st.success('✅ 自動検知: 正常と判定')

            st.divider()

            # 元画像と二値化画像を並べて表示
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.subheader('元画像')
                # BGR -> RGB に変換して表示
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
            with col_img2:
                st.subheader('二値化画像')
                st.image(bw, use_container_width=True, clamp=True)

            # グラフ表示
            st.subheader('📈 グラフ表示')
            
            # グラフを2列で表示
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                # グラフ: フラクタル次元の折れ線（sizes vs counts から可視化）
                fig1, ax1 = plt.subplots()
                ax1.plot(np.log(1.0/sizes), np.log(counts), marker='o', linewidth=2, markersize=8)
                ax1.set_xlabel('log(1/size)')
                ax1.set_ylabel('log(count)')
                ax1.set_title('フラクタル次元解析グラフ')
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                plt.close(fig1)

            with graph_col2:
                # 円グラフ: 空間占有率
                fig2, ax2 = plt.subplots()
                # 白ピクセル（占有）と黒ピクセル（非占有）の割合
                # 色を白と黒に対応させる
                colors = ['white', 'black']
                wedgeprops = {'edgecolor': 'gray', 'linewidth': 1}
                ax2.pie([occupancy, 1-occupancy], 
                       labels=['白ピクセル', '黒ピクセル'], 
                       autopct='%1.1f%%',
                       colors=colors,
                       wedgeprops=wedgeprops,
                       startangle=90)
                ax2.set_title('空間占有率（白ピクセル vs 黒ピクセル）')
                st.pyplot(fig2)
                plt.close(fig2)

            # 予測と実測の比較プロット（あれば）
            if pred is not None:
                st.subheader('🔍 予測 vs 実測の比較')
                compare_col1, compare_col2 = st.columns(2)
                
                with compare_col1:
                    # 比較フラクタル次元グラフ
                    fig3, ax3 = plt.subplots()
                    ax3.plot([0,1],[fractal_d, pred['fractal']], marker='o', linewidth=2, markersize=10)
                    ax3.set_xticks([0,1])
                    ax3.set_xticklabels(['実測','予測'])
                    ax3.set_ylabel('フラクタル次元')
                    ax3.set_title('フラクタル次元の比較')
                    ax3.grid(True, alpha=0.3)
                    st.pyplot(fig3)
                    plt.close(fig3)
                
                with compare_col2:
                    # 比較占有率
                    if pred['occupancy'] is not None:
                        fig4, ax4 = plt.subplots()
                        ax4.plot([0,1],[occupancy, pred['occupancy']], marker='o', linewidth=2, markersize=10)
                        ax4.set_xticks([0,1])
                        ax4.set_xticklabels(['実測','予測'])
                        ax4.set_ylabel('占有率')
                        ax4.set_title('空間占有率の比較')
                        ax4.grid(True, alpha=0.3)
                        st.pyplot(fig4)
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

        # 複数ファイル時、Excelにまとめて書き込み（append）
        if len(results_list) >= 2:
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

        # 学習件数の表示
        train_df = load_train_data()
        if train_df is not None:
            st.sidebar.write(f'学習データ件数: {len(train_df)}')
        else:
            st.sidebar.write('学習データはまだありません。')

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