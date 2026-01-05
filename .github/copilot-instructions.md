# Fractal-Analyzer-V2: Copilot Instructions

目的: このリポジトリで AI コーディングエージェントが即戦力となるための、プロジェクト固有の設計・ワークフロー・慣習の要点を簡潔に共有します。

## ビッグピクチャー
- UI は Streamlit の [fractal_app.py](fractal_app.py) が中心。画像取り込み→FD 算出→AI 予測→検証→レポート生成までを一貫処理。
- FD 算出は高速ベクトル化実装が [fractal_app.py](fractal_app.py) に、参照版は [fd_boxcount.py](fd_boxcount.py) に実装（`fd_std_boxcount`, `fd_3d_dbc`）。CuPy があれば自動で GPU を使用（`xp` エイリアス、`USE_CUPY`）。
- 低画質→高画質 FD の補正は LightGBM（`LGBMRegressor`）。モデルと履歴は永続化（`trained_fd_model.pkl`, `training_history.json`）。
- 画像解析と補助: 顔検出・部位抽出は [skin_analysis.py](skin_analysis.py)、実験ログ・相関・図作成は [experiment_analysis.py](experiment_analysis.py)。レポート生成は [generate_updated_report.py](generate_updated_report.py)（PDF、ReportLab）と [scripts/make_template_docx.py](scripts/make_template_docx.py)（Word テンプレート）。

## 主要ワークフロー
- アプリ起動（Windows）：`streamlit run fractal_app.py`（必要なら `--server.port 8501`）。
- FD 算出の CLI 検証：[fd_boxcount.py](fd_boxcount.py) を `python fd_boxcount.py <image_path>` で実行し 2D/DBC の FD を確認。
- 実験ログ: `ExperimentDataManager.save_data()` が [experimental_data.csv](experimental_data.csv) に追記。相関・散布図作成は `calculate_correlations()`, `create_scatter_plot()`。
- レポート生成: PDF は [generate_updated_report.py](generate_updated_report.py)、Word テンプレは `python scripts\make_template_docx.py` 実行後に編集（[docs/templates](docs/templates)）。

## データと永続化
- 画像ペア命名: `IMG_XXXX.jpg` ↔ `IMG_XXXX_low1.jpg`（Low4–7 含む）。UI はこの規約を前提とするフローあり。
- モデルと履歴: ルート直下 `trained_fd_model.pkl` と `training_history.json` を自動ロード/保存。
- データセット: [SKIN_DATA/](SKIN_DATA)（`Facial Skin Condition Dataset.csv` と例画像）。日本語パス対応の堅牢ローダー（OpenCV のバッファ読み）を使用。
- EXIF 読み取りは [fractal_app.py](fractal_app.py) 内 `extract_exif_data()`（PIL があれば活用）。

## 慣習・パターン
- 画像は内部 BGR（`cv2.imdecode`）。FD 算出時は `cv2.COLOR_BGR2GRAY` でグレースケール化。
- FD は複数スケール箱サイズで算出し、傾きから FD を推定。外れ値保護しつつ 2.0–3.0 にクリップ。
- GPU 切替は自動（`cupy` 有無）。数値演算は `xp`（numpy/cupy）の別名で記述。ホスト転送用の補助関数に合わせる実装を推奨。
- 顔領域: MediaPipe ランドマーク優先、OpenCV Haar/dlib フォールバック、最終手段でヒューリスティック矩形。`extract_face_regions(image, landmarks)` API に合わせる。
- データ拡張（`augment_image`）は左右反転、回転、明暗/コントラスト、ガンマ、ノイズ/ブラー/シャープなどを高/低画質ペアに対して対称適用。
- 可視化は Matplotlib 基本、Plotly は `PLOTLY_AVAILABLE` が True の場合に使用。

## 連携ポイント・外部依存
- 任意モジュールは条件付き取込: `skin_quality_evaluator.py`, `image_quality_assessor.py` が無ければ機能を自動的に無効化。壊れない降格設計を維持。
- Streamlit 状態: `st.session_state` で設定・中間結果を保持。UI セクションは `st.subheader`, `st.metric`, `st.dataframe`, `st.plotly_chart` の既存構造に従う。
- ReportLab（PDF）は Windows 日本語フォント（Meiryo/YuGothic/MSMincho）を優先登録して日本語表示を確保。

## 具体例（抜粋）
- FD を計算（2D 標準偏差ボックスカウント）:
  ```python
  import cv2, numpy as np
  from fd_boxcount import fd_std_boxcount
  img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
  fd, scales, Nh, *_ = fd_std_boxcount(img)
  ```
- LightGBM 学習（アプリ内）: 低画質特徴→高画質 FD を予測。`LGBMRegressor(..., n_jobs=-1)` を用い、`trained_fd_model.pkl` へ保存。

## 注意点
- `opencv-python-headless` を使用（GUI ウィンドウ非表示）。表示は Streamlit/プロットに依存。
- 日本語パス/フォントを想定。画像はバッファ読み、PDF は Windows フォント登録で対応。
- 相関は分散ほぼゼロで `nan` になりうるため、UI でガード表示。単一画像の部位別は不安定で方向性確認に留める設計（詳細は [generate_updated_report.py](generate_updated_report.py)）。
- FD 傾き範囲の妥当性を担保し、スパイク値をクリップ。撮影条件差は解析・検出に影響するため標準化を推奨。

## クイックコマンド（PowerShell）
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
streamlit run fractal_app.py --server.port 8501
python fd_boxcount.py SKIN_DATA\1\IMG_0001.jpg
python scripts\make_template_docx.py
```

不明瞭・不足しているセクションがあれば指摘してください。実際の運用フローや UI 流れに合わせて追記・修正します。
