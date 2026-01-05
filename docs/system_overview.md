# 顔画像評価システム: アーキテクチャと処理概要 (Fractal-Analyzer-V2)

本ドキュメントは、ChatGPT等のLLMが本システムを迅速に理解できるように、目的、設計思想、主要構成、データフロー、アルゴリズム、UI・ワークフロー、依存関係、運用上の慣習を体系化して説明します。

## 1. 目的と設計思想
- 目的: 低画質画像からフラクタル次元(FD)を解析し、AI補正により高画質相当のFDを推定。肌状態指標とFDの関係を検証し、研究・教育で活用。
- 設計思想:
  - 一貫したUIフロー: 画像取り込み→FD算出→AI予測→検証→レポート生成。
  - 降格設計: 外部モジュールが無い場合も壊れず機能縮退して継続。
  - 実運用重視: 日本語パス、Windowsフォント、GPU自動切替、統計的可視化に対応。

## 2. 全体構成 (Big Picture)
- 中心UI: [fractal_app.py](fractal_app.py) (Streamlit)。セッション管理、入出力、プロット、モデル永続化。
- FD算出: [fd_boxcount.py](fd_boxcount.py)
  - `fd_std_boxcount(img_bgr)`: 2D強度場の標準偏差ボックスカウント法。
  - `fd_3d_dbc(img_bgr)`: 擬似3DのDifferential Box Counting (高さ=グレースケール)。
- 画像解析: [skin_analysis.py](skin_analysis.py)
  - 顔検出: MediaPipe優先 → OpenCV Haar → dlib → ヒューリスティック矩形のフォールバック。
  - 部位抽出・肌トラブル推定の補助関数。
- 実験・図表: [experiment_analysis.py](experiment_analysis.py)
  - CSV追記、相関計算、散布図/ヒートマップ作成、実験要約。
- レポート生成: [generate_updated_report.py](generate_updated_report.py) (PDF/ReportLab)、[scripts/make_template_docx.py](scripts/make_template_docx.py) (Wordテンプレ)。
- モデル永続化: ルート直下に `trained_fd_model.pkl`, `training_history.json` を保存/復元。

## 3. データフロー (入力→特徴→推定→可視化→保存)
1) 画像取り込み: バッファ読込で日本語パス対応 (`cv2.imdecode(np.fromfile(...))`)。
2) FD算出:
   - 2D: 箱サイズスケール `h` ごとにブロック標準偏差→正規化合計 `N(h)`。
   - 擬似3D: 高さ量子化でボックス数 `N(r)` を数え上げ。
   - 回帰: 対数空間で直線当てはめ。$\log N(h)$ vs $\log h$ の傾き $a$。
     - 2D FD: $FD = |a|$, クリップ $FD \in [2, 3]$。
     - DBC FD: $FD_{3} = 3 - |a|$, クリップ $FD_{3} \in [2, 3]$。
3) AI補正: LightGBM (`LGBMRegressor`) により低画質特徴から高画質FDを予測。
4) 可視化/検証: 散布図・相関・回帰線、FD比較ラインプロット、誤差指標(MAE, R²等)。
5) 保存: 実験行を [experimental_data.csv](experimental_data.csv) に追記。モデル・履歴を永続化。

## 4. アルゴリズム詳細 (FD 算出)
- 2D標準偏差ボックスカウント:
  - グレースケール化→スケール `h` で画像を (h×h) ブロックへ分割。
  - ブロック標準偏差 `std_blk` を `nh = std_blk / h` に正規化、全ブロック合計 `N(h)`。
  - $\log N(h)$ と $\log h$ の線形回帰で傾きを取得し、$FD=|\text{slope}|$ を [2,3] にクリップ。
- 擬似3D DBC:
  - グレースケールを [0,1] 高さに正規化し、ブロックごとに最小/最大を量子化。
  - `n_r = ceil(max/G) - floor(min/G)` (G=1/r) を合計して `N(r)` を得る。
  - $FD_{3} = 3 - |\text{slope}|$ を [2,3] にクリップ。

## 5. GPU 切替と数値演算
- `cupy` が利用可能なら `USE_CUPY=True` とし、`xp` (numpy/cupy) 別名で数値処理。
- ホスト転送の必要に応じて `xp`→`numpy` 変換を行う。GPUが無い環境でも同一APIで動作。

## 6. 顔検出とフォールバック設計
- MediaPipe Face Mesh (利用可なら最優先)。複数前処理を試行 (CLAHE, ガンマ補正等)。
- OpenCV Haar Cascade: パラメータをリトライし、最良の矩形を選択。
- dlib検出器: 68点ランドマークがあれば活用。
- ヒューリスティック矩形: 中央推定で頑健に解析継続。
- `extract_face_regions(image, landmarks)` の互換APIで部位抽出に接続。

## 7. データ拡張 (augment_image)
- 左右/上下反転、回転、明暗・コントラスト調整、ガンマ補正、ノイズ/ブラー/シャープ、彩度変更など。
- 高/低画質ペアに対して対称適用し、学習データ不足を補う。

## 8. UIとワークフロー (Streamlit)
- セッション状態 `st.session_state` に中間結果や設定を保持。
- セクション構成: `st.subheader`, `st.metric`, `st.dataframe`, `st.plotly_chart` を踏襲。
- 表示は `opencv-python-headless` 前提。GUIは使用せず、ブラウザで可視化。
- クイック起動:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  streamlit run fractal_app.py --server.port 8501
  ```

## 9. 実験ログ・相関・図作成
- 追記保存: `ExperimentDataManager.save_data()` で [experimental_data.csv](experimental_data.csv)。
- 相関計算: `calculate_correlations()` でFDと各指標の相関・p値を算出。
- 散布図/回帰線: `create_scatter_plot()` により関係性を可視化。
- 注意: 分散が極小の場合は `nan` → UIでガード。

## 10. レポート生成
- PDFレポート: [generate_updated_report.py](generate_updated_report.py) (Windows日本語フォント登録: Meiryo/YuGothic/MSMincho)。
- Wordテンプレ生成: [scripts/make_template_docx.py](scripts/make_template_docx.py) → [docs/templates](docs/templates)。

## 11. 命名規約と永続化
- 画像ペア命名: `IMG_XXXX.jpg` ↔ `IMG_XXXX_low1.jpg`（Low4–7含む）。UIフローが依存。
- モデル/履歴: `trained_fd_model.pkl`, `training_history.json` を自動ロード/保存。
- データセット: [SKIN_DATA/](SKIN_DATA) にCSVと例画像。日本語パス対応のバッファ読み。

## 12. 依存関係 (requirements)
- 中核: streamlit, numpy, pandas, opencv-python-headless, scikit-learn, matplotlib, Pillow, scipy, plotly, seaborn, lightgbm, python-docx。
- オプション: cupy (GPU加速), mediapipe, dlib (存在すれば利用)。

## 13. 具体例 (FD算出とAI補正)
- 2D FD計算:
  ```python
  import cv2, numpy as np
  from fd_boxcount import fd_std_boxcount
  img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
  fd, scales, Nh, *_ = fd_std_boxcount(img)
  ```
- LightGBMで予測:
  ```python
  from lightgbm import LGBMRegressor
  model = LGBMRegressor(n_estimators=200, max_depth=-1, n_jobs=-1)
  model.fit(X_low_quality, y_high_quality_fd)
  # 保存
  import pickle
  with open('trained_fd_model.pkl','wb') as f:
      pickle.dump(model, f)
  ```

## 14. 研究的注意点
- FDの傾き範囲とスパイク値はクリップ・検証。
- 撮影条件差(照明/露出/焦点)はFD・検出に影響→標準化運用を推奨。
- 単一画像の部位別は不安定→方向性確認に留め、結論は集団分析で。

## 15. まとめ
- 本システムは、FD解析とAI補正を中核に、降格設計・日本語環境対応・統計可視化を備えた顔画像評価プラットフォーム。
- 研究・教育に適し、Streamlit UIで再現性と運用を両立。PDF/Docxで成果を出力。

---

## 16. 肌評価の考え方とシステム実装（追記）

### 評価の狙い
- 表面の不規則性が増えるほど FD が高くなるという仮説に基づき、肌テクスチャ（毛穴・シワ・色ムラなど）を FD で定量化し、画像解析のトラブル検出指標と統合して評価。

### 評価パイプライン
- 入力画像の取得: 日本語パスでも安全なバッファ読み（例は fd_boxcount.py）。
- 顔・部位抽出: MediaPipe優先、OpenCV/dlibフォールバック、最終手段の中央矩形（skin_analysis.py）。
- FD算出: 2D標準偏差ボックスカウントと擬似3D DBC（fd_boxcount.py）。
- トラブル検出: 毛穴・シワ・赤み・色ムラ等の指標を画像特徴から推定（skin_analysis.py）。
- 集計と保存: 部位別のFDとトラブル指標を集計し、スコア化・保存・可視化（fractal_app.py、experiment_analysis.py）。

### FDの計算（考え方）
- 多段スケールの箱サイズ h でブロック標準偏差から合計値 N(h) を算出し、回帰で傾きを得る。
- 2D: $FD = |a|$ をクリップして $FD \in [2,3]$。 DBC: $FD_{3} = 3 - |a|$ を同様に $[2,3]$。
- 実装: `fd_std_boxcount()` と `fd_3d_dbc()`。数値演算は `cupy` があれば自動でGPU（`USE_CUPY`, `xp`）。

### スコア化ロジック
- FD由来トラブルスコア（0–100点）: $\text{score} = \max(0, \min(100, (\text{average\_fd} - 2.0) \times 100))$。
- 肌トラブル総合スコア: 部位別の指標（毛穴・シワ・色ムラ・赤み・クマ・テカリ等）を平均して `trouble_total_score` を生成。

### 部位別評価の考え方
- `extract_face_regions(image, landmarks)` で額/目/鼻/頬/口/顎などを抽出し領域ごとにFD・トラブルを算出。
- 単一画像の部位別は不安定→方向性確認用途。結論は集団データで検証。

### AI補正の位置づけ
- 低画質→高画質FDの補正を LightGBM で推定、`trained_fd_model.pkl` と `training_history.json` に保存・復元。
- 推論・検証モードで相関・MAE・R²・散布図などを表示し、補正妥当性を検証。

### 可視化と検証
- 傾きが正なら理論（不規則性↑→FD↑→トラブル↑）と整合。分散極小で `nan` になりうるためUIでガード。
- 散布図・回帰直線: `create_scatter_plot()`、ヒートマップ: `create_correlation_heatmap()`。

### 設計上の配慮
- 降格設計: 任意モジュールが不在でも機能縮退で継続。
- 日本語パス・フォント: 画像はバッファ読み、PDFはWindows日本語フォント登録。
- GPU自動切替: `cupy` あればGPU、無ければ `numpy` で同一API。

### キーファイル
- UI/永続化・表示: fractal_app.py
- FD算出: fd_boxcount.py
- 顔検出・部位抽出・トラブル推定: skin_analysis.py
- 実験ログ・相関・図作成: experiment_analysis.py
- PDFレポート: generate_updated_report.py

---

## 17. 使用データおよび評価項目
- データセット: `SKIN_DATA/` と `Facial Skin Condition Dataset.csv`（例画像・メタ情報）。
- 画像命名規約: `IMG_XXXX.jpg` ↔ `IMG_XXXX_low1.jpg`（Low4–7含む）。UIフローがこの規約に依存。
- 保存先: 実験行は `experimental_data.csv` に追記（`ExperimentDataManager.save_data`）。
- 評価用主要カラム（相関計算に使用; `calculate_correlations` より）:
  - FD: `average_fd`（部位平均）
  - 主観/測定: `roughness_score`, `dryness_score`, `moisture_level`, `sebum_level`, `pore_score`, `wrinkle_score`, `redness_score`, `dark_circle_score`, `age`
  - 自動検出: `trouble_pore_visibility`, `trouble_wrinkles`, `trouble_color_unevenness`, `trouble_redness_acne`, `trouble_dark_circles`, `trouble_oiliness`, `trouble_total_score`
  - 派生スコア: `overall_score`（`(average_fd-2.0)*100` を0–100にクリップ）
- FDアルゴリズム: `fd_std_boxcount`（2D）, `fd_3d_dbc`（擬似3D）。

## 18. 実験の流れ
- 画像取得→顔検出/部位抽出（MediaPipe優先, OpenCV/dlib/ヒューリスティックでフォールバック）。
- FD算出（複数スケールで回帰; 2D/DBC）と部位別/全体の集計。
- 低画質→高画質FDのAI補正（LightGBM; モデルと履歴を保存・復元）。
- トラブル指標の算出と `trouble_total_score` の作成。
- 実験行を `experimental_data.csv` に追記し、散布図・回帰線・相関を可視化。
- 必要に応じて PDF/Docx レポートを生成。

## 19. 相関評価の方法
- 手法: ピアソン相関係数 $r$ と p値（`scipy.stats.pearsonr`）。
- 前処理: 欠損行を除外し、`len(data) \ge 3` を満たすときに計算。分散が極小で `nan` になる場合はUIでガード。
- 可視化: 散布図と最小二乗回帰直線（`np.polyfit`, `create_scatter_plot`）。
- 有意性: 目安として $p < 0.05$ を有意。データ数 `n` と併記。
- 集団 vs 部位内: 集団（被験者・時点を跨ぐ）では正の関係が安定。一方、単一画像の部位内はばらつきが大きく方向性確認用途。

---

## 20. 評価閾値と算出方法（詳細）

### 閾値・ガード（実装上の基準）
- FDのクリップ: 算出後に $FD, FD_3 \\in [2, 3]$ にクリップ（外れ値抑制）。
- 相関の計算要件: 欠損除外後にサンプル数 $n \\ge 3$ を満たす場合のみ算出。
- 有意性の目安: ピアソン相関の p値で $p < 0.05$ を有意と表示。
- 近似定数系列のガード: 片方の分散がほぼ0のとき `nan` になり得るためUIで文言ガード。
- スコアのレンジ: FD由来トラブルスコアは $\\text{score}=\\max(0,\\min(100,(\\text{average\\_fd}-2.0)\\times100))$。

### ピアソン相関とp値
- 定義: $r = \\dfrac{\\mathrm{cov}(X,Y)}{\\sigma_X \\sigma_Y}$。
- 標本式: $r = \\dfrac{\\sum_i (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_i (x_i - \\bar{x})^2} \\sqrt{\\sum_i (y_i - \\bar{y})^2}}$。
- p値の導出（参考）: $t = r \\sqrt{\\dfrac{n-2}{1-r^2}}$ として自由度 $n-2$ の t 分布から算出（実装は `scipy.stats.pearsonr`）。

### 最小二乗法（OLS）
- 回帰直線: $y = a x + b$。
- 傾き: $a = \\dfrac{\\sum_i (x_i-\\bar{x})(y_i-\\bar{y})}{\\sum_i (x_i-\\bar{x})^2}$、切片: $b = \\bar{y} - a\\bar{x}$（実装は `np.polyfit(x,y,1)`）。
- FD算出時の回帰: $x=\\log h,\\ y=\\log N(h)$（2D）や $x=\\log r,\\ y=\\log N(r)$（DBC）。
  - 2D: $FD = |a|$、DBC: $FD_3 = 3 - |a|$（ともに [2,3] にクリップ）。
- 評価可視化の回帰: $x=\\text{average\\_fd},\\ y$ はトラブル総合や派生スコア。傾きの符号で関係方向を読む。

### 誤差指標（必要に応じて表示）
- $\\mathrm{MAE} = \\dfrac{1}{n} \\sum_i |y_i - \\hat{y}_i|$、$\\mathrm{RMSE} = \\sqrt{ \\dfrac{1}{n} \\sum_i (y_i - \\hat{y}_i)^2 }$。
- $R^2 = 1 - \\dfrac{\\sum_i (y_i-\\hat{y}_i)^2}{\\sum_i (y_i-\\bar{y})^2}$。

### 実装対応（所在）
- 相関/回帰の可視化: `create_scatter_plot()`（experiment_analysis.py）。
- 相関の算出: `calculate_correlations()`（scipy.stats.pearsonr を内部利用）。
- 回帰の実装: `numpy.polyfit` に基づく一次回帰（両モジュールにて使用）。

---

## 4.3 評価システムの仕様および使用手順

### 4.3.1 評価システムの概要
- 目的: 肌表面の不規則性をFDで定量化し、評価指標と統合して比較・経時検証を可能にする。
- 入力と出力:
  - 入力: 顔画像（カラー、一般的な静止画）。
  - 出力: 部位別FD、画像代表FD（部位平均）、補助評価値（例: 肌トラブル総合スコア）。メタデータ（時刻・被験者ID等）とともに1レコードとして保存。
- 基本方針: 顔全体を対象としつつ領域分割で部位特性を保持。降格設計で外部モジュール不在時も解析継続。

### 4.3.2 画像処理および分割の考え方
- 顔全体を入力: 顔領域抽出後、以降の解析を顔領域内で実施。
- 領域分割: 額・頬・鼻・口周囲・顎・目周辺などの意味的部分に分割し、各部位を独立に評価。
- 注記: 本節はアルゴリズム詳細には踏み込まず、分割の狙い（局所差の顕在化と全体傾向の両立）を示すのみ。

### 4.3.3 フラクタル次元算出と評価値の取得
- 各部分でFDを算出: 複数スケールでFDを推定し、部位ごとのFDを得る。
- 画像代表値: 部位別FDの代表統計量（平均）を画像代表FDとして採用。
- 評価項目の同時記録: 主観/測定/自動検出による評価項目や総合スコア、メタデータを同一レコードで保存。

### 4.3.4 実験時の使用手順
- 画像入力: 対象の顔画像を読み込み、顔領域抽出と分割を実行。
- 評価実行: 分割領域ごとにFDを推定し、画像代表FDを生成。必要に応じてトラブル指標を統合。
- 結果の保存: 画像代表FD・部位別FD・評価項目・メタデータを保存（UI操作詳細やコード説明は本節では扱わない）。

---

## 4.4 システムの使い方（番号付き手順）

### 重複について
- 本節は「実験の流れ」を補足する、もう一段具体的な手順です。UIの詳細操作やコード解説は省略し、概念的な順序のみ示します。

### 何をしたか（概要）
- 顔画像から顔領域を抽出し、意味的な複数部位に分割。
- 各部位でFDを算出し、画像代表FD（部位平均）を作成。
- 必要に応じて肌トラブル指標を算出・統合し、総合スコア化。
- 評価レコード（FD・評価項目・メタデータ）として保存し、可視化・相関解析・レポートに利用。

### システムを使ったときの流れ（もう一段具体）
1) 準備: 評価対象の顔画像を用意し、システムを起動する。
2) 画像入力: 顔画像を読み込み、顔領域が抽出されることを確認する。
3) 分割確認: 顔領域が額・頬・鼻・口周囲・顎・目周辺などに分割されることを確認する。
4) FD計算: 各分割領域に対してFDが算出され、部位別FDが得られる。
5) 代表値作成: 部位別FDの代表統計量（平均）から画像代表FDを作成する。
6) 評価項目統合: 主観/測定/自動検出の指標（必要に応じて）を統合し、総合スコアを作成する。
7) 保存: 画像代表FD・部位別FD・評価項目・メタデータを1レコードとして保存する。
8) 可視化/検証: 散布図・回帰線・相関などで関係性を確認し、必要に応じてレポートに反映する。
