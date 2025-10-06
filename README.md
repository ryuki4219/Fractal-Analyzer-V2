# Fractal-Analyzer-V2

フラクタル次元を用いた画像解析アプリケーション（Streamlit版）

## 📋 概要

このアプリケーションは、Box-Counting法を用いて画像のフラクタル次元を解析します。
また、機械学習による予測機能や解像度補正AI機能を搭載しています。

## 🚀 クイックスタート

### 方法1: バッチファイルで起動（推奨・最も簡単）

1. `起動.bat` をダブルクリック
2. 自動的にブラウザが開きます
3. ブラウザで http://localhost:8501 にアクセス（自動で開かない場合）

### 方法2: コマンドラインから起動

```powershell
# 1. 必要なパッケージをインストール（初回のみ）
pip install -r requirements.txt

# 2. アプリを起動
streamlit run fractal_app.py
```

その後、ブラウザで http://localhost:8501 を開いてください。

## 📦 必要な環境

- Python 3.8以上（Python 3.13.5で動作確認済み）
- 必要なパッケージ: requirements.txt を参照

## ✨ 主な機能

### 基本機能
- **Box-Counting法によるフラクタル次元解析**
- **二値化処理と空間占有率の計算**
- **複数画像の一括解析とExcel出力**
- **機械学習による予測機能**
- **異常値の自動検知**

### 🤖 解像度補正AI機能
低解像度画像でも高解像度相当のフラクタル次元を推定する機能

#### 使い方
1. 高解像度画像を用意
2. サイドバー「学習データ生成モード」ON → アップロード
3. 20～100枚繰り返す
4. 「解像度補正モデルを学習」ボタンをクリック
5. 「解像度補正を有効化」ON
6. 低解像度画像をアップロード
7. 補正結果を確認！

### ⚡ パフォーマンス設定
- **高速プレビューモード**: 計算量削減、素早く結果表示
- **高精度解析モード**: 全計算実行、正確な結果出力

## 📁 出力ファイル

- `model_joblib.pkl`: 学習済みモデル
- `scaler_joblib.pkl`: スケーラ
- `classifier_joblib.pkl`: 分類器
- `results.xlsx`: 解析結果（複数画像処理時）
- `train_data.csv`: 学習データ
- `resolution_correction_model.pkl`: 解像度補正モデル
- `resolution_correction_scaler.pkl`: 解像度補正スケーラ

## 🔧 トラブルシューティング

アプリが開けない、エラーが発生する場合は、[TROUBLESHOOTING.md](TROUBLESHOOTING.md) を参照してください。

### よくある問題

#### ブラウザが自動的に開かない
→ 手動で http://localhost:8501 にアクセスしてください

#### パッケージが見つからない
```powershell
pip install --upgrade -r requirements.txt
```

#### ポートが使用中
```powershell
streamlit run fractal_app.py --server.port 8502
```

## 📖 詳細ドキュメント

- [実装概要](IMPLEMENTATION_SUMMARY.md)
- [最適化ガイド](OPTIMIZATION_GUIDE.md)
- [クイックスタート最適化](QUICKSTART_OPTIMIZATION.md)
- [解像度補正AIガイド](RESOLUTION_AI_GUIDE.md)
- [新UI README](新UI_README.md)
- [README v2](README_v2.md)

## 🧪 動作確認

テスト用アプリで基本動作を確認:
```powershell
streamlit run test_streamlit.py
```

## 💡 使用例

1. **単一画像の解析**
   - 画像をアップロード
   - 閾値とリサイズ設定を調整
   - フラクタル次元と占有率を確認

2. **複数画像の一括解析**
   - 複数の画像を選択してアップロード
   - 自動的にExcelファイルに結果を保存

3. **学習モデルの活用**
   - 解析を繰り返すと自動的に学習データが蓄積
   - 「学習を実行」で予測モデルを構築
   - 新しい画像で予測値と実測値を比較

## ⚠️ 注意事項

本プログラムはサンプル実装です。
画像サイズ、特徴量、異常判定基準、モデル選定は用途に応じて調整してください。

## 📝 ライセンス

（ライセンス情報を記載してください）

## 🤝 貢献

バグ報告や機能リクエストは Issue でお知らせください。

---

最終更新: 2025年10月6日
