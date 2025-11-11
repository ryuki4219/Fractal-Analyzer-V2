# 🗂️ モデル管理ガイド

複数の学習モデルを管理し、用途に応じて使い分ける方法を説明します。

---

## 📋 モデル管理の基本

### モデルファイルとは

学習済みAIモデルは`.pkl`ファイルとして保存されます:

```
trained_fd_model.pkl  ← デフォルトのモデル
```

このファイルには以下の情報が含まれます:
- AIの学習結果(重み・パラメータ)
- 使用した品質レベル
- 学習時の設定

---

## 🔄 複数モデルの作成と管理

### ステップ1: 品質レベルごとにモデルを作成

#### low1モデル (高精度用)
```
1. 学習モードで品質レベル「low1」を選択
2. データ拡張28種類すべて選択
3. 学習を実行
4. 完了後、モデルを保存:
   → ダウンロードボタンをクリック
   → ファイル名: trained_fd_model_low1.pkl
```

**性能**:
- 相関係数: 97.25%
- MAE: 0.024
- 用途: 高精度予測が必要な場合

---

#### low4モデル (バランス型)
```
1. 学習モードで品質レベル「low4」を選択
2. データ拡張28種類すべて選択
3. 学習を実行
4. 完了後、モデルを保存:
   → ダウンロードボタンをクリック
   → ファイル名: trained_fd_model_low4.pkl
```

**性能**:
- 相関係数: 97.05%
- MAE: 0.024
- 用途: バランスの取れた予測

---

#### low8モデル (低品質用)
```
1. 学習モードで品質レベル「low8」を選択
2. データ拡張28種類すべて選択
3. 学習を実行
4. 完了後、モデルを保存:
   → ダウンロードボタンをクリック
   → ファイル名: trained_fd_model_low8.pkl
```

**性能**:
- 相関係数: 92.90%
- MAE: 0.041
- 用途: 非常に低品質な画像の予測

---

#### グループモデル (汎用型)
```
1. 学習モードで品質グループ「low4-7」を選択
2. データ拡張28種類すべて選択
3. 学習を実行
4. 完了後、モデルを保存:
   → ダウンロードボタンをクリック
   → ファイル名: trained_fd_model_low4-7.pkl
```

**性能**:
- 相関係数: 94.14%
- MAE: 0.037
- 用途: 様々な品質レベルに対応

---

### ステップ2: モデルを保存

#### 方法1: アプリからダウンロード
```
1. 学習完了画面で「モデルをダウンロード」をクリック
2. ファイル名を変更して保存
   例: trained_fd_model_low1.pkl
```

#### 方法2: ファイルを手動コピー
```powershell
# モデルファイルを別名でコピー
Copy-Item "trained_fd_model.pkl" -Destination "trained_fd_model_low1.pkl"
```

---

### ステップ3: モデルを切り替える

#### 方法1: ファイル名を変更
```powershell
# 現在のモデルをバックアップ
Rename-Item "trained_fd_model.pkl" -NewName "trained_fd_model_backup.pkl"

# 使いたいモデルをリネーム
Rename-Item "trained_fd_model_low1.pkl" -NewName "trained_fd_model.pkl"
```

#### 方法2: ファイルをコピー
```powershell
# low1モデルを使用
Copy-Item "trained_fd_model_low1.pkl" -Destination "trained_fd_model.pkl" -Force
```

---

## 📊 モデル比較表

作成した複数のモデルを比較する表:

| モデル | ファイル名 | 相関係数 | MAE | 用途 | 推奨度 |
|--------|-----------|----------|-----|------|--------|
| **low1** | `trained_fd_model_low1.pkl` | 97.25% | 0.024 | 高精度予測 | ⭐⭐⭐⭐⭐ |
| low4 | `trained_fd_model_low4.pkl` | 97.05% | 0.024 | バランス型 | ⭐⭐⭐⭐ |
| low4-7 | `trained_fd_model_low4-7.pkl` | 94.14% | 0.037 | 汎用型 | ⭐⭐⭐ |
| low8 | `trained_fd_model_low8.pkl` | 92.90% | 0.041 | 低品質用 | ⭐⭐ |

---

## 🎯 用途別おすすめモデル

### ケース1: 論文・発表用の高精度予測
```
推奨モデル: low1
ファイル: trained_fd_model_low1.pkl

理由:
- 相関係数97.25% (最高精度)
- 平均誤差わずか2.4%
- 信頼性の高い結果
```

### ケース2: 日常的な解析作業
```
推奨モデル: low4 または low4-7グループ
ファイル: trained_fd_model_low4.pkl

理由:
- 十分な精度 (94-97%)
- 様々な品質レベルに対応
- バランスが良い
```

### ケース3: 非常に低品質な画像
```
推奨モデル: low8
ファイル: trained_fd_model_low8.pkl

理由:
- 低品質画像専用に最適化
- 相関係数92.90%でも実用可能
```

---

## 📁 ファイル構成例

```
Fractal-Analyzer-V2/
├── trained_fd_model.pkl              ← 現在使用中のモデル
├── models/                           ← モデル保存フォルダ(作成推奨)
│   ├── trained_fd_model_low1.pkl    ← 高精度モデル
│   ├── trained_fd_model_low4.pkl    ← バランス型
│   ├── trained_fd_model_low4-7.pkl  ← 汎用型
│   ├── trained_fd_model_low8.pkl    ← 低品質用
│   └── README.txt                    ← モデルの説明
└── training_history.json             ← 学習履歴
```

### modelsフォルダの作成
```powershell
# フォルダを作成
New-Item -ItemType Directory -Path "models"

# 説明ファイルを作成
@"
# モデルファイル一覧

- trained_fd_model_low1.pkl: 高精度モデル (相関97.25%)
- trained_fd_model_low4.pkl: バランス型 (相関97.05%)
- trained_fd_model_low4-7.pkl: 汎用型 (相関94.14%)
- trained_fd_model_low8.pkl: 低品質用 (相関92.90%)

作成日: 2025-11-11
"@ | Out-File -FilePath "models\README.txt" -Encoding UTF8
```

---

## 🔄 モデル切り替えスクリプト

便利なスクリプトを作成して、簡単にモデルを切り替えられるようにします。

### switch_model.ps1 (PowerShellスクリプト)

```powershell
# モデル切り替えスクリプト
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("low1", "low4", "low4-7", "low8")]
    [string]$ModelType
)

$modelFile = "models\trained_fd_model_$ModelType.pkl"

if (Test-Path $modelFile) {
    # 現在のモデルをバックアップ
    if (Test-Path "trained_fd_model.pkl") {
        Copy-Item "trained_fd_model.pkl" -Destination "trained_fd_model_backup.pkl" -Force
        Write-Host "✅ 現在のモデルをバックアップしました" -ForegroundColor Green
    }
    
    # 指定モデルをコピー
    Copy-Item $modelFile -Destination "trained_fd_model.pkl" -Force
    Write-Host "✅ $ModelType モデルに切り替えました" -ForegroundColor Green
    Write-Host "   ファイル: $modelFile" -ForegroundColor Cyan
} else {
    Write-Host "❌ エラー: $modelFile が見つかりません" -ForegroundColor Red
    Write-Host "   先に学習モードでモデルを作成してください" -ForegroundColor Yellow
}
```

### 使い方

```powershell
# low1モデルに切り替え
.\switch_model.ps1 -ModelType low1

# low4モデルに切り替え
.\switch_model.ps1 -ModelType low4

# グループモデルに切り替え
.\switch_model.ps1 -ModelType low4-7
```

---

## 📊 モデルの性能比較方法

### 精度検証モードで比較

```
1. モデル1をセット (例: low1)
2. 精度検証モードで10-20組を検証
3. 結果をCSVでダウンロード: validation_results_low1.csv

4. モデル2に切り替え (例: low4)
5. 同じ画像で精度検証を実行
6. 結果をCSVでダウンロード: validation_results_low4.csv

7. 結果を比較:
   - 相関係数
   - MAE
   - RMSE
   - 個別画像の誤差
```

### 比較例

| 画像 | 実測FD | low1予測 | low1誤差 | low4予測 | low4誤差 | 優位モデル |
|------|--------|----------|----------|----------|----------|-----------|
| IMG_5023 | 2.6801 | 2.8347 | +5.77% | 2.8500 | +6.34% | low1 ✅ |
| IMG_5024 | 2.7845 | 2.8116 | +0.97% | 2.8200 | +1.27% | low1 ✅ |
| IMG_5025 | 2.8466 | 2.8866 | +1.41% | 2.8900 | +1.52% | low1 ✅ |

---

## ⚠️ 注意事項

### 1. モデルと画像の品質レベルを一致させる

```
❌ 間違い:
  モデル: low1で学習
  入力画像: IMG_5023_low8.jpg  ← 品質レベルが違う
  結果: 精度が大幅に低下

✅ 正しい:
  モデル: low1で学習
  入力画像: IMG_5023_low1.jpg  ← 品質レベルが一致
  結果: 高精度な予測
```

### 2. モデルファイルのバックアップ

```
重要なモデルは必ずバックアップを取る:

# 定期的にバックアップ
Copy-Item "trained_fd_model.pkl" -Destination "backup\trained_fd_model_$(Get-Date -Format 'yyyyMMdd').pkl"
```

### 3. 学習履歴との対応

```
training_history.json には全モデルの学習履歴が含まれます:

{
  "timestamp": "2025-11-10 15:38:10",
  "quality_level": "low1",
  "correlation_pred": 0.9725,
  "mae_pred": 0.0243
}

モデルファイルと学習履歴を対応させて管理してください。
```

---

## 🚀 実践: 完全なモデルセットを作成

### ゴール
用途に応じて使い分けられる4つのモデルを作成

### 手順 (約40分)

```
⏱️ 10分: low1モデル作成
1. 学習モード → low1選択 → データ拡張28種類
2. 学習実行
3. ダウンロード → trained_fd_model_low1.pkl

⏱️ 10分: low4モデル作成
1. 学習モード → low4選択 → データ拡張28種類
2. 学習実行
3. ダウンロード → trained_fd_model_low4.pkl

⏱️ 15分: low4-7グループモデル作成
1. 学習モード → low4-7グループ選択 → データ拡張28種類
2. 学習実行
3. ダウンロード → trained_fd_model_low4-7.pkl

⏱️ 5分: モデル整理
1. modelsフォルダを作成
2. すべてのモデルを移動
3. README.txtを作成
```

### 完成後
```
✅ 4つの用途別モデル完成
✅ いつでも切り替え可能
✅ 最適なモデルで高精度予測
```

---

## 更新履歴

- 2025-11-11: 初版作成
  - 基本的なモデル管理方法
  - 切り替えスクリプト
  - 性能比較方法
