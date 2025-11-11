# モデルファイル保存フォルダ

このフォルダには、異なる品質レベルで学習した複数のモデルを保存します。

## 📁 推奨ファイル構成

```
models/
├── trained_fd_model_low1.pkl     # 高精度モデル (相関97.25%)
├── trained_fd_model_low4.pkl     # バランス型 (相関97.05%)
├── trained_fd_model_low4-7.pkl   # 汎用型 (相関94.14%)
└── trained_fd_model_low8.pkl     # 低品質用 (相関92.90%)
```

## 🎯 各モデルの特性

### low1モデル (trained_fd_model_low1.pkl)
- **相関係数**: 97.25%
- **MAE**: 0.024
- **用途**: 論文・発表用の高精度予測
- **推奨度**: ⭐⭐⭐⭐⭐

### low4モデル (trained_fd_model_low4.pkl)
- **相関係数**: 97.05%
- **MAE**: 0.024
- **用途**: 日常的な解析作業
- **推奨度**: ⭐⭐⭐⭐

### low4-7グループモデル (trained_fd_model_low4-7.pkl)
- **相関係数**: 94.14%
- **MAE**: 0.037
- **用途**: 様々な品質レベルに対応
- **推奨度**: ⭐⭐⭐

### low8モデル (trained_fd_model_low8.pkl)
- **相関係数**: 92.90%
- **MAE**: 0.041
- **用途**: 非常に低品質な画像の予測
- **推奨度**: ⭐⭐

## 🔄 モデルの切り替え方法

### 方法1: スクリプトを使用（推奨）

```powershell
# low1モデルに切り替え
.\switch_model.ps1 -ModelType low1

# low4モデルに切り替え
.\switch_model.ps1 -ModelType low4
```

### 方法2: 手動でコピー

```powershell
# low1モデルを使用
Copy-Item "models\trained_fd_model_low1.pkl" -Destination "trained_fd_model.pkl" -Force
```

## 📝 モデル作成記録

学習したモデルの情報を記録してください。

### モデル作成テンプレート

```
作成日: YYYY-MM-DD
品質レベル: lowX
元画像数: XX組
データ拡張: XX種類
総データ数: XXX組
相関係数: X.XXXX
MAE: X.XXXX
備考: 
```

---

## 更新履歴

- 2025-11-11: フォルダ作成
