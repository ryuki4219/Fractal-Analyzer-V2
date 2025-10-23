import pickle
import numpy as np

# モデルファイルのパス
model_path = r"C:\Users\iikrk\OneDrive\デスクトップ\trained_fd_model (1).pkl"

print("=" * 60)
print("🔍 モデル解析結果")
print("=" * 60)

# モデル読み込み
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"\n✅ モデル読み込み成功!")
print(f"\n【基本情報】")
print(f"  型: {type(model).__name__}")
print(f"  推定器数: {model.n_estimators}")
print(f"  最大深度: {model.max_depth}")
print(f"  学習率: {model.learning_rate}")
print(f"  入力特徴量数: {model.n_features_in_}")

print(f"\n【特徴量の重要度】")
feature_names = ["平均値", "標準偏差", "エッジ強度", "ノイズ", "エントロピー"]
for name, imp in zip(feature_names, model.feature_importances_):
    print(f"  {name:12s}: {imp:8.1f}")

print(f"\n【予測テスト】")
test = np.array([[0.5, 0.1, 0.3, 0.05, 6.5]])
pred = model.predict(test)
print(f"  入力: {test[0]}")
print(f"  予測FD: {pred[0]:.4f}")

print(f"\n【用途】")
print("  このモデルは低画質画像の特徴から")
print("  高画質相当のフラクタル次元を予測します")
print("=" * 60)
