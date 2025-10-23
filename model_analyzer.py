"""
学習済みモデル解析ツール
trained_fd_model.pkl の内容を詳しく解析します
"""

import pickle
import os

def analyze_model(model_path):
    """
    学習済みモデルを読み込んで詳細情報を表示
    """
    print("=" * 80)
    print("🔍 学習済みモデル解析ツール")
    print("=" * 80)
    
    # ファイルの存在確認
    if not os.path.exists(model_path):
        print(f"❌ エラー: ファイルが見つかりません: {model_path}")
        return
    
    # ファイルサイズ
    file_size = os.path.getsize(model_path)
    print(f"\n📁 ファイル情報:")
    print(f"  - パス: {model_path}")
    print(f"  - サイズ: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # モデルの読み込み
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"  - ✅ 読み込み成功")
    except Exception as e:
        print(f"  - ❌ 読み込み失敗: {e}")
        return
    
    # モデルの基本情報
    print(f"\n🤖 モデル基本情報:")
    print(f"  - 型: {type(model).__name__}")
    print(f"  - クラス: {model.__class__.__module__}.{model.__class__.__name__}")
    
    # LightGBMモデルの詳細情報
    if hasattr(model, 'n_estimators'):
        print(f"\n📊 LightGBMパラメータ:")
        print(f"  - 推定器数 (n_estimators): {model.n_estimators}")
        print(f"  - 最大深度 (max_depth): {model.max_depth}")
        print(f"  - 学習率 (learning_rate): {model.learning_rate}")
        print(f"  - 並列数 (n_jobs): {model.n_jobs}")
    
    # 訓練済みモデルの情報
    if hasattr(model, 'booster_'):
        print(f"\n🌳 訓練済みモデル情報:")
        booster = model.booster_
        
        # 特徴量の数
        if hasattr(model, 'n_features_in_'):
            print(f"  - 入力特徴量数: {model.n_features_in_}")
        
        # 木の数
        num_trees = model.n_estimators
        print(f"  - 決定木の数: {num_trees}")
        
        # モデルのダンプ（最初の木のみ）
        try:
            model_dump = booster.dump_model()
            print(f"\n📋 モデル構造情報:")
            print(f"  - 特徴量名: {model_dump.get('feature_names', 'N/A')}")
            print(f"  - 木の数: {len(model_dump.get('tree_info', []))}")
            
            # 最初の木の情報
            if model_dump.get('tree_info'):
                first_tree = model_dump['tree_info'][0]
                print(f"\n🌲 最初の決定木の詳細:")
                print(f"  - 木のインデックス: {first_tree.get('tree_index', 'N/A')}")
                print(f"  - ノード数: {first_tree.get('num_leaves', 'N/A')}")
        except Exception as e:
            print(f"  - モデル構造の取得に失敗: {e}")
    
    # 特徴量の重要度
    if hasattr(model, 'feature_importances_'):
        print(f"\n⭐ 特徴量の重要度:")
        importances = model.feature_importances_
        
        # 特徴量名（このアプリの場合）
        feature_names = [
            "平均値 (mean)",
            "標準偏差 (std)",
            "エッジ強度 (edge_strength)",
            "ノイズレベル (noise_level)",
            "エントロピー (entropy)"
        ]
        
        for i, (name, importance) in enumerate(zip(feature_names, importances)):
            bar_length = int(importance / max(importances) * 40)
            bar = "█" * bar_length
            print(f"  [{i+1}] {name:30s}: {importance:8.1f} {bar}")
        
        # 最も重要な特徴量
        most_important_idx = importances.argmax()
        print(f"\n  🏆 最も重要な特徴量: {feature_names[most_important_idx]}")
    
    # 予測関数のテスト
    print(f"\n🧪 予測機能テスト:")
    try:
        # ダミーデータで予測テスト
        import numpy as np
        test_input = np.array([[0.5, 0.1, 0.3, 0.05, 6.5]])  # 5次元特徴ベクトル
        prediction = model.predict(test_input)
        print(f"  - ✅ 予測成功")
        print(f"  - サンプル入力: {test_input[0]}")
        print(f"  - 予測出力 (FD): {prediction[0]:.4f}")
    except Exception as e:
        print(f"  - ❌ 予測失敗: {e}")
    
    # モデルの用途説明
    print(f"\n💡 このモデルの用途:")
    print(f"""
  このモデルは「フラクタル次元予測AI」です。
  
  【入力】: 低画質画像の5つの特徴量
    1. 平均値 (画像の明るさ)
    2. 標準偏差 (明るさのばらつき)
    3. エッジ強度 (輪郭の鮮明さ)
    4. ノイズレベル (画像のノイズ量)
    5. エントロピー (情報量)
  
  【出力】: 高画質相当のフラクタル次元 (1.0〜2.0程度)
  
  【使い方】:
    - 推論モードで低画質画像をアップロード
    - このモデルが自動的に高品質FDを予測
    - 高画質画像がなくても正確な値を取得可能
  """)
    
    # 学習データに関する推定
    print(f"\n📚 学習データの推定:")
    if hasattr(model, 'n_features_in_'):
        print(f"  - 特徴量数: {model.n_features_in_}次元")
    if hasattr(model, 'n_estimators'):
        print(f"  - 使用した決定木: {model.n_estimators}個")
        print(f"  - 推定学習時間: {model.n_estimators * 0.01:.2f}秒 (概算)")
    
    print("\n" + "=" * 80)
    print("✅ 解析完了")
    print("=" * 80)
    
    return model


if __name__ == "__main__":
    # デスクトップのモデルファイルを解析
    model_path = r"C:\Users\iikrk\OneDrive\デスクトップ\trained_fd_model (1).pkl"
    
    # または現在のディレクトリにある場合
    if not os.path.exists(model_path):
        model_path = "trained_fd_model.pkl"
    
    model = analyze_model(model_path)
    
    # オプション: モデルをさらに詳しく調べたい場合
    if model is not None:
        print(f"\n💬 追加情報が必要な場合:")
        print(f"  - モデルオブジェクト: 変数 'model' に保存されています")
        print(f"  - 属性一覧: dir(model)")
        print(f"  - ヘルプ: help(model)")
