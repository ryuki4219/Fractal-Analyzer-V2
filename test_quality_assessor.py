# テストスクリプト - 画像品質判定システム
import os
from image_quality_assessor import assess_image_quality, check_device_compatibility

print("=" * 70)
print("画像品質判定システム - テスト実行")
print("=" * 70)

# テスト対象の画像リスト
test_images = [
    "SKIN_DATA/1/front.jpg",
    "SKIN_DATA/2/front.jpg",
    "SKIN_DATA/5/front.jpg",
    "SKIN_DATA/9/front.jpg",
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"\n⚠️  ファイルが見つかりません: {img_path}")
        continue
    
    print(f"\n{'=' * 70}")
    print(f"📁 テスト画像: {img_path}")
    print(f"{'=' * 70}")
    
    # 品質評価
    result = assess_image_quality(img_path)
    
    if 'error' in result:
        print(f"❌ エラー: {result['message']}")
        continue
    
    # 結果表示
    rec = result['recommendation']
    print(f"\n{rec['icon']} 判定結果: {rec['title']}")
    print(f"品質レベル: {result['quality_level']}")
    print(f"処理可否: {'✅ 可能' if result['can_process'] else '❌ 不可'}")
    
    print(f"\n📊 品質指標:")
    metrics = result['metrics']
    print(f"  解像度: {metrics['resolution']} ({metrics['total_pixels']:,}画素)")
    print(f"  解像度スコア: {metrics['resolution_score']}/100")
    print(f"  鮮明度: {metrics['sharpness']:.2f}")
    print(f"  ノイズレベル: {metrics['noise_level']:.2f}")
    print(f"  推定JPEG品質: {metrics['estimated_jpeg_quality']}")
    
    print(f"\n💬 メッセージ: {rec['message']}")
    
    if 'advice' in rec:
        print(f"💡 アドバイス: {rec['advice']}")
    
    print(f"\n⚙️  処理方法: {rec['processing_method']}")
    print(f"🎯 信頼度: {rec['confidence']}")

print(f"\n{'=' * 70}")
print("✅ テスト完了")
print(f"{'=' * 70}")
