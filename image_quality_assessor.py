# image_quality_assessor.py
# ============================================================
# 画像品質自動判定システム
# - 画像の品質レベルを自動判定（High / Low4-7 / Low1-3 / Low8-10）
# - 処理可否を判定
# - 推奨デバイスとの適合性チェック
# ============================================================

import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ============================================================
# 品質基準の定義
# ============================================================

HIGH_QUALITY_CRITERIA = {
    "min_resolution": {
        "width": 1920,
        "height": 1080,
        "total_pixels": 2073600  # 約200万画素
    },
    "jpeg_quality": {
        "min_quality": 85,
        "max_compression_ratio": 10
    },
    "quality_metrics": {
        "min_sharpness": 100,    # ラプラシアン分散
        "max_noise_level": 15,   # ノイズレベル
        "min_snr": 20,           # SNR (dB)
    },
    "color": {
        "bit_depth": 24,
        "color_space": "sRGB"
    }
}

# 品質レベル分類基準（ユーザーの実際の画像データ330枚に基づいて調整）
# シャープネス値はTenengrad法による実測値を基準
# データセット分析: シャープネス範囲 4.5-61.2、解像度範囲 22×22-2865×2865
# 250×250 (62,500px) の解像度スコア: 5.09%
QUALITY_THRESHOLDS = {
    'high': {
        'resolution_score': 80,   # 90→80 大きな画像を受け入れやすく
        'sharpness': 30,          # 40→30 実データの高品質画像に合わせる
        'noise_max': 30,
        'jpeg_quality_min': 75
    },
    'low4-7': {
        'resolution_score': 5,    # 30→5 250×250(5.09%)を含むように調整
        'sharpness': 8,           # 15→8 実データの約70%が10-25の範囲にあるため
        'noise_max': 80,
        'jpeg_quality_min': 30
    },
    'low1-3': {
        # 高JPEG品質だが他の指標が基準未達
        'jpeg_quality_min': 75,
        'sharpness_max': 30       # 40→30 実データに合わせる
    },
    'low8-10': {
        # 上記のいずれにも該当しない低品質
        'default': True
    }
}

# ============================================================
# 画質評価関数
# ============================================================

def calculate_sharpness(image):
    """
    画像の鮮明度を計算（Tenengrad法 - 勾配ベース）
    
    従来のラプラシアン分散法は画像サイズや肌のテクスチャに敏感すぎるため、
    より安定的なTenengrad法（勾配の平均値）を使用。
    
    Args:
        image: BGR画像（numpy array）
    
    Returns:
        float: 鮮明度スコア（大きいほど鮮明）
              典型値: 15-50（肌画像の場合）
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sobelフィルタで勾配を計算（Tenengrad法）
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 勾配の大きさを計算
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # 平均勾配強度（画像サイズに依存しない）
    sharpness = np.mean(gradient_magnitude)
    
    return sharpness


def estimate_noise_level(image):
    """
    画像のノイズレベルを推定
    
    Args:
        image: BGR画像
    
    Returns:
        float: ノイズレベル（標準偏差）
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 高周波成分を抽出してノイズを推定
    # Sobel フィルタで微分
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # ノイズレベル = エッジ強度の標準偏差
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    noise_level = np.std(edge_magnitude)
    
    return noise_level


def estimate_jpeg_quality(image_path):
    """
    JPEG品質を推定
    
    Args:
        image_path: 画像ファイルパス
    
    Returns:
        int: 推定JPEG品質（0-100）
    """
    try:
        # PILで画像を開く
        pil_img = Image.open(image_path)
        
        # JPEG品質情報を取得（可能な場合）
        if hasattr(pil_img, 'info') and 'quality' in pil_img.info:
            return pil_img.info['quality']
        
        # ファイルサイズから推定
        file_size = os.path.getsize(image_path)
        img_array = np.array(pil_img)
        
        if len(img_array.shape) == 3:
            total_pixels = img_array.shape[0] * img_array.shape[1]
        else:
            total_pixels = img_array.shape[0] * img_array.shape[1]
        
        # バイト/ピクセル比から品質を推定
        bytes_per_pixel = file_size / total_pixels
        
        # 経験的な推定式
        if bytes_per_pixel > 3.0:
            estimated_quality = 95
        elif bytes_per_pixel > 2.0:
            estimated_quality = 85
        elif bytes_per_pixel > 1.0:
            estimated_quality = 75
        elif bytes_per_pixel > 0.5:
            estimated_quality = 65
        else:
            estimated_quality = 50
        
        return estimated_quality
        
    except Exception as e:
        print(f"JPEG品質推定エラー: {e}")
        return 70  # デフォルト値


def check_resolution(width, height):
    """
    解像度をスコア化
    
    Args:
        width: 幅（ピクセル）
        height: 高さ（ピクセル）
    
    Returns:
        int: 解像度スコア（0-100）
    """
    total_pixels = width * height
    target_pixels = HIGH_QUALITY_CRITERIA['min_resolution']['total_pixels']
    
    # ピクセル数の比率からスコアを計算
    ratio = total_pixels / target_pixels
    
    if ratio >= 1.0:
        score = 100
    else:
        score = int(ratio * 100)
    
    return score


# ============================================================
# 品質レベル分類
# ============================================================

def classify_quality_level(resolution_score, sharpness, noise, jpeg_quality, width=None, height=None):
    """
    各指標から品質レベルを分類
    
    【重要】解像度を最優先の判定基準とする
    - 50×50未満: 問答無用でlow8-10（解析拒否推奨）
    - 50×50以上で他の条件を満たす: 順次判定
    
    Args:
        resolution_score: 解像度スコア (0-100)
        sharpness: 鮮明度
        noise: ノイズレベル
        jpeg_quality: JPEG品質推定値 (0-100)
        width: 画像幅（オプション）
        height: 画像高さ（オプション）
    
    Returns:
        str: 'high' | 'low4-7' | 'low1-3' | 'low8-10'
    """
    
    # 【最優先】解像度チェック
    # 50×50未満（2500画素未満）は問答無用でlow8-10
    if width is not None and height is not None:
        total_pixels = width * height
        if width < 50 or height < 50 or total_pixels < 2500:
            return 'low8-10'
    
    # High品質判定（すべての条件を満たす必要あり）
    high_criteria = QUALITY_THRESHOLDS['high']
    if (resolution_score >= high_criteria['resolution_score'] and
        sharpness >= high_criteria['sharpness'] and
        noise <= high_criteria['noise_max'] and
        jpeg_quality >= high_criteria['jpeg_quality_min']):
        return 'high'
    
    # Low4-7判定（Golden Zone）- 解像度が十分なら緩く判定
    low47_criteria = QUALITY_THRESHOLDS['low4-7']
    
    # 解像度が十分にあれば、シャープネスとノイズの基準を緩和
    if resolution_score >= low47_criteria['resolution_score']:
        # 解像度が高い場合: シャープネスとJPEG品質の最低基準のみチェック
        if (sharpness >= low47_criteria['sharpness'] and
            noise <= low47_criteria['noise_max'] and
            jpeg_quality >= low47_criteria['jpeg_quality_min']):
            return 'low4-7'
    
    # Low1-3判定（品質過剰領域）
    # 【重要】解像度が十分にある（low4-7基準以上）が、
    # JPEG品質だけが異常に高く、他の指標が低い場合のみ
    low13_criteria = QUALITY_THRESHOLDS['low1-3']
    if (resolution_score >= low47_criteria['resolution_score'] and  # 解像度は十分
        jpeg_quality >= low13_criteria['jpeg_quality_min'] and      # JPEG品質が高い
        sharpness < low13_criteria['sharpness_max']):               # でもシャープネスが低い
        return 'low1-3'
    
    # Low8-10（解析拒否推奨）
    # 解像度が基準未満、またはその他の条件を満たさない
    return 'low8-10'


def assess_image_quality(image_path):
    """
    画像品質を総合的に評価
    
    Args:
        image_path: 画像ファイルパス
    
    Returns:
        dict: 品質評価結果
    """
    try:
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            return {
                'error': 'image_read_failed',
                'message': '画像を読み込めませんでした'
            }
        
        # 各指標を計算
        height, width = image.shape[:2]
        resolution_score = check_resolution(width, height)
        sharpness = calculate_sharpness(image)
        noise = estimate_noise_level(image)
        jpeg_quality = estimate_jpeg_quality(image_path)
        
        # 品質レベル分類（解像度を最優先するため、幅と高さも渡す）
        quality_level = classify_quality_level(
            resolution_score,
            sharpness,
            noise,
            jpeg_quality,
            width,
            height
        )
        
        # 推奨事項を取得
        recommendation = get_recommendation(quality_level)
        
        # 信頼度の詳細判定（メトリクスに基づいて動的に決定）
        confidence_level = determine_confidence_level(
            quality_level,
            resolution_score,
            sharpness,
            noise,
            jpeg_quality
        )
        
        # 信頼度をrecommendationに反映
        recommendation['confidence'] = confidence_level
        
        # 処理可否を判定（low8-10は解析拒否推奨）
        can_process = recommendation.get('can_analyze', True)
        
        return {
            'quality_level': quality_level,
            'can_process': can_process,
            'metrics': {
                'resolution': f"{width}x{height}",
                'resolution_score': resolution_score,
                'total_pixels': width * height,
                'sharpness': round(sharpness, 2),
                'noise_level': round(noise, 2),
                'estimated_jpeg_quality': jpeg_quality
            },
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'error': 'assessment_failed',
            'message': f'品質評価中にエラーが発生しました: {str(e)}'
        }


def determine_confidence_level(quality_level, resolution_score, sharpness, noise, jpeg_quality):
    """
    メトリクスに基づいて信頼度レベルを決定
    品質レベルと一貫性を持たせ、直観的にわかりやすくする
    
    Args:
        quality_level: 品質レベル ('high', 'low4-7', 'low1-3', 'low8-10')
        resolution_score: 解像度スコア (0-100)
        sharpness: シャープネス値
        noise: ノイズレベル
        jpeg_quality: JPEG品質 (0-100)
    
    Returns:
        str: 'very_high' | 'high' | 'middle' | 'low'
    
    判定基準:
        - high品質 → very_high信頼度
        - low4-7品質 → high/middle信頼度 (解像度とシャープネスの両方で判定)
        - low1-3品質 → high信頼度 (品質過剰、直接解析推奨)
        - low8-10品質 → low信頼度 (信頼度低下)
    """
    if quality_level == 'high':
        # high品質は常にvery_high信頼度
        return 'very_high'
    
    elif quality_level == 'low4-7':
        # low4-7（Golden Zone）内で細分化
        # 重要: 解像度とシャープネスの両方を考慮して判定
        
        # high信頼度: 解像度とシャープネスの両方が良好
        if resolution_score >= 30 and sharpness >= 15:
            return 'high'
        
        # middle信頼度: それ以外のlow4-7範囲内の画像
        # （解像度が低い、またはシャープネスが低い、または両方が中程度）
        else:
            return 'middle'
    
    elif quality_level == 'low1-3':
        # 品質過剰は高信頼度（直接解析推奨）
        # 解像度は低いがJPEG品質が高すぎる画像
        # AI予測には不向きだが、直接解析なら高精度
        return 'high'
    
    else:  # low8-10
        # 信頼度低下の可能性がある品質は常にlow信頼度
        return 'low'


def get_recommendation(quality_level):
    """
    品質レベルに応じた推奨事項を返す
    
    Args:
        quality_level: 品質レベル
    
    Returns:
        dict: 推奨事項
    """
    recommendations = {
        'high': {
            'status': 'excellent',
            'icon': '✅',
            'title': '高品質画像',
            'message': 'この画像は推奨品質です。直接解析を行います。',
            'processing_method': 'direct_analysis',
            'confidence': 'very_high',  # デフォルト値（動的に更新される）
            'can_analyze': True
        },
        'low4-7': {
            'status': 'good',
            'icon': '✅',
            'title': '推奨品質範囲',
            'message': 'この画像はGolden Zone（Low4-7）です。AI予測を使用します。',
            'processing_method': 'ai_prediction',
            'confidence': 'high',  # デフォルト値（メトリクスに応じてvery_high/high/middleに動的更新）
            'can_analyze': True
        },
        'low1-3': {
            'status': 'good',
            'icon': '✅',
            'title': '品質過剰（直接解析推奨）',
            'message': 'JPEG品質が高すぎるため、AI予測には不向きですが、直接解析で正確な結果が得られます。',
            'processing_method': 'direct_analysis',
            'confidence': 'high',  # デフォルト値（動的に更新される）
            'advice': '直接解析を使用してください。またはJPEG品質を70-85%程度に調整して再撮影することもできます',
            'can_analyze': True
        },
        'low8-10': {
            'status': 'rejected',
            'icon': '❌',
            'title': '解析拒否推奨（解像度不足）',
            'message': '画像の解像度が著しく低いため、解析結果の信頼性が極めて低くなります。',
            'processing_method': 'rejected',
            'confidence': 'low',  # デフォルト値（動的に更新される）
            'advice': '50×50ピクセル以上の画像で再撮影してください。推奨デバイスを使用することをお勧めします。',
            'can_analyze': False
        }
    }
    
    return recommendations.get(quality_level, recommendations['low8-10'])


# ============================================================
# 推奨デバイスチェック
# ============================================================

RECOMMENDED_DEVICES = {
    'excellent': {
        'smartphones': [
            'iPhone 11以降', 'iPhone XS以降', 'iPhone 8以降',
            'Galaxy S10以降', 'Galaxy S8以降',
            'Pixel 4以降', 'Pixel 2以降',
            'Xperia 1以降', 'Xperia XZ以降'
        ],
        'cameras': [
            '一眼レフ全般',
            'ミラーレス全般',
            'コンパクトカメラ（2018年以降）'
        ],
        'quality_range': 'low4-7またはhigh',
        'confidence': '高い'
    },
    'acceptable': {
        'smartphones': [
            'iPhone 6/6s/7',
            'Galaxy S6/S7',
            '2015-2017年のミドルレンジ機種'
        ],
        'cameras': [
            'コンパクトカメラ（2013-2017年）'
        ],
        'quality_range': 'low4-7（下限）',
        'confidence': '中程度'
    },
    'not_recommended': {
        'smartphones': [
            'iPhone 5以前',
            '2013年以前の機種',
            '500万画素以下のカメラ'
        ],
        'cameras': [
            'トイカメラ',
            '古い携帯電話カメラ'
        ],
        'quality_range': 'low8-10',
        'confidence': '使用不可'
    }
}


def check_device_compatibility(image_path=None, device_name=None):
    """
    デバイスの互換性をチェック
    
    Args:
        image_path: 画像パス（品質から逆算）
        device_name: デバイス名（マニュアル確認）
    
    Returns:
        dict: 互換性情報
    """
    if image_path:
        # 画像品質から判定
        quality_assessment = assess_image_quality(image_path)
        
        if 'error' in quality_assessment:
            return quality_assessment
        
        quality_level = quality_assessment['quality_level']
        
        if quality_level in ['high', 'low4-7']:
            return {
                'status': 'excellent',
                'icon': '✅',
                'message': 'この画像は推奨品質範囲です',
                'can_use': True,
                'quality_assessment': quality_assessment
            }
        elif quality_level == 'low1-3':
            return {
                'status': 'caution',
                'icon': '⚠️',
                'message': '品質が高すぎます（過学習リスク）',
                'can_use': 'experimental',
                'quality_assessment': quality_assessment
            }
        else:  # low8-10
            return {
                'status': 'rejected',
                'icon': '❌',
                'message': '品質が低すぎます（使用不可）',
                'can_use': False,
                'suggestion': '推奨デバイスで撮影し直してください',
                'recommended_devices': RECOMMENDED_DEVICES['excellent'],
                'quality_assessment': quality_assessment
            }
    
    # デバイス名から判定（簡易版）
    if device_name:
        for category, info in RECOMMENDED_DEVICES.items():
            all_devices = info.get('smartphones', []) + info.get('cameras', [])
            for device in all_devices:
                if device.lower() in device_name.lower():
                    return {
                        'status': category,
                        'device_category': info,
                        'can_use': category != 'not_recommended'
                    }
    
    return {
        'status': 'unknown',
        'message': '判定できませんでした',
        'can_use': None
    }


# ============================================================
# テスト用メイン関数
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用法: python image_quality_assessor.py <画像ファイルパス>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 60)
    print("画像品質自動判定システム")
    print("=" * 60)
    
    # 品質評価
    result = assess_image_quality(image_path)
    
    if 'error' in result:
        print(f"\n❌ エラー: {result['message']}")
        sys.exit(1)
    
    # 結果表示
    print(f"\n{result['recommendation']['icon']} 判定結果: {result['recommendation']['title']}")
    print(f"品質レベル: {result['quality_level']}")
    print(f"処理可否: {'可能' if result['can_process'] else '不可'}")
    
    print(f"\n📊 品質指標:")
    metrics = result['metrics']
    print(f"  解像度: {metrics['resolution']} ({metrics['total_pixels']:,}画素)")
    print(f"  解像度スコア: {metrics['resolution_score']}/100")
    print(f"  鮮明度: {metrics['sharpness']:.2f}")
    print(f"  ノイズレベル: {metrics['noise_level']:.2f}")
    print(f"  推定JPEG品質: {metrics['estimated_jpeg_quality']}")
    
    print(f"\n💬 メッセージ:")
    print(f"  {result['recommendation']['message']}")
    
    if 'advice' in result['recommendation']:
        print(f"\n💡 アドバイス:")
        print(f"  {result['recommendation']['advice']}")
    
    print(f"\n⚙️  処理方法: {result['recommendation']['processing_method']}")
    print(f"信頼度: {result['recommendation']['confidence']}")
    
    print("\n" + "=" * 60)
