# -*- coding: utf-8 -*-
"""
肌分析モジュール
顔検出、部位分割、肌トラブル検出機能を提供
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# MediaPipe Face Meshの初期化（遅延ロード）
_face_mesh = None

def get_face_mesh():
    """MediaPipe Face Meshのシングルトンインスタンスを取得"""
    global _face_mesh
    if _face_mesh is None:
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            _face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except ImportError:
            return None
    return _face_mesh


def detect_face_landmarks(image):
    """
    顔のランドマークを検出
    
    Args:
        image: BGR画像
    
    Returns:
        landmarks: 顔のランドマーク（478点）、検出失敗時はNone
    """
    face_mesh = get_face_mesh()
    if face_mesh is None:
        return None
    
    # BGR→RGB変換
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 顔検出
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return None
    
    return results.multi_face_landmarks[0]


def extract_face_regions(image, landmarks):
    """
    顔画像から各部位を抽出
    
    Args:
        image: BGR画像
        landmarks: MediaPipeの顔ランドマーク
    
    Returns:
        dict: 各部位の画像と座標 {region_name: {'image': img, 'bbox': (x, y, w, h)}}
    """
    h, w = image.shape[:2]
    
    # ランドマークを画像座標に変換
    points = []
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))
    points = np.array(points)
    
    regions = {}
    
    # 額（おでこ）: 顔上部
    forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    if all(i < len(points) for i in forehead_indices):
        forehead_points = points[forehead_indices]
        x, y, w_region, h_region = cv2.boundingRect(forehead_points)
        # 上方向に拡張（髪の生え際まで）
        y_start = max(0, y - h_region // 2)
        y_end = y + h_region
        regions['forehead'] = {
            'image': image[y_start:y_end, x:x+w_region],
            'bbox': (x, y_start, w_region, y_end - y_start)
        }
    
    # 左頬
    left_cheek_indices = [116, 123, 147, 213, 192, 214, 212, 202, 204, 194, 135, 
                          210, 169, 170, 171, 208, 32, 49, 48, 64, 98]
    if all(i < len(points) for i in left_cheek_indices):
        left_points = points[left_cheek_indices]
        x, y, w_region, h_region = cv2.boundingRect(left_points)
        regions['left_cheek'] = {
            'image': image[y:y+h_region, x:x+w_region],
            'bbox': (x, y, w_region, h_region)
        }
    
    # 右頬
    right_cheek_indices = [345, 352, 376, 433, 416, 434, 432, 422, 424, 418, 364,
                           430, 394, 395, 396, 428, 262, 279, 278, 294, 327]
    if all(i < len(points) for i in right_cheek_indices):
        right_points = points[right_cheek_indices]
        x, y, w_region, h_region = cv2.boundingRect(right_points)
        regions['right_cheek'] = {
            'image': image[y:y+h_region, x:x+w_region],
            'bbox': (x, y, w_region, h_region)
        }
    
    # 鼻
    nose_indices = [4, 5, 195, 197, 196, 174, 198, 236, 3, 51, 45, 6, 168, 122, 
                    188, 412, 351, 419, 248, 281, 275, 440, 456, 399, 437]
    if all(i < len(points) for i in nose_indices):
        nose_points = points[nose_indices]
        x, y, w_region, h_region = cv2.boundingRect(nose_points)
        regions['nose'] = {
            'image': image[y:y+h_region, x:x+w_region],
            'bbox': (x, y, w_region, h_region)
        }
    
    # 口周り
    mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
                     318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    if all(i < len(points) for i in mouth_indices):
        mouth_points = points[mouth_indices]
        x, y, w_region, h_region = cv2.boundingRect(mouth_points)
        # 上下に拡張
        padding = h_region // 3
        y_start = max(0, y - padding)
        y_end = min(h, y + h_region + padding)
        regions['mouth_area'] = {
            'image': image[y_start:y_end, x:x+w_region],
            'bbox': (x, y_start, w_region, y_end - y_start)
        }
    
    # 顎
    chin_indices = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356,
                    389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162,
                    127, 234, 93, 132, 58, 172, 136, 150, 149]
    if all(i < len(points) for i in chin_indices):
        chin_points = points[chin_indices]
        x, y, w_region, h_region = cv2.boundingRect(chin_points)
        # 下半分のみ（顎部分）
        y_mid = y + h_region // 2
        regions['chin'] = {
            'image': image[y_mid:y+h_region, x:x+w_region],
            'bbox': (x, y_mid, w_region, h_region // 2)
        }
    
    # 目の下（クマ・くすみ検出用）
    left_under_eye_indices = [226, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24]
    right_under_eye_indices = [446, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254]
    
    if all(i < len(points) for i in left_under_eye_indices):
        left_ue_points = points[left_under_eye_indices]
        x, y, w_region, h_region = cv2.boundingRect(left_ue_points)
        regions['left_under_eye'] = {
            'image': image[y:y+h_region, x:x+w_region],
            'bbox': (x, y, w_region, h_region)
        }
    
    if all(i < len(points) for i in right_under_eye_indices):
        right_ue_points = points[right_under_eye_indices]
        x, y, w_region, h_region = cv2.boundingRect(right_ue_points)
        regions['right_under_eye'] = {
            'image': image[y:y+h_region, x:x+w_region],
            'bbox': (x, y, w_region, h_region)
        }
    
    return regions


def detect_skin_troubles(region_image, region_name: str) -> Dict:
    """
    肌トラブルを検出（フラクタル次元ベース + 画像処理）
    
    検出できる肌トラブル:
    - 乾燥（キメの乱れ）: 高FD値
    - 毛穴の目立ち: 高FD値 + 暗点検出
    - シワ: 高FD値 + エッジ検出
    - 色ムラ（くすみ）: 色の標準偏差
    - ニキビ・吹き出物: 赤み検出
    - クマ: 暗さ検出（目の下のみ）
    - テカリ（皮脂過多）: 明度の高い領域
    
    Args:
        region_image: 部位の画像
        region_name: 部位名
    
    Returns:
        dict: 検出された肌トラブル情報
    """
    if region_image is None or region_image.size == 0:
        return {'error': '画像が不正です'}
    
    troubles = {}
    
    # 基本的な画像統計
    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # HSV色空間に変換
    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
    
    # 1. 毛穴の目立ち検出（暗点カウント）
    _, dark_spots = cv2.threshold(gray, mean_brightness - std_brightness, 255, cv2.THRESH_BINARY_INV)
    dark_spot_ratio = np.sum(dark_spots > 0) / dark_spots.size
    troubles['pore_visibility'] = {
        'score': min(dark_spot_ratio * 100, 100),
        'level': '高' if dark_spot_ratio > 0.15 else '中' if dark_spot_ratio > 0.08 else '低'
    }
    
    # 2. シワ検出（エッジ強度）
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    troubles['wrinkles'] = {
        'score': min(edge_ratio * 500, 100),  # スケーリング
        'level': '高' if edge_ratio > 0.12 else '中' if edge_ratio > 0.06 else '低'
    }
    
    # 3. 色ムラ・くすみ検出
    color_std = np.std(region_image, axis=(0, 1)).mean()
    troubles['color_unevenness'] = {
        'score': min(color_std / 40 * 100, 100),
        'level': '高' if color_std > 35 else '中' if color_std > 20 else '低'
    }
    
    # 4. ニキビ・赤み検出（赤チャンネル優位）
    b, g, r = cv2.split(region_image)
    redness = np.mean(r) - np.mean(g)
    troubles['redness_acne'] = {
        'score': min(max(redness, 0) / 30 * 100, 100),
        'level': '高' if redness > 25 else '中' if redness > 15 else '低'
    }
    
    # 5. クマ検出（目の下のみ）
    if 'under_eye' in region_name:
        troubles['dark_circles'] = {
            'score': min((255 - mean_brightness) / 2.55, 100),
            'level': '高' if mean_brightness < 100 else '中' if mean_brightness < 130 else '低'
        }
    
    # 6. テカリ検出（明度の高い領域）
    _, bright_spots = cv2.threshold(gray, mean_brightness + std_brightness, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright_spots > 0) / bright_spots.size
    troubles['oiliness'] = {
        'score': min(bright_ratio * 200, 100),
        'level': '高' if bright_ratio > 0.20 else '中' if bright_ratio > 0.10 else '低'
    }
    
    # 7. 肌のテクスチャ粗さ（後でFD値を追加）
    troubles['texture_roughness'] = {
        'score': 0,  # FD値で上書きされる
        'level': '未計算'
    }
    
    return troubles


def create_trouble_report(troubles_by_region: Dict, fd_by_region: Dict) -> str:
    """
    肌トラブルレポートを生成
    
    Args:
        troubles_by_region: 部位別の肌トラブル情報
        fd_by_region: 部位別のフラクタル次元
    
    Returns:
        str: レポートテキスト
    """
    report_lines = ["# 🔍 肌トラブル検出レポート\n"]
    
    # 各部位のレポート
    region_names_jp = {
        'forehead': '額',
        'left_cheek': '左頬',
        'right_cheek': '右頬',
        'nose': '鼻',
        'mouth_area': '口周り',
        'chin': '顎',
        'left_under_eye': '左目の下',
        'right_under_eye': '右目の下'
    }
    
    for region, troubles in troubles_by_region.items():
        region_jp = region_names_jp.get(region, region)
        report_lines.append(f"\n## 📍 {region_jp}\n")
        
        # FD値
        if region in fd_by_region:
            fd_val = fd_by_region[region]
            report_lines.append(f"- **フラクタル次元**: {fd_val:.4f}")
        
        # 主要なトラブル
        high_troubles = []
        for trouble_name, trouble_data in troubles.items():
            if trouble_data.get('level') == '高':
                trouble_jp = {
                    'pore_visibility': '毛穴の目立ち',
                    'wrinkles': 'シワ',
                    'color_unevenness': '色ムラ・くすみ',
                    'redness_acne': 'ニキビ・赤み',
                    'dark_circles': 'クマ',
                    'oiliness': 'テカリ',
                    'texture_roughness': 'キメの粗さ'
                }.get(trouble_name, trouble_name)
                high_troubles.append(trouble_jp)
        
        if high_troubles:
            report_lines.append(f"- ⚠️ **検出されたトラブル**: {', '.join(high_troubles)}")
        else:
            report_lines.append("- ✅ **特に問題なし**")
    
    return '\n'.join(report_lines)


# 部位名の日本語マッピング
REGION_NAMES_JP = {
    'forehead': '額',
    'left_cheek': '左頬',
    'right_cheek': '右頬',
    'nose': '鼻',
    'mouth_area': '口周り',
    'chin': '顎',
    'left_under_eye': '左目の下',
    'right_under_eye': '右目の下'
}

# 肌トラブル名の日本語マッピング
TROUBLE_NAMES_JP = {
    'pore_visibility': '毛穴の目立ち',
    'wrinkles': 'シワ',
    'color_unevenness': '色ムラ・くすみ',
    'redness_acne': 'ニキビ・赤み',
    'dark_circles': 'クマ',
    'oiliness': 'テカリ',
    'texture_roughness': 'キメの粗さ（FD値）'
}
