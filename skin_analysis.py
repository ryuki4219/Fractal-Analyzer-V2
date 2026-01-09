# -*- coding: utf-8 -*-
"""
肌分析モジュール
顔検出、部位分割、肌トラブル検出機能を提供
Python 3.13対応版 - OpenCV + dlib(オプション)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 検出器の初期化（遅延ロード）
_face_mesh = None
_face_cascade = None
_dlib_detector = None
_dlib_predictor = None
_mediapipe_available = False

def _init_face_detectors():
    """顔検出器を初期化"""
    global _face_cascade, _dlib_detector, _dlib_predictor, _mediapipe_available, _face_mesh
    
    # OpenCV Haar Cascade（必ず動く）
    if _face_cascade is None:
        candidates = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
        ]
        for cascade_path in candidates:
            if os.path.exists(cascade_path):
                clf = cv2.CascadeClassifier(cascade_path)
                if not clf.empty():
                    _face_cascade = clf
                    break
        if _face_cascade is None:
            print("Warning: Haar cascade not found. OpenCV face detection may fail.")
    
    # MediaPipe（Python 3.12以下のみ）
    if _face_mesh is None:
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            _face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.1,
                min_tracking_confidence=0.1
            )
            _mediapipe_available = True
            print("MediaPipe初期化成功")
        except ImportError:
            _mediapipe_available = False
            print("MediaPipe利用不可 - OpenCVベースの検出を使用")
    
    # dlib（オプション）
    if _dlib_detector is None:
        try:
            import dlib
            _dlib_detector = dlib.get_frontal_face_detector()
            # 68点ランドマーク予測器（存在すれば）
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                _dlib_predictor = dlib.shape_predictor(predictor_path)
        except ImportError:
            pass


def get_face_mesh():
    """MediaPipe Face Meshのシングルトンインスタンスを取得"""
    global _face_mesh, _mediapipe_available
    _init_face_detectors()
    return _face_mesh if _mediapipe_available else None


def get_face_cascade():
    """OpenCV Haar Cascade顔検出器を取得"""
    global _face_cascade
    _init_face_detectors()
    return _face_cascade


def detect_face_opencv(image):
    """
    OpenCVを使用して顔を検出し、顔領域の矩形を返す
    
    Returns:
        (x, y, w, h) or None
    """
    _init_face_detectors()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 複数の前処理を試行
    preprocessing_methods = [
        ("original", gray),
        ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)),
        ("equalized", cv2.equalizeHist(gray)),
    ]
    
    for method_name, processed_gray in preprocessing_methods:
        if _face_cascade is not None and not _face_cascade.empty():
            # 画像サイズに応じて最小サイズを可変設定
            h_img, w_img = processed_gray.shape[:2]
            min_side = max(30, int(min(h_img, w_img) * 0.08))
            min_size = (min_side, min_side)

            # 複数パラメータでリトライ
            param_sets = [
                dict(scaleFactor=1.1, minNeighbors=5, minSize=min_size),
                dict(scaleFactor=1.05, minNeighbors=4, minSize=min_size),
                dict(scaleFactor=1.2, minNeighbors=3, minSize=(30, 30)),
            ]
            for params in param_sets:
                faces = _face_cascade.detectMultiScale(
                    processed_gray,
                    **params,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    print(f"OpenCV Haar Cascade ({method_name}, params={params})で顔検出成功")
                    return (x, y, w, h)
    
    # dlibを試行
    if _dlib_detector is not None:
        try:
            faces = _dlib_detector(gray, 1)
            if len(faces) > 0:
                face = max(faces, key=lambda r: r.width() * r.height())
                x, y = face.left(), face.top()
                w, h = face.width(), face.height()
                print("dlibで顔検出成功")
                return (x, y, w, h)
        except:
            pass
    
    # 画像中央のヒューリスティック・フォールバック
    # 顔が大きく、前処理では検出できないケース向け
    try:
        h_img, w_img = image.shape[:2]
        cx, cy = w_img // 2, h_img // 2
        fw = int(w_img * 0.6)
        fh = int(h_img * 0.7)
        x = max(0, cx - fw // 2)
        y = max(0, cy - int(fh * 0.45))  # 眉〜顎あたりを中心にやや上寄せ
        w = min(fw, w_img - x)
        h = min(fh, h_img - y)
        print("フォールバック: 中央推定矩形で顔領域を仮定")
        return (x, y, w, h)
    except Exception:
        pass

    return None


def detect_face_landmarks(image):
    """
    顔のランドマークを検出
    MediaPipeが利用可能なら478点、そうでなければ簡易的な顔領域情報を返す
    
    Args:
        image: BGR画像
    
    Returns:
        landmarks: MediaPipeランドマーク、または簡易FaceRegionオブジェクト、失敗時はNone
    """
    _init_face_detectors()
    
    original = image.copy()
    h, w = original.shape[:2]
    
    # MediaPipeが利用可能な場合
    if _mediapipe_available and _face_mesh is not None:
        preprocessing_methods = [
            ("original", lambda img: img),
            ("clahe", apply_clahe),
            ("gamma_bright", lambda img: apply_gamma(img, 1.5)),
            ("gamma_dark", lambda img: apply_gamma(img, 0.7)),
            ("histogram_eq", apply_histogram_equalization),
        ]
        
        for method_name, preprocess_func in preprocessing_methods:
            try:
                processed = preprocess_func(original.copy())
                img = normalize_image_size(processed)
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = _face_mesh.process(rgb_image)
                
                if results.multi_face_landmarks:
                    print(f"MediaPipe ({method_name})で顔検出成功")
                    return results.multi_face_landmarks[0]
            except Exception as e:
                continue
    
    # OpenCVベースのフォールバック
    face_rect = detect_face_opencv(original)
    if face_rect is not None:
        # 簡易的なFaceRegionオブジェクトを作成
        return SimpleFaceRegion(face_rect, (w, h))
    
    return None


class SimpleFaceRegion:
    """OpenCVで検出した顔領域を表すシンプルなクラス"""
    
    def __init__(self, face_rect, image_size):
        """
        Args:
            face_rect: (x, y, w, h) 顔の矩形
            image_size: (width, height) 画像サイズ
        """
        self.face_rect = face_rect
        self.image_size = image_size
        self.is_simple = True  # MediaPipeではないことを示すフラグ
        
        # 仮想的なランドマークを生成
        self.landmark = self._generate_virtual_landmarks()
    
    def _generate_virtual_landmarks(self):
        """顔の矩形から仮想的なランドマークを生成"""
        x, y, w, h = self.face_rect
        img_w, img_h = self.image_size
        
        # 正規化座標で仮想ランドマークを生成（478点に近似）
        landmarks = []
        
        # 顔の各部位の相対位置（経験的な比率）
        # 額: 上部15-35%
        # 目: 35-45%
        # 鼻: 45-70%
        # 口: 70-85%
        # 顎: 85-100%
        
        for i in range(478):
            # 簡略化のため、顔の矩形内にランダムに点を配置
            # 実際には各部位の典型的な位置に基づいて配置
            rel_x = (i % 22) / 22.0  # 0-1の範囲
            rel_y = (i // 22) / 22.0
            
            norm_x = (x + w * rel_x) / img_w
            norm_y = (y + h * rel_y) / img_h
            
            landmarks.append(VirtualLandmark(norm_x, norm_y))
        
        return landmarks


class VirtualLandmark:
    """仮想的なランドマーク点"""
    def __init__(self, x, y):
        self.x = x
        self.y = y


def normalize_image_size(image, target_max=1024, target_min=480):
    """画像サイズを正規化"""
    h, w = image.shape[:2]
    
    if max(w, h) > target_max:
        scale = target_max / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif max(w, h) < target_min:
        scale = target_min / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image


def apply_clahe(image):
    """CLAHEコントラスト強調"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def apply_gamma(image, gamma):
    """ガンマ補正"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_histogram_equalization(image):
    """ヒストグラム均一化"""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def extract_face_regions(image, landmarks):
    """
    顔画像から各部位を抽出
    MediaPipeランドマークまたはSimpleFaceRegion（OpenCVベース）に対応
    
    Args:
        image: BGR画像
        landmarks: MediaPipeの顔ランドマーク または SimpleFaceRegion
    
    Returns:
        dict: 各部位の画像と座標 {region_name: {'image': img, 'bbox': (x, y, w, h)}}
    """
    h, w = image.shape[:2]
    regions = {}
    
    # SimpleFaceRegion（OpenCVベース）の場合は、顔の矩形から部位を推定
    if hasattr(landmarks, 'is_simple') and landmarks.is_simple:
        return extract_face_regions_from_rect(image, landmarks.face_rect)
    
    # MediaPipeランドマークの場合
    # ランドマークを画像座標に変換
    points = []
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))
    points = np.array(points)
    
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
        # 鼻をさらにわずかに右下へ（+2.5%幅、+2%高）
        shift_x = max(1, int(0.025 * w))
        shift_y = max(1, int(0.020 * h))
        x = max(0, min(x + shift_x, w - w_region))
        y = max(0, min(y + shift_y, h - h_region))
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
        # 周辺（口周り）パディング：横10%、上9%、下16%（もう少し下寄せ）
        pad_x = max(2, int(w_region * 0.10))
        pad_top = max(1, int(h_region * 0.09))
        pad_bot = max(2, int(h_region * 0.16))
        x1 = max(0, x - pad_x)
        x2 = min(w, x + w_region + pad_x)
        y1 = max(0, y - pad_top)
        y2 = min(h, y + h_region + pad_bot)
        # 全体をさらに下へ2%（画像高基準）
        shift_down = max(1, int(0.02 * h))
        y1 = min(h - 1, y1 + shift_down)
        y2 = min(h, max(y1 + 1, y2 + max(1, int(0.01 * h))))
        regions['mouth_area'] = {
            'image': image[y1:y2, x1:x2],
            'bbox': (x1, y1, x2 - x1, y2 - y1)
        }
        mouth_y_end = y2
    
    # 顎
    # 顎（下顎ラインのみに限定して矩形化）
    chin_jaw_indices = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356]
    if all(i < len(points) for i in chin_jaw_indices):
        chin_points = points[chin_jaw_indices]
        cx, cy, cw, ch = cv2.boundingRect(chin_points)
        jaw_max_y = int(np.max(chin_points[:, 1]))
        jaw_min_y = int(np.min(chin_points[:, 1]))
        # 口領域の下端より少し下から開始。顎ラインを下回りすぎないよう制限。
        margin = max(2, int(0.02 * h))
        base_start = cy + int(ch * 0.18)
        try:
            base_start = max(base_start, mouth_y_end + margin)
        except NameError:
            pass
        # 顎ライン付近にクランプ（下へは+2%hまで）
        y_start = max(jaw_min_y, min(base_start, jaw_max_y - int(0.04 * h)))
        y_end = min(h, min(jaw_max_y + int(0.02 * h), y_start + int(0.08 * h)))
        # 横は少し内側に（頬を含みすぎないように）
        x1 = max(0, cx + int(cw * 0.05))
        x2 = min(w, cx + cw - int(cw * 0.05))
        if y_end > y_start and x2 > x1:
            regions['chin'] = {
                'image': image[y_start:y_end, x1:x2],
                'bbox': (x1, y_start, x2 - x1, y_end - y_start)
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


def extract_face_regions_from_rect(image, face_rect):
    """
    顔の矩形から各部位を推定して抽出（OpenCVフォールバック用）
    顔の一般的な比率に基づいて部位を推定
    
    Args:
        image: BGR画像
        face_rect: (x, y, w, h) 顔の矩形
    
    Returns:
        dict: 各部位の画像と座標
    """
    h, w = image.shape[:2]
    fx, fy, fw, fh = face_rect
    
    regions = {}
    
    # 顔の比率に基づいて部位を推定
    # 額: 顔の上部 0-25%
    forehead_y1 = fy
    forehead_y2 = fy + int(fh * 0.25)
    forehead_x1 = fx + int(fw * 0.15)
    forehead_x2 = fx + int(fw * 0.85)
    if forehead_y2 > forehead_y1 and forehead_x2 > forehead_x1:
        regions['forehead'] = {
            'image': image[forehead_y1:forehead_y2, forehead_x1:forehead_x2],
            'bbox': (forehead_x1, forehead_y1, forehead_x2-forehead_x1, forehead_y2-forehead_y1)
        }
    
    # 左頬: 顔の左側 35-65%の高さ、5-35%の幅
    lc_y1 = fy + int(fh * 0.35)
    lc_y2 = fy + int(fh * 0.70)
    lc_x1 = fx + int(fw * 0.05)
    lc_x2 = fx + int(fw * 0.35)
    if lc_y2 > lc_y1 and lc_x2 > lc_x1:
        regions['left_cheek'] = {
            'image': image[lc_y1:lc_y2, lc_x1:lc_x2],
            'bbox': (lc_x1, lc_y1, lc_x2-lc_x1, lc_y2-lc_y1)
        }
    
    # 右頬: 顔の右側 35-65%の高さ、65-95%の幅
    rc_y1 = fy + int(fh * 0.35)
    rc_y2 = fy + int(fh * 0.70)
    rc_x1 = fx + int(fw * 0.65)
    rc_x2 = fx + int(fw * 0.95)
    if rc_y2 > rc_y1 and rc_x2 > rc_x1:
        regions['right_cheek'] = {
            'image': image[rc_y1:rc_y2, rc_x1:rc_x2],
            'bbox': (rc_x1, rc_y1, rc_x2-rc_x1, rc_y2-rc_y1)
        }
    
    # 鼻: 顔の中央 32-65%の高さ、幅はさらに右寄り（36-60%）
    nose_y1 = fy + int(fh * 0.32)
    nose_y2 = fy + int(fh * 0.65)
    nose_x1 = fx + int(fw * 0.36)
    nose_x2 = fx + int(fw * 0.60)
    if nose_y2 > nose_y1 and nose_x2 > nose_x1:
        regions['nose'] = {
            'image': image[nose_y1:nose_y2, nose_x1:nose_x2],
            'bbox': (nose_x1, nose_y1, nose_x2-nose_x1, nose_y2-nose_y1)
        }
    
    # 口周り: 下げすぎて首にかからないように制限
    mouth_y1 = max(fy + int(fh * 0.60), nose_y2 - int(fh * 0.06))
    mouth_y2 = min(fy + int(fh * 0.78), fy + int(fh * 0.82))
    mouth_x1 = fx + int(fw * 0.32)
    mouth_x2 = fx + int(fw * 0.68)
    if mouth_y2 > mouth_y1 and mouth_x2 > mouth_x1:
        regions['mouth_area'] = {
            'image': image[mouth_y1:mouth_y2, mouth_x1:mouth_x2],
            'bbox': (mouth_x1, mouth_y1, mouth_x2-mouth_x1, mouth_y2-mouth_y1)
        }
    
    # 顎: 高さを最大8%fh、下端は顔矩形の93-96%に制限
    chin_x1 = fx + int(fw * 0.35)
    chin_x2 = fx + int(fw * 0.65)
    chin_y1 = min(fy + int(fh * 0.88), mouth_y2 + int(0.03 * fh))
    chin_y2 = min(fy + int(fh * 0.96), chin_y1 + int(fh * 0.08))
    if chin_y2 > chin_y1 and chin_x2 > chin_x1:
        regions['chin'] = {
            'image': image[chin_y1:chin_y2, chin_x1:chin_x2],
            'bbox': (chin_x1, chin_y1, chin_x2-chin_x1, chin_y2-chin_y1)
        }
    
    # 左目の下（クマの部分）: 42-55%の高さ、15-40%の幅
    # 目は約25-35%の高さにあり、その下のクマ・たるみ部分
    lue_y1 = fy + int(fh * 0.42)
    lue_y2 = fy + int(fh * 0.55)
    lue_x1 = fx + int(fw * 0.12)
    lue_x2 = fx + int(fw * 0.42)
    if lue_y2 > lue_y1 and lue_x2 > lue_x1:
        regions['left_under_eye'] = {
            'image': image[lue_y1:lue_y2, lue_x1:lue_x2],
            'bbox': (lue_x1, lue_y1, lue_x2-lue_x1, lue_y2-lue_y1)
        }
    
    # 右目の下（クマの部分）: 42-55%の高さ、60-88%の幅
    rue_y1 = fy + int(fh * 0.42)
    rue_y2 = fy + int(fh * 0.55)
    rue_x1 = fx + int(fw * 0.58)
    rue_x2 = fx + int(fw * 0.88)
    if rue_y2 > rue_y1 and rue_x2 > rue_x1:
        regions['right_under_eye'] = {
            'image': image[rue_y1:rue_y2, rue_x1:rue_x2],
            'bbox': (rue_x1, rue_y1, rue_x2-rue_x1, rue_y2-rue_y1)
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
