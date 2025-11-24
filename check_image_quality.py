# -*- coding: utf-8 -*-
# 画像品質診断ツール
# フォルダ内の全画像の品質を一覧表示

import os
import sys
import io
from image_quality_assessor import assess_image_quality
import glob

# Windows環境でUTF-8出力を強制
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_folder_quality(folder_path):
    """フォルダ内の全画像の品質を診断"""
    
    # 画像ファイルを取得
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"[エラー] フォルダに画像が見つかりません: {folder_path}")
        return
    
    print(f"[フォルダ] {folder_path}")
    print(f"[画像数] {len(image_files)}枚\n")
    print("=" * 100)
    
    # 品質レベル別にカウント
    quality_counts = {'high': 0, 'low4-7': 0, 'low1-3': 0, 'low8-10': 0, 'error': 0}
    
    results = []
    
    for i, img_path in enumerate(sorted(image_files), 1):
        filename = os.path.basename(img_path)
        result = assess_image_quality(img_path)
        
        if 'error' in result:
            quality_counts['error'] += 1
            print(f"{i}. [ERROR] {filename}: エラー - {result.get('message', 'Unknown error')}")
            continue
        
        quality_level = result['quality_level']
        quality_counts[quality_level] += 1
        
        metrics = result['metrics']
        rec = result['recommendation']
        
        # アイコンと処理方法
        icon = rec['icon']
        method = rec['processing_method']
        
        print(f"\n{i}. {icon} {filename}")
        print(f"   品質レベル: {quality_level} ({rec['title']})")
        print(f"   解像度: {metrics['resolution']} (スコア: {metrics['resolution_score']:.1f}%)")
        print(f"   シャープネス: {metrics['sharpness']:.2f}")
        print(f"   ノイズレベル: {metrics['noise_level']:.2f}")
        print(f"   JPEG品質: {metrics['estimated_jpeg_quality']}")
        print(f"   処理方法: {method}")
        print(f"   解析可否: {'[OK] 可能' if result['can_process'] else '[NG] 不可'}")
        
        results.append({
            'filename': filename,
            'quality_level': quality_level,
            'metrics': metrics,
            'can_process': result['can_process']
        })
    
    # サマリー表示
    print("\n" + "=" * 100)
    print("\n[品質レベル別サマリー]")
    print(f"   [OK] high (直接解析):        {quality_counts['high']:3d}枚 ({quality_counts['high']/len(image_files)*100:5.1f}%)")
    print(f"   [AI] low4-7 (AI予測):        {quality_counts['low4-7']:3d}枚 ({quality_counts['low4-7']/len(image_files)*100:5.1f}%)")
    print(f"   [OK] low1-3 (直接解析推奨):  {quality_counts['low1-3']:3d}枚 ({quality_counts['low1-3']/len(image_files)*100:5.1f}%)")
    print(f"   [NG] low8-10 (解析不可):     {quality_counts['low8-10']:3d}枚 ({quality_counts['low8-10']/len(image_files)*100:5.1f}%)")
    if quality_counts['error'] > 0:
        print(f"   [!!] エラー:                 {quality_counts['error']:3d}枚")
    
    analyzable = quality_counts['high'] + quality_counts['low4-7'] + quality_counts['low1-3']
    print(f"\n   合計解析可能: {analyzable}枚 / {len(image_files)}枚 ({analyzable/len(image_files)*100:.1f}%)")
    
    # 統計情報
    if results:
        print("\n[統計情報]")
        resolutions = [r['metrics']['resolution_score'] for r in results if r['can_process']]
        sharpnesses = [r['metrics']['sharpness'] for r in results if r['can_process']]
        
        if resolutions:
            print(f"   解像度スコア: 最小 {min(resolutions):.1f}%, 最大 {max(resolutions):.1f}%, 平均 {sum(resolutions)/len(resolutions):.1f}%")
        if sharpnesses:
            print(f"   シャープネス: 最小 {min(sharpnesses):.2f}, 最大 {max(sharpnesses):.2f}, 平均 {sum(sharpnesses)/len(sharpnesses):.2f}")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # デフォルトパス
        folder_path = r"E:\画質別頬画像(元画像＋10段階)"
    
    if not os.path.exists(folder_path):
        print(f"❌ フォルダが見つかりません: {folder_path}")
        print("\n使い方: python check_image_quality.py [フォルダパス]")
        sys.exit(1)
    
    check_folder_quality(folder_path)
