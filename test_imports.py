"""
fractal_app.pyのインポートテスト
"""

print("=" * 60)
print("インポートテスト開始")
print("=" * 60)

try:
    print("\n1. skin_quality_evaluator をインポート...")
    from skin_quality_evaluator import SkinQualityEvaluator
    print("   ✅ 成功")
    
    evaluator = SkinQualityEvaluator()
    print("   ✅ SkinQualityEvaluator インスタンス作成成功")
    
    grade = evaluator.get_grade(2.75)
    print(f"   ✅ get_grade(2.75) = {grade}")
    
except Exception as e:
    print(f"   ❌ エラー: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. image_quality_assessor をインポート...")
    from image_quality_assessor import assess_image_quality
    print("   ✅ 成功")
    
except Exception as e:
    print(f"   ❌ エラー: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. fractal_app の主要モジュールをインポート...")
    import streamlit as st
    print("   ✅ streamlit インポート成功")
    
    import numpy as np
    print("   ✅ numpy インポート成功")
    
    import pandas as pd
    print("   ✅ pandas インポート成功")
    
    import cv2
    print("   ✅ cv2 インポート成功")
    
except Exception as e:
    print(f"   ❌ エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("インポートテスト完了")
print("=" * 60)
