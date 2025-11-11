"""
診断用シンプルアプリ
"""
import streamlit as st

st.title("🔍 診断テスト")

st.write("## 1. 基本動作テスト")
st.success("✅ Streamlit は正常に動作しています")

st.write("## 2. モジュールインポートテスト")

# skin_quality_evaluator
try:
    from skin_quality_evaluator import SkinQualityEvaluator
    evaluator = SkinQualityEvaluator()
    grade = evaluator.get_grade(2.75)
    st.success(f"✅ skin_quality_evaluator: 正常 (test grade={grade})")
except Exception as e:
    st.error(f"❌ skin_quality_evaluator: エラー - {e}")

# image_quality_assessor  
try:
    from image_quality_assessor import assess_image_quality, RECOMMENDED_DEVICES
    st.success(f"✅ image_quality_assessor: 正常")
except Exception as e:
    st.error(f"❌ image_quality_assessor: エラー - {e}")

# 他の主要モジュール
try:
    import numpy as np
    import pandas as pd
    import cv2
    st.success("✅ numpy, pandas, cv2: 正常")
except Exception as e:
    st.error(f"❌ モジュールエラー - {e}")

st.write("## 3. 肌品質評価テスト")

try:
    from skin_quality_evaluator import SkinQualityEvaluator
    evaluator = SkinQualityEvaluator()
    
    test_fd = st.slider("テストFD値", 2.0, 3.0, 2.75, 0.01)
    
    grade = evaluator.get_grade(test_fd)
    grade_info = evaluator.grade_criteria[grade]
    
    st.metric("グレード", f"{grade_info['icon']} {grade}")
    st.info(f"**説明:** {grade_info['description']}")
    st.write(f"**解釈:** {grade_info['interpretation']}")
    
except Exception as e:
    st.error(f"エラー: {e}")
    import traceback
    st.code(traceback.format_exc())

st.write("---")
st.write("診断完了!")
