import streamlit as st

st.set_page_config(page_title="診断テスト", page_icon="🔍")

st.title("🔍 Fractal Analyzer V2 - 診断")

st.write("## ステップ1: 基本動作確認")
if st.button("✅ Streamlit は動作しています"):
    st.success("Streamlit正常!")

st.write("## ステップ2: モジュールインポート確認")

# skin_quality_evaluator
with st.expander("skin_quality_evaluator テスト", expanded=True):
    try:
        from skin_quality_evaluator import SkinQualityEvaluator
        evaluator = SkinQualityEvaluator()
        
        st.success("✅ インポート成功")
        
        # 動作テスト
        test_fd = 2.75
        grade = evaluator.get_grade(test_fd)
        grade_info = evaluator.grade_criteria[grade]
        
        st.info(f"テストFD={test_fd} → グレード: {grade_info['icon']} {grade}")
        st.write(f"**説明:** {grade_info['description']}")
        
    except Exception as e:
        st.error(f"❌ エラー: {e}")
        import traceback
        st.code(traceback.format_exc())

# image_quality_assessor
with st.expander("image_quality_assessor テスト"):
    try:
        from image_quality_assessor import assess_image_quality, RECOMMENDED_DEVICES
        st.success("✅ インポート成功")
        st.write(f"推奨デバイス数: {len(RECOMMENDED_DEVICES)}カテゴリ")
        
    except Exception as e:
        st.error(f"❌ エラー: {e}")
        import traceback
        st.code(traceback.format_exc())

st.write("---")
st.write("### 🎯 次のステップ")
st.info("""
すべてのテストが✅なら、fractal_app.pyも正常に動作するはずです。

❌エラーが表示されている場合は、そのエラーメッセージを確認してください。
""")
