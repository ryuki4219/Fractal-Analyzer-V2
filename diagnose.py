import sys
import traceback

print("=" * 60)
print("エラー診断スクリプト")
print("=" * 60)

# 1. skin_quality_evaluatorのテスト
print("\n[1] skin_quality_evaluatorのインポートテスト...")
try:
    from skin_quality_evaluator import SkinQualityEvaluator
    print("✅ インポート成功")
    
    evaluator = SkinQualityEvaluator()
    print("✅ インスタンス作成成功")
    
    # get_gradeのテスト
    test_value = 2.75
    grade = evaluator.get_grade(test_value)
    print(f"✅ get_grade({test_value}) = {grade}")
    
    # grade_criteriaのテスト
    grade_info = evaluator.grade_criteria[grade]
    print(f"✅ grade_criteria['{grade}'] = {grade_info['icon']} {grade_info['description']}")
    
except Exception as e:
    print(f"❌ エラー発生:")
    print(f"   {type(e).__name__}: {e}")
    traceback.print_exc()

# 2. image_quality_assessorのテスト
print("\n[2] image_quality_assessorのインポートテスト...")
try:
    from image_quality_assessor import assess_image_quality
    print("✅ インポート成功")
except Exception as e:
    print(f"❌ エラー発生:")
    print(f"   {type(e).__name__}: {e}")
    traceback.print_exc()

# 3. fractal_appのインポートテスト
print("\n[3] fractal_appのインポートテスト...")
try:
    # Streamlitを使わずにインポート可能な部分をテスト
    import importlib.util
    spec = importlib.util.spec_from_file_location("fractal_app", "fractal_app.py")
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        print("✅ モジュールスペック作成成功")
        # 実際のロードはStreamlitが必要なのでスキップ
        print("⚠️  完全なロードはStreamlit環境が必要")
except Exception as e:
    print(f"❌ エラー発生:")
    print(f"   {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("診断完了")
print("=" * 60)
