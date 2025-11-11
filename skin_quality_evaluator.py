"""
肌品質評価モジュール
フラクタル次元に基づいて肌の品質を評価します
"""

import numpy as np
from typing import Dict, List, Tuple

class SkinQualityEvaluator:
    """
    フラクタル次元を用いた肌品質評価クラス
    
    【参考文献】中川匡弘「肌のフラクタル構造解析」光学 39巻11号 (2010)
    
    フラクタル次元(FD)の解釈（中川氏の研究に基づく）:
    - 高いFD値 (2.7-3.0): フラクタル構造が複雑 → きめ細かく滑らかな肌
    - 中程度FD値 (2.4-2.7): 普通の複雑さ → 一般的な肌
    - 低いFD値 (2.0-2.4): 構造が単純 → 粗い肌、シワが目立つ
    
    ※「滑らかな肌ほどフラクタル次元が3に近い」という知見に基づく
    """
    
    def __init__(self):
        # 肌品質の基準値（中川氏の研究に基づいて修正）
        # FD値が高い(3に近い)ほど滑らかで綺麗な肌
        self.standards = {
            'excellent': {'min': 2.80, 'label': '非常に良い（S）', 'emoji': '⭐⭐⭐⭐⭐'},
            'very_good': {'min': 2.70, 'label': '良い（A）', 'emoji': '⭐⭐⭐⭐'},
            'good': {'min': 2.60, 'label': 'やや良い（B）', 'emoji': '⭐⭐⭐'},
            'fair': {'min': 2.50, 'label': '普通（C）', 'emoji': '⭐⭐'},
            'poor': {'min': 0.0, 'label': '要改善（D）', 'emoji': '⭐'}
        }
        
        # グレード基準（S/A/B/C/Dシステム）
        self.grade_criteria = {
            'S': {
                'range': (2.80, 3.00),
                'description': '非常に滑らか',
                'icon': '🌟',
                'interpretation': 'きめ細かく、非常に滑らかな肌質です。フラクタル構造が複雑で理想的な状態です。',
                'recommendation': '現在のスキンケアを継続し、紫外線対策を怠らないようにしましょう。'
            },
            'A': {
                'range': (2.70, 2.80),
                'description': '滑らか',
                'icon': '✨',
                'interpretation': 'きめ細かく滑らかな肌質です。良好な状態を維持しています。',
                'recommendation': '現状維持を心がけ、保湿と紫外線対策を継続しましょう。'
            },
            'B': {
                'range': (2.60, 2.70),
                'description': '普通',
                'icon': '👍',
                'interpretation': '一般的な肌質です。さらなる改善の余地があります。',
                'recommendation': '保湿ケアを強化し、規則正しい生活習慣を心がけましょう。'
            },
            'C': {
                'range': (2.50, 2.60),
                'description': 'やや粗い',
                'icon': '💧',
                'interpretation': 'やや粗めの肌質です。スキンケアで改善が期待できます。',
                'recommendation': '集中保湿ケア、ビタミンC誘導体配合化粧品の使用をお勧めします。'
            },
            'D': {
                'range': (0.0, 2.50),
                'description': '粗い',
                'icon': '⚠️',
                'interpretation': '肌のフラクタル構造が単純化しています。積極的なケアが必要です。',
                'recommendation': '皮膚科専門医への相談、集中的な保湿ケア、生活習慣の見直しをお勧めします。'
            }
        }
        
        # 年齢別の平均値（中川氏の研究に基づいて修正）
        # 若い肌ほどFD値が高い（構造が複雑）
        self.age_reference = {
            '10-20': {'avg': 2.75, 'std': 0.08},
            '20-30': {'avg': 2.70, 'std': 0.10},
            '30-40': {'avg': 2.60, 'std': 0.12},
            '40-50': {'avg': 2.50, 'std': 0.15},
            '50+': {'avg': 2.40, 'std': 0.18}
        }
    
    def get_grade(self, fd_value: float) -> str:
        """
        FD値からグレード(S/A/B/C/D)を取得
        
        Args:
            fd_value: フラクタル次元値
            
        Returns:
            グレード文字列
        """
        if fd_value >= 2.80:
            return 'S'
        elif fd_value >= 2.70:
            return 'A'
        elif fd_value >= 2.60:
            return 'B'
        elif fd_value >= 2.50:
            return 'C'
        else:
            return 'D'
    
    def evaluate_single(self, fd_value: float) -> Dict:
        """
        単一のフラクタル次元値を評価
        
        Args:
            fd_value: フラクタル次元値
            
        Returns:
            評価結果の辞書
        """
        # グレード判定
        grade = self.get_grade(fd_value)
        grade_info = self.grade_criteria[grade]
        
        # スコア化 (0-100)
        score = self._calculate_score(fd_value)
        
        # 特徴分析
        features = self._analyze_features(fd_value)
        
        return {
            'fd_value': fd_value,
            'grade': grade,
            'grade_info': grade_info,
            'grade_emoji': grade_info['icon'],
            'score': score,
            'interpretation': grade_info['interpretation'],
            'features': features,
            'recommendations': self._get_recommendations(fd_value)
        }
    
    def evaluate_multiple(self, fd_values: List[float], labels: List[str] = None) -> Dict:
        """
        複数の画像のフラクタル次元を評価
        
        Args:
            fd_values: フラクタル次元値のリスト
            labels: 画像のラベル（オプション）
            
        Returns:
            総合評価結果
        """
        if not fd_values:
            return None
        
        fd_array = np.array(fd_values)
        
        # 統計情報
        stats = {
            'mean': float(np.mean(fd_array)),
            'median': float(np.median(fd_array)),
            'std': float(np.std(fd_array)),
            'min': float(np.min(fd_array)),
            'max': float(np.max(fd_array)),
            'range': float(np.max(fd_array) - np.min(fd_array))
        }
        
        # 平均値での評価
        overall_eval = self.evaluate_single(stats['mean'])
        
        # 個別評価
        individual_evals = []
        for i, fd in enumerate(fd_values):
            label = labels[i] if labels and i < len(labels) else f"画像{i+1}"
            eval_result = self.evaluate_single(fd)
            eval_result['label'] = label
            individual_evals.append(eval_result)
        
        # 一貫性の評価
        consistency = self._evaluate_consistency(fd_array)
        
        return {
            'overall': overall_eval,
            'statistics': stats,
            'individual': individual_evals,
            'consistency': consistency
        }
    
    def compare_with_age_group(self, fd_value: float, age_group: str) -> Dict:
        """
        年齢層の平均値と比較
        
        Args:
            fd_value: フラクタル次元値
            age_group: 年齢層 ('10-20', '20-30', '30-40', '40-50', '50+')
            
        Returns:
            比較結果
        """
        if age_group not in self.age_reference:
            return {'error': '無効な年齢層です'}
        
        ref = self.age_reference[age_group]
        diff = fd_value - ref['avg']
        z_score = diff / ref['std']
        
        # パーセンタイル推定
        percentile = self._z_to_percentile(z_score)
        
        return {
            'age_group': age_group,
            'your_value': fd_value,
            'age_average': ref['avg'],
            'difference': diff,
            'z_score': z_score,
            'percentile': percentile,
            'interpretation': self._interpret_comparison(z_score)
        }
    
    def _get_grade(self, fd_value: float) -> Dict:
        """グレード判定（旧関数・互換性のため残す）"""
        grade_letter = self.get_grade(fd_value)
        return self.grade_criteria[grade_letter]
    
    def _calculate_score(self, fd_value: float) -> float:
        """
        スコア計算 (0-100)
        【修正】FD値が高い(3に近い)ほど高スコア
        FD 3.0 = 100点, FD 2.0 = 0点で線形補間
        """
        score = ((fd_value - 2.0) / (3.0 - 2.0)) * 100
        return max(0, min(100, score))
    
    def _analyze_features(self, fd_value: float) -> Dict:
        """
        肌の特徴分析（中川氏の研究に基づいて修正）
        FD値が高い = フラクタル構造が複雑 = きめ細かく滑らか
        """
        features = {
            'smoothness': 'とてもスムーズ' if fd_value >= 2.75 else 'スムーズ' if fd_value >= 2.60 else '普通' if fd_value >= 2.50 else 'やや粗い',
            'texture': 'きめ細かい' if fd_value >= 2.75 else 'やや細かい' if fd_value >= 2.60 else '普通' if fd_value >= 2.50 else 'やや粗い',
            'complexity': '高（理想的）' if fd_value >= 2.70 else '中' if fd_value >= 2.50 else '低（要ケア）'
        }
        return features
    
    def _get_interpretation(self, fd_value: float) -> str:
        """解釈メッセージ（中川氏の研究に基づいて修正）"""
        grade = self.get_grade(fd_value)
        return self.grade_criteria[grade]['interpretation']
    
    def _get_recommendations(self, fd_value: float) -> List[str]:
        """改善提案（中川氏の研究に基づいて修正）"""
        recommendations = []
        
        if fd_value >= 2.75:  # S, A グレード
            recommendations = [
                "✅ 現在のスキンケアを継続",
                "✅ 紫外線対策を怠らない",
                "✅ 十分な睡眠と水分補給"
            ]
        elif fd_value >= 2.60:  # B グレード
            recommendations = [
                "💧 保湿ケアを強化",
                "🌞 紫外線対策を徹底",
                "😴 規則正しい生活習慣"
            ]
        elif fd_value >= 2.50:  # C グレード
            recommendations = [
                "💧 集中保湿ケアが必要",
                "🧴 ビタミンC誘導体配合化粧品の使用",
                "🌞 日焼け止めの徹底",
                "💤 十分な睡眠時間の確保"
            ]
        else:  # D グレード
            recommendations = [
                "⚠️ 皮膚科専門医への相談を推奨",
                "💧 集中的な保湿ケア",
                "🧴 レチノール配合化粧品の検討",
                "🌞 徹底的な紫外線対策",
                "💤 十分な睡眠時間の確保",
                "🥗 バランスの取れた食事",
                "💊 必要に応じてサプリメント"
            ]
        
        return recommendations
    
    def _evaluate_consistency(self, fd_array: np.ndarray) -> Dict:
        """一貫性評価"""
        std = np.std(fd_array)
        
        if std < 0.05:
            consistency = '非常に均一'
            message = '肌質が非常に均一です'
        elif std < 0.10:
            consistency = '均一'
            message = '肌質はおおむね均一です'
        elif std < 0.15:
            consistency = 'やや不均一'
            message = '部位によってやや差があります'
        else:
            consistency = '不均一'
            message = '部位による差が大きいです'
        
        return {
            'level': consistency,
            'std_dev': float(std),
            'message': message
        }
    
    def _z_to_percentile(self, z_score: float) -> float:
        """Z-scoreをパーセンタイルに変換（近似）"""
        from scipy import stats
        try:
            return float(stats.norm.cdf(z_score) * 100)
        except:
            # scipyが利用できない場合の近似
            return 50 + z_score * 15
    
    def _interpret_comparison(self, z_score: float) -> str:
        """
        年齢層比較の解釈（中川氏の研究に基づいて修正）
        FD値が高い = 良好
        """
        if z_score > 1.5:
            return "年齢層の平均より非常に良好です（フラクタル構造が複雑）"
        elif z_score > 0.5:
            return "年齢層の平均より良好です"
        elif z_score > -0.5:
            return "年齢層の平均的な範囲内です"
        elif z_score > -1.5:
            return "年齢層の平均よりやや低めです"
        else:
            return "年齢層の平均より低めです。スキンケアの見直しをお勧めします"


# 使用例
if __name__ == "__main__":
    evaluator = SkinQualityEvaluator()
    
    print("=" * 70)
    print("肌品質評価システム（中川匡弘氏の研究に基づく）")
    print("=" * 70)
    print("\n【重要】FD値が高い(3に近い)ほど、フラクタル構造が複雑 = 滑らかな肌\n")
    
    # 単一評価の例
    test_values = [
        (2.85, "若い女性の頬（理想的）"),
        (2.72, "20代女性の頬"),
        (2.65, "30代女性の頬"),
        (2.55, "40代女性の頬"),
        (2.45, "スキンケア要改善")
    ]
    
    for fd, description in test_values:
        result = evaluator.evaluate_single(fd)
        print(f"\n=== {description} ===")
        print(f"FD値: {result['fd_value']:.2f}")
        print(f"グレード: {result['grade_emoji']} {result['grade']} - {result['grade_info']['description']}")
        print(f"スコア: {result['score']:.1f}/100点")
        print(f"解釈: {result['interpretation']}")
        print(f"特徴: 滑らかさ={result['features']['smoothness']}, "
              f"きめ={result['features']['texture']}, "
              f"複雑さ={result['features']['complexity']}")
    
    # 複数評価の例
    print("\n" + "=" * 70)
    print("=== 複数部位の評価例 ===")
    print("=" * 70)
    fd_values = [2.75, 2.72, 2.68, 2.70, 2.73]
    labels = ["頬", "額", "顎", "鼻", "目元"]
    multi_result = evaluator.evaluate_multiple(fd_values, labels)
    
    print(f"\n平均FD: {multi_result['statistics']['mean']:.3f}")
    print(f"総合グレード: {multi_result['overall']['grade_emoji']} {multi_result['overall']['grade']}")
    print(f"総合スコア: {multi_result['overall']['score']:.1f}/100点")
    print(f"\n一貫性: {multi_result['consistency']['level']} (標準偏差: {multi_result['consistency']['std_dev']:.3f})")
    
    # 年齢層比較の例
    print("\n" + "=" * 70)
    print("=== 年齢層比較例 ===")
    print("=" * 70)
    comparison = evaluator.compare_with_age_group(2.75, '20-30')
    print(f"あなたのFD値: {comparison['your_value']:.2f}")
    print(f"20-30歳代平均: {comparison['age_average']:.2f}")
    print(f"差: {comparison['difference']:+.2f}")
    print(f"評価: {comparison['interpretation']}")
