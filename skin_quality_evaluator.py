"""
肌品質評価モジュール
フラクタル次元に基づいて肌の品質を評価します
"""

import numpy as np
from typing import Dict, List, Tuple

class SkinQualityEvaluator:
    """
    フラクタル次元を用いた肌品質評価クラス
    
    フラクタル次元(FD)の解釈:
    - 高いFD値 (2.6-3.0): より複雑なテクスチャ → シワ、毛穴が目立つ
    - 低いFD値 (2.0-2.4): よりスムーズなテクスチャ → きめ細かい肌
    """
    
    def __init__(self):
        # 肌品質の基準値 (研究に基づく標準値)
        self.standards = {
            'excellent': {'max': 2.20, 'label': '非常に良い', 'emoji': '⭐⭐⭐⭐⭐'},
            'very_good': {'max': 2.35, 'label': '良い', 'emoji': '⭐⭐⭐⭐'},
            'good': {'max': 2.50, 'label': 'やや良い', 'emoji': '⭐⭐⭐'},
            'fair': {'max': 2.65, 'label': '普通', 'emoji': '⭐⭐'},
            'poor': {'max': 3.00, 'label': '要改善', 'emoji': '⭐'}
        }
        
        # 年齢別の平均値 (参考値)
        self.age_reference = {
            '10-20': {'avg': 2.15, 'std': 0.08},
            '20-30': {'avg': 2.25, 'std': 0.10},
            '30-40': {'avg': 2.40, 'std': 0.12},
            '40-50': {'avg': 2.55, 'std': 0.15},
            '50+': {'avg': 2.70, 'std': 0.18}
        }
    
    def evaluate_single(self, fd_value: float) -> Dict:
        """
        単一のフラクタル次元値を評価
        
        Args:
            fd_value: フラクタル次元値
            
        Returns:
            評価結果の辞書
        """
        # グレード判定
        grade = self._get_grade(fd_value)
        
        # スコア化 (0-100)
        score = self._calculate_score(fd_value)
        
        # 特徴分析
        features = self._analyze_features(fd_value)
        
        return {
            'fd_value': fd_value,
            'grade': grade['label'],
            'grade_emoji': grade['emoji'],
            'score': score,
            'interpretation': self._get_interpretation(fd_value),
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
        """グレード判定"""
        for grade, info in self.standards.items():
            if fd_value <= info['max']:
                return info
        return self.standards['poor']
    
    def _calculate_score(self, fd_value: float) -> float:
        """
        スコア計算 (0-100)
        FD 2.0 = 100点, FD 3.0 = 0点で線形補間
        """
        score = 100 - ((fd_value - 2.0) / (3.0 - 2.0)) * 100
        return max(0, min(100, score))
    
    def _analyze_features(self, fd_value: float) -> Dict:
        """肌の特徴分析"""
        features = {
            'smoothness': 'とてもスムーズ' if fd_value < 2.3 else 'スムーズ' if fd_value < 2.5 else '普通',
            'texture': 'きめ細かい' if fd_value < 2.3 else 'やや細かい' if fd_value < 2.5 else '普通',
            'complexity': '低' if fd_value < 2.4 else '中' if fd_value < 2.6 else '高'
        }
        return features
    
    def _get_interpretation(self, fd_value: float) -> str:
        """解釈メッセージ"""
        if fd_value < 2.20:
            return "非常にきめ細かく、スムーズな肌質です。理想的な状態です。"
        elif fd_value < 2.35:
            return "きめ細かく、良好な肌質です。現状を維持しましょう。"
        elif fd_value < 2.50:
            return "やや良好な肌質です。さらなる改善の余地があります。"
        elif fd_value < 2.65:
            return "普通の肌質です。保湿やケアで改善が期待できます。"
        else:
            return "肌のテクスチャがやや粗くなっています。スキンケアの見直しをお勧めします。"
    
    def _get_recommendations(self, fd_value: float) -> List[str]:
        """改善提案"""
        recommendations = []
        
        if fd_value < 2.30:
            recommendations = [
                "✅ 現在のスキンケアを継続",
                "✅ 紫外線対策を怠らない",
                "✅ 十分な睡眠と水分補給"
            ]
        elif fd_value < 2.50:
            recommendations = [
                "💧 保湿ケアを強化",
                "🌞 紫外線対策を徹底",
                "😴 規則正しい生活習慣"
            ]
        else:
            recommendations = [
                "💧 集中保湿ケアが必要",
                "🧴 ビタミンC誘導体配合化粧品の使用",
                "🌞 日焼け止めの徹底",
                "💤 十分な睡眠時間の確保",
                "🥗 バランスの取れた食事"
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
        """年齢層比較の解釈"""
        if z_score < -1.5:
            return "年齢層の平均より非常に良好です"
        elif z_score < -0.5:
            return "年齢層の平均より良好です"
        elif z_score < 0.5:
            return "年齢層の平均的な範囲内です"
        elif z_score < 1.5:
            return "年齢層の平均よりやや高めです"
        else:
            return "年齢層の平均より高めです。ケアの見直しをお勧めします"


# 使用例
if __name__ == "__main__":
    evaluator = SkinQualityEvaluator()
    
    # 単一評価
    result = evaluator.evaluate_single(2.25)
    print("=== 単一評価 ===")
    print(f"FD値: {result['fd_value']}")
    print(f"評価: {result['grade_emoji']} {result['grade']}")
    print(f"スコア: {result['score']:.1f}点")
    print(f"解釈: {result['interpretation']}")
    
    # 複数評価
    fd_values = [2.20, 2.25, 2.30, 2.22, 2.28]
    labels = ["頬", "額", "顎", "鼻", "目元"]
    multi_result = evaluator.evaluate_multiple(fd_values, labels)
    
    print("\n=== 総合評価 ===")
    print(f"平均FD: {multi_result['statistics']['mean']:.3f}")
    print(f"総合評価: {multi_result['overall']['grade']}")
