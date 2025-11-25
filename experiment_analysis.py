# -*- coding: utf-8 -*-
"""
実験データ収集・相関分析モジュール
肌状態とフラクタル次元の関係を科学的に検証
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


class ExperimentDataManager:
    """実験データの管理クラス"""
    
    def __init__(self, csv_file='experimental_data.csv'):
        self.csv_file = csv_file
    
    def save_data(self, data_entry: Dict) -> bool:
        """
        実験データを保存
        
        Args:
            data_entry: データエントリ（辞書形式）
        
        Returns:
            bool: 保存成功したかどうか
        """
        try:
            df_new = pd.DataFrame([data_entry])
            
            if os.path.exists(self.csv_file):
                df_existing = pd.read_csv(self.csv_file, encoding='utf-8-sig')
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            df_combined.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
            return True
        
        except Exception as e:
            print(f"データ保存エラー: {e}")
            return False
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        実験データを読み込み
        
        Returns:
            DataFrame: 実験データ、存在しない場合はNone
        """
        if not os.path.exists(self.csv_file):
            return None
        
        try:
            return pd.read_csv(self.csv_file, encoding='utf-8-sig')
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def get_subject_history(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        特定の被験者の履歴を取得
        
        Args:
            subject_id: 被験者ID
        
        Returns:
            DataFrame: 被験者の測定履歴
        """
        df = self.load_data()
        if df is None:
            return None
        
        return df[df['subject_id'] == subject_id].sort_values('timestamp')


def calculate_correlations(df: pd.DataFrame) -> Dict:
    """
    フラクタル次元と各肌指標の相関を計算
    
    Args:
        df: 実験データのDataFrame
    
    Returns:
        dict: 相関係数とp値
    """
    from scipy import stats
    
    correlations = {}
    
    if 'average_fd' not in df.columns:
        return correlations
    
    # FD値と相関を計算する項目
    correlation_items = {
        # 主観評価スコア
        'roughness_score': '肌荒れ度（主観）',
        'dryness_score': '乾燥度（主観）',
        'moisture_level': '水分量',
        'sebum_level': '皮脂量',
        'pore_score': '毛穴（主観）',
        'wrinkle_score': 'シワ（主観）',
        'redness_score': '赤み（主観）',
        'dark_circle_score': 'クマ（主観）',
        'age': '年齢',
        # 自動検出肌トラブルスコア（画像解析結果）
        'trouble_pore_visibility': '毛穴の目立ち（自動検出）',
        'trouble_wrinkles': 'シワ（自動検出）',
        'trouble_color_unevenness': '色ムラ・くすみ（自動検出）',
        'trouble_redness_acne': 'ニキビ・赤み（自動検出）',
        'trouble_dark_circles': 'クマ（自動検出）',
        'trouble_oiliness': 'テカリ（自動検出）',
        'trouble_total_score': '肌トラブル総合スコア（自動検出）'
    }
    
    for col, name_jp in correlation_items.items():
        if col in df.columns:
            # 欠損値を除外
            valid_data = df[['average_fd', col]].dropna()
            
            if len(valid_data) >= 3:  # 最低3データ点必要
                r, p_value = stats.pearsonr(valid_data['average_fd'], valid_data[col])
                correlations[name_jp] = {
                    'r': r,
                    'p_value': p_value,
                    'n': len(valid_data),
                    'significant': p_value < 0.05
                }
    
    return correlations


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       x_label: str, y_label: str, title: str):
    """
    散布図を作成（回帰直線付き）
    
    Args:
        df: データフレーム
        x_col: X軸の列名
        y_col: Y軸の列名
        x_label: X軸ラベル
        y_label: Y軸ラベル
        title: グラフタイトル
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 欠損値を除外
    valid_data = df[[x_col, y_col]].dropna()
    
    if len(valid_data) == 0:
        ax.text(0.5, 0.5, 'データが不足しています', 
               ha='center', va='center', fontsize=14)
        return fig
    
    x = valid_data[x_col]
    y = valid_data[y_col]
    
    # 散布図
    ax.scatter(x, y, s=100, alpha=0.6, color='steelblue', 
              edgecolors='darkblue', linewidth=1.5)
    
    # 回帰直線
    if len(valid_data) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
               label=f'回帰直線: y={z[0]:.4f}x+{z[1]:.4f}')
        
        # 相関係数
        from scipy import stats
        r, p_value = stats.pearsonr(x, y)
        
        # テキストボックス
        textstr = f'相関係数 r = {r:.3f}\np値 = {p_value:.4f}\nデータ数 n = {len(valid_data)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
    
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if len(valid_data) >= 2:
        ax.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    return fig


def create_correlation_heatmap(correlations: Dict):
    """
    相関係数のヒートマップを作成
    
    Args:
        correlations: 相関係数の辞書
    
    Returns:
        matplotlib figure
    """
    import seaborn as sns
    
    if not correlations:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'データが不足しています', 
               ha='center', va='center', fontsize=14)
        return fig
    
    # データを整形
    names = list(correlations.keys())
    r_values = [correlations[name]['r'] for name in names]
    p_values = [correlations[name]['p_value'] for name in names]
    
    # DataFrameを作成
    df_corr = pd.DataFrame({
        '項目': names,
        '相関係数': r_values,
        'p値': p_values
    })
    
    # 相関係数でソート
    df_corr = df_corr.sort_values('相関係数', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # バープロット
    colors = ['red' if r < 0 else 'blue' for r in df_corr['相関係数']]
    bars = ax.barh(df_corr['項目'], df_corr['相関係数'], color=colors, alpha=0.7)
    
    # 有意性マーカー
    for i, (idx, row) in enumerate(df_corr.iterrows()):
        if row['p値'] < 0.01:
            marker = '**'
        elif row['p値'] < 0.05:
            marker = '*'
        else:
            marker = ''
        
        x_pos = row['相関係数']
        offset = 0.05 if x_pos >= 0 else -0.05
        ax.text(x_pos + offset, i, f"{row['相関係数']:.3f}{marker}", 
               va='center', fontweight='bold', fontsize=10)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('相関係数 (r)', fontsize=12, fontweight='bold')
    ax.set_title('フラクタル次元との相関分析\n(* p<0.05, ** p<0.01)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-1, 1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def generate_experiment_summary(df: pd.DataFrame) -> str:
    """
    実験サマリーレポートを生成
    
    Args:
        df: 実験データのDataFrame
    
    Returns:
        str: マークダウン形式のレポート
    """
    lines = ["# 📊 実験データサマリー\n"]
    
    # 基本統計
    lines.append(f"## 📋 基本情報")
    lines.append(f"- **総測定回数**: {len(df)}回")
    lines.append(f"- **被験者数**: {df['subject_id'].nunique()}人")
    
    if 'average_fd' in df.columns:
        lines.append(f"- **FD値範囲**: {df['average_fd'].min():.4f} - {df['average_fd'].max():.4f}")
        lines.append(f"- **FD値平均**: {df['average_fd'].mean():.4f} ± {df['average_fd'].std():.4f}")
    
    lines.append("")
    
    # 被験者の属性
    if 'gender' in df.columns:
        lines.append(f"## 👥 被験者属性")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            lines.append(f"- {gender}: {count}人")
    
    if 'age' in df.columns:
        lines.append(f"- **平均年齢**: {df['age'].mean():.1f}歳 (範囲: {df['age'].min():.0f}-{df['age'].max():.0f}歳)")
    
    lines.append("")
    
    # 測定条件の分布
    if 'condition' in df.columns:
        lines.append(f"## 🌡️ 測定条件")
        condition_counts = df['condition'].value_counts()
        for condition, count in condition_counts.items():
            lines.append(f"- {condition}: {count}回")
    
    return '\n'.join(lines)
