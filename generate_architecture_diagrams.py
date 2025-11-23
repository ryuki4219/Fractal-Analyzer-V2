"""
システムアーキテクチャ図を画像として生成するスクリプト
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'MS Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

def create_module_composition_diagram():
    """モジュール構成図"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 9.5, 'Fractal Analyzer V2\nモジュール構成図', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    
    # メインシステム
    main_box = FancyBboxPatch((1, 7.5), 8, 1.2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#2C3E50', facecolor='#3498DB', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(main_box)
    ax.text(5, 8.1, 'Fractal Analyzer V2\n(肌品質評価システム)', 
            ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    # 3層アーキテクチャ
    layers = [
        {'name': 'UI Layer\n(Streamlit)', 'x': 1.5, 'color': '#E74C3C'},
        {'name': 'Core Engine\n(分析処理)', 'x': 4.5, 'color': '#27AE60'},
        {'name': 'Data Layer\n(永続化)', 'x': 7.5, 'color': '#F39C12'}
    ]
    
    for layer in layers:
        box = FancyBboxPatch((layer['x']-0.8, 4.5), 1.6, 1.5, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#34495E', facecolor=layer['color'], 
                             linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(layer['x'], 5.25, layer['name'], 
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # 下部ラベル
    labels = [
        {'text': '[ユーザー]', 'x': 2.3},
        {'text': '[アルゴリズム]', 'x': 5},
        {'text': '[ファイル]', 'x': 8.3}
    ]
    
    for label in labels:
        ax.text(label['x'], 3.5, label['text'], 
                ha='center', va='center', fontsize=11, 
                style='italic', color='#2C3E50')
    
    # 矢印（メインシステムから各層へ）
    for layer in layers:
        arrow = FancyArrowPatch((5, 7.5), (layer['x'], 6.0),
                               arrowstyle='->', mutation_scale=25, 
                               linewidth=2, color='#2C3E50', alpha=0.6)
        ax.add_patch(arrow)
        
        # 各層から下部ラベルへ
        arrow2 = FancyArrowPatch((layer['x'], 4.5), (layer['x'], 3.8),
                                arrowstyle='->', mutation_scale=20, 
                                linewidth=1.5, color='#34495E', alpha=0.5)
        ax.add_patch(arrow2)
    
    plt.tight_layout()
    plt.savefig('diagram_1_module_composition.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ モジュール構成図を保存しました: diagram_1_module_composition.png")
    plt.close()


def create_project_structure_diagram():
    """プロジェクト構成図"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 13.5, 'Fractal Analyzer V2\nプロジェクト構成', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    
    # ルートフォルダ
    root_box = FancyBboxPatch((0.5, 12), 9, 0.8, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#2C3E50', facecolor='#95A5A6', 
                              linewidth=2, alpha=0.5)
    ax.add_patch(root_box)
    ax.text(1, 12.4, '📁 Fractal-Analyzer-V2/', 
            ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 各セクション
    sections = [
        {
            'title': '📄 コアモジュール (3ファイル)',
            'y': 10.5,
            'items': [
                'fractal_app.py (5,719行)',
                'image_quality_assessor.py (610行)',
                'skin_quality_evaluator.py (388行)'
            ],
            'color': '#E74C3C'
        },
        {
            'title': '🎨 ビューアーツール (3ファイル)',
            'y': 8.5,
            'items': [
                'image_viewer.py',
                'skin_data_viewer.py',
                'skin_viewer.py'
            ],
            'color': '#3498DB'
        },
        {
            'title': '💾 データ・モデル',
            'y': 6.5,
            'items': [
                'trained_fd_model.pkl',
                'training_history.json',
                'requirements.txt',
                'SKIN_DATA/ (画像データセット)',
                'models/ (モデル保存)'
            ],
            'color': '#27AE60'
        },
        {
            'title': '📚 ドキュメント (18ファイル)',
            'y': 3.8,
            'items': [
                'README.md, USER_GUIDE.md, QUICK_START.md',
                'TROUBLESHOOTING.md, SYSTEM_ARCHITECTURE.md',
                'VALIDATION_MODE_GUIDE.md, その他各種ガイド',
                'PRIVACY_POLICY.md, TERMS_OF_SERVICE.md',
                'LICENSE (MIT)'
            ],
            'color': '#F39C12'
        },
        {
            'title': '🚀 起動スクリプト (4ファイル)',
            'y': 1.5,
            'items': [
                '起動.bat, 簡単起動.bat',
                'トラブルシューティング.bat',
                'デスクトップショートカット作成.bat'
            ],
            'color': '#9B59B6'
        }
    ]
    
    for section in sections:
        # セクションタイトル
        title_box = FancyBboxPatch((1, section['y']+0.5), 8, 0.5, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='#2C3E50', facecolor=section['color'], 
                                   linewidth=2, alpha=0.7)
        ax.add_patch(title_box)
        ax.text(1.2, section['y']+0.75, section['title'], 
                ha='left', va='center', fontsize=11, color='white', fontweight='bold')
        
        # 項目リスト
        for i, item in enumerate(section['items']):
            y_pos = section['y'] - 0.15 - (i * 0.3)
            ax.text(1.5, y_pos, f'• {item}', 
                    ha='left', va='center', fontsize=9, color='#2C3E50')
    
    # フッター
    ax.text(5, 0.3, '総ファイル数: 約30ファイル (開発ファイル削除後)', 
            ha='center', va='center', fontsize=10, 
            style='italic', color='#7F8C8D', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('diagram_2_project_structure.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ プロジェクト構成図を保存しました: diagram_2_project_structure.png")
    plt.close()


def create_detailed_module_diagram():
    """詳細モジュール図（3つのコアモジュール）"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 20))
    
    # 1. fractal_app.py
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # タイトル
    title_box = FancyBboxPatch((0.5, 7), 9, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#E74C3C', facecolor='#E74C3C', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(title_box)
    ax.text(5, 7.4, 'fractal_app.py (5,719行)\nメインアプリケーション', 
            ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    # 4つのモード
    modes = [
        {'name': '学習モード\nTraining', 'x': 1.5, 'color': '#3498DB'},
        {'name': '推論モード\nInference', 'x': 3.5, 'color': '#27AE60'},
        {'name': '肌品質評価\nAssessment', 'x': 5.5, 'color': '#F39C12'},
        {'name': '検証モード\nValidation', 'x': 7.5, 'color': '#9B59B6'}
    ]
    
    for mode in modes:
        box = FancyBboxPatch((mode['x']-0.6, 5.5), 1.2, 1, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#2C3E50', facecolor=mode['color'], 
                             linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(mode['x'], 6, mode['name'], 
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # コア機能
    core_box = FancyBboxPatch((1, 3.5), 8, 1.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#34495E', facecolor='#ECF0F1', 
                              linewidth=2, alpha=0.9)
    ax.add_patch(core_box)
    ax.text(5, 5.1, 'コア機能モジュール', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    
    core_features = [
        {'name': 'Box-Counting\nフラクタル解析\n(GPU対応)', 'x': 2.5},
        {'name': 'LightGBM\nAI予測エンジン\n(並列処理)', 'x': 5},
        {'name': '画像前処理\nリサイズ等', 'x': 7.5}
    ]
    
    for feature in core_features:
        box = FancyBboxPatch((feature['x']-0.7, 3.7), 1.4, 0.9, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#7F8C8D', facecolor='white', 
                             linewidth=1, alpha=0.9)
        ax.add_patch(box)
        ax.text(feature['x'], 4.15, feature['name'], 
                ha='center', va='center', fontsize=8)
    
    # 外部モジュール連携
    external_box = FancyBboxPatch((1, 1.8), 8, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  edgecolor='#34495E', facecolor='#D5DBDB', 
                                  linewidth=2, alpha=0.9)
    ax.add_patch(external_box)
    ax.text(5, 2.9, '外部モジュール連携', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    
    externals = [
        {'name': 'image_quality\n_assessor\n(画質判定)', 'x': 3},
        {'name': 'skin_quality\n_evaluator\n(肌品質評価)', 'x': 7}
    ]
    
    for ext in externals:
        box = FancyBboxPatch((ext['x']-0.9, 1.95), 1.8, 0.7, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#7F8C8D', facecolor='white', 
                             linewidth=1, alpha=0.9)
        ax.add_patch(box)
        ax.text(ext['x'], 2.3, ext['name'], 
                ha='center', va='center', fontsize=8)
    
    # データ永続化
    data_box = FancyBboxPatch((1, 0.3), 8, 1, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#34495E', facecolor='#AED6F1', 
                              linewidth=2, alpha=0.9)
    ax.add_patch(data_box)
    ax.text(5, 1.15, 'データ永続化', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(5, 0.65, '• trained_fd_model.pkl (LightGBMモデル)\n• training_history.json (訓練履歴)\n• 推論結果CSV出力', 
            ha='center', va='center', fontsize=8)
    
    # 2. image_quality_assessor.py
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 5), 9, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#3498DB', facecolor='#3498DB', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(title_box)
    ax.text(5, 5.4, 'image_quality_assessor.py (610行)\n画像品質自動判定システム', 
            ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    # 品質評価エンジン
    assess_items = [
        {'name': '解像度\nチェック', 'x': 1.2, 'y': 3.5},
        {'name': 'シャープネス\n測定(FFT)', 'x': 2.6, 'y': 3.5},
        {'name': 'ノイズ\nレベル測定', 'x': 4, 'y': 3.5},
        {'name': 'JPEG品質\n推定', 'x': 5.8, 'y': 3.5},
        {'name': 'ブレ検出\n(ラプラシアン)', 'x': 7.2, 'y': 3.5},
        {'name': '総合判定\n(合格/不合格)', 'x': 8.6, 'y': 3.5}
    ]
    
    for item in assess_items:
        box = FancyBboxPatch((item['x']-0.5, item['y']-0.3), 1, 0.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#2980B9', facecolor='#EBF5FB', 
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(item['x'], item['y'], item['name'], 
                ha='center', va='center', fontsize=7.5, fontweight='bold')
    
    # 推奨デバイス
    device_box = FancyBboxPatch((1, 1.8), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#16A085', facecolor='#D5F4E6', 
                                linewidth=2, alpha=0.9)
    ax.add_patch(device_box)
    ax.text(5, 2.65, '推奨デバイス情報', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(5, 2.2, '• iPhone 13 Pro以降 (48MP ProRAW)  • Google Pixel 7以降 (50MP)\n• Samsung Galaxy S23以降 (200MP)  • 一眼レフカメラ (2000万画素以上)', 
            ha='center', va='center', fontsize=7.5)
    
    # 出力情報
    output_box = FancyBboxPatch((1, 0.3), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#D68910', facecolor='#FCF3CF', 
                                linewidth=2, alpha=0.9)
    ax.add_patch(output_box)
    ax.text(5, 1.15, '出力情報', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(5, 0.7, '• is_high_quality (bool)  • quality_score (0-100)\n• issues (リスト)  • recommendations (推奨事項)', 
            ha='center', va='center', fontsize=7.5)
    
    # 3. skin_quality_evaluator.py
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 5), 9, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#27AE60', facecolor='#27AE60', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(title_box)
    ax.text(5, 5.4, 'skin_quality_evaluator.py (388行)\n肌品質評価・グレーディングシステム', 
            ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    # FD値ベース評価
    fd_box = FancyBboxPatch((1, 3.2), 8, 1.3, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='#229954', facecolor='#D5F4E6', 
                            linewidth=2, alpha=0.9)
    ax.add_patch(fd_box)
    ax.text(5, 4.35, 'フラクタル次元ベース評価', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    fd_grades = [
        '• FD 2.60以上: 優秀 (90-100点)',
        '• FD 2.55-2.60: 良好 (80-89点)',
        '• FD 2.50-2.55: 標準 (70-79点)',
        '• FD 2.50未満: 要改善 (70点未満)'
    ]
    
    for i, grade in enumerate(fd_grades):
        ax.text(5, 3.9 - i*0.25, grade, 
                ha='center', va='center', fontsize=8)
    
    # 総合評価レポート
    report_items = [
        {'name': 'グレード\n(S/A/B/C/D)', 'x': 2.5, 'color': '#F39C12'},
        {'name': 'スコア\n(0-100点)', 'x': 5, 'color': '#3498DB'},
        {'name': 'コメント\n(アドバイス)', 'x': 7.5, 'color': '#9B59B6'}
    ]
    
    ax.text(5, 2.5, '総合評価レポート', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    for item in report_items:
        box = FancyBboxPatch((item['x']-0.7, 1.5), 1.4, 0.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#2C3E50', facecolor=item['color'], 
                             linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        ax.text(item['x'], 1.8, item['name'], 
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # 可視化機能
    viz_box = FancyBboxPatch((1, 0.3), 8, 0.9, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#8E44AD', facecolor='#EBDEF0', 
                             linewidth=2, alpha=0.9)
    ax.add_patch(viz_box)
    ax.text(5, 1.05, '可視化機能', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(5, 0.7, '• レーダーチャート (多次元評価)  • スコアヒストリー (時系列変化)  • 比較分析 (複数画像)', 
            ha='center', va='center', fontsize=7.5)
    
    plt.tight_layout()
    plt.savefig('diagram_3_detailed_modules.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ 詳細モジュール図を保存しました: diagram_3_detailed_modules.png")
    plt.close()


def create_dataflow_diagrams():
    """データフロー図（4モード）"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    modes = [
        {
            'ax': axes[0, 0],
            'title': '学習モード',
            'color': '#3498DB',
            'steps': [
                ('高画質画像フォルダ', 7.5),
                ('画像読み込み\n(glob検索)', 6.5),
                ('前処理\n• リサイズ\n• グレー変換', 5.3),
                ('Box-Counting\nフラクタル解析\n(GPU加速)', 3.9),
                ('特徴量抽出\n• FD値\n• 統計量', 2.5),
                ('LightGBM訓練\n(並列処理)', 1.3),
                ('trained_fd_model.pkl\ntraining_history.json', 0.2)
            ]
        },
        {
            'ax': axes[0, 1],
            'title': '推論モード',
            'color': '#27AE60',
            'steps': [
                ('低画質画像', 7.5),
                ('画質判定(自動)\n← image_quality_assessor.py', 6.5),
                ('[合格] Box-Counting\n(低画質FD算出)', 5),
                ('LightGBMモデル AI予測\n← trained_fd_model.pkl', 3.8),
                ('補正FD値出力', 2.8),
                ('肌品質評価\n← skin_quality_evaluator.py', 1.8),
                ('評価レポート\nCSV出力', 0.5)
            ]
        },
        {
            'ax': axes[1, 0],
            'title': '肌品質評価モード',
            'color': '#F39C12',
            'steps': [
                ('画像アップロード', 7.5),
                ('画質チェック', 6.5),
                ('FD値計算\n(Box-Counting)', 5.3),
                ('AI補正(オプション)\n(低画質の場合)', 4),
                ('肌品質評価\n• グレード\n• スコア\n• アドバイス', 2.3),
                ('レポート表示\n可視化チャート', 0.5)
            ]
        },
        {
            'ax': axes[1, 1],
            'title': '検証モード',
            'color': '#9B59B6',
            'steps': [
                ('訓練データ', 7.5),
                ('モデル読み込み\n← trained_fd_model.pkl', 6.5),
                ('予測実行\n(全データ)', 5.3),
                ('精度評価\n• MAE\n• R²スコア\n• 相関係数', 3.5),
                ('検証レポート\n散布図・残差図', 1.5)
            ]
        }
    ]
    
    for mode in modes:
        ax = mode['ax']
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8.5)
        ax.axis('off')
        
        # タイトル
        title_box = FancyBboxPatch((1, 7.8), 8, 0.6, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor=mode['color'], facecolor=mode['color'], 
                                   linewidth=3, alpha=0.8)
        ax.add_patch(title_box)
        ax.text(5, 8.1, f"{mode['title']}のデータフロー", 
                ha='center', va='center', fontsize=13, color='white', fontweight='bold')
        
        # ステップ描画
        for i, (step_text, y_pos) in enumerate(mode['steps']):
            # ボックス
            is_terminal = i == 0 or i == len(mode['steps']) - 1
            box_color = '#ECF0F1' if not is_terminal else '#D5DBDB'
            edge_color = mode['color'] if is_terminal else '#7F8C8D'
            
            box = FancyBboxPatch((2, y_pos-0.35), 6, 0.7, 
                                 boxstyle="round,pad=0.08", 
                                 edgecolor=edge_color, facecolor=box_color, 
                                 linewidth=2 if is_terminal else 1.5, 
                                 alpha=0.9)
            ax.add_patch(box)
            ax.text(5, y_pos, step_text, 
                    ha='center', va='center', fontsize=9, fontweight='bold' if is_terminal else 'normal')
            
            # 矢印（最後のステップ以外）
            if i < len(mode['steps']) - 1:
                next_y = mode['steps'][i+1][1]
                arrow = FancyArrowPatch((5, y_pos-0.4), (5, next_y+0.4),
                                       arrowstyle='->', mutation_scale=20, 
                                       linewidth=2, color=mode['color'], alpha=0.7)
                ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('diagram_4_dataflow.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ データフロー図を保存しました: diagram_4_dataflow.png")
    plt.close()


def create_dependencies_diagram():
    """外部依存関係図"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 9.5, 'Fractal Analyzer V2\n外部依存関係', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Pythonパッケージ
    pkg_box = FancyBboxPatch((0.5, 5.5), 4.3, 3.3, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#2980B9', facecolor='#EBF5FB', 
                             linewidth=2, alpha=0.9)
    ax.add_patch(pkg_box)
    ax.text(2.65, 8.6, '必須Pythonパッケージ', 
            ha='center', va='top', fontsize=12, fontweight='bold', color='#2980B9')
    
    packages = [
        'streamlit >= 1.28.0',
        'lightgbm >= 4.0.0',
        'opencv-python-headless >= 4.8.0',
        'numpy >= 1.24.0',
        'scipy >= 1.11.0',
        'pandas >= 2.0.0',
        'matplotlib >= 3.7.0',
        'plotly >= 5.17.0',
        'scikit-learn >= 1.3.0',
        'pillow >= 10.0.0'
    ]
    
    for i, pkg in enumerate(packages):
        y_pos = 8.1 - i * 0.25
        ax.text(1, y_pos, f'• {pkg}', 
                ha='left', va='center', fontsize=8)
    
    # オプションパッケージ
    opt_box = FancyBboxPatch((5.2, 7.5), 4.3, 1.3, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#27AE60', facecolor='#D5F4E6', 
                             linewidth=2, alpha=0.9)
    ax.add_patch(opt_box)
    ax.text(7.35, 8.6, 'オプションパッケージ', 
            ha='center', va='top', fontsize=12, fontweight='bold', color='#27AE60')
    ax.text(7.35, 8, '• cupy-cuda12x\n  (GPU加速用)', 
            ha='center', va='center', fontsize=9)
    
    # システム要件
    sys_box = FancyBboxPatch((5.2, 5.5), 4.3, 1.6, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#8E44AD', facecolor='#EBDEF0', 
                             linewidth=2, alpha=0.9)
    ax.add_patch(sys_box)
    ax.text(7.35, 6.95, 'システム要件', 
            ha='center', va='top', fontsize=12, fontweight='bold', color='#8E44AD')
    ax.text(7.35, 6.2, 'Python: 3.9以上 (推奨: 3.11)\nメモリ: 4GB以上 (推奨: 8GB)\nGPU: CUDA対応 (オプション)\nOS: Windows/macOS/Linux', 
            ha='center', va='center', fontsize=8)
    
    # デプロイ環境
    deploy_box = FancyBboxPatch((0.5, 2.5), 9, 2.6, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#E74C3C', facecolor='#FADBD8', 
                                linewidth=2, alpha=0.9)
    ax.add_patch(deploy_box)
    ax.text(5, 4.95, 'デプロイ環境', 
            ha='center', va='top', fontsize=12, fontweight='bold', color='#E74C3C')
    
    # ローカル実行
    local_box = FancyBboxPatch((1, 3.5), 3.8, 1.2, 
                               boxstyle="round,pad=0.08", 
                               edgecolor='#34495E', facecolor='white', 
                               linewidth=1.5, alpha=0.9)
    ax.add_patch(local_box)
    ax.text(2.9, 4.35, 'ローカル実行', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(2.9, 3.9, 'streamlit run fractal_app.py', 
            ha='center', va='center', fontsize=8, family='monospace', 
            bbox=dict(boxstyle='round', facecolor='#F8F9F9', edgecolor='#BDC3C7'))
    
    # クラウド実行
    cloud_box = FancyBboxPatch((5.2, 3.5), 3.8, 1.2, 
                               boxstyle="round,pad=0.08", 
                               edgecolor='#34495E', facecolor='white', 
                               linewidth=1.5, alpha=0.9)
    ax.add_patch(cloud_box)
    ax.text(7.1, 4.35, 'Streamlit Community Cloud', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(7.1, 3.9, 'URL: fractal-analyzer-v2\n.streamlit.app\n自動デプロイ: GitHub push時', 
            ha='center', va='center', fontsize=7.5)
    
    # アーキテクチャ
    arch_box = FancyBboxPatch((0.5, 0.5), 9, 1.6, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#F39C12', facecolor='#FCF3CF', 
                              linewidth=2, alpha=0.9)
    ax.add_patch(arch_box)
    ax.text(5, 1.95, 'アーキテクチャ構成', 
            ha='center', va='top', fontsize=12, fontweight='bold', color='#D68910')
    
    arch_items = [
        {'name': 'Streamlit\nWeb UI', 'x': 1.8, 'color': '#E74C3C'},
        {'name': 'LightGBM\nAI Engine', 'x': 3.6, 'color': '#27AE60'},
        {'name': 'OpenCV\n画像処理', 'x': 5.4, 'color': '#3498DB'},
        {'name': 'NumPy/SciPy\n数値計算', 'x': 7.2, 'color': '#9B59B6'},
        {'name': 'CuPy (opt.)\nGPU加速', 'x': 9, 'color': '#16A085'}
    ]
    
    for item in arch_items:
        box = FancyBboxPatch((item['x']-0.5, 0.7), 1, 0.7, 
                             boxstyle="round,pad=0.05", 
                             edgecolor=item['color'], facecolor='white', 
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(item['x'], 1.05, item['name'], 
                ha='center', va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('diagram_5_dependencies.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ 外部依存関係図を保存しました: diagram_5_dependencies.png")
    plt.close()


def create_algorithm_diagram():
    """主要アルゴリズム図"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Box-Counting法
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 9), 9, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#3498DB', facecolor='#3498DB', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(title_box)
    ax.text(5, 9.4, 'Box-Counting法\n(フラクタル次元計算)', 
            ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    box_content = FancyBboxPatch((1, 1), 8, 7.5, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#2980B9', facecolor='#EBF5FB', 
                                 linewidth=2, alpha=0.9)
    ax.add_patch(box_content)
    
    sections = [
        ('入力:', 7.8, ['• img: グレースケール画像', '• size: 解析サイズ (256推奨)', '• box_sizes: [2,4,8,16,32,64]']),
        ('処理:', 6.3, ['1. 画像を2値化 (閾値127)', '2. 各ボックスサイズで格子分割', '3. ボックス内の占有率を計算', '4. log(N) vs log(1/r) の傾きを算出']),
        ('出力:', 4, ['• FD値 (Fractal Dimension): 2.0-3.0']),
        ('高速化技術:', 3, ['✓ GPU並列演算 (CuPy)', '✓ ベクトル化 (NumPyブロードキャスト)', '✓ バッチ処理 (複数スケール同時計算)']),
        ('性能:', 1.3, ['計算量: O(n²) → O(n²/p) ※p=並列度', '精度: ±0.01 (標準偏差)'])
    ]
    
    for section_title, y_start, items in sections:
        ax.text(1.5, y_start, section_title, 
                ha='left', va='top', fontsize=9, fontweight='bold', color='#2C3E50')
        for i, item in enumerate(items):
            ax.text(2, y_start - 0.35 - i*0.35, item, 
                    ha='left', va='center', fontsize=7.5)
    
    # 2. LightGBM予測モデル
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 9), 9, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#27AE60', facecolor='#27AE60', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(title_box)
    ax.text(5, 9.4, 'LightGBM予測モデル\n(AI補正エンジン)', 
            ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # 特徴量
    feature_box = FancyBboxPatch((1, 5.5), 8, 3, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#229954', facecolor='#D5F4E6', 
                                 linewidth=2, alpha=0.9)
    ax.add_patch(feature_box)
    ax.text(5, 8.3, '特徴量 (10次元)', 
            ha='center', va='top', fontsize=10, fontweight='bold', color='#27AE60')
    
    features = [
        '1. FD値 (Box-Counting)',
        '2. 平均輝度',
        '3. 標準偏差',
        '4. 歪度 (Skewness)',
        '5. 尖度 (Kurtosis)',
        '6. エッジ強度 (Sobel)',
        '7. テクスチャ (GLCM)',
        '8. 周波数成分 (FFT)',
        '9. 解像度 (ピクセル数)',
        '10. JPEG品質推定'
    ]
    
    for i, feature in enumerate(features):
        col = i // 5
        row = i % 5
        ax.text(2.2 + col*4, 7.8 - row*0.5, feature, 
                ha='left', va='center', fontsize=7.5)
    
    # モデル構成
    model_box = FancyBboxPatch((1, 1), 8, 4, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#229954', facecolor='#EAF2F8', 
                               linewidth=2, alpha=0.9)
    ax.add_patch(model_box)
    ax.text(5, 4.8, 'モデル構成', 
            ha='center', va='top', fontsize=10, fontweight='bold', color='#27AE60')
    
    model_info = [
        ('ハイパーパラメータ:', ['• n_estimators: 100-500', '• max_depth: 5-15', '• learning_rate: 0.01-0.1', '• num_leaves: 31', '• n_jobs: -1 (全コア使用)']),
        ('訓練データ:', ['• 入力: 低画質画像の特徴量', '• 出力: 対応する高画質FD値']),
        ('評価指標:', ['• MAE (Mean Absolute Error)', '• R² Score (決定係数)', '• Pearson相関係数'])
    ]
    
    y_pos = 4.3
    for section_title, items in model_info:
        ax.text(1.5, y_pos, section_title, 
                ha='left', va='top', fontsize=8, fontweight='bold', color='#2C3E50')
        for item in items:
            y_pos -= 0.35
            ax.text(2, y_pos, item, 
                    ha='left', va='center', fontsize=7)
        y_pos -= 0.2
    
    # 3. 画像品質判定
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 9), 9, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#F39C12', facecolor='#F39C12', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(title_box)
    ax.text(5, 9.4, '画像品質判定アルゴリズム\n(総合品質スコア)', 
            ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    quality_checks = [
        ('1. 解像度チェック', ['✓ 推奨: 2000万画素以上', '✓ 最低: 800万画素'], 7.5),
        ('2. シャープネス測定 (FFT)', ['✓ 高周波成分比率 > 0.3'], 6.3),
        ('3. ノイズレベル測定', ['✓ 標準偏差 / 平均 < 0.5'], 5.3),
        ('4. ブレ検出 (ラプラシアン)', ['✓ 分散値 > 100'], 4.3),
        ('5. JPEG品質推定', ['✓ 品質係数 > 80'], 3.3)
    ]
    
    for check_title, criteria, y_pos in quality_checks:
        check_box = FancyBboxPatch((1, y_pos-0.6), 8, 0.7, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='#D68910', facecolor='#FCF3CF', 
                                   linewidth=1.5, alpha=0.9)
        ax.add_patch(check_box)
        ax.text(1.5, y_pos-0.15, check_title, 
                ha='left', va='top', fontsize=8, fontweight='bold')
        for i, criterion in enumerate(criteria):
            ax.text(2, y_pos-0.45-i*0.2, criterion, 
                    ha='left', va='center', fontsize=7)
    
    # スコア計算
    score_box = FancyBboxPatch((1, 1.3), 8, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#E67E22', facecolor='#FDEBD0', 
                               linewidth=2, alpha=0.9)
    ax.add_patch(score_box)
    ax.text(5, 2.65, 'スコア計算・判定', 
            ha='center', va='top', fontsize=9, fontweight='bold', color='#D68910')
    ax.text(5, 2.1, 'quality_score = Σ(各項目スコア × 重み) / 100', 
            ha='center', va='center', fontsize=8, family='monospace')
    ax.text(5, 1.65, 'is_high_quality = (quality_score >= 70)', 
            ha='center', va='center', fontsize=8, family='monospace', 
            bbox=dict(boxstyle='round', facecolor='#F8F9F9', edgecolor='#BDC3C7'))
    
    # 4. 肌品質評価
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 9), 9, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#9B59B6', facecolor='#9B59B6', 
                               linewidth=3, alpha=0.8)
    ax.add_patch(title_box)
    ax.text(5, 9.4, '肌品質評価アルゴリズム\n(グレーディングシステム)', 
            ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # スコア変換
    formula_box = FancyBboxPatch((1, 7.5), 8, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#8E44AD', facecolor='#F4ECF7', 
                                 linewidth=2, alpha=0.9)
    ax.add_patch(formula_box)
    ax.text(5, 8.5, 'FD値 → スコア変換', 
            ha='center', va='top', fontsize=10, fontweight='bold', color='#9B59B6')
    ax.text(5, 8.05, 'score = (fd_value - 2.40) × 100', 
            ha='center', va='center', fontsize=9, family='monospace', 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#BDC3C7'))
    
    # グレード判定
    grades = [
        {'grade': 'S', 'fd': 'FD ≥ 2.60', 'score': '90-100点', 'desc': '非常に良好', 'color': '#27AE60', 'y': 6.5},
        {'grade': 'A', 'fd': 'FD ≥ 2.55', 'score': '80-89点', 'desc': '良好', 'color': '#3498DB', 'y': 5.5},
        {'grade': 'B', 'fd': 'FD ≥ 2.50', 'score': '70-79点', 'desc': '標準', 'color': '#F39C12', 'y': 4.5},
        {'grade': 'C', 'fd': 'FD ≥ 2.45', 'score': '60-69点', 'desc': 'やや問題あり', 'color': '#E67E22', 'y': 3.5},
        {'grade': 'D', 'fd': 'FD < 2.45', 'score': '60点未満', 'desc': '要改善', 'color': '#E74C3C', 'y': 2.5}
    ]
    
    ax.text(5, 7, 'グレード判定基準', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='#9B59B6')
    
    for grade_info in grades:
        grade_box = FancyBboxPatch((1, grade_info['y']-0.35), 8, 0.7, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor=grade_info['color'], 
                                   facecolor='white', 
                                   linewidth=2, alpha=0.9)
        ax.add_patch(grade_box)
        
        # グレード
        grade_circle = FancyBboxPatch((1.3, grade_info['y']-0.25), 0.5, 0.5, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor=grade_info['color'], 
                                      facecolor=grade_info['color'], 
                                      linewidth=2, alpha=0.8)
        ax.add_patch(grade_circle)
        ax.text(1.55, grade_info['y'], grade_info['grade'], 
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        
        # 情報
        ax.text(2.2, grade_info['y'], f"{grade_info['fd']}  |  {grade_info['score']}  |  {grade_info['desc']}", 
                ha='left', va='center', fontsize=8)
    
    # コメント生成
    comment_box = FancyBboxPatch((1, 0.5), 8, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#8E44AD', facecolor='#EBDEF0', 
                                 linewidth=2, alpha=0.9)
    ax.add_patch(comment_box)
    ax.text(5, 1.85, 'コメント生成機能', 
            ha='center', va='top', fontsize=9, fontweight='bold', color='#9B59B6')
    ax.text(5, 1.3, '• グレード別のアドバイス\n• 改善提案\n• 年齢層との比較', 
            ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('diagram_6_algorithms.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ 主要アルゴリズム図を保存しました: diagram_6_algorithms.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("システムアーキテクチャ図の生成を開始します...")
    print("=" * 60)
    
    create_module_composition_diagram()
    create_project_structure_diagram()
    create_detailed_module_diagram()
    create_dataflow_diagrams()
    create_dependencies_diagram()
    create_algorithm_diagram()
    
    print("=" * 60)
    print("すべての図の生成が完了しました！")
    print("=" * 60)
    print("\n生成されたファイル:")
    print("  1. diagram_1_module_composition.png - モジュール構成図")
    print("  2. diagram_2_project_structure.png - プロジェクト構成図")
    print("  3. diagram_3_detailed_modules.png - 詳細モジュール図")
    print("  4. diagram_4_dataflow.png - データフロー図")
    print("  5. diagram_5_dependencies.png - 外部依存関係図")
    print("  6. diagram_6_algorithms.png - 主要アルゴリズム図")
    print("=" * 60)
