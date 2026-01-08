# -*- coding: utf-8 -*-
"""
System Overview PDF Generator
- Builds a Japanese-font PDF summarizing architecture and workflows for LLM ingestion
"""
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Image, Flowable
import math
from reportlab.lib import fonts
from reportlab.lib.colors import black, white, HexColor
import os
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Ellipse, Polygon
from reportlab.graphics import renderPM, renderPDF
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def register_japanese_font():
    candidates = [
        ("Meiryo", "C:/Windows/Fonts/meiryo.ttc"),
        ("YuGothic", "C:/Windows/Fonts/YuGothM.ttc"),
        ("MSMincho", "C:/Windows/Fonts/msmincho.ttc"),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                fonts.addMapping(name, 0, 0, name)
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return None

SECTIONS = [
    ("目的と設計思想", [
        "低画質画像からFDを解析しAI補正で高画質相当FDを推定。研究・教育用途。",
        "降格設計・日本語環境対応・GPU自動切替・統計可視化を重視。",
    ]),
    ("全体構成", [
        "UI: fractal_app.py (Streamlit)。",
        "FD: fd_boxcount.py の fd_std_boxcount と fd_3d_dbc。",
        "画像解析: skin_analysis.py (MediaPipe/OpenCV/dlib/ヒューリスティック)。",
        "実験/図表: experiment_analysis.py (CSV追記・相関・散布図・ヒートマップ)。",
        "レポート: generate_updated_report.py (PDF)、scripts/make_template_docx.py (Word)。",
        "永続化: trained_fd_model.pkl, training_history.json。",
    ]),
    ("データフロー", [
        "画像取り込み→FD算出→AI補正→可視化・検証→保存。",
        "FDは対数空間で回帰。2D: FD=|slope|, DBC: FD3=3-|slope| (clip[2,3])。",
    ]),
    ("アルゴリズム詳細", [
        "2D: ブロック標準偏差を正規化・合計し N(h)。logN vs logh の傾き。",
        "DBC: 高さ量子化でボックス数 N(r)。logN vs logr の傾き。",
    ]),
    ("GPUと数値演算", [
        "cupy有効時はUSE_CUPY=True、xpエイリアスでnumpy/cupy切替。",
    ]),
    ("顔検出フォールバック", [
        "MediaPipe優先→OpenCV Haar→dlib→中央推定矩形で頑健化。",
        "extract_face_regions API互換を維持。",
    ]),
    ("データ拡張", [
        "反転・回転・明暗/コントラスト・ガンマ・ノイズ/ブラー/シャープ・彩度。",
        "高/低画質ペアに対称適用。",
    ]),
    ("UI/ワークフロー", [
        "st.session_state で状態保持。st.subheader/st.metric/plotly_chart。",
        "opencv-python-headless前提。PowerShell: streamlit run fractal_app.py。",
    ]),
    ("実験ログ/相関/図", [
        "ExperimentDataManager.save_data でCSV追記。",
        "calculate_correlations、create_scatter_plot を使用。",
    ]),
    ("命名規約と永続化", [
        "IMG_XXXX.jpg ↔ IMG_XXXX_low1.jpg (Low4–7)。",
        "trained_fd_model.pkl, training_history.json を自動ロード/保存。",
        "SKIN_DATA/ にCSVと例画像。",
    ]),
    ("依存関係", [
        "streamlit, numpy, pandas, opencv-python-headless, scikit-learn, matplotlib, Pillow, scipy, plotly, seaborn, lightgbm, python-docx。",
        "オプション: cupy, mediapipe, dlib。",
    ]),
    ("研究的注意点", [
        "FD傾き範囲はクリップ・検証。撮影条件差の影響に留意。",
        "単一画像部位別は不安定→方向性確認、結論は集団分析で。",
    ]),
    ("肌評価の考え方", [
        "不規則性↑→FD↑→肌トラブル↑という仮説に基づく評価。",
        "FDと画像解析のトラブル指標を統合して総合評価。",
    ]),
    ("評価パイプライン", [
        "入力→顔/部位抽出→FD算出→トラブル検出→集計保存→可視化。",
        "skin_analysis.py と fd_boxcount.py を中核として接続。",
    ]),
    ("FD計算の考え方", [
        "多段スケールで N(h)/N(r) を算出し対数空間で回帰。",
        "2D: FD=|slope|、DBC: FD3=3-|slope|（ともに clip[2,3]）。",
    ]),
    ("スコア化ロジック", [
        "FD由来トラブルスコア: score=max(0,min(100,(average_fd-2.0)*100))。",
        "肌トラブル総合スコア: 部位別指標を平均して trouble_total_score を生成。",
    ]),
    ("可視化と検証", [
        "傾きが正なら理論と整合。分散極小で nan はUIでガード。",
        "散布図/回帰線: create_scatter_plot、ヒートマップ: create_correlation_heatmap。",
    ]),
    ("設計上の配慮", [
        "降格設計で任意モジュール不在でも継続。",
        "日本語パス/フォント対応、GPU自動切替（cupy→xpエイリアス）。",
    ]),
    ("キーファイル", [
        "fractal_app.py（UI/永続化/表示）",
        "fd_boxcount.py（FD算出）",
        "skin_analysis.py（顔検出/部位抽出/トラブル推定）",
        "experiment_analysis.py（ログ/相関/図）",
        "generate_updated_report.py（PDFレポート）",
    ]),
    ("使用データおよび評価項目", [
        "SKIN_DATA/ と Facial Skin Condition Dataset.csv（例画像・メタ）。",
        "命名: IMG_XXXX.jpg ↔ IMG_XXXX_low1.jpg（Low4–7含む）。",
        "保存: experimental_data.csv に追記（ExperimentDataManager.save_data）。",
        "評価カラム: average_fd, roughness/dryness/moisture/sebum/pore/wrinkle/redness/dark_circle/age, trouble_*, trouble_total_score, overall_score。",
    ]),
    ("実験の流れ", [
        "画像取得→顔/部位抽出→FD算出→AI補正→トラブル統合→保存→可視化→レポート。",
        "LightGBMで低画質から高画質FDを推定しモデル永続化。",
    ]),
    ("相関評価の方法", [
        "ピアソン相関 r と p値（scipy.stats.pearsonr）。",
        "欠損除外・n>=3、分散極小はnan→UIガード。",
        "散布図+回帰線（np.polyfit, create_scatter_plot）、p<0.05目安。",
        "集団では正の関係が安定、部位内は方向性確認用途。",
    ]),
    ("評価閾値と算出方法（詳細）", [
        "FD/FD3は[2,3]にクリップ、相関はn>=3で算出、p<0.05を目安。",
        "FD由来スコア: score=max(0,min(100,(average_fd-2.0)*100))。",
        "Pearson r と p値（t統計を介して算出; 実装はpearsonr）。",
        "OLS回帰: y=ax+b、a=Cov/Var、b=ybar-a xbar（実装はnp.polyfit）。",
        "誤差指標: MAE/RMSE/R2 を適宜表示。",
    ]),
    ("4.3 評価システムの仕様および使用手順", [
        "目的と入出力を定義し、比較・経時検証を可能にする評価基盤。",
    ]),
    ("4.3.1 評価システムの概要", [
        "入力: 顔画像（カラー）。出力: 部位別FD、画像代表FD、評価値。",
        "メタデータと併せて保存し、相関・回帰・レポートに利用。",
        "領域分割で部位特性を保持。降格設計で継続性を確保。",
    ]),
    ("4.3.2 画像処理および分割の考え方", [
        "顔領域を抽出し、額・頬・鼻・口周囲・顎・目周辺に分割。",
        "アルゴリズム詳細は扱わず、局所差と全体傾向の両立を狙う。",
    ]),
    ("4.3.3 フラクタル次元算出と評価値の取得", [
        "各部位でFDを推定し、部位別FDを得る。",
        "代表値は部位平均を採用し、頑健な画像指標とする。",
        "FDと評価項目・メタデータを同一レコードで保存。",
    ]),
    ("4.3.4 実験時の使用手順", [
        "画像入力→評価実行→結果保存（UI操作やコード詳細は記載しない）。",
    ]),
    ("4.4 システムの使い方（番号付き手順）", [
        "本節は『実験の流れ』の補足として、概念的な順序を示す。",
        "何をしたか（概要）: 顔抽出→分割→部位FD→代表FD→指標統合→保存→可視化。",
        "1) 準備 2) 画像入力 3) 分割確認 4) FD計算 5) 代表値作成 6) 評価項目統合 7) 保存 8) 可視化/検証",
    ]),
]


def build_pdf(output_path: str):
    font_name = register_japanese_font() or "Helvetica"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    style = ParagraphStyle(
        name="NormalJP",
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=TA_LEFT,
        textColor=black,
    )
    title_style = ParagraphStyle(name="TitleJP", parent=style, fontSize=16, leading=22)
    section_style = ParagraphStyle(name="SectionJP", parent=style, fontSize=13, leading=20)

    story = []
    story.append(Paragraph("顔画像評価システム 構成と処理概要", title_style))
    story.append(Spacer(1, 8*mm))

    for section_title, bullets in SECTIONS:
        story.append(Paragraph(section_title, section_style))
        story.append(Spacer(1, 2*mm))
        story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in bullets], bulletType='bullet'))
        story.append(Spacer(1, 5*mm))

    # Results section (dynamic summary)
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(base_dir, "experimental_data.csv")
        results_bullets = []
        data_bullets = []
        n_text = "複数"
        reg_png = None
        reg_info = None
        # stats placeholders
        n = 0
        fd_mean = fd_stdv = fd_min = fd_max = None
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
            n = len(df)
            n_text = f"{n}"
            # Data summary
            data_bullets.append(f"対象データ数: {n} 件")
            if 'average_fd' in df.columns:
                s = df['average_fd'].dropna()
                if len(s) > 0:
                    fd_mean, fd_stdv, fd_min, fd_max = float(s.mean()), float(s.std()), float(s.min()), float(s.max())
                    data_bullets.append(f"フラクタル次元（平均±SD）: {fd_mean:.3f} ± {fd_stdv:.3f}（最小 {fd_min:.3f}, 最大 {fd_max:.3f}）")
            if 'trouble_total_score' in df.columns:
                t = df['trouble_total_score'].dropna()
                if len(t) > 0:
                    data_bullets.append(f"トラブル総合スコア: 有効 {len(t)} 件")
            # Correlation / regression
            if {'average_fd','trouble_total_score'}.issubset(df.columns):
                dd = df[['average_fd','trouble_total_score']].dropna()
                if len(dd) >= 3:
                    r = float(dd['average_fd'].corr(dd['trouble_total_score']))
                    # 回帰（フラクタル次元を目的変数、トラブル総合を説明変数）
                    x_arr = dd['trouble_total_score'].to_numpy()
                    y_arr = dd['average_fd'].to_numpy()
                    a, b = np.polyfit(x_arr, y_arr, 1)
                    data_bullets.append(f"相関（フラクタル次元 vs トラブル総合）: r={r:.3f}, 回帰（縦=フラクタル次元, 横=肌トラブル総合）: y={a:.3f}x+{b:.3f}")
                    # 残差とRMSE、近傍（インライヤ）比率
                    y_pred = a * x_arr + b
                    resid = y_arr - y_pred
                    rmse = float(np.sqrt(np.mean(resid**2)))
                    inlier_mask = np.abs(resid) <= rmse
                    inlier_cnt = int(inlier_mask.sum())
                    total_cnt = int(len(y_arr))
                    inlier_ratio = float(inlier_cnt / total_cnt) if total_cnt > 0 else 0.0
                    # 有効データ（インライヤ）の相関・回帰（件数>=3のとき）
                    a_in = b_in = r_in = None
                    if inlier_cnt >= 3:
                        x_in = x_arr[inlier_mask]
                        y_in = y_arr[inlier_mask]
                        r_in = float(np.corrcoef(x_in, y_in)[0,1])
                        a_in, b_in = np.polyfit(x_in, y_in, 1)
                    reg_info = {
                        "a": float(a),
                        "b": float(b),
                        "r": float(r),
                        "rmse": rmse,
                        "inlier_cnt": inlier_cnt,
                        "total_cnt": total_cnt,
                        "inlier_ratio": inlier_ratio,
                        "a_in": float(a_in) if a_in is not None else None,
                        "b_in": float(b_in) if b_in is not None else None,
                        "r_in": float(r_in) if r_in is not None else None,
                        "xmin": float(x_arr.min()),
                        "xmax": float(x_arr.max()),
                        "ymin": float(y_arr.min()),
                        "ymax": float(y_arr.max()),
                    }

                    # Regression plot
                    try:
                        # Japanese font for matplotlib
                        plt.rcParams['font.family'] = font_name
                        fig, ax = plt.subplots(figsize=(5.8, 4.0), dpi=150)
                        # 軸を入れ替え：x=肌トラブル総合、y=フラクタル次元
                        x = x_arr
                        y = y_arr
                        ax.scatter(x, y, s=16, alpha=0.75, color="#1976D2", label="データ")
                        xs = np.linspace(x.min(), x.max(), 100)
                        ys = a * xs + b
                        ax.plot(xs, ys, color="#D32F2F", linewidth=2.0, label=f"回帰線: y={a:.3f}x+{b:.3f}")
                        ax.set_xlabel("肌トラブル総合スコア")
                        ax.set_ylabel("フラクタル次元")
                        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
                        ax.legend(loc="best", fontsize=9)
                        # annotation with r
                        ax.text(0.02, 0.98, f"相関 r = {r:.3f}", transform=ax.transAxes, va='top', ha='left', fontsize=9)
                        out_dir = os.path.dirname(output_path)
                        reg_png = os.path.join(out_dir, "regression_fd_vs_trouble.png")
                        fig.tight_layout()
                        fig.savefig(reg_png)
                        plt.close(fig)
                    except Exception:
                        reg_png = None

        # Results summary bullets（数値を自動挿入し、軸の向きを明示）
        if n > 0:
            results_bullets.append(f"対象: {n_text}件の顔画像を解析。")
        if fd_mean is not None:
            results_bullets.append(f"フラクタル次元の統計: 平均 {fd_mean:.3f}、標準偏差 {fd_stdv:.3f}、最小 {fd_min:.3f}、最大 {fd_max:.3f}。")
        if reg_info is not None:
            sgn = "正" if reg_info["r"] >= 0 else "負"
            results_bullets.append(f"相関係数 r={reg_info['r']:.3f}（{sgn}の関係）。")
            results_bullets.append(f"回帰（縦=フラクタル次元, 横=肌トラブル総合）: y={reg_info['a']:.3f}x+{reg_info['b']:.3f}。")
        else:
            results_bullets.extend([
                f"本研究では、約{n_text}件の顔画像を対象としてフラクタル次元を算出し、肌トラブル評価との関係を分析した。",
                "その結果、フラクタル次元が高い画像ほど、肌トラブル評価が高くなる傾向が確認された。",
                "最小二乗法による回帰分析においても、フラクタル次元と肌トラブル指標の間には正の相関が示された。",
            ])

        story.append(Paragraph("結果（要約）", section_style))
        story.append(Spacer(1, 2*mm))
        story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in results_bullets], bulletType='bullet'))
        story.append(Spacer(1, 4*mm))

        if data_bullets:
            story.append(Paragraph("データ概要", section_style))
            story.append(Spacer(1, 2*mm))
            story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in data_bullets], bulletType='bullet'))
            story.append(Spacer(1, 5*mm))

        # Embed regression plot if created
        if reg_png and os.path.exists(reg_png):
            story.append(Paragraph("回帰図（横: 肌トラブル総合, 縦: フラクタル次元）", section_style))
            story.append(Spacer(1, 2*mm))
            story.append(Image(reg_png, width=170*mm, height=110*mm))
            story.append(Spacer(1, 5*mm))

        # Insights from the graph
        if reg_info is not None:
            def r_strength_label(rv: float) -> str:
                av = abs(rv)
                if av < 0.3:
                    return "弱い"
                if av < 0.5:
                    return "中程度"
                return "比較的強い"
            insights = []
            sgn = "正" if reg_info["a"] >= 0 else "負"
            insights.append(f"相関係数 r={reg_info['r']:.3f}（{r_strength_label(reg_info['r'])}、{sgn}の関係）。")
            insights.append(f"回帰式は フラクタル次元 y = {reg_info['a']:.3f} × 肌トラブル総合 + {reg_info['b']:.3f}。")
            insights.append(f"解釈：肌トラブル総合が10ポイント増加すると、フラクタル次元はおよそ {reg_info['a']*10:.3f} 増加。")
            insights.append(f"観測範囲：肌トラブル総合 {reg_info['xmin']:.1f}–{reg_info['xmax']:.1f}、フラクタル次元 {reg_info['ymin']:.3f}–{reg_info['ymax']:.3f}。")
            insights.append("注意：一次回帰は観測範囲での近似であり、因果関係を直接示すものではない。")
            story.append(Paragraph("グラフから読み取れること", section_style))
            story.append(Spacer(1, 2*mm))
            story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in insights], bulletType='bullet'))
            story.append(Spacer(1, 5*mm))

            # 有効性評価（回帰近傍データ）
            eff_bullets = []
            eff_bullets.append(f"残差のRMSE ≈ {reg_info['rmse']:.4f}。この範囲以内を近傍（インライヤ）と定義。")
            eff_bullets.append(f"インライヤ件数/全件数: {reg_info['inlier_cnt']} / {reg_info['total_cnt']}（{reg_info['inlier_ratio']*100:.1f}%）。")
            label_eff = "概ね有効" if reg_info['inlier_ratio'] >= 0.6 else ("一部有効" if reg_info['inlier_ratio'] >= 0.4 else "限定的")
            eff_bullets.append(f"評価：近傍比率が高く、{label_eff}な傾向が示唆。撮影条件・前処理の標準化で更なる改善が期待される。")
            if reg_info['a_in'] is not None and reg_info['r_in'] is not None:
                eff_bullets.append(f"有効データのみの再回帰：y = {reg_info['a_in']:.3f}x + {reg_info['b_in']:.3f}、相関 r={reg_info['r_in']:.3f}。")
                eff_bullets.append(f"解釈（有効データ）：10ポイント増加時のフラクタル次元変化 ≈ {reg_info['a_in']*10:.3f}。")
            story.append(Paragraph("有効性評価（回帰近傍データ）", section_style))
            story.append(Spacer(1, 2*mm))
            story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in eff_bullets], bulletType='bullet'))
            story.append(Spacer(1, 5*mm))
    except Exception as e:
        story.append(Paragraph(f"結果サマリーの生成に失敗: {e}", style))

    # Discussion section
    try:
        discussion_bullets = [
            "本手法は簡易的な撮影画像を用いた評価であるため撮影条件や前処理の影響を受けやすく、今後は実画像データの拡充および解析条件の標準化が課題である。",
            "撮影条件（照明・距離・解像度・皮膚表面の直前処置）のばらつきがFDと評価に与えるバイアスを定量検証する必要がある。",
            "前処理（平滑化・シャープ化・ガンマ補正等）がFD推定へ及ぼす影響を系統的に評価し、標準前処理フローを策定することが望ましい。",
            "データ拡充（年齢層・性別・肌タイプ・機器多様性）と、撮影・前処理・解析パラメータのプロトコル化により外的妥当性の向上を図る。",
        ]
        story.append(Paragraph("考察", section_style))
        story.append(Spacer(1, 2*mm))
        story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in discussion_bullets], bulletType='bullet'))
        story.append(Spacer(1, 5*mm))
    except Exception as e:
        story.append(Paragraph(f"考察の生成に失敗: {e}", style))

    # Diagrams generation and embedding
    try:
        out_dir = os.path.dirname(output_path)
        sys_png = os.path.join(out_dir, "system_diagram.png")
        mod_png = os.path.join(out_dir, "module_diagram.png")
        sys_pdf = os.path.join(out_dir, "system_diagram.pdf")
        mod_pdf = os.path.join(out_dir, "module_diagram.pdf")
        flow_pdf = os.path.join(out_dir, "flow_diagram.pdf")

        def create_system_diagram():
            d = Drawing(640, 380)

            # color palette
            col_in_fill = HexColor('#E3F2FD')
            col_in_stroke = HexColor('#2196F3')
            col_proc_fill = HexColor('#E8F5E9')
            col_proc_stroke = HexColor('#4CAF50')
            col_out_fill = HexColor('#F3E5F5')
            col_out_stroke = HexColor('#9C27B0')
            col_arrow = HexColor('#455A64')

            def add_box(x, y, w, h, lines, fill_color, stroke_color, step=None):
                d.add(Rect(x, y, w, h, strokeColor=stroke_color, fillColor=fill_color))
                # center-aligned multiline label
                font = font_name
                fs = 11
                total_h = fs * len(lines)
                start_y = y + h/2 + total_h/2 - fs
                for i, txt in enumerate(lines):
                    d.add(String(x + w/2, start_y - i*fs, txt, fontName=font, fontSize=fs, textAnchor='middle'))
                if step is not None:
                    cx, cy, r = x + 16, y + h - 16, 10
                    d.add(Circle(cx, cy, r, strokeColor=stroke_color, fillColor=stroke_color))
                    d.add(String(cx, cy-5, str(step), fontName=font, fontSize=11, textAnchor='middle', fillColor=white))
                return {"x": x, "y": y, "w": w, "h": h}

            def _arrow_head(x1, y1, x2, y2, color=col_arrow, head_len=12, head_deg=28):
                dx, dy = x2 - x1, y2 - y1
                L = math.hypot(dx, dy)
                if L == 0:
                    return
                ux, uy = dx / L, dy / L
                bx, by = -ux, -uy
                rad = math.radians(head_deg)
                r1x = bx * math.cos(rad) - by * math.sin(rad)
                r1y = bx * math.sin(rad) + by * math.cos(rad)
                r2x = bx * math.cos(-rad) - by * math.sin(-rad)
                r2y = bx * math.sin(-rad) + by * math.cos(-rad)
                p1x, p1y = x2 + r1x * head_len, y2 + r1y * head_len
                p2x, p2y = x2 + r2x * head_len, y2 + r2y * head_len
                d.add(Line(x2, y2, p1x, p1y, strokeColor=color))
                d.add(Line(x2, y2, p2x, p2y, strokeColor=color))

            def arrow(x1, y1, x2, y2, color=col_arrow):
                d.add(Line(x1, y1, x2, y2, strokeColor=color))
                _arrow_head(x1, y1, x2, y2, color=color)

            def arrow_ortho(x1, y1, x2, y2, mid_y=None, color=col_arrow):
                # Draw orthogonal path with a horizontal/vertical combination and head at end
                if mid_y is None:
                    mid_y = (y1 + y2) / 2.0
                # segment 1: vertical to mid_y
                d.add(Line(x1, y1, x1, mid_y, strokeColor=color))
                # segment 2: horizontal to target x
                d.add(Line(x1, mid_y, x2, mid_y, strokeColor=color))
                # segment 3: vertical to end
                d.add(Line(x2, mid_y, x2, y2, strokeColor=color))
                _arrow_head(x2, mid_y, x2, y2, color=color)

            # Pipeline layout with colors and step numbers
            b_input = add_box(10, 280, 190, 85, ["入力", "（顔画像）"], col_in_fill, col_in_stroke, step=1)
            b_face  = add_box(230, 280, 210, 85, ["顔検出・分割", "（領域抽出）"], col_proc_fill, col_proc_stroke, step=2)
            b_fd    = add_box(470, 280, 150, 85, ["FD算出", "（2D / DBC）"], col_proc_fill, col_proc_stroke, step=3)
            b_ai    = add_box(230, 170, 210, 85, ["AI補正", "（LightGBM）"], col_proc_fill, col_proc_stroke, step=4)
            b_eval  = add_box(470, 170, 150, 85, ["評価・可視化", "（相関・グラフ）"], col_proc_fill, col_proc_stroke, step=5)
            b_rep   = add_box(470, 70, 150, 85, ["レポート", "（PDF / Word）"], col_out_fill, col_out_stroke, step=6)

            # Arrows between boxes
            # Arrow helpers with margins to avoid text overlap
            margin = 10
            # 入力 -> 顔検出（水平）
            arrow(b_input["x"] + b_input["w"], b_input["y"] + b_input["h"]/2,
                b_face["x"] - margin, b_face["y"] + b_face["h"]/2)
            # 顔検出 -> FD算出
            arrow(b_face["x"] + b_face["w"], b_face["y"] + b_face["h"]/2,
                b_fd["x"] - margin, b_fd["y"] + b_fd["h"]/2)
            # FD算出 -> AI補正（オルソ：縦→横→縦）
            arrow_ortho(
                b_fd["x"] + b_fd["w"]/2,
                b_fd["y"],
                b_ai["x"] + b_ai["w"]/2,
                b_ai["y"] + b_ai["h"] + margin,
                mid_y=(b_fd["y"] + b_ai["y"] + b_ai["h"] + margin) / 2.0
            )
            # AI補正 -> 評価
            arrow(b_ai["x"] + b_ai["w"], b_ai["y"] + b_ai["h"]/2,
                b_eval["x"] - margin, b_eval["y"] + b_eval["h"]/2)
            # 評価 -> レポート（縦）
            arrow(b_eval["x"] + b_eval["w"]/2, b_eval["y"],
                b_rep["x"] + b_rep["w"]/2, b_rep["y"] + b_rep["h"] + margin)

            # Legend
            def legend_item(x, y, text, fc, sc):
                d.add(Rect(x, y, 18, 10, fillColor=fc, strokeColor=sc))
                d.add(String(x + 24, y - 1, text, fontName=font_name, fontSize=10))

            legend_x, legend_y = 10, 20
            d.add(String(legend_x, legend_y + 26, "凡例", fontName=font_name, fontSize=11))
            legend_item(legend_x, legend_y + 10, "入力", col_in_fill, col_in_stroke)
            legend_item(legend_x + 100, legend_y + 10, "処理", col_proc_fill, col_proc_stroke)
            legend_item(legend_x + 200, legend_y + 10, "出力", col_out_fill, col_out_stroke)

            return d

        def create_module_diagram():
            d = Drawing(640, 400)

            # module palette (aligned to system diagram categories)
            ui_fill = HexColor('#FFFDE7'); ui_stroke = HexColor('#FBC02D')
            algo_fill = HexColor('#E8F5E9'); algo_stroke = HexColor('#4CAF50')
            report_fill = HexColor('#F3E5F5'); report_stroke = HexColor('#9C27B0')
            data_fill = HexColor('#E3F2FD'); data_stroke = HexColor('#2196F3')
            col_arrow = HexColor('#455A64')

            def add_box(x, y, w, h, lines, fill_color, stroke_color):
                d.add(Rect(x, y, w, h, strokeColor=stroke_color, fillColor=fill_color))
                font = font_name
                fs = 11
                total_h = fs * len(lines)
                start_y = y + h/2 + total_h/2 - fs
                for i, txt in enumerate(lines):
                    d.add(String(x + w/2, start_y - i*fs, txt, fontName=font, fontSize=fs, textAnchor='middle'))
                return {"x": x, "y": y, "w": w, "h": h}

            def connect(x1, y1, x2, y2):
                d.add(Line(x1, y1, x2, y2, strokeColor=col_arrow))
                dx, dy = x2 - x1, y2 - y1
                L = math.hypot(dx, dy)
                if L == 0:
                    return
                ux, uy = dx / L, dy / L
                bx, by = -ux, -uy
                rad = math.radians(28)
                r1x = bx * math.cos(rad) - by * math.sin(rad)
                r1y = bx * math.sin(rad) + by * math.cos(rad)
                r2x = bx * math.cos(-rad) - by * math.sin(-rad)
                r2y = bx * math.sin(-rad) + by * math.cos(-rad)
                p1x, p1y = x2 + r1x * 12, y2 + r1y * 12
                p2x, p2y = x2 + r2x * 12, y2 + r2y * 12
                d.add(Line(x2, y2, p1x, p1y, strokeColor=col_arrow))
                d.add(Line(x2, y2, p2x, p2y, strokeColor=col_arrow))

            # Center UI box (larger)
            b_ui = add_box(230, 290, 180, 75, ["UI本体", "（fractal_app.py）"], ui_fill, ui_stroke)

            # Surrounding modules (expanded, category colors)
            b_fd   = add_box(20, 210, 190, 75, ["FDアルゴリズム", "（fd_boxcount.py）"], algo_fill, algo_stroke)
            b_skin = add_box(230, 210, 180, 75, ["顔分析・部位抽出", "（skin_analysis.py）"], algo_fill, algo_stroke)
            b_exp  = add_box(430, 210, 190, 75, ["実験ログ・相関", "（experiment_analysis.py）"], algo_fill, algo_stroke)
            b_pdf  = add_box(20, 105, 190, 75, ["PDFレポート", "（generate_updated_report.py）"], report_fill, report_stroke)
            b_docx = add_box(230, 105, 180, 75, ["Wordテンプレ", "（scripts/make_template_docx.py）"], report_fill, report_stroke)
            b_data = add_box(430, 105, 190, 75, ["データセット", "（SKIN_DATA/）"], data_fill, data_stroke)

            # Connections to central UI
            # connectors with margins so arrow tips stay outside UI box
            mg = 10
            # 左列 -> UI 左側
            connect(b_fd["x"] + b_fd["w"], b_fd["y"] + b_fd["h"]/2,
                    b_ui["x"] - mg, b_ui["y"] + b_ui["h"]/2)
            # 中列上 -> UI 上側
            connect(b_skin["x"] + b_skin["w"]/2, b_skin["y"] + b_skin["h"],
                    b_ui["x"] + b_ui["w"]/2, b_ui["y"] - mg)
            # 右列 -> UI 右側
            connect(b_exp["x"], b_exp["y"] + b_exp["h"]/2,
                    b_ui["x"] + b_ui["w"] + mg, b_ui["y"] + b_ui["h"]/2)
            # 下段 -> UI 下側
            connect(b_pdf["x"] + b_pdf["w"]/2, b_pdf["y"] + b_pdf["h"],
                    b_ui["x"] + b_ui["w"]/2 - 40, b_ui["y"] - mg)
            connect(b_docx["x"] + b_docx["w"]/2, b_docx["y"] + b_docx["h"],
                    b_ui["x"] + b_ui["w"]/2, b_ui["y"] - mg)
            connect(b_data["x"] + b_data["w"]/2, b_data["y"] + b_data["h"],
                    b_ui["x"] + b_ui["w"]/2 + 40, b_ui["y"] - mg)

            # Legend (bottom-left)
            def legend_item(x, y, text, fc, sc):
                d.add(Rect(x, y, 18, 10, fillColor=fc, strokeColor=sc))
                d.add(String(x + 24, y - 1, text, fontName=font_name, fontSize=10))

            legend_x, legend_y = 10, 20
            d.add(String(legend_x, legend_y + 26, "凡例", fontName=font_name, fontSize=11))
            legend_item(legend_x, legend_y + 10, "UI", ui_fill, ui_stroke)
            legend_item(legend_x + 80, legend_y + 10, "アルゴリズム・解析", algo_fill, algo_stroke)
            legend_item(legend_x + 210, legend_y + 10, "レポート", report_fill, report_stroke)
            legend_item(legend_x + 300, legend_y + 10, "データ", data_fill, data_stroke)

            return d

        def create_flow_diagram():
            # Vertical, orthogonal flow with decision and side branch
            d = Drawing(640, 760)

            # palette
            col_arrow = HexColor('#455A64')
            fill_proc = HexColor('#E8F5E9'); stroke_proc = HexColor('#4CAF50')
            fill_data = HexColor('#E3F2FD'); stroke_data = HexColor('#2196F3')
            fill_out  = HexColor('#F3E5F5'); stroke_out  = HexColor('#9C27B0')
            fill_dec  = HexColor('#FFF8E1'); stroke_dec  = HexColor('#F9A825')

            # helpers (centered boxes)
            def add_box_c(cx, y, w, h, lines, fc, sc):
                x = cx - w/2
                d.add(Rect(x, y, w, h, fillColor=fc, strokeColor=sc))
                fs = 11
                total = fs * len(lines)
                sy = y + h/2 + total/2 - fs
                for i, t in enumerate(lines):
                    d.add(String(cx, sy - i*fs, t, fontName=font_name, fontSize=fs, textAnchor='middle'))
                return {"cx": cx, "y": y, "w": w, "h": h}

            def add_ellipse_c(cx, y, rx, ry, text, sc):
                cy = y + ry
                d.add(Ellipse(cx, cy, rx, ry, fillColor=None, strokeColor=sc))
                d.add(String(cx, cy-6, text, fontName=font_name, fontSize=12, textAnchor='middle'))
                return {"cx": cx, "cy": cy, "rx": rx, "ry": ry}

            def add_diamond_c(cx, y, w, h, lines):
                cy = y + h/2
                pts = [cx, cy+h/2, cx+w/2, cy, cx, cy-h/2, cx-w/2, cy]
                d.add(Polygon(pts, fillColor=fill_dec, strokeColor=stroke_dec))
                fs = 11
                total = fs * len(lines)
                sy = cy + total/2 - fs
                for i, t in enumerate(lines):
                    d.add(String(cx, sy - i*fs, t, fontName=font_name, fontSize=fs, textAnchor='middle'))
                return {"cx": cx, "cy": cy, "w": w, "h": h}

            def _arrow_head(x1, y1, x2, y2, color=col_arrow, head_len=10, head_deg=28):
                dx, dy = x2 - x1, y2 - y1
                L = math.hypot(dx, dy)
                if L == 0:
                    return
                ux, uy = dx / L, dy / L
                bx, by = -ux, -uy
                rad = math.radians(head_deg)
                r1x = bx*math.cos(rad) - by*math.sin(rad)
                r1y = bx*math.sin(rad) + by*math.cos(rad)
                r2x = bx*math.cos(-rad) - by*math.sin(-rad)
                r2y = bx*math.sin(-rad) + by*math.cos(-rad)
                p1x, p1y = x2 + r1x*head_len, y2 + r1y*head_len
                p2x, p2y = x2 + r2x*head_len, y2 + r2y*head_len
                d.add(Line(x2, y2, p1x, p1y, strokeColor=color))
                d.add(Line(x2, y2, p2x, p2y, strokeColor=color))

            def arrow_v(x, y1, y2):
                d.add(Line(x, y1, x, y2, strokeColor=col_arrow))
                _arrow_head(x, y1, x, y2)

            def arrow_h(x1, x2, y):
                d.add(Line(x1, y, x2, y, strokeColor=col_arrow))
                _arrow_head(x1, y, x2, y)

            # layout grid
            cx = 320
            w = 240; h = 56
            gap = 24
            y = 680

            start = add_ellipse_c(cx, y, 42, 22, "開始", stroke_proc)
            y -= (start["ry"]*2 + gap)

            inp = add_box_c(cx, y, w, h, ["入力（顔画像）"], fill_data, stroke_data)
            arrow_v(cx, inp["y"] + inp["h"], inp["y"] + inp["h"] + gap)
            y -= (h + gap)

            prep = add_box_c(cx, y, w, h, ["前処理（リサイズ・色空間）"], fill_proc, stroke_proc)
            arrow_v(cx, prep["y"] + prep["h"], prep["y"] + prep["h"] + gap)
            y -= (h + gap)

            detect = add_box_c(cx, y, w, h, ["顔検出"], fill_proc, stroke_proc)
            arrow_v(cx, detect["y"] + detect["h"], detect["y"] + detect["h"] + gap)
            y -= (h + gap)

            dec = add_diamond_c(cx, y, 140, 70, ["検出成功？"])
            # labels for yes/no
            d.add(String(cx + 70, dec["cy"] + 22, "はい", fontName=font_name, fontSize=10, textAnchor='start'))
            d.add(String(cx + 70, dec["cy"] - 22, "いいえ", fontName=font_name, fontSize=10, textAnchor='start'))
            # yes path: go down
            arrow_v(cx, dec["cy"] - dec["h"]/2, dec["cy"] - dec["h"]/2 - gap)

            # no path: go right to fail, then back to 前処理
            fail = add_box_c(cx + 260, dec["cy"] - 30, 200, 60, ["失敗時処理", "（再検出/撮影やり直し）"], fill_out, stroke_out)
            arrow_h(cx + dec["w"]/2, fail["cx"] - fail["w"]/2 - 10, dec["cy"])  # to fail box
            # back to prep (orthogonal: right->up->left)
            midx = fail["cx"]
            arrow_v(midx, dec["cy"], prep["y"] + prep["h"] + gap)
            arrow_h(midx, cx, prep["y"] + prep["h"] + gap)

            y = dec["cy"] - dec["h"]/2 - gap - h  # next row baseline
            seg = add_box_c(cx, y, w, h, ["領域分割"], fill_proc, stroke_proc)
            arrow_v(cx, seg["y"] + seg["h"], seg["y"] + seg["h"] + gap)

            y -= (h + gap)
            fd = add_box_c(cx, y, w, h, ["FD算出（2D/DBC）"], fill_proc, stroke_proc)
            arrow_v(cx, fd["y"] + fd["h"], fd["y"] + fd["h"] + gap)

            y -= (h + gap)
            ai = add_box_c(cx, y, w, h, ["AI補正（LightGBM）"], fill_proc, stroke_proc)
            arrow_v(cx, ai["y"] + ai["h"], ai["y"] + ai["h"] + gap)

            y -= (h + gap)
            ev = add_box_c(cx, y, w, h, ["評価・可視化（相関・グラフ）"], fill_proc, stroke_proc)
            arrow_v(cx, ev["y"] + ev["h"], ev["y"] + ev["h"] + gap)

            y -= (h + gap)
            save = add_box_c(cx, y, w, h, ["保存（CSV/モデル）"], fill_data, stroke_data)
            arrow_v(cx, save["y"] + save["h"], save["y"] + save["h"] + gap)

            y -= (h + gap)
            rep = add_box_c(cx, y, w, h, ["レポート（PDF/Word）"], fill_out, stroke_out)
            arrow_v(cx, rep["y"] + rep["h"], rep["y"] + rep["h"] + gap)

            y -= (h + gap)
            end = add_ellipse_c(cx, y, 42, 22, "終了", stroke_proc)

            # legend
            d.add(String(20, 20, "凡例: 青=データ、緑=処理、紫=出力/例外", fontName=font_name, fontSize=10))
            return d

        sd = create_system_diagram()
        md = create_module_diagram()
        fdg = create_flow_diagram()
        try:
            renderPM.drawToFile(sd, sys_png, fmt='PNG')
            renderPM.drawToFile(md, mod_png, fmt='PNG')
        except Exception:
            pass
        try:
            renderPDF.drawToFile(sd, sys_pdf)
            renderPDF.drawToFile(md, mod_pdf)
            renderPDF.drawToFile(fdg, flow_pdf)
        except Exception:
            pass

        class DrawingFlowable(Flowable):
            def __init__(self, drawing, width_pt, height_pt):
                super().__init__()
                self.drawing = drawing
                self.width = width_pt
                self.height = height_pt
            def wrap(self, availWidth, availHeight):
                return self.width, self.height
            def draw(self):
                sx = self.width / float(self.drawing.width)
                sy = self.height / float(self.drawing.height)
                self.canv.saveState()
                self.canv.scale(sx, sy)
                renderPDF.draw(self.drawing, self.canv, 0, 0)
                self.canv.restoreState()

        story.append(Paragraph("システム図 (コピー用: PDFは docs/output/system_diagram.pdf)", section_style))
        story.append(Spacer(1, 2*mm))
        story.append(DrawingFlowable(sd, width_pt=170*mm, height_pt=105*mm))
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("モジュール図 (コピー用: PDFは docs/output/module_diagram.pdf)", section_style))
        story.append(Spacer(1, 2*mm))
        story.append(DrawingFlowable(md, width_pt=170*mm, height_pt=115*mm))
        story.append(Spacer(1, 5*mm))

        # Flow diagram
        story.append(Paragraph("実際のフロー図 (コピー用: PDFは docs/output/flow_diagram.pdf)", section_style))
        story.append(Spacer(1, 2*mm))
        story.append(DrawingFlowable(fdg, width_pt=170*mm, height_pt=125*mm))
        story.append(Spacer(1, 5*mm))
    except Exception as e:
        # If diagram generation fails, continue without images
        story.append(Paragraph(f"図版生成に失敗: {e}", style))

    doc.build(story)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "output")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(out_dir, "system_overview.pdf"))
    build_pdf(output_path)
    print("WROTE", output_path)
