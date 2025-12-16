# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib import fonts
from reportlab.lib.colors import black
import os

# Windows日本語フォント設定（Meiryoがあれば使用）
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


def build_pdf(output_path):
    font_name = register_japanese_font() or "Helvetica"
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)

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

    # タイトル
    story.append(Paragraph("フラクタル次元による肌状態評価の研究（改訂版 v5）", title_style))
    story.append(Spacer(1, 8*mm))

    # 追加知見（要旨）
    story.append(Paragraph("要旨", section_style))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "本研究では、低画質画像からのフラクタル次元（FD）推定および画像解析による肌トラブル検出を統合し、FDと肌状態の関係性を検証した。"
        "被験者間・時点間を含む集計では、FDと肌トラブル総合スコアの間に正の関係が一貫して観察され、理論（表面の不規則性↑→FD↑→肌トラブル↑）と整合した。"
        "一方、単一画像の部位内比較では、総合スコアの平均化や部位特異性により、弱い負傾向が観察される場合があり、集団傾向との差異が明確化された。",
        style
    ))
    story.append(Spacer(1, 5*mm))

    # 新たに分かったこと（箇条書き）
    story.append(Paragraph("新たに分かったこと", section_style))
    story.append(Spacer(1, 2*mm))
    bullets = [
        "集団レベルの分析では、FDと肌トラブル総合スコアに正の相関が得られやすく、データ数の増加に伴い統計的有意性が確保される。",
        "単体画像（部位別）の回帰では、総合スコア化に伴う相殺効果や部位特異性により、負傾向が生じることがある。これは集団傾向と矛盾しない。",
        "毛穴・シワなどテクスチャ関連指標はFDと整合的に正の関係を示しやすい一方、テカリ等は撮影条件の影響を受け、関係のばらつきが大きい。",
        "撮影条件（照明・露出・フォーカス）と前処理（グレースケール・コントラスト調整）がFD・トラブル検出に影響するため、標準化が重要。",
        "OpenCV顔検出のロバスト化（パラメータ自動調整・中央推定フォールバック）により、実運用での解析継続性が向上した。",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in bullets], bulletType='bullet'))
    story.append(Spacer(1, 5*mm))

    # 読み取り指針（最小二乗法のグラフ）
    story.append(Paragraph("最小二乗法グラフの読み取り指針", section_style))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "回帰直線 y=ax+b の傾き a が正なら理論と整合、負なら部位特異性や総合化の影響を考慮する。"
        "点群が直線に沿うほど関係が強い。単体画像の部位別は n が小さく不安定なため、方向性の確認に留め、結論は集団分析で行う。",
        style
    ))
    story.append(Spacer(1, 3*mm))

    # 提案（今後の拡張）
    story.append(Paragraph("今後の拡張提案", section_style))
    story.append(Spacer(1, 2*mm))
    bullets2 = [
        "トラブル種別（毛穴・シワ・色ムラ・赤み・テカリ）ごとの回帰を追加し、部位特異性を切り分ける。",
        "相関係数 r・p値・傾きの95%信頼区間を併記し、統計的妥当性を強化する。",
        "ロバスト回帰（Huber、Theil–Sen）を付録として提示し、外れ値感度を低減する。",
        "同一被験者の時系列データで傾向の一貫性を検証する。",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, style)) for b in bullets2], bulletType='bullet'))

    doc.build(story)


if __name__ == "__main__":
    out_dir = os.path.expanduser("C:/Users/iikrk/OneDrive - 神奈川工科大学/ドキュメント")
    output_path = os.path.join(out_dir, "フラクタル次元による肌状態評価の研究_v5.pdf")
    build_pdf(output_path)
    print("WROTE", output_path)
