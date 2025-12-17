# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import black
from reportlab.lib.units import mm
from reportlab.lib import fonts
import os

# 日本語フォント登録（Windows想定）
def register_japanese_font():
    candidates = [
        ("Meiryo", "C:/Windows/Fonts/meiryo.ttc"),
        ("YuGothic", "C:/Windows/Fonts/YuGothM.ttc"),
        ("MSMincho", "C:/Windows/Fonts/msmincho.ttc"),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
                pdfmetrics.registerFont(TTFont(name, path))
                fonts.addMapping(name, 0, 0, name)
                return name
            except Exception:
                continue
    return None


def build_pdf(output_path):
    font_name = register_japanese_font() or "Helvetica"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    base = ParagraphStyle(
        name="Base",
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=TA_LEFT,
        textColor=black,
    )
    title = ParagraphStyle(name="Title", parent=base, fontSize=16, leading=22)
    section = ParagraphStyle(name="Section", parent=base, fontSize=13, leading=20)

    story = []

    # タイトル
    story.append(Paragraph("フラクタル次元による肌状態評価の研究（読みやすい版 v5-easy）", title))
    story.append(Spacer(1, 8*mm))

    # 要旨（やさしい版）
    story.append(Paragraph("要旨（やさしい版）", section))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "私たちは、写真からフラクタル次元（FD）を計算し、同時に肌の状態（毛穴・シワ・色ムラ・赤み・テカリ）を自動で数値化して比べました。"
        "たくさんの人や日付の違うデータをまとめて見ると、FDが高いと肌トラブルも増える傾向が見られました（いっしょに増える）。"
        "一方、1枚の顔写真の中で部位ごとに比べると、平均のとり方や部位の違いの影響で、FDが高い部位ほどトラブルが少なく見えることもありました（逆向きに見えることがある）。"
        "これはデータの見方の違いによるもので、全体の結果と矛盾しません。",
        base
    ))
    story.append(Spacer(1, 5*mm))

    # 新しく分かったこと（やさしい表現）
    story.append(Paragraph("新しく分かったこと", section))
    story.append(Spacer(1, 1*mm))
    bullets1 = [
        "多くのデータを集めるほど、『FDが高いほど肌の問題も増える』という関係がはっきりします。",
        "1枚の写真の中で部位ごとに比べると、“平均して1つにまとめた点数”の影響で、関係が弱くなったり、逆向きに見えることがあります。",
        "毛穴やシワのように『細かいデコボコ』に関係するものは、FDといっしょに増えやすいです。テカリは光の当たり方など撮影条件に左右されやすいです。",
        "明るさ・ピントなど撮り方や、グレースケール化などの前処理は、結果に影響するのでそろえることが大切です。",
        "顔検出の工夫（設定を自動で変える／最後は中央を顔として扱う）で、止まらず解析を進められるようになりました。",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, base)) for b in bullets1], bulletType='bullet'))
    story.append(Spacer(1, 5*mm))

    # グラフ（最小二乗法）の読み方
    story.append(Paragraph("グラフ（最小二乗法）の読み方", section))
    story.append(Spacer(1, 1*mm))
    bullets2 = [
        "右上がりの線: FDが高いほど、肌トラブルも大きくなる（いっしょに増える）。",
        "右下がりの線: FDが高いほど、肌トラブルは小さくなる（逆向き）。",
        "点が線の近くに集まっている: 関係が強い。バラバラ: 関係が弱い。",
        "1枚の写真の部位だけのグラフは、点の数が少ないので『方向の目安』を見る程度にして、結論はたくさんのデータで判断します。",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, base)) for b in bullets2], bulletType='bullet'))
    story.append(Spacer(1, 5*mm))

    # 今後やるとよいこと
    story.append(Paragraph("今後やるとよいこと", section))
    story.append(Spacer(1, 1*mm))
    bullets3 = [
        "毛穴だけ、シワだけ、など『種類ごと』にグラフを作って比べる。",
        "関係の強さを数で添える（例: 強い/中くらい/弱い、偶然でなさそう等）。",
        "外れ値に強い方法のグラフも参考にする。",
        "同じ人を日を変えて測り、変化が同じ方向かを確かめる。",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, base)) for b in bullets3], bulletType='bullet'))
    story.append(Spacer(1, 5*mm))

    # 言い換えメモ
    story.append(Paragraph("言い換えメモ（本文で使える表現）", section))
    story.append(Spacer(1, 1*mm))
    bullets4 = [
        "正の傾向 → いっしょに増える／右上がり",
        "負の傾向 → 片方が増えるともう片方は減る／右下がり",
        "相関 → 関係の強さ",
        "有意差がある → 偶然ではなさそう／はっきりした差がある",
        "回帰直線 → データの流れを表す1本の線",
        "傾き → 線の傾き（右上がり/右下がり）",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, base)) for b in bullets4], bulletType='bullet'))

    doc.build(story)


if __name__ == "__main__":
    out_dir = os.path.expanduser("C:/Users/iikrk/OneDrive - 神奈川工科大学/ドキュメント")
    output_path = os.path.join(out_dir, "フラクタル次元による肌状態評価の研究_v5_easy.pdf")
    build_pdf(output_path)
    print("WROTE", output_path)
