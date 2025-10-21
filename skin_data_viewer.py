"""
肌画像データセットビューアー - Streamlitアプリ
15人の顔の肌画像（正面・左側・右側）を簡単に閲覧できます
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import os

# ページ設定
st.set_page_config(
    page_title="肌画像データセットビューアー",
    page_icon="👤",
    layout="wide"
)

st.title("👤 肌画像データセットビューアー")
st.markdown("**15人の顔の肌画像（正面・左側・右側）を閲覧できます**")

# パス設定
base_dir = Path(r"c:\Users\iikrk\OneDrive - 神奈川工科大学\ドキュメント\GitHub\Fractal-Analyzer-V2")
skin_data_dir = base_dir / "SKIN_DATA"

# データが存在するか確認
if not skin_data_dir.exists():
    st.error("❌ SKIN_DATAフォルダが見つかりません。")
    st.stop()

# 人物フォルダを取得（数字のみ）
person_dirs = sorted(
    [d for d in skin_data_dir.iterdir() if d.is_dir() and d.name.isdigit()],
    key=lambda x: int(x.name)
)

if len(person_dirs) == 0:
    st.error("❌ 画像データが見つかりませんでした。")
    st.stop()

st.success(f"✅ {len(person_dirs)}人分のデータが見つかりました")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # 人物選択
    person_id = st.selectbox(
        "👤 人物を選択",
        options=[int(d.name) for d in person_dirs],
        format_func=lambda x: f"人物 {x}"
    )
    
    st.markdown("---")
    st.subheader("📊 データセット情報")
    st.write(f"- 総人数: {len(person_dirs)}人")
    st.write(f"- 各人の画像: 3枚")
    st.write(f"  - 正面 (front.jpg)")
    st.write(f"  - 左側 (left-side.jpg)")
    st.write(f"  - 右側 (right-side.jpg)")
    st.write(f"- 総画像数: {len(person_dirs) * 3}枚")
    
    st.markdown("---")
    st.subheader("🎨 表示設定")
    show_info = st.checkbox("画像情報を表示", value=True)
    show_all = st.checkbox("全員をギャラリー表示", value=False)

# メインコンテンツ
st.markdown("---")

if not show_all:
    # 個別表示モード
    st.subheader(f"👤 人物 {person_id}")
    
    person_dir = skin_data_dir / str(person_id)
    
    # 3枚の画像を横並びで表示
    col1, col2, col3 = st.columns(3)
    
    images_info = [
        ("front.jpg", "正面", col1),
        ("left-side.jpg", "左側", col2),
        ("right-side.jpg", "右側", col3)
    ]
    
    for img_name, label, col in images_info:
        img_path = person_dir / img_name
        
        with col:
            st.markdown(f"### {label}")
            
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    
                    if show_info:
                        st.write("**画像情報:**")
                        st.write(f"- サイズ: {img.size[0]} × {img.size[1]}")
                        st.write(f"- フォーマット: {img.format}")
                        st.write(f"- モード: {img.mode}")
                    
                    # ダウンロードボタン
                    with open(img_path, "rb") as f:
                        st.download_button(
                            label=f"📥 {label}をダウンロード",
                            data=f,
                            file_name=f"person_{person_id}_{img_name}",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"画像の読み込みに失敗: {e}")
            else:
                st.warning(f"{label}の画像が見つかりません")
    
    # ナビゲーション
    st.markdown("---")
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    
    current_idx = [int(d.name) for d in person_dirs].index(person_id)
    
    with col_prev:
        if st.button("⬅️ 前の人", use_container_width=True, disabled=(current_idx == 0)):
            st.rerun()
    
    with col_info:
        st.write(f"**人物 {person_id} / {len(person_dirs)}人**")
    
    with col_next:
        if st.button("次の人 ➡️", use_container_width=True, disabled=(current_idx == len(person_dirs) - 1)):
            st.rerun()

else:
    # ギャラリー表示モード
    st.subheader("📚 全員ギャラリー")
    
    for person_dir in person_dirs:
        person_num = int(person_dir.name)
        
        with st.expander(f"👤 人物 {person_num}", expanded=(person_num == person_id)):
            col1, col2, col3 = st.columns(3)
            
            images_info = [
                ("front.jpg", "正面", col1),
                ("left-side.jpg", "左側", col2),
                ("right-side.jpg", "右側", col3)
            ]
            
            for img_name, label, col in images_info:
                img_path = person_dir / img_name
                
                with col:
                    st.markdown(f"**{label}**")
                    
                    if img_path.exists():
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"エラー: {e}")
                    else:
                        st.warning("画像なし")

# フッター
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <p>👤 肌画像データセットビューアー | {len(person_dirs)}人 × 3枚 = {len(person_dirs) * 3}枚</p>
</div>
""", unsafe_allow_html=True)
