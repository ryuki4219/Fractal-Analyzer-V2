"""
肌画像ビューアー - Streamlitアプリ
検出された肌画像を簡単に閲覧できます
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import os

# ページ設定
st.set_page_config(
    page_title="肌画像ビューアー",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ 肌画像ビューアー")
st.markdown("**検出された肌画像を閲覧できます**")

# パス設定
base_dir = Path(r"c:\Users\iikrk\OneDrive - 神奈川工科大学\ドキュメント\GitHub\Fractal-Analyzer-V2")
skin_images_dir = base_dir / "BIGDATE" / "skin_images"

# 肌画像が存在するか確認
if not skin_images_dir.exists() or len(list(skin_images_dir.glob("skin_*.png"))) == 0:
    st.warning("⚠️ 肌画像がまだ検出されていません。")
    st.info("以下のコマンドを実行して肌画像を検出してください：")
    st.code("python detect_skin_images.py", language="bash")
    st.stop()

# 肌画像を読み込み
image_files = sorted(
    skin_images_dir.glob("skin_*.png"),
    key=lambda x: int(x.stem.split('_')[1])
)

if len(image_files) == 0:
    st.error("❌ 肌画像が見つかりませんでした。")
    st.stop()

st.success(f"✅ {len(image_files)}枚の肌画像が見つかりました")

# 統計ファイルを読み込み
stats_file = skin_images_dir / "detection_stats.txt"
if stats_file.exists():
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats_content = f.read()
    
    with st.expander("📊 検出統計情報を表示"):
        st.text(stats_content)

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # 画像番号選択
    image_index = st.number_input(
        "画像番号",
        min_value=1,
        max_value=len(image_files),
        value=1,
        step=1
    )
    
    # 表示列数
    cols = st.slider("表示列数", 1, 5, 3)
    
    # ページサイズ
    page_size = st.slider("1ページの画像数", 1, 50, 12)
    
    # ページ番号
    total_pages = (len(image_files) + page_size - 1) // page_size
    page = st.number_input(
        f"ページ番号 (1～{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )
    
    # フィルター
    st.markdown("---")
    st.subheader("🔍 フィルター")
    min_skin_ratio = st.slider(
        "最小肌色割合 (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0
    )

# 肌色割合でフィルタリング
filtered_files = []
for img_file in image_files:
    # ファイル名から肌色割合を抽出
    try:
        ratio_str = img_file.stem.split('_')[2].replace('%', '')
        ratio = float(ratio_str)
        if ratio >= min_skin_ratio:
            filtered_files.append((img_file, ratio))
    except:
        filtered_files.append((img_file, 0.0))

if len(filtered_files) == 0:
    st.warning(f"フィルター条件（肌色割合 >= {min_skin_ratio}%）に一致する画像がありません。")
    st.stop()

st.info(f"フィルター結果: {len(filtered_files)}枚 / {len(image_files)}枚")

# メインコンテンツ
st.markdown("---")

# タブ表示
tab1, tab2 = st.tabs(["📷 単一画像表示", "📚 ギャラリー表示"])

with tab1:
    if image_index <= len(filtered_files):
        img_path, ratio = filtered_files[image_index - 1]
        st.subheader(f"画像 #{image_index} (肌色: {ratio:.1f}%)")
        
        try:
            img = Image.open(img_path)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(img, use_container_width=True)
            with col2:
                st.write("**画像情報:**")
                st.write(f"- ファイル名: {img_path.name}")
                st.write(f"- 肌色割合: {ratio:.1f}%")
                st.write(f"- サイズ: {img.size[0]} × {img.size[1]}")
                st.write(f"- フォーマット: {img.format}")
                
                # ダウンロードボタン
                with open(img_path, "rb") as f:
                    st.download_button(
                        label="📥 ダウンロード",
                        data=f,
                        file_name=img_path.name,
                        mime="image/png"
                    )
            
            # ナビゲーション
            col_prev, col_info, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("⬅️ 前の画像", use_container_width=True, disabled=(image_index == 1)):
                    st.rerun()
            with col_info:
                st.write(f"**{image_index} / {len(filtered_files)}**")
            with col_next:
                if st.button("次の画像 ➡️", use_container_width=True, disabled=(image_index == len(filtered_files))):
                    st.rerun()
        
        except Exception as e:
            st.error(f"画像の読み込みに失敗しました: {e}")
    else:
        st.warning("画像番号が範囲外です。")

with tab2:
    st.subheader(f"ページ {page} / {total_pages}")
    
    # ページの画像を取得
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_files))
    page_images = filtered_files[start_idx:end_idx]
    
    # グリッド表示
    rows = (len(page_images) + cols - 1) // cols
    for row in range(rows):
        columns = st.columns(cols)
        for col_idx in range(cols):
            img_idx = row * cols + col_idx
            if img_idx < len(page_images):
                with columns[col_idx]:
                    try:
                        img_path, ratio = page_images[img_idx]
                        img = Image.open(img_path)
                        st.image(
                            img,
                            use_container_width=True,
                            caption=f"#{start_idx + img_idx + 1} (肌色: {ratio:.1f}%)"
                        )
                    except Exception as e:
                        st.error(f"エラー: {e}")
    
    # ページネーション
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ 前のページ", use_container_width=True, disabled=(page == 1)):
            st.rerun()
    with col2:
        st.write(f"**ページ {page} / {total_pages}**")
    with col3:
        if st.button("次のページ ➡️", use_container_width=True, disabled=(page == total_pages)):
            st.rerun()

# フッター
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <p>🖼️ 肌画像ビューアー | {len(filtered_files)}枚の肌画像</p>
</div>
""", unsafe_allow_html=True)
