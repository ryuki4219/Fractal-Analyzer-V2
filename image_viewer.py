"""
BIGDATEフォルダ画像ビューアー - Streamlitアプリ
17,000枚以上の画像を簡単に閲覧できます
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import os

# ページ設定
st.set_page_config(
    page_title="BIGDATE 画像ビューアー",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ BIGDATE 画像ビューアー")
st.markdown("**17,425枚の画像を簡単に閲覧できます**")

# パス設定
base_dir = Path(r"c:\Users\iikrk\OneDrive - 神奈川工科大学\ドキュメント\GitHub\Fractal-Analyzer-V2")
images_dir = base_dir / "BIGDATE" / "images"

# 画像が整理されていない場合の処理
if not images_dir.exists() or len(list(images_dir.glob("image*.png"))) == 0:
    st.warning("⚠️ 画像がまだ整理されていません。")
    st.info("以下のコマンドを実行して画像を整理してください：")
    st.code("python organize_images.py", language="bash")
    
    # 代替：元のフォルダから直接読み込み
    st.markdown("---")
    st.subheader("📁 元フォルダから直接表示")
    temp_dir = base_dir / "BIGDATE" / "temp"
    if temp_dir.exists():
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(temp_dir.rglob(f'*{ext}')))
            image_files.extend(list(temp_dir.rglob(f'*{ext.upper()}')))
        
        if len(image_files) > 0:
            st.success(f"✅ {len(image_files)}枚の画像が見つかりました")
            
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
                page_size = st.slider("1ページの画像数", 1, 50, 9)
                
                # ページ番号
                total_pages = (len(image_files) + page_size - 1) // page_size
                page = st.number_input(
                    f"ページ番号 (1～{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1
                )
            
            # メインコンテンツ
            st.markdown("---")
            
            # タブ表示
            tab1, tab2 = st.tabs(["📷 単一画像表示", "📚 ギャラリー表示"])
            
            with tab1:
                st.subheader(f"画像 #{image_index}")
                try:
                    img_path = image_files[image_index - 1]
                    img = Image.open(img_path)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(img, use_container_width=True)
                    with col2:
                        st.write("**画像情報:**")
                        st.write(f"- 元ファイル名: {img_path.name}")
                        st.write(f"- サイズ: {img.size[0]} × {img.size[1]}")
                        st.write(f"- フォーマット: {img.format}")
                        st.write(f"- モード: {img.mode}")
                        
                        # ダウンロードボタン
                        with open(img_path, "rb") as f:
                            st.download_button(
                                label="📥 ダウンロード",
                                data=f,
                                file_name=img_path.name,
                                mime=f"image/{img.format.lower()}"
                            )
                    
                    # ナビゲーション
                    col_prev, col_info, col_next = st.columns([1, 2, 1])
                    with col_prev:
                        if st.button("⬅️ 前の画像", use_container_width=True):
                            if image_index > 1:
                                st.rerun()
                    with col_info:
                        st.write(f"**{image_index} / {len(image_files)}**")
                    with col_next:
                        if st.button("次の画像 ➡️", use_container_width=True):
                            if image_index < len(image_files):
                                st.rerun()
                
                except Exception as e:
                    st.error(f"画像の読み込みに失敗しました: {e}")
            
            with tab2:
                st.subheader(f"ページ {page} / {total_pages}")
                
                # ページの画像を取得
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(image_files))
                page_images = image_files[start_idx:end_idx]
                
                # グリッド表示
                rows = (len(page_images) + cols - 1) // cols
                for row in range(rows):
                    columns = st.columns(cols)
                    for col_idx in range(cols):
                        img_idx = row * cols + col_idx
                        if img_idx < len(page_images):
                            with columns[col_idx]:
                                try:
                                    img_path = page_images[img_idx]
                                    img = Image.open(img_path)
                                    st.image(img, use_container_width=True, caption=f"#{start_idx + img_idx + 1}")
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
        else:
            st.error("❌ 画像が見つかりませんでした。")
    else:
        st.error("❌ BIGDATEフォルダが見つかりません。")
else:
    # 整理済み画像を表示
    image_files = sorted(images_dir.glob("image*.png"), key=lambda x: int(x.stem.replace('image', '')))
    st.success(f"✅ {len(image_files)}枚の画像が整理されています")
    
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
        page_size = st.slider("1ページの画像数", 1, 50, 9)
        
        # ページ番号
        total_pages = (len(image_files) + page_size - 1) // page_size
        page = st.number_input(
            f"ページ番号 (1～{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )
    
    # メインコンテンツ
    st.markdown("---")
    
    # タブ表示
    tab1, tab2 = st.tabs(["📷 単一画像表示", "📚 ギャラリー表示"])
    
    with tab1:
        st.subheader(f"image{image_index}")
        try:
            img_path = image_files[image_index - 1]
            img = Image.open(img_path)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(img, use_container_width=True)
            with col2:
                st.write("**画像情報:**")
                st.write(f"- ファイル名: {img_path.name}")
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
        
        except Exception as e:
            st.error(f"画像の読み込みに失敗しました: {e}")
    
    with tab2:
        st.subheader(f"ページ {page} / {total_pages}")
        
        # ページの画像を取得
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(image_files))
        page_images = image_files[start_idx:end_idx]
        
        # グリッド表示
        rows = (len(page_images) + cols - 1) // cols
        for row in range(rows):
            columns = st.columns(cols)
            for col_idx in range(cols):
                img_idx = row * cols + col_idx
                if img_idx < len(page_images):
                    with columns[col_idx]:
                        try:
                            img_path = page_images[img_idx]
                            img = Image.open(img_path)
                            img_num = int(img_path.stem.replace('image', ''))
                            st.image(img, use_container_width=True, caption=f"image{img_num}")
                        except Exception as e:
                            st.error(f"エラー")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🖼️ BIGDATE 画像ビューアー | 17,425枚の画像データセット</p>
</div>
""", unsafe_allow_html=True)
