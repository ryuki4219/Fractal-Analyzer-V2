"""
Streamlit動作確認用のシンプルなテストファイル
"""
import streamlit as st

st.title('Streamlit動作テスト')
st.write('このメッセージが表示されれば、Streamlitは正常に動作しています。')
st.success('✅ Streamlitは正常に動作しています')

# 次に各ライブラリのインポートテスト
try:
    import numpy as np
    st.write('✓ NumPy:', np.__version__)
except Exception as e:
    st.error(f'✗ NumPy: {e}')

try:
    import pandas as pd
    st.write('✓ Pandas:', pd.__version__)
except Exception as e:
    st.error(f'✗ Pandas: {e}')

try:
    import cv2
    st.write('✓ OpenCV:', cv2.__version__)
except Exception as e:
    st.error(f'✗ OpenCV: {e}')

try:
    from sklearn import __version__ as sklearn_version
    st.write('✓ Scikit-learn:', sklearn_version)
except Exception as e:
    st.error(f'✗ Scikit-learn: {e}')

try:
    import matplotlib
    st.write('✓ Matplotlib:', matplotlib.__version__)
except Exception as e:
    st.error(f'✗ Matplotlib: {e}')

try:
    import skimage
    st.write('✓ Scikit-image:', skimage.__version__)
except Exception as e:
    st.error(f'✗ Scikit-image: {e}')
