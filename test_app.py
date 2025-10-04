import streamlit as st

st.title("テストアプリ")
st.write("Hello, Streamlit!")

if st.button("テストボタン"):
    st.success("ボタンが押されました！")