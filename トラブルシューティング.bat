@echo off
chcp 65001 > nul
echo.
echo ========================================
echo   トラブルシューティング診断
echo ========================================
echo.

echo [1] Pythonバージョン確認...
python --version
echo.

echo [2] 仮想環境確認...
if exist .venv\Scripts\activate.bat (
    echo       ✓ 仮想環境: 存在します
) else (
    echo       ✗ 仮想環境: 見つかりません
)
echo.

echo [3] Streamlitインストール確認...
python -c "import streamlit; print('✓ Streamlit version:', streamlit.__version__)"
if %errorlevel% neq 0 (
    echo       ✗ Streamlitがインストールされていません
)
echo.

echo [4] 必要なライブラリ確認...
python -c "import cv2, numpy, sklearn, lightgbm; print('✓ 必須ライブラリ: OK')" 2>nul
if %errorlevel% neq 0 (
    echo       ✗ 一部のライブラリが不足しています
    echo       以下を実行してください:
    echo       pip install -r requirements.txt
)
echo.

echo [5] ポート8501確認...
netstat -ano | findstr ":8501" > nul
if %errorlevel% equ 0 (
    echo       ✓ ポート8501: 使用中（アプリが起動している可能性）
    echo       http://localhost:8501 をブラウザで開いてください
) else (
    echo       ○ ポート8501: 空いています
)
echo.

echo [6] fractal_app.py確認...
if exist fractal_app.py (
    echo       ✓ fractal_app.py: 存在します
    python -m py_compile fractal_app.py 2>nul
    if %errorlevel% equ 0 (
        echo       ✓ 構文エラー: ありません
    ) else (
        echo       ✗ 構文エラー: 検出されました
    )
) else (
    echo       ✗ fractal_app.py: 見つかりません
)
echo.

echo ========================================
echo   診断完了
echo ========================================
echo.
echo 【解決策】
echo.
echo ● アプリが起動しない場合:
echo   1. 起動.bat を実行
echo   2. http://localhost:8501 をブラウザで開く
echo.
echo ● ライブラリエラーの場合:
echo   pip install -r requirements.txt
echo.
echo ● それでも解決しない場合:
echo   このウィンドウをスクリーンショットして
echo   エラー内容を確認してください
echo.
pause
