@echo off
cd /d "%~dp0"

echo ========================================
echo   Fractal Analyzer V2
echo ========================================
echo.
echo アプリを起動しています...
echo.

REM 仮想環境を確認
if not exist .venv\Scripts\activate.bat (
    echo エラー: 仮想環境が見つかりません
    echo.
    echo 解決方法:
    echo 1. python -m venv .venv
    echo 2. .venv\Scripts\activate.bat
    echo 3. pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 仮想環境を有効化
call .venv\Scripts\activate.bat

REM Streamlitを起動
echo.
echo ブラウザで以下のURLを開いてください:
echo http://localhost:8501
echo.
echo ※ 自動で開かない場合は手動でコピーして開いてください
echo ※ 終了するには Ctrl+C を押してください
echo.

timeout /t 3 /nobreak > nul
start http://localhost:8501

streamlit run fractal_app.py --server.port 8501

pause
