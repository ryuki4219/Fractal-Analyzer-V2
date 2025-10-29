@echo off
chcp 65001 > nul

echo.
echo ========================================
echo   Fractal Analyzer V2 起動中...
echo ========================================
echo.

REM 既存のStreamlitプロセスを終了
echo [1/3] 既存プロセスをチェック中...
taskkill /F /IM streamlit.exe > nul 2>&1
timeout /t 1 /nobreak > nul

REM 仮想環境を有効化
echo [2/3] 仮想環境を有効化中...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo       仮想環境: 有効化完了
) else (
    echo       警告: 仮想環境が見つかりません
)

REM Streamlitを起動
echo [3/3] アプリケーション起動中...
echo.
echo ========================================
echo   アプリが起動しました！
echo   ブラウザで開いてください:
echo   http://localhost:8501
echo ========================================
echo.
echo ※ ブラウザが自動で開かない場合は、
echo    上記URLを手動でコピーして開いてください
echo.

start http://localhost:8501
streamlit run fractal_app.py --server.port 8501 --server.headless true

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo   エラー: 起動に失敗しました
    echo ========================================
    echo.
    pause
)
