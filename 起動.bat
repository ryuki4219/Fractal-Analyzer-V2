@echo off
chcp 65001 > nul

echo.
echo ========================================
echo   Fractal Analyzer V2 起動中...
echo ========================================
echo.

REM 現在のディレクトリを確認
cd /d "%~dp0"
echo 作業ディレクトリ: %CD%
echo.

REM 既存のStreamlitプロセスを終了
echo [1/4] 既存プロセスをチェック中...
taskkill /F /IM streamlit.exe > nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq streamlit*" > nul 2>&1
timeout /t 1 /nobreak > nul

REM 仮想環境の存在確認
echo [2/4] 仮想環境を確認中...
if not exist .venv\Scripts\activate.bat (
    echo.
    echo ========================================
    echo   エラー: 仮想環境が見つかりません
    echo ========================================
    echo.
    echo 解決方法:
    echo 1. 以下のコマンドを実行してください:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 仮想環境を有効化
echo [3/4] 仮想環境を有効化中...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo エラー: 仮想環境の有効化に失敗しました
    pause
    exit /b 1
)
echo       仮想環境: 有効化完了
echo.

REM Pythonとstreamlitの確認
echo [4/4] アプリケーション起動準備中...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo エラー: Pythonが見つかりません
    pause
    exit /b 1
)

streamlit --version > nul 2>&1
if %errorlevel% neq 0 (
    echo エラー: Streamlitがインストールされていません
    echo 実行: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Streamlitを起動
echo.
echo ========================================
echo   アプリケーション起動中...
echo ========================================
echo.
echo ブラウザで以下のURLを開いてください:
echo   http://localhost:8501
echo.
echo ※ 3秒後にブラウザが自動で開きます...
echo ※ 終了するには Ctrl+C を押してください
echo.

REM 3秒待ってからブラウザを開く
timeout /t 3 /nobreak > nul
start http://localhost:8501

REM Streamlitを起動（フォアグラウンドで実行）
streamlit run fractal_app.py --server.port 8501

REM エラーハンドリング
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo   エラー: 起動に失敗しました
    echo ========================================
    echo.
    echo トラブルシューティング:
    echo 1. ポート8501が使用中の可能性があります
    echo 2. 以下のコマンドを試してください:
    echo    netstat -ano ^| findstr :8501
    echo 3. または別のポートを使用:
    echo    streamlit run fractal_app.py --server.port 8502
    echo.
    pause
)
