@echo off
chcp 65001 > nul

echo.
echo [Start] Fractal Analyzer V2
echo.

taskkill /F /IM streamlit.exe > nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *streamlit*" > nul 2>&1

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Opening browser...
echo URL: http://localhost:8501
echo.
start http://localhost:8501
streamlit run fractal_app.py --server.port 8501 --server.headless true

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start.
    pause
)
