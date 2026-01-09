param(
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Separator {
    Write-Host "========================================"
}

function Write-Step([string]$step, [string]$message) {
    Write-Host "[$step] $message"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptRoot
Set-Location $workspaceRoot

Write-Host ""
Write-Separator
Write-Host "  Fractal Analyzer V2 起動中..."
Write-Separator
Write-Host ""
Write-Host "作業ディレクトリ: $workspaceRoot"
Write-Host ""

# Step 1: Kill existing Streamlit/python processes
Write-Step "1/4" "既存プロセスをチェック中..."
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process python   -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like 'streamlit*' -or $_.Path -like "*$($workspaceRoot)\\.venv*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 800

# Step 2: Ensure virtual environment exists
Write-Step "2/4" "仮想環境を確認中..."
$activateBat = [System.IO.Path]::Combine($workspaceRoot, '.venv', 'Scripts', 'activate.bat')
if (-not (Test-Path $activateBat)) {
    Write-Separator
    Write-Host "  エラー: 仮想環境が見つかりません" -ForegroundColor Red
    Write-Separator
    Write-Host ""
    Write-Host "解決方法:" -ForegroundColor Yellow
    Write-Host " 1) python -m venv .venv"
    Write-Host " 2) .venv\\Scripts\\activate.bat"
    Write-Host " 3) pip install -r requirements.txt"
    Read-Host "Enterキーで終了"
    exit 1
}

# Step 3: Verify Python/Streamlit from the venv
Write-Step "3/4" "仮想環境を検証中..."
$pythonExe = [System.IO.Path]::Combine($workspaceRoot, '.venv', 'Scripts', 'python.exe')
if (-not (Test-Path $pythonExe)) {
    Write-Host "エラー: .venv 内に python.exe が見つかりません" -ForegroundColor Red
    Read-Host "Enterキーで終了"
    exit 1
}

try {
    & $pythonExe --version > $null
} catch {
    Write-Host "エラー: Python の起動に失敗しました" -ForegroundColor Red
    Read-Host "Enterキーで終了"
    exit 1
}

try {
    & $pythonExe -m streamlit --version > $null
} catch {
    Write-Host "エラー: Streamlit がインストールされていません" -ForegroundColor Red
    Write-Host "実行例: pip install -r requirements.txt"
    Read-Host "Enterキーで終了"
    exit 1
}

# Step 4: Launch application
Write-Step "4/4" "アプリケーション起動準備中..."
Write-Host ""
Write-Separator
Write-Host "  アプリケーション起動中..."
Write-Separator
Write-Host ""
Write-Host "ブラウザで以下のURLを開いてください:"
Write-Host "  http://localhost:$Port"
Write-Host ""
Write-Host "※ 数秒後にブラウザが自動で開きます"
Write-Host "※ 終了するには Ctrl+C を押してください"
Write-Host ""

Start-Sleep -Seconds 3
Start-Process "http://localhost:$Port" | Out-Null

$streamlitArgs = @('-m', 'streamlit', 'run', 'fractal_app.py', '--server.port', $Port)
& $pythonExe $streamlitArgs

if ($LASTEXITCODE -ne 0) {
    Write-Separator
    Write-Host "  エラー: Streamlit の起動に失敗しました" -ForegroundColor Red
    Write-Separator
    Write-Host ""
    Write-Host "トラブルシューティング:" -ForegroundColor Yellow
    Write-Host " 1) ポート $Port が使用中でないか確認 (netstat -ano | findstr :$Port)"
    Write-Host " 2) 別ポートで起動: streamlit run fractal_app.py --server.port 8502"
    Read-Host "Enterキーで終了"
}
