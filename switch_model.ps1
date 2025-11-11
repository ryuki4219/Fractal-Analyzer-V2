# モデル切り替えスクリプト
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("low1", "low2", "low3", "low4", "low5", "low6", "low7", "low8", "low9", "low10", "low4-7", "backup")]
    [string]$ModelType
)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Fractal Analyzer V2 - モデル切り替え" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($ModelType -eq "backup") {
    # バックアップから復元
    if (Test-Path "trained_fd_model_backup.pkl") {
        Copy-Item "trained_fd_model_backup.pkl" -Destination "trained_fd_model.pkl" -Force
        Write-Host "✅ バックアップから復元しました" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "❌ エラー: バックアップファイルが見つかりません" -ForegroundColor Red
        Write-Host ""
        exit 1
    }
} else {
    $modelFile = "models\trained_fd_model_$ModelType.pkl"
    
    # modelsフォルダがない場合は作成
    if (-not (Test-Path "models")) {
        New-Item -ItemType Directory -Path "models" | Out-Null
        Write-Host "📁 modelsフォルダを作成しました" -ForegroundColor Yellow
    }
    
    if (Test-Path $modelFile) {
        # 現在のモデルをバックアップ
        if (Test-Path "trained_fd_model.pkl") {
            Copy-Item "trained_fd_model.pkl" -Destination "trained_fd_model_backup.pkl" -Force
            Write-Host "💾 現在のモデルをバックアップしました" -ForegroundColor Green
            Write-Host "   → trained_fd_model_backup.pkl" -ForegroundColor Gray
            Write-Host ""
        }
        
        # 指定モデルをコピー
        Copy-Item $modelFile -Destination "trained_fd_model.pkl" -Force
        Write-Host "✅ $ModelType モデルに切り替えました" -ForegroundColor Green
        Write-Host "   → $modelFile" -ForegroundColor Cyan
        Write-Host ""
        
        # モデル情報を表示
        Write-Host "📊 モデル情報:" -ForegroundColor Yellow
        $fileInfo = Get-Item $modelFile
        Write-Host "   サイズ: $([math]::Round($fileInfo.Length / 1KB, 2)) KB" -ForegroundColor Gray
        Write-Host "   更新日時: $($fileInfo.LastWriteTime)" -ForegroundColor Gray
        Write-Host ""
        
    } else {
        Write-Host "❌ エラー: $modelFile が見つかりません" -ForegroundColor Red
        Write-Host ""
        Write-Host "💡 解決方法:" -ForegroundColor Yellow
        Write-Host "   1. 学習モードで品質レベル「$ModelType」を選択" -ForegroundColor Gray
        Write-Host "   2. データ拡張を設定して学習を実行" -ForegroundColor Gray
        Write-Host "   3. 学習完了後、モデルをダウンロード" -ForegroundColor Gray
        Write-Host "   4. ファイル名を「trained_fd_model_$ModelType.pkl」に変更" -ForegroundColor Gray
        Write-Host "   5. modelsフォルダに保存" -ForegroundColor Gray
        Write-Host ""
        exit 1
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 完了! アプリを再起動してください" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
