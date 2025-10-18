@echo off
chcp 65001 > nul
echo.
echo デスクトップショートカットを作成しています...
echo.

REM 現在のディレクトリ
set "SCRIPT_DIR=%~dp0"
set "TARGET=%SCRIPT_DIR%起動.bat"

REM VBScriptでショートカット作成
echo Set WshShell = WScript.CreateObject("WScript.Shell") > "%TEMP%\shortcut.vbs"
echo Set Shortcut = WshShell.CreateShortcut(WshShell.SpecialFolders("Desktop") ^& "\Fractal Analyzer V2.lnk") >> "%TEMP%\shortcut.vbs"
echo Shortcut.TargetPath = "%TARGET%" >> "%TEMP%\shortcut.vbs"
echo Shortcut.WorkingDirectory = "%SCRIPT_DIR%" >> "%TEMP%\shortcut.vbs"
echo Shortcut.Description = "Fractal Analyzer V2 - AI画像補完システム" >> "%TEMP%\shortcut.vbs"
echo Shortcut.Save >> "%TEMP%\shortcut.vbs"

cscript //nologo "%TEMP%\shortcut.vbs"
del "%TEMP%\shortcut.vbs"

echo.
echo ✓ 完了！
echo.
echo デスクトップに「Fractal Analyzer V2」が作成されました。
echo このアイコンをダブルクリックして起動してください。
echo.
pause
