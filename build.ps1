# Baut dist\ToneMatch\ToneMatch.exe
# Start:  powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller ToneMatch.spec --noconfirm --clean

# Mitgelieferte Tones neben die EXE kopieren
$dst = "dist\ToneMatch\tones"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "tones\*.json" $dst -Force

Write-Host "Fertig: dist\ToneMatch\ToneMatch.exe"
