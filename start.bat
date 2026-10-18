@echo off
chcp 65001 >nul
rem 雙擊這個檔案來訓練模型（等同於 python src\main.py）
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo 找不到 Python。請先安裝 Python 3.10 以上，安裝時勾選 "Add python.exe to PATH"。
    pause
    exit /b 1
)

python -c "import torch, pandas, sklearn, matplotlib, joblib" >nul 2>&1
if errorlevel 1 (
    echo 正在安裝需要的套件，第一次會花幾分鐘...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo 套件安裝失敗，請把上面的錯誤訊息貼給我。
        pause
        exit /b 1
    )
)

python src\main.py %*
rem 不論成功或失敗都停下來，讓你看得到結果或錯誤訊息
pause
