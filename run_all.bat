@echo off
chcp 65001 >nul
rem 雙擊這個檔案，一次產生報告用的所有表格和圖（全部跑完約需 1 小時以上）
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

echo [1/4] 訓練最終模型 -^> outputs\
python src\main.py || goto :failed
echo [2/4] 資料分析 -^> outputs\data\
python src\explore.py || goto :failed
echo [3/4] 錯誤分析 -^> outputs\errors\
python src\error_analysis.py || goto :failed
echo [4/4] 多 seed 實驗（最久）-^> outputs\experiments\
python src\experiments.py || goto :failed

echo.
echo 全部完成，結果都在 outputs 資料夾。
pause
exit /b 0

:failed
echo.
echo 執行失敗，請把上面的錯誤訊息貼給我。
pause
exit /b 1
