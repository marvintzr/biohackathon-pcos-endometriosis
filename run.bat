@echo off
cd /d "%~dp0"

if not exist "data\(Main_Dataset)_PCOS_data_without_infertility.xlsx" (
    echo.
    echo ERROR: Missing dataset files in the "data" folder.
    echo Copy both .xlsx files into: %~dp0data
    echo See SETUP.txt for details.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo Starting Streamlit at http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py --server.address localhost --server.port 8501
