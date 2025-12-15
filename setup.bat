@echo off
REM Setup script for RAG Chatbot on Windows
REM This script must be run from the APPLICATION CORE folder where requirements.txt is located

cls
echo 🤖 RAG Chatbot - Automated Setup
echo =================================
echo.

REM Change to APPLICATION CORE directory
REM This assumes setup.bat is being run from SETUP SCRIPT or somewhere else
REM We need to navigate to where requirements.txt actually is
cd /d "C:\Users\jange\OneDrive\Desktop\RAG Chatbot\APPLICATION CORE"

if not exist "requirements.txt" (
    echo ❌ ERROR: Could not find requirements.txt
    echo.
    echo Make sure you're in the APPLICATION CORE folder that contains:
    echo   - app.py
    echo   - config.py
    echo   - requirements.txt
    echo.
    echo Expected location: C:\Users\jange\OneDrive\Desktop\RAG Chatbot\APPLICATION CORE
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python found: %PYTHON_VERSION%
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment created

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated
echo.

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo.
    echo Make sure requirements.txt exists in: %cd%
    pause
    exit /b 1
)
echo ✅ Dependencies instal