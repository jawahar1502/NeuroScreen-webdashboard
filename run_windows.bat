@echo off
rem Starts NeuroScreen. The browser opens at http://localhost:8000
cd /d "%~dp0"
echo Installing requirements (first run only)...
python -m pip install -r requirements.txt
python app.py
pause
