@echo off

REM Initial setup
if not exist "bin" mkdir "bin"

if exist "bin\Standalone" rmdir /s /q "bin\Standalone"
if exist "bin\Dir" rmdir /s /q "bin\Dir"


REM Standalone
pyinstaller --name FaceAttendaceSystem_Standalone --add-data "models;models" --distpath ".\bin\Windows\Standalone" --onefile main.py -y


REM Folder
pyinstaller --name FaceAttendaceSystem --add-data "models;models" --distpath ".\bin\Windows\Dir" --onedir main.py -y


REM Cleanup
del /q "*.spec" 2>nul
if exist "build" rmdir /s /q "build"