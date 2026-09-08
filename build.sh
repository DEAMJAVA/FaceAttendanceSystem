#!/bin/bash

# Initial setup
mkdir -p "bin/Linux_Mac"
rm -rf "bin/Linux_Mac/*"


#Standalone
pyinstaller --name FaceAttendaceSystem_Standalone --add-data "models:models" --distpath "./bin/Linux_Mac/Standalone" --onefile main.py -y


#Folder
pyinstaller --name FaceAttendaceSystem --add-data "models:models" --distpath "./bin/Linux_Mac/Dir" --onedir main.py -y


#Cleanup
rm ./*.spec
rm -rf ./build