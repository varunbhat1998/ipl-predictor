@echo off
cd /d "C:\Users\varun\OneDrive\Desktop\ipl_predictor\V1"
echo [%date% %time%] Starting IPL Match Bot V1 >> start_bot.log
"C:\Python314\python.exe" match_bot.py >> start_bot.log 2>&1
echo [%date% %time%] Bot exited >> start_bot.log
