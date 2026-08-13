@echo off
REM 拍一张当前 slam 地图快照：远程抓 /map -> 本地渲染 PNG -> 自动打开。
cd /d %~dp0
.venv\Scripts\python.exe scripts\snapshot_map.py
pause
