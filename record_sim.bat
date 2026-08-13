@echo off
REM 录制远程 MuJoCo 3D 仿真画面为 mp4 并下载播放。机器狗必须在动（firefly 探索中）。
REM 用法: record_sim.bat [秒数] [fps] [maze]   默认 60秒 10fps rooms5x5
cd /d %~dp0
.venv\Scripts\python.exe scripts\record_sim.py %*
pause
