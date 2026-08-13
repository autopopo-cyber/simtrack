@echo off
REM 实时监控远程 firefly_explorer 自主探索日志（每3秒刷新）。Ctrl-C 退出。
cd /d %~dp0
.venv\Scripts\python.exe scripts\watch_firefly.py
pause
