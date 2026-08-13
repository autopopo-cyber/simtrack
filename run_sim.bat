@echo off
REM =====================================================================
REM  simtrack runner - two modes:
REM    LIVE : opens a MuJoCo window, follow-cam tracks the dog (WATCH it run)
REM    RECOR: headless render to PNG, then ffmpeg to 3d_cam.mp4 + 2d_map.mp4
REM  Usage: double-click and answer prompts, OR
REM         run_sim.bat <scene> <seconds> <mode>   mode = live | recor
REM         e.g.  run_sim.bat 1 300 live
REM  Output (RECOR mode): runs\<scene>\seed<N>_<ts>\ *.mp4 + scorecard.json
REM =====================================================================

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
if not exist %PY% ( echo [ERR] not found: %PY% - check venv & pause & exit /b 1 )

set SCEN=%1
set DUR=%2
set MODE=%3

if not "%SCEN%"=="" goto got_scen
echo.
echo ============ simtrack runner ============
echo  scenes:
echo    1) pure walls      slow, shows wedge/drift - best for diagnosis
echo    2) feature obs     random fixed obs on straight runs (target case)
echo    3) mixed           bend fixed + straight moving obs (DWA tracker)
echo    4) pure walls 60s  quick check
echo.
set /p SCEN=pick scene [1-4, default 1]:
:got_scen
if "%SCEN%"=="" set SCEN=1

if not "%MODE%"=="" goto got_mode
echo.
echo  mode:
echo    live  open a MuJoCo window, follow-cam tracks the dog (WATCH)
echo    recor headless render, build 3d_cam.mp4 + 2d_map.mp4 (RECORD)
set /p MODE=mode [live/recor, default live]:
:got_mode
if "%MODE%"=="" set MODE=live

if "%SCEN%"=="4" ( if "%DUR%"=="" set DUR=60 )
if not "%DUR%"=="" goto got_dur
set /p DUR=duration seconds [default 300]:
:got_dur
if "%DUR%"=="" set DUR=300

set SCEN_ARG=--no-obs 1
set SCEN_NAME=wall
if "%SCEN%"=="2" ( set SCEN_ARG=--obs-feature 1& set SCEN_NAME=feature )
if "%SCEN%"=="3" ( set SCEN_ARG=--obs-mix 1& set SCEN_NAME=mix )
if "%SCEN%"=="4" ( set SCEN_ARG=--no-obs 1& set SCEN_NAME=wall_quick )

set SEED=7
set /p SEED=seed [default 7, change for a different drift realization]:
if "%SEED%"=="" set SEED=7

set VIEWER_FLAG=
set OUTFLAG=
if /i "%MODE%"=="live" (
  set VIEWER_FLAG=--viewer 1
  set MUJOCO_GL=
  goto run
)
REM record mode: need offscreen GL + frame output dir
set MUJOCO_GL=glfw
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%i
set TSDIR=runs\%SCEN_NAME%\seed%SEED%_%STAMP%
mkdir %TSDIR% 2>nul
set OUTFLAG=--render-every 50 --out-dir %TSDIR% --save-name %TSDIR%\scorecard.json

:run
echo.
echo ============================================================
echo  scene=%SCEN_NAME%  seed=%SEED%  duration=%DUR%s  mode=%MODE%
if /i "%MODE%"=="live" (
  echo  LIVE: a MuJoCo window opens, camera follows the dog. Close window to stop.
) else (
  echo  RECOR: frames to %TSDIR%, then build 3d_cam.mp4 + 2d_map.mp4
  echo  2d_map legend: blue=est pose green=true pose red=drift black=wall white=free
)
echo ============================================================
echo.

%PY% test_scripts\algo3_headless.py %SCEN_ARG% --seed %SEED% --timeout %DUR% ^
  %VIEWER_FLAG% %OUTFLAG% --max-steps 999999 --trail-every 20
if errorlevel 1 ( echo [ERR] sim exited with error & pause & exit /b 1 )

if /i not "%MODE%"=="recor" ( echo. & echo done. & pause & exit /b 0 )

echo.
echo ============ building videos ============
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [WARN] ffmpeg not on PATH - skipping video. Frames are in %TSDIR%
  explorer %TSDIR% & pause & exit /b 0
)
pushd %TSDIR%
echo [ffmpeg] 3D follow-cam  -> 3d_cam.mp4 ...
ffmpeg -y -framerate 30 -i frame_%%06d.png -vf "scale=1280:-2" -pix_fmt yuv420p -loglevel error 3d_cam.mp4
echo [ffmpeg] 2D map         -> 2d_map.mp4 ...
ffmpeg -y -framerate 30 -i map_%%06d.png -vf "scale=1280:-2" -pix_fmt yuv420p -loglevel error 2d_map.mp4
popd
echo.
echo  done. videos in %TSDIR%\3d_cam.mp4 and 2d_map.mp4
explorer %TSDIR%
pause
