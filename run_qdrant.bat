@echo off
chcp 65001 >nul
title AI_EveryNyan - Qdrant Launcher

:: === Пути относительно папки с батником (корень проекта) ===
set "ROOT=%~dp0"
set "QDRANT_DATA=%ROOT%data\qdrant_storage"
set "CONTAINER_NAME=ai_everynyan-qdrant"

:: Создаём папку хранения, если нет
if not exist "%QDRANT_DATA%" mkdir "%QDRANT_DATA%"

:: Проверка Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

:: Проверяем, существует ли контейнер
docker inspect %CONTAINER_NAME% >nul 2>&1
if %errorlevel% == 0 (
    docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Status}}" | findstr "Up" >nul
    if %errorlevel% == 0 (
        echo [INFO] Qdrant is already running.
        echo     Web UI: http://localhost:6333/dashboard/
        pause
        exit /b 0
    ) else (
        echo [INFO] Starting existing container...
        docker start %CONTAINER_NAME%
        goto :wait
    )
)

:: Запуск нового контейнера
echo [INFO] Creating & starting Qdrant...
echo     Storage: %QDRANT_DATA%
docker run -d ^
  --name %CONTAINER_NAME% ^
  -p 6333:6333 ^
  -p 6334:6334 ^
  -v "%QDRANT_DATA%:/qdrant/storage:z" ^
  --memory 2g ^
  --cpus 2 ^
  --restart unless-stopped ^
  qdrant/qdrant:latest

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container.
    pause
    exit /b 1
)

:wait
echo [INFO] Waiting for Qdrant to be ready...
for /L %%i in (1,1,30) do (
    curl -s http://localhost:6333/readyz >nul 2>&1
    if %errorlevel% == 0 goto :ready
    timeout /t 2 /nobreak >nul
)
echo [WARN] Timeout, but continuing...

:ready
echo [SUCCESS] Qdrant is ready!
echo     API: http://localhost:6333
echo     UI:  http://localhost:6333/dashboard/
pause