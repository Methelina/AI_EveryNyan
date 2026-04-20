@echo off
chcp 65001 >nul
title AI_EveryNyan - Qdrant Launcher

:: ===========================================
:: Qdrant Launcher (Relative Paths)
:: Запускает Qdrant в Docker с хранением данных
:: в папке проекта: data/qdrant_storage
:: ===========================================

:: Определяем путь к папке, где лежит этот батник
set "PROJECT_ROOT=%~dp0"
set "QDRANT_DATA=%PROJECT_ROOT%data\qdrant_storage"
set "CONTAINER_NAME=ai_everynyan-qdrant"

:: Создаём папку для данных, если нет
if not exist "%QDRANT_DATA%" (
    mkdir "%QDRANT_DATA%"
    echo [i] Created directory: data\qdrant_storage
)

:: Проверка: запущен ли Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

:: Проверка: существует ли контейнер
docker inspect %CONTAINER_NAME% >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Container "%CONTAINER_NAME%" already exists.
    docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Status}}" | findstr "Up" >nul
    if %errorlevel% == 0 (
        echo [SUCCESS] Qdrant is already running.
        echo     Web UI: http://localhost:6333/dashboard/
        pause
        exit /b 0
    ) else (
        echo [WARN] Container exists but is not running. Starting...
        docker start %CONTAINER_NAME%
        goto :wait_ready
    )
)

:: Запуск нового контейнера
echo [INFO] Starting Qdrant container...
echo     Storage: %QDRANT_DATA%
echo     Web UI:  http://localhost:6333/dashboard/

docker run -d ^
  --name %CONTAINER_NAME% ^
  -p 6333:6333 ^
  -p 6334:6334 ^
  -v "%QDRANT_DATA%:/qdrant/storage:z" ^
  -e QDRANT__SERVICE__API_KEY= ^
  --memory 2g ^
  --cpus 2 ^
  --restart unless-stopped ^
  qdrant/qdrant:v1.11.0

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Qdrant container.
    pause
    exit /b 1
)

:wait_ready
echo [INFO] Waiting for Qdrant to be ready...
for /L %%i in (1,1,30) do (
    curl -s http://localhost:6333/readyz >nul 2>&1
    if %errorlevel% == 0 (
        echo [SUCCESS] Qdrant is ready!
        goto :done
    )
    timeout /t 2 /nobreak >nul
)
echo [WARN] Qdrant may not be fully ready yet, but continuing...

:done
echo.
echo ╔════════════════════════════════════════╗
echo ║   Qdrant запущен успешно!              ║
echo ║   API: http://localhost:6333          ║
echo ║   UI:  http://localhost:6333/dashboard/║
echo ╚════════════════════════════════════════╝
echo.
echo Для остановки: docker stop %CONTAINER_NAME%
echo.
pause