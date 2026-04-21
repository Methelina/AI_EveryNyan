@echo off
chcp 65001 >nul
title AI_EveryNyan - SearXNG Launcher

:: === Пути относительно папки с батником (корень проекта) ===
set "ROOT=%~dp0"
set "SEARXNG_DATA=%ROOT%data\searxng_config"
set "CONTAINER_NAME=ai_everynyan-searxng"

:: === Настройки по умолчанию для SeanXNG ===
set "SEARXNG_PORT=8080"
set "SEARXNG_BASE_URL=http://localhost:8080/"
set "SEARXNG_SECRET="
set "SEARXNG_LIMIT_MEMORY=1g"
set "SEARXNG_LIMIT_CPUS=1"

:: === Загрузка и инициализация конфига (создается при настройке) ===
set "CONFIG_FILE=%ROOT%config\searxng_env.bat"
if exist "%CONFIG_FILE%" call "%CONFIG_FILE%"

:: === МЕНЮ ===
:menu
cls
echo ==========================================
echo   SearXNG Launcher - AI_EveryNyan
echo ==========================================
echo  Current config:
echo     Container : %CONTAINER_NAME%
echo     Port      : %SEARXNG_PORT%
echo     Base URL  : %SEARXNG_BASE_URL%
echo     Memory    : %SEARXNG_LIMIT_MEMORY%
echo     CPUs      : %SEARXNG_LIMIT_CPUS%
echo ==========================================
echo 1. Start SearXNG
echo 2. Configure (interactive setup)
echo 3. Clean Storage and Restart (WIPE DATA)
echo 4. Regenerate settings.yml
echo 5. Exit
echo ==========================================
choice /c 12345 /n /m "Select option: "

if %errorlevel% == 2 goto :configure
if %errorlevel% == 3 goto :clean_and_restart
if %errorlevel% == 4 goto :regenerate_settings
if %errorlevel% == 5 exit /b 0
goto :start_searxng

:: === КОНФИГУРАЦИЯ ===
:configure
cls
echo ==========================================
echo   SearXNG Configuration Setup
echo ==========================================
echo.

:: Локальный порт сервиса
echo Current port: %SEARXNG_PORT%
set /p "SEARXNG_PORT=Enter port [default: 8080]: "
if "%SEARXNG_PORT%"=="" set "SEARXNG_PORT=8080"

:: Базовый URL
echo.
echo Current base URL: %SEARXNG_BASE_URL%
set /p "SEARXNG_BASE_URL=Enter base URL [default: http://localhost:%SEARXNG_PORT%/]: "
if "%SEARXNG_BASE_URL%"=="" set "SEARXNG_BASE_URL=http://localhost:%SEARXNG_PORT%/"

:: Секретный ключ (опционально)
echo.
set /p "SEARXNG_SECRET=Enter secret key (leave empty for auto-generation): "

:: Лимит памяти
echo.
echo Current memory limit: %SEARXNG_LIMIT_MEMORY%
set /p "SEARXNG_LIMIT_MEMORY=Enter memory limit [default: 1g]: "
if "%SEARXNG_LIMIT_MEMORY%"=="" set "SEARXNG_LIMIT_MEMORY=1g"

:: CPU лимит
echo.
echo Current CPU limit: %SEARXNG_LIMIT_CPUS%
set /p "SEARXNG_LIMIT_CPUS=Enter CPU limit [default: 1]: "
if "%SEARXNG_LIMIT_CPUS%"=="" set "SEARXNG_LIMIT_CPUS=1"

:: Сохранение конфига
echo.
echo [INFO] Saving configuration to %CONFIG_FILE%...
if not exist "%ROOT%config" mkdir "%ROOT%config"

(
echo @echo off
echo set "SEARXNG_PORT=%SEARXNG_PORT%"
echo set "SEARXNG_BASE_URL=%SEARXNG_BASE_URL%"
echo set "SEARXNG_SECRET=%SEARXNG_SECRET%"
echo set "SEARXNG_LIMIT_MEMORY=%SEARXNG_LIMIT_MEMORY%"
echo set "SEARXNG_LIMIT_CPUS=%SEARXNG_LIMIT_CPUS%"
) > "%CONFIG_FILE%"

echo [OK] Configuration saved.
echo.
pause
goto :menu

:: === ОЧИСТКА И ПЕРЕЗАПУСК ===
:clean_and_restart
echo.
echo [INFO] Stopping container...
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1

echo [INFO] Deleting old config data...
if exist "%SEARXNG_DATA%" (
    rmdir /s /q "%SEARXNG_DATA%"
    echo [OK] Config data deleted.
) else (
    echo [SKIP] Config folder not found.
)

echo [INFO] Restarting script to apply changes...
timeout /t 2 /nobreak >nul
cls
goto :start_searxng

:: === ВОССТАНОВЛЕНИЕ НАСТРОЕК ===
:regenerate_settings
echo.
echo [INFO] Stopping container to regenerate settings...
docker stop %CONTAINER_NAME% >nul 2>&1

if not exist "%SEARXNG_DATA%\settings.yml" (
    echo [WARN] settings.yml not found at %SEARXNG_DATA%\settings.yml
    echo        Starting container to generate one...
    goto :start_searxng
)

echo [INFO] Adding JSON format to settings.yml...

:: Проверка что settings.yml существует
findstr /C:"- json" "%SEARXNG_DATA%\settings.yml" >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] JSON format already present in settings.yml.
) else (
    powershell -Command "(Get-Content '%SEARXNG_DATA%\settings.yml') -replace '(?<=- html)', '- html`n    - json' | Set-Content '%SEARXNG_DATA%\settings.yml'"
    echo [OK] JSON format added to settings.yml.
)

:: Обновить base_url в settings.yml если существует
powershell -Command "$c = Get-Content '%SEARXNG_DATA%\settings.yml' -Raw; if ($c -match 'base_url:') { $c = $c -replace '(?<=base_url:\s*).*', '%SEARXNG_BASE_URL%'; Set-Content '%SEARXNG_DATA%\settings.yml' $c }"
echo [OK] Base URL updated.

echo [INFO] Starting container...
goto :start_searxng

:: === ГЛАВНАЯ ЛОГИКА ===
:start_searxng
echo.

:: Создание директории конфига, если нет
if not exist "%SEARXNG_DATA%" mkdir "%SEARXNG_DATA%"

:: Проверка docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    goto :menu
)

:: Проверка, что контейнер уже есть
docker inspect %CONTAINER_NAME% >nul 2>&1
if %errorlevel% == 0 (
    docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Status}}" | findstr "Up" >nul
    if %errorlevel% == 0 (
        echo [INFO] SearXNG is already running.
        echo     Web UI: http://localhost:%SEARXNG_PORT%
        echo     Search API: http://localhost:%SEARXNG_PORT%/search?q=test^&format=json
        pause
        goto :menu
    ) else (
        echo [INFO] Starting existing container...
        docker start %CONTAINER_NAME%
        goto :wait
    )
)

:: Создание limiter.toml (без ограничений для локального использования)
(
echo [botdetection.ip_limit]
echo link_token = false
echo.
echo [botdetection.ip_lists]
echo block_ip = []
echo pass_ip = []
) > "%SEARXNG_DATA%\limiter.toml"

:: Создание и запуск контейнера
set "RUN_CMD=docker run -d ^
    --name %CONTAINER_NAME% ^
    -p %SEARXNG_PORT%:8080 ^
    -v "%SEARXNG_DATA%:/etc/searxng:rw" ^
    --memory %SEARXNG_LIMIT_MEMORY% ^
    --cpus %SEARXNG_LIMIT_CPUS% ^
    --restart unless-stopped ^
    --cap-drop ALL ^
    --cap-add CHOWN ^
    --cap-add SETGID ^
    --cap-add SETUID ^
    --cap-add DAC_OVERRIDE ^
    --log-driver json-file ^
    --log-opt max-size=1m ^
    --log-opt max-file=1 ^
    -e SEARXNG_BASE_URL=%SEARXNG_BASE_URL%"

:: Добавление секретного ключа, если есть
if not "%SEARXNG_SECRET%"=="" (
    set "RUN_CMD=%RUN_CMD% -e SEARXNG_SECRET=%SEARXNG_SECRET%"
)

set "RUN_CMD=%RUN_CMD% searxng/searxng:latest"

:: Запуск контейнера
echo [INFO] Creating ^& starting SearXNG...
echo     Config : %SEARXNG_DATA%
echo     Port   : %SEARXNG_PORT%
%RUN_CMD%

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container.
    echo         On first run, try removing --cap-drop ALL and retrying.
    pause
    goto :menu
)

:: Первый запуск: дождаться settings.yml, затем исправление формата JSON
:wait
echo [INFO] Waiting for SearXNG to be ready...
for /L %%i in (1,1,30) do (
    curl -s http://localhost:%SEARXNG_PORT%/healthz >nul 2>&1
    if %errorlevel% == 0 goto :ready
    timeout /t 2 /nobreak >nul
)
echo [WARN] Timeout, but continuing...

:ready
:: Патчинг settings.yml
if exist "%SEARXNG_DATA%\settings.yml" (
    findstr /C:"- json" "%SEARXNG_DATA%\settings.yml" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [INFO] Patching settings.yml with JSON format...
        powershell -Command "$c = Get-Content '%SEARXNG_DATA%\settings.yml' -Raw; if ($c -match '- html') { $c = $c -replace '(?m)^(\s*)- html', '$1- html`n$1- json'; Set-Content '%SEARXNG_DATA%\settings.yml' $c -NoNewline }"
        echo [INFO] Restarting container to apply JSON format...
        docker restart %CONTAINER_NAME% >nul 2>&1
    )
)

echo [SUCCESS] SearXNG is ready!
echo     Web UI    : http://localhost:%SEARXNG_PORT%
echo     Search API: http://localhost:%SEARXNG_PORT%/search?q=test^&format=json
echo.
echo     Query URL for integration:
echo     http://localhost:%SEARXNG_PORT%/search?q=^<query^>
echo.
pause
goto :menu
