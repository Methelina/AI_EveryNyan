:: ==========================================
:: SearXNG Launcher v1.0.4 - AI_EveryNyan
:: Version: 1.0.4
:: Author: Soror L.'.L.'. (Linda)
:: Updated: 2026-04-23
::
:: Patchnote v1.0.4:
::   [FIX] Все команды в одну строку (убраны ^, т.к. ломали парсинг)
::   [FIX] Исправлены PowerShell команды: экранирование двоеточия, одинарные кавычки
::   [FIX] Убран амперсанд & из echo (вместо "Creating & starting" -> "Creating and starting")
::
:: Patchnote v1.0.3:
::   [FIX] Исправлена ошибка invalid mode: rw при монтировании томов
::   [FIX] Пути Windows конвертируются в формат K:/path (двоеточие сохраняется)
::   [FIX] Убрано лишнее экранирование кавычек в docker run
::
:: Patchnote v1.0.2:
::   [+] Конвертация путей Windows->Docker: K:\path -> /k/path (фикс "too many colons")
::   [+] Маркер .fixes_applied для однократного применения сетевых фиксов
::
:: Patchnote v1.0.1:
::   [-] Убрано авто-отключение движков (конфигурация только через settings.yml)
::   [*] Сетевые фиксы применяются опционально через меню
::
:: Patchnote v1.0.0:
::   [+] Первичный релиз лаунчера для Docker-развертывания
:: ==========================================

@echo off
setlocal EnableDelayedExpansion
title AI_EveryNyan - SearXNG Launcher v1.0.4

:: === Глобальные настройки ===
set "VERSION=1.0.4"
set "ROOT=%~dp0"
set "LOG_DIR=%ROOT%logs"
set "LOG_FILE=%LOG_DIR%\searxng_launcher.log"
set "SEARXNG_DATA=%ROOT%data\searxng_config"
set "CONTAINER_NAME=ai_everynyan-searxng"
set "CONFIG_FILE=%ROOT%config\searxng_env.bat"

:: === Настройки по умолчанию ===
set "SEARXNG_PORT=2597"
set "SEARXNG_BASE_URL=http://localhost:2597/"
set "SEARXNG_SECRET="
set "SEARXNG_LIMIT_MEMORY=1g"
set "SEARXNG_LIMIT_CPUS=1"
set "SEARXNG_IMAGE=ghcr.io/searxng/searxng:latest"

:: === Инициализация логирования ===
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
call :log "=== SearXNG Launcher v%VERSION% started at %DATE% %TIME% ==="

:: === Загрузка сохранённого конфига ===
if exist "%CONFIG_FILE%" call "%CONFIG_FILE%"

:: === Пересчёт base_url если порт изменился ===
if not "!SEARXNG_BASE_URL!"=="http://localhost:%SEARXNG_PORT%/" (
    set "SEARXNG_BASE_URL=http://localhost:%SEARXNG_PORT%/"
    call :log "Auto-corrected base_url to http://localhost:%SEARXNG_PORT%/"
)

goto :menu

:: === ФУНКЦИИ ===
:log
echo [%TIME%] %~1 >> "%LOG_FILE%"
goto :eof

:error_exit
call :log "[ERROR] %~1"
echo [ERROR] %~1
pause
goto :menu

:: === МЕНЮ ===
:menu
cls
echo ==========================================
echo   SearXNG Launcher v%VERSION% - AI_EveryNyan
echo ==========================================
echo  Configuration:
echo     Container : %CONTAINER_NAME%
echo     Port      : %SEARXNG_PORT%
echo     Base URL  : %SEARXNG_BASE_URL%
echo     Memory    : %SEARXNG_LIMIT_MEMORY%
echo     CPUs      : %SEARXNG_LIMIT_CPUS%
echo     Image     : %SEARXNG_IMAGE%
echo ==========================================
echo 1. Start SearXNG
echo 2. Configure (interactive setup)
echo 3. Clean Storage and Restart (WIPE DATA)
echo 4. Apply minimal network fixes (timeouts + IPv4)
echo 5. View container logs
echo 6. Run diagnostics (inside container)
echo 7. Edit settings.yml manually
echo 8. Exit
echo ==========================================
choice /c 12345678 /n /m "Select option [1-8]: "

if %errorlevel% == 2 goto :configure
if %errorlevel% == 3 goto :clean_and_restart
if %errorlevel% == 4 goto :apply_network_fixes
if %errorlevel% == 5 goto :view_logs
if %errorlevel% == 6 goto :diagnostics
if %errorlevel% == 7 goto :edit_settings
if %errorlevel% == 8 exit /b 0
goto :start_searxng

:: === КОНФИГУРАЦИЯ ===
:configure
cls
echo ==========================================
echo   SearXNG Configuration Setup
echo ==========================================
echo.
echo Current port: %SEARXNG_PORT%
set /p "SEARXNG_PORT=Enter host port [default: 2597]: "
if "%SEARXNG_PORT%"=="" set "SEARXNG_PORT=2597"
echo.
echo Current base URL: %SEARXNG_BASE_URL%
set /p "SEARXNG_BASE_URL=Enter base URL [default: http://localhost:%SEARXNG_PORT%/]: "
if "%SEARXNG_BASE_URL%"=="" set "SEARXNG_BASE_URL=http://localhost:%SEARXNG_PORT%/"
echo.
set /p "SEARXNG_SECRET=Enter secret key (leave empty for auto): "
echo.
echo Current memory: %SEARXNG_LIMIT_MEMORY%
set /p "SEARXNG_LIMIT_MEMORY=Enter memory limit [default: 1g]: "
if "%SEARXNG_LIMIT_MEMORY%"=="" set "SEARXNG_LIMIT_MEMORY=1g"
echo.
echo Current CPUs: %SEARXNG_LIMIT_CPUS%
set /p "SEARXNG_LIMIT_CPUS=Enter CPU limit [default: 1]: "
if "%SEARXNG_LIMIT_CPUS%"=="" set "SEARXNG_LIMIT_CPUS=1"
echo.
call :log "Saving configuration to %CONFIG_FILE%"
if not exist "%ROOT%config" mkdir "%ROOT%config"
(
echo @echo off
echo set "SEARXNG_PORT=%SEARXNG_PORT%"
echo set "SEARXNG_BASE_URL=%SEARXNG_BASE_URL%"
echo set "SEARXNG_SECRET=%SEARXNG_SECRET%"
echo set "SEARXNG_LIMIT_MEMORY=%SEARXNG_LIMIT_MEMORY%"
echo set "SEARXNG_LIMIT_CPUS=%SEARXNG_LIMIT_CPUS%"
) > "%CONFIG_FILE%"
call :log "Configuration saved successfully"
echo [OK] Configuration saved.
pause
goto :menu

:: === ОЧИСТКА И ПЕРЕЗАПУСК ===
:clean_and_restart
echo.
call :log "Starting clean restart procedure"
echo [INFO] Stopping and removing container...
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1
echo [INFO] Deleting old config data...
if exist "%SEARXNG_DATA%" (
    rmdir /s /q "%SEARXNG_DATA%"
    call :log "Config data deleted: %SEARXNG_DATA%"
    echo [OK] Config data deleted.
) else (
    echo [SKIP] Config folder not found.
)
echo [INFO] Restarting script...
timeout /t 2 /nobreak >nul
cls
call :log "Clean restart completed, returning to menu"
goto :start_searxng

:: === СЕТЕВЫЕ ФИКСЫ (минимальные, без отключения движков) ===
:apply_network_fixes
echo.
call :log "Applying minimal network fixes to settings.yml"
if not exist "%SEARXNG_DATA%\settings.yml" (
    echo [WARN] settings.yml not found. Start container first to generate it.
    pause
    goto :menu
)
powershell -Command "$c=Get-Content '%SEARXNG_DATA%\settings.yml' -Raw; if($c -match 'request_timeout:\s*[\d.]+'){ if($c -match 'request_timeout:\s*[0-9](\.[0-9]+)?\s*$'){ $c=$c -replace '(?m)^(\s*)request_timeout:\s*[\d.]+$', '$1request_timeout: 15.0' } } else { $c=$c -replace '(?m)^(\s*)outgoing:', '$1outgoing:`n$1  request_timeout: 15.0' }; Set-Content '%SEARXNG_DATA%\settings.yml' $c -NoNewline"
powershell -Command "$c=Get-Content '%SEARXNG_DATA%\settings.yml' -Raw; if($c -notmatch 'max_request_timeout:'){ $c=$c -replace '(?m)^(\s*)outgoing:', '$1outgoing:`n$1  max_request_timeout: 30.0' }; Set-Content '%SEARXNG_DATA%\settings.yml' $c -NoNewline"
powershell -Command "$c=Get-Content '%SEARXNG_DATA%\settings.yml' -Raw; if($c -notmatch 'enable_http2:'){ $c=$c -replace '(?m)^(\s*)outgoing:', '$1outgoing:`n$1  enable_http2: false`n$1  source_ips:`n$1    - 0.0.0.0' }; Set-Content '%SEARXNG_DATA%\settings.yml' $c -NoNewline"
call :log "Minimal network fixes applied"
echo [INFO] Restarting container to apply changes...
docker restart %CONTAINER_NAME% >nul 2>&1
timeout /t 3 /nobreak >nul
echo [OK] Fixes applied. Note: Engine selection is manual via settings.yml.
pause
goto :menu

:: === ЛОГИ ===
:view_logs
echo.
echo [INFO] Showing last 50 lines of container logs...
echo Press Ctrl+C to exit log view.
pause
docker logs --tail 50 -f %CONTAINER_NAME%
goto :menu

:: === ДИАГНОСТИКА ===
:diagnostics
echo.
echo [INFO] Running diagnostics inside container...
docker exec -it %CONTAINER_NAME% /bin/sh -c "echo '=== DNS Test ===' && nslookup google.com 2>&1 | findstr 'Address' && echo '=== HTTP Test (sync) ===' && python3 -c \"import urllib.request; print('urllib:', urllib.request.urlopen('https://duckduckgo.com', timeout=5).status)\" && echo '=== HTTP Test (async) ===' && python3 -c \"import asyncio, httpx; asyncio.run(print('httpx:', httpx.get('https://duckduckgo.com', timeout=5).status_code))\" && echo '=== Settings Check ===' && grep -A 8 'outgoing:' /etc/searxng/settings.yml 2>/dev/null || echo 'No outgoing block found'"
pause
goto :menu

:: === РЕДАКТИРОВАНИЕ settings.yml ===
:edit_settings
echo.
if exist "%SEARXNG_DATA%\settings.yml" (
    echo [INFO] Opening settings.yml in Notepad...
    notepad "%SEARXNG_DATA%\settings.yml"
    echo [INFO] After editing, use option 4 to restart container.
) else (
    echo [WARN] settings.yml not found. Start container first.
)
pause
goto :menu

:: === ГЛАВНАЯ ЛОГИКА ===
:start_searxng
echo.
call :log "Starting SearXNG launch sequence"
if not exist "%SEARXNG_DATA%" mkdir "%SEARXNG_DATA%"
if not exist "%SEARXNG_DATA%\cache" mkdir "%SEARXNG_DATA%\cache"

:: === КОНВЕРТАЦИЯ ПУТЕЙ ДЛЯ DOCKER (Windows -> Unix style, с сохранением двоеточия) ===
set "DOCKER_DATA=%SEARXNG_DATA:\=/%"
:: Например, K:\work\... -> K:/work/...
set "DOCKER_CACHE=%DOCKER_DATA%/cache"

:: Проверка Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    call :error_exit "Docker is not running. Please start Docker Desktop first."
)

:: Проверка состояния контейнера
docker inspect %CONTAINER_NAME% >nul 2>&1
if %errorlevel% == 0 (
    docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Status}}" | findstr "Up" >nul
    if %errorlevel% == 0 (
        call :log "SearXNG already running on port %SEARXNG_PORT%"
        echo [INFO] SearXNG is already running.
        echo     Web UI    : http://localhost:%SEARXNG_PORT%
        echo     Search API: http://localhost:%SEARXNG_PORT%/search?q=test^&format=json
        pause
        goto :menu
    ) else (
        call :log "Starting existing container"
        echo [INFO] Starting existing container...
        docker start %CONTAINER_NAME%
        goto :wait
    )
)

:: Генерация limiter.toml
(
echo [botdetection.ip_limit]
echo link_token = false
echo.
echo [botdetection.ip_lists]
echo block_ip = []
echo pass_ip = []
) > "%SEARXNG_DATA%\limiter.toml"

:: === ЗАПУСК КОНТЕЙНЕРА (одной строкой, без ^) ===
call :log "Executing docker run command"
echo [INFO] Creating and starting SearXNG...
echo     Config : %SEARXNG_DATA%
echo     Port   : %SEARXNG_PORT%
echo     Image  : %SEARXNG_IMAGE%

docker run -d --name %CONTAINER_NAME% --dns 8.8.8.8 --dns 1.1.1.1 -p %SEARXNG_PORT%:8080 -v "%DOCKER_DATA%:/etc/searxng:rw" -v "%DOCKER_CACHE%:/var/cache/searxng:rw" --memory %SEARXNG_LIMIT_MEMORY% --cpus %SEARXNG_LIMIT_CPUS% --restart unless-stopped --cap-drop ALL --cap-add CHOWN --cap-add SETGID --cap-add SETUID --cap-add DAC_OVERRIDE --log-driver json-file --log-opt max-size=1m --log-opt max-file=1 -e SEARXNG_BASE_URL=%SEARXNG_BASE_URL% -e SEARXNG_OUTGOING_ENABLE_HTTP2=false %SEARXNG_IMAGE%

set "LAUNCH_RESULT=%errorlevel%"
if %LAUNCH_RESULT% neq 0 (
    call :log "First launch failed (code %LAUNCH_RESULT%), attempting pull"
    echo [WARN] Failed to start. Pulling image...
    docker pull %SEARXNG_IMAGE%
    if %errorlevel% neq 0 (
        call :error_exit "Failed to pull image. Check network or image name."
    )
    docker run -d --name %CONTAINER_NAME% --dns 8.8.8.8 --dns 1.1.1.1 -p %SEARXNG_PORT%:8080 -v "%DOCKER_DATA%:/etc/searxng:rw" -v "%DOCKER_CACHE%:/var/cache/searxng:rw" --memory %SEARXNG_LIMIT_MEMORY% --cpus %SEARXNG_LIMIT_CPUS% --restart unless-stopped --cap-drop ALL --cap-add CHOWN --cap-add SETGID --cap-add SETUID --cap-add DAC_OVERRIDE --log-driver json-file --log-opt max-size=1m --log-opt max-file=1 -e SEARXNG_BASE_URL=%SEARXNG_BASE_URL% -e SEARXNG_OUTGOING_ENABLE_HTTP2=false %SEARXNG_IMAGE%
    set "LAUNCH_RESULT=%errorlevel%"
)
if %LAUNCH_RESULT% neq 0 (
    call :error_exit "Failed to start container after pull. Try removing --cap-drop ALL for first run."
)
call :log "Container started successfully"

:: === ОЖИДАНИЕ ГОТОВНОСТИ ===
:wait
echo [INFO] Waiting for SearXNG to be ready...
call :log "Waiting for health endpoint"
for /L %%i in (1,1,30) do (
    curl -s http://localhost:%SEARXNG_PORT%/healthz >nul 2>&1
    if %errorlevel% == 0 goto :ready
    timeout /t 2 /nobreak >nul
)
call :log "Health check timeout, continuing anyway"
echo [WARN] Timeout waiting for healthz, but continuing...

:: === ПАТЧ ПОСЛЕ ПЕРВОГО ЗАПУСКА (только сетевые параметры, с маркером) ===
:ready
if exist "%SEARXNG_DATA%\settings.yml" (
    call :log "Checking for post-start network patches"
    
    :: JSON формат (всегда применяем)
    findstr /C:"- json" "%SEARXNG_DATA%\settings.yml" >nul 2>&1
    if %errorlevel% neq 0 (
        powershell -Command "$nl=[char]10; $c=Get-Content '%SEARXNG_DATA%\settings.yml' -Raw; if($c -match '- html'){ $c=$c -replace '(?m)^(\s*)- html', ('$1- html' + $nl + '$1- json'); Set-Content '%SEARXNG_DATA%\settings.yml' $c -NoNewline }"
        call :log "Added JSON format to output"
    )
    
    :: Сетевые фиксы только при первом запуске (маркер .fixes_applied)
    if not exist "%SEARXNG_DATA%\.fixes_applied" (
        call :log "First run: applying network fixes"
        powershell -Command "$c=Get-Content '%SEARXNG_DATA%\settings.yml' -Raw; $c=$c -replace '(?m)^(\s*)request_timeout:.*$', '$1request_timeout: 15.0'; if($c -notmatch 'max_request_timeout:'){ $c=$c -replace '(?m)^(\s*)outgoing:', '$1outgoing:`n$1  max_request_timeout: 30.0' }; if($c -notmatch 'enable_http2:'){ $c=$c -replace '(?m)^(\s*)outgoing:', '$1outgoing:`n$1  enable_http2: false`n$1  source_ips:`n$1    - 0.0.0.0' }; Set-Content '%SEARXNG_DATA%\settings.yml' $c -NoNewline"
        type nul > "%SEARXNG_DATA%\.fixes_applied"
        call :log "Network fixes applied, marker created"
        call :log "Restarting container to apply settings"
        docker restart %CONTAINER_NAME% >nul 2>&1
        timeout /t 3 /nobreak >nul
    ) else (
        call :log "Network fixes already applied (marker exists)"
    )
)

call :log "SearXNG launch sequence completed successfully"
echo [SUCCESS] SearXNG is ready!
echo     Web UI    : http://localhost:%SEARXNG_PORT%
echo     Search API: http://localhost:%SEARXNG_PORT%/search?q=test^&format=json
echo.
echo     Query URL for integration:
echo     http://localhost:%SEARXNG_PORT%/search?q=^<query^>
echo.
echo [TIP] To enable/disable engines, edit settings.yml via option 7.
echo [TIP] If searches timeout, try option 4 (minimal network fixes).
pause
goto :menu

:: === ЗАВЕРШЕНИЕ ===
:exit_launcher
call :log "Launcher exited by user"
exit /b 0