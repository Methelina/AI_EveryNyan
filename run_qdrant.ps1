<#
Qdrant launcher for AI_EveryNyan: Docker-first with automatic portable fallback.

Tries Docker backend first; if Docker is unavailable (or $use_portable_qd = 1),
falls back to the portable qdrant.exe unpacked into bin\qd\. Both backends serve
http://localhost:6333 with storage in data\qdrant_storage, so the Python side
(src\runtime.py via qdrant_client) is fully backend-agnostic.

run_qdrant.ps1
Version:     1.0.1
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.1 (Soror L'.L'.):
  [*] Re-created after repo rollback (2026-09-29). Logic identical to v1.0.0.

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] New launcher: Docker-first, automatic portable fallback (bin\qd\), menu Start / Clean / Exit.
  [+] Hardcoded switch $use_portable_qd forces the portable backend.
  [+] Portable download from GitHub releases (qdrant v1.19.1, x86_64-pc-windows-msvc).
  [*] Ready-check via curl.exe (Invoke-WebRequest can false-fail via system proxy).
#>

# ---------------------------------------------------------------------------
# Configuration.
# $use_portable_qd = 1  -> always use portable backend (skip Docker entirely).
# $use_portable_qd = 0  -> Docker first, portable fallback if Docker is missing/down.
# ---------------------------------------------------------------------------
$use_portable_qd = 0

$ErrorActionPreference = 'Stop'
$Root      = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir   = Join-Path $Root 'data\qdrant_storage'
$BinDir    = Join-Path $Root 'bin\qd'
$QdrantExe = Join-Path $BinDir 'qdrant.exe'
$PidFile   = Join-Path $BinDir 'qdrant.pid'
$LogDir    = Join-Path $Root 'logs'
$LogFile   = Join-Path $LogDir 'qdrant_portable.log'
$ZipUrl    = 'https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-x86_64-pc-windows-msvc.zip'
$ZipFile   = Join-Path $BinDir 'qdrant.zip'
$ApiUrl    = 'http://localhost:6333'

function Write-Status([string]$Msg, [string]$Level = 'INFO') {
    Write-Host ("[{0}] {1}" -f $Level, $Msg)
}

function Test-Docker {
    try {
        docker info 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Test-QdrantReady {
    # curl.exe вместо Invoke-WebRequest: IWR может идти через системный прокси
    # и ложно фейлить localhost-проверку на некоторых машинах.
    curl.exe -s -o NUL --max-time 3 "$ApiUrl/readyz" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Wait-QdrantReady([int]$TimeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-QdrantReady) { return $true }
        Start-Sleep -Seconds 2
    }
    return (Test-QdrantReady)
}

function Stop-PortableQdrant {
    if (Test-Path $PidFile) {
        $oldPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($oldPid) {
            $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if ($proc) { Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue }
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
    # Fallback: kill any qdrant.exe started from our bin dir.
    Get-Process -Name 'qdrant' -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -like "$BinDir\*" } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

function Ensure-PortableInstalled {
    if (Test-Path $QdrantExe) { return }
    Write-Status "Portable Qdrant not found. Downloading from GitHub..." "INFO"
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipFile -UseBasicParsing
    Write-Status "Extracting to $BinDir ..." "INFO"
    Expand-Archive -Path $ZipFile -DestinationPath $BinDir -Force
    Remove-Item $ZipFile -Force -ErrorAction SilentlyContinue
    if (!(Test-Path $QdrantExe)) {
        Write-Status "qdrant.exe not found after extraction. Archive layout may have changed." "ERROR"
        exit 1
    }
    Write-Status "Portable Qdrant installed." "SUCCESS"
}

function Start-PortableQdrant {
    Ensure-PortableInstalled
    New-Item -ItemType Directory -Path $DataDir, $LogDir -Force | Out-Null
    $env:QDRANT__STORAGE__STORAGE_PATH = $DataDir
    Write-Status "Starting portable Qdrant (storage: $DataDir)..." "INFO"
    $proc = Start-Process -FilePath $QdrantExe `
        -WorkingDirectory $BinDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError (Join-Path $LogDir 'qdrant_portable.err.log') `
        -PassThru
    Set-Content -Path $PidFile -Value $proc.Id
    Write-Status "Logs: $LogFile" "INFO"
}

function Start-DockerQdrant {
    $container = 'ai_everynyan-qdrant'
    docker inspect $container 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $status = (docker ps --filter "name=$container" --format '{{.Status}}')
        if ($status -match 'Up') {
            Write-Status "Qdrant container is already running." "INFO"
            return
        }
        Write-Status "Starting existing container..." "INFO"
        docker start $container | Out-Null
        return
    }
    Write-Status "Creating & starting Qdrant container (storage: $DataDir)..." "INFO"
    docker run -d `
        --name $container `
        -p 6333:6333 `
        -p 6334:6334 `
        -v "${DataDir}:/qdrant/storage:z" `
        --memory 2g `
        --cpus 2 `
        --restart unless-stopped `
        qdrant/qdrant:latest | Out-Null
}

function Start-QdrantBackend {
    if (Test-QdrantReady) {
        Write-Status "Qdrant is already ready." "INFO"
        Show-Endpoints
        return
    }
    if ($use_portable_qd -eq 1) {
        Write-Status "[RUNNER] fallback: portable mode forced (use_portable_qd = 1)" "WARN"
        Start-PortableQdrant
    } elseif (Test-Docker) {
        Start-DockerQdrant
    } else {
        Write-Status "[RUNNER] fallback: Docker unavailable, switching to portable Qdrant (bin\qd)" "WARN"
        Start-PortableQdrant
    }
    Write-Status "Waiting for Qdrant to become ready..."
    if (Wait-QdrantReady) {
        Write-Status "Qdrant is ready!" "SUCCESS"
        Show-Endpoints
    } else {
        Write-Status "Timeout waiting for /readyz. Check logs: $LogFile" "WARN"
    }
}

function Show-Endpoints {
    Write-Status "    API: $ApiUrl"
    Write-Status "    UI:  $ApiUrl/dashboard/"
}

function Clear-StorageAndRestart {
    Write-Status "Stopping backends..." "INFO"
    Stop-PortableQdrant
    if (Test-Docker) {
        docker stop 'ai_everynyan-qdrant' 2>$null | Out-Null
        docker rm 'ai_everynyan-qdrant' 2>$null | Out-Null
    }
    Write-Status "Deleting storage data (WIPE)..." "INFO"
    if (Test-Path $DataDir) {
        Remove-Item -Path $DataDir -Recurse -Force
        Write-Status "Storage deleted." "SUCCESS"
    } else {
        Write-Status "Storage folder not found, skipping." "INFO"
    }
    Start-QdrantBackend
}

# ---------------------------------------------------------------------------
# Menu loop.
# ---------------------------------------------------------------------------
while ($true) {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "  Qdrant Launcher - AI_EveryNyan"
    Write-Host "  Backend: $(if ($use_portable_qd -eq 1) { 'PORTABLE (forced)' } elseif (Test-Docker) { 'Docker (portable fallback ready)' } else { 'PORTABLE (Docker unavailable)' })"
    Write-Host "=========================================="
    Write-Host "1. Start Qdrant"
    Write-Host "2. Clean Storage and Restart (WIPE DATA)"
    Write-Host "3. Exit"
    Write-Host "=========================================="
    $choice = Read-Host "Select option"
    switch ($choice) {
        '1' { Start-QdrantBackend }
        '2' { Clear-StorageAndRestart }
        '3' { exit 0 }
        default { Write-Status "Unknown option." "WARN" }
    }
}
