#!/usr/bin/env python3
"""
isolated_fetch_test.py
Тест извлечения контента с URL с выбором метода:

  2) Только bs4 + markdownify (без трафилатуры) – быстрый, работает на статичных страницах.
  4) Playwright + bs4 + markdownify – рендерит JavaScript, работает на динамичных SPA.
  5) Nodriver + bs4 + markdownify – рендерит JavaScript, использует Chromium Playwright.

Использование:
    python isolated_fetch_test.py <URL> [method]

    method: 2, 4 или 5 (по умолчанию 2)
    Если method не указан, скрипт предложит выбрать интерактивно.

Результат сохраняется в папку logs (создаётся автоматически) в файл
fetched_<method_name>_<timestamp>.txt.
Лог всех операций – logs/mcp_fetch_test.log
"""

import os
import sys
import asyncio
import httpx
from datetime import datetime
from pathlib import Path

# ============================================================================
# Определение корня проекта (где лежит папка logs).
# Считаем, что скрипт лежит в tools/mcp/isolated_fetch_test.py.
# Корень проекта – тремя уровнями выше.
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent   # K:\work\AI\AI_EveryNyan\Reposit
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "mcp_fetch_test.log"

TIMEOUT_HTTP = 30
TIMEOUT_PLAYWRIGHT = 60
TIMEOUT_NODRIVER = 60
DEFAULT_MAX_LENGTH = None   # None = не обрезать

# ----------------------------------------------------------------------
# Импорт необходимых библиотек
# ----------------------------------------------------------------------
BS4_AVAILABLE = False
MD_AVAILABLE = False
PLAYWRIGHT_AVAILABLE = False
NODRIVER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    print("[ERROR] BeautifulSoup4 не установлена. Установите: pip install beautifulsoup4")
    sys.exit(1)

try:
    from markdownify import markdownify as md
    MD_AVAILABLE = True
except ImportError:
    print("[ERROR] markdownify не установлена. Установите: pip install markdownify")
    sys.exit(1)

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    print("[WARN] Playwright не установлена. Для метода 4 установите: pip install playwright && playwright install chromium")

try:
    import nodriver as uc
    NODRIVER_AVAILABLE = True
except ImportError:
    print("[WARN] Nodriver не установлена. Для метода 5 установите: pip install nodriver")

# ----------------------------------------------------------------------
# Логирование в файл
# ----------------------------------------------------------------------
def log_message(msg: str):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {msg}\n")
    print(msg)

# ----------------------------------------------------------------------
# Очистка HTML через BeautifulSoup (общая для всех методов)
# ----------------------------------------------------------------------
def clean_html_with_bs4(html: str) -> str:
    """Удалить шумные теги с помощью BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception:
        soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['script', 'style', 'meta', 'link', 'noscript',
                     'header', 'footer', 'nav', 'aside', 'form', 'button',
                     'iframe', 'svg', 'canvas', 'command', 'embed', 'object']):
        tag.decompose()

    for tag in soup.find_all():
        if len(tag.get_text(strip=True)) == 0 and tag.name not in ['br', 'p', 'hr']:
            tag.decompose()
    return str(soup)

# ----------------------------------------------------------------------
# Преобразование очищенного HTML в текст (bs4 + markdownify)
# ----------------------------------------------------------------------
async def extract_with_bs4_md(html: str) -> str:
    """Очистка через bs4 и преобразование в Markdown синхронно (без сети)."""
    cleaned = clean_html_with_bs4(html)
    text = md(cleaned, strip=["img", "script", "style", "nav", "footer", "header", "aside"])
    text = "\n".join(line for line in text.splitlines() if line.strip())
    if not text.strip() and BS4_AVAILABLE:
        soup = BeautifulSoup(cleaned, 'lxml' if 'lxml' in sys.modules else 'html.parser')
        raw_text = soup.get_text(separator='\n', strip=True)
        text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
    return text

# ----------------------------------------------------------------------
# Метод 2: bs4 + markdownify (HTTP через httpx)
# ----------------------------------------------------------------------
async def fetch_method_bs4_md(url: str) -> str:
    log_message(f"=== Метод 2: bs4+markdownify, URL: {url} ===")
    async with httpx.AsyncClient(timeout=TIMEOUT_HTTP, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    log_message(f"  Загружено HTML: {len(html)} символов")
    text = await extract_with_bs4_md(html)
    log_message(f"  Извлечено текста: {len(text)} символов")
    return text

# ----------------------------------------------------------------------
# Метод 4: Playwright + bs4 + markdownify
# ----------------------------------------------------------------------
async def fetch_method_playwright(url: str) -> str:
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright не установлен. Установите: pip install playwright && playwright install chromium")
    log_message(f"=== Метод 4: Playwright+bs4+md, URL: {url} ===")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ]
        )
        page = await browser.new_page()
        log_message("  Запущен headless Chromium, загрузка страницы...")
        await page.goto(url, wait_until="networkidle", timeout=TIMEOUT_PLAYWRIGHT * 1000)
        html = await page.content()
        await browser.close()
    log_message(f"  Получено HTML (отрендеренный): {len(html)} символов")
    text = await extract_with_bs4_md(html)
    log_message(f"  Извлечено текста: {len(text)} символов")
    return text

# ----------------------------------------------------------------------
# Поиск пути к Chromium (для Nodriver)
# ----------------------------------------------------------------------
def find_chromium_executable() -> str:
    """Ищет Chromium, установленный Playwright, или системный Chrome."""
    # 1. Chromium от Playwright
    playwright_chrome = os.path.expanduser("~\\AppData\\Local\\ms-playwright\\chromium-*\\chrome-win64\\chrome.exe")
    import glob
    matches = glob.glob(playwright_chrome)
    if matches:
        matches.sort(reverse=True)
        return matches[0]
    
    # 2. Системный Chrome (Windows)
    possible_paths = [
        os.environ.get("PROGRAMFILES", "C:\\Program Files") + "\\Google\\Chrome\\Application\\chrome.exe",
        os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)") + "\\Google\\Chrome\\Application\\chrome.exe",
        os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    
    raise FileNotFoundError("Не найден исполняемый файл Chrome/Chromium. Установите Chrome или Playwright с chromium.")

# ----------------------------------------------------------------------
# Метод 5: Nodriver + bs4 + markdownify (исправленный)
# ----------------------------------------------------------------------
async def fetch_method_nodriver(url: str) -> str:
    if not NODRIVER_AVAILABLE:
        raise RuntimeError("Nodriver не установлен. Установите: pip install nodriver")
    log_message(f"=== Метод 5: Nodriver+bs4+md, URL: {url} ===")
    
    chrome_path = find_chromium_executable()
    log_message(f"  Используется браузер: {chrome_path}")
    
    # Используем контекстный менеджер для автоматического закрытия браузера
    browser = None
    try:
        browser = await uc.start(
            headless=True,
            browser_executable_path=chrome_path
        )
        page = await browser.get(url)
        log_message("  Браузер запущен, ожидание рендеринга...")
        await asyncio.sleep(3)  # даём время на прогрузку динамики
        html = await page.get_content()
        log_message(f"  Получено HTML (отрендеренный): {len(html)} символов")
        text = await extract_with_bs4_md(html)
        log_message(f"  Извлечено текста: {len(text)} символов")
        return text
    except Exception as e:
        log_message(f"  ❌ Ошибка Nodriver: {e}")
        raise
    finally:
        # Безопасно закрываем браузер, если он существует
        if browser is not None:
            try:
                await browser.stop()
            except Exception as stop_err:
                log_message(f"  При закрытии браузера ошибка (игнорируется): {stop_err}")
        # Даём время на завершение процессов
        await asyncio.sleep(0.5)

# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
async def main():
    if len(sys.argv) < 2:
        print("Использование: python isolated_fetch_test.py <URL> [method]")
        print("  method: 2 - bs4+markdownify (быстрый, статика)")
        print("          4 - Playwright+bs4+md (рендер JS, надёжный)")
        print("          5 - Nodriver+bs4+md (рендер JS, обход детекции)")
        print("  Если method не указан, будет запрошен интерактивно.")
        sys.exit(1)

    url = sys.argv[1]

    if len(sys.argv) >= 3:
        method_choice = sys.argv[2]
    else:
        print("\nВыберите метод извлечения:")
        print("  2 - bs4+markdownify (быстрый, статика)")
        print("  4 - Playwright+bs4+md (рендер JS, надёжный)")
        print("  5 - Nodriver+bs4+md (рендер JS, обход детекции)")
        method_choice = input("Ваш выбор (2/4/5): ").strip()

    if method_choice == "2":
        full_text = await fetch_method_bs4_md(url)
        method_name = "bs4_markdownify"
    elif method_choice == "4":
        full_text = await fetch_method_playwright(url)
        method_name = "playwright_bs4_md"
    elif method_choice == "5":
        full_text = await fetch_method_nodriver(url)
        method_name = "nodriver_bs4_md"
    else:
        print("Неверный выбор. Используйте 2, 4 или 5.")
        sys.exit(1)

    if DEFAULT_MAX_LENGTH and len(full_text) > DEFAULT_MAX_LENGTH:
        full_text = full_text[:DEFAULT_MAX_LENGTH] + "\n\n... (truncated)"
        log_message(f"Обрезано до {DEFAULT_MAX_LENGTH} символов")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = LOG_DIR / f"fetched_{method_name}_{timestamp}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Метод: {method_name}\n")
        f.write(f"Извлечено: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(full_text)

    log_message(f"✅ Результат сохранён в файл: {out_file}")
    log_message(f"   Размер файла: {len(full_text)} символов")
    print(f"\nПросмотр первых 5000000 символов:\n{full_text[:500000]}...")


if __name__ == "__main__":
    asyncio.run(main())