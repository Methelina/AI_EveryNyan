"""
MCP server providing web search (via SearXNG) and URL content extraction tools.
Exposes two tools: web_search for meta-search, fetch_url for page content retrieval.

\\tools\\mcp\\tool_searxng.py

Version:     0.3.0
Author:      pytraveler
Updated:     2026-04-23

Patch Notes v0.3.0 (by Soror L.'.L.'):
  [CHANGE] fetch_url полностью переписан на метод "bs4 + markdownify" (без trafilatura).
  [+] fetch_url: предварительная очистка HTML через BeautifulSoup (удаление script/style/nav/footer/...).
  [+] fetch_url: преобразование очищенного HTML в Markdown через markdownify.
  [+] fetch_url: добавлен краткий вывод в stderr (URL и размер извлечённого текста).
  [REMOVED] trafilatura больше не используется (оказался менее стабильным на сложных сайтах).
  [+] Конфигурационные переменные вынесены в начало файла (лимиты, таймауты, пути).
"""

import os
import sys
import httpx
from datetime import datetime
from fastmcp import FastMCP

# ============================================================================
# КОНФИГУРАЦИЯ – все настройки в одном месте
# ============================================================================

# Лимиты для fetch_url (размер извлекаемого текста)
DEFAULT_MAX_FETCH_LENGTH = 500000          # Максимальное количество символов по умолчанию (можно увеличить до 50000)
MAX_FETCH_LENGTH_GLOBAL = 500000          # Абсолютный максимум, если пользователь запросит больше (защита)

# Лимиты для web_search
DEFAULT_MAX_SEARCH_RESULTS = 10          # Количество результатов поиска по умолчанию
MAX_SEARCH_RESULTS_GLOBAL = 50           # Абсолютный максимум результатов

# Таймауты
HTTP_TIMEOUT = 30                        # Таймаут HTTP-запросов в секундах

# Пути и логирование
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:2597")
DEBUG_LOG = os.path.join(os.path.dirname(__file__), "../../logs/mcp_debug.log")
DEBUG_LOG = os.path.abspath(DEBUG_LOG)

# Настройки очистки HTML (теги, которые удаляются)
TAGS_TO_REMOVE = ['script', 'style', 'meta', 'link', 'noscript',
                  'header', 'footer', 'nav', 'aside', 'form', 'button',
                  'iframe', 'svg', 'canvas', 'command', 'embed', 'object']

# Список тегов для `markdownify(strip=...)`
MD_STRIP_TAGS = ['img', 'script', 'style', 'nav', 'footer', 'header', 'aside']

# ============================================================================

# Проверка зависимостей
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[MCP] WARNING: BeautifulSoup4 not installed. fetch_url will be limited.", file=sys.stderr)

try:
    from markdownify import markdownify as md
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    print("[MCP] WARNING: markdownify not installed. fetch_url will be limited.", file=sys.stderr)

mcp = FastMCP("searxng")

# Создаём папку для логов, если её нет
os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)

def log_debug(msg: str):
    """Безопасно пишем в файл, не трогая stdout/stderr."""
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")

# ----------------------------------------------------------------------
# Вспомогательная функция очистки HTML (bs4)
# ----------------------------------------------------------------------
def clean_html_with_bs4(html: str) -> str:
    """Удаляет шумные теги, возвращает очищенный HTML."""
    if not BS4_AVAILABLE:
        return html
    try:
        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')

        for tag in TAGS_TO_REMOVE:
            for t in soup.find_all(tag):
                t.decompose()

        # Удаляем пустые элементы (кроме br, p, hr)
        for tag in soup.find_all():
            if len(tag.get_text(strip=True)) == 0 and tag.name not in ['br', 'p', 'hr']:
                tag.decompose()

        return str(soup)
    except Exception as e:
        log_debug(f"Ошибка в clean_html_with_bs4: {e}")
        return html


# ----------------------------------------------------------------------
# Инструменты MCP
# ----------------------------------------------------------------------

@mcp.tool()
async def web_search(
    query: str,
    categories: str = "general",
    language: str = "auto",
    max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
) -> str:
    """Meta-search the web via SearXNG and return markdown-formatted results."""
    log_debug(f"=== web_search called ===")
    log_debug(f"Query: {query}")
    log_debug(f"Categories: {categories}, language: {language}, max_results: {max_results}")

    # Защита от слишком больших запросов
    if max_results > MAX_SEARCH_RESULTS_GLOBAL:
        max_results = MAX_SEARCH_RESULTS_GLOBAL
        log_debug(f"max_results ограничено до {MAX_SEARCH_RESULTS_GLOBAL}")

    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "language": language,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])[:max_results]
    log_debug(f"Found {len(results)} raw results")

    if not results:
        log_debug("No results, returning empty")
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"### {i}. {title}")
        lines.append(f"**URL:** {url}")
        if snippet:
            lines.append(f"\n{snippet}")
        lines.append("")

    formatted = "\n".join(lines)
    log_debug(f"Formatted result length: {len(formatted)} characters")
    log_debug(f"Preview: {formatted[:500]}...")
    log_debug("=== web_search finished ===\n")
    return formatted


@mcp.tool()
async def fetch_url(url: str, max_length: int = DEFAULT_MAX_FETCH_LENGTH) -> str:
    """
    Fetch a URL, clean it with BeautifulSoup, and return markdown.
    Method: bs4 pre‑cleaning + markdownify (без trafilatura).
    """
    log_debug("=== fetch_url called ===")
    log_debug(f"URL: {url}, max_length: {max_length}")

    # Ограничиваем max_length глобальным максимумом
    if max_length > MAX_FETCH_LENGTH_GLOBAL:
        max_length = MAX_FETCH_LENGTH_GLOBAL
        log_debug(f"max_length ограничено до {MAX_FETCH_LENGTH_GLOBAL}")

    # 1. Загрузка HTML
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    log_debug(f"Downloaded HTML: {len(html)} characters")

    # 2. Очистка через bs4
    cleaned_html = clean_html_with_bs4(html)
    log_debug(f"Cleaned HTML (bs4): {len(cleaned_html)} characters")

    if not MD_AVAILABLE:
        log_debug("markdownify not installed, returning raw cleaned text with fallback")
        try:
            soup = BeautifulSoup(cleaned_html, 'lxml' if BS4_AVAILABLE else 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        except Exception:
            text = "Error: markdownify missing and BeautifulSoup extraction failed."
    else:
        # 3. Преобразование в markdown с агрессивным удалением оставшихся тегов
        text = md(cleaned_html, strip=MD_STRIP_TAGS)
        text = "\n".join(line for line in text.splitlines() if line.strip())

        # 4. Если результат пустой – пробуем прямой текст через bs4
        if not text.strip() and BS4_AVAILABLE:
            log_debug("markdownify gave empty result, falling back to bs4.get_text()")
            try:
                soup = BeautifulSoup(cleaned_html, 'lxml' if 'lxml' in sys.modules else 'html.parser')
                raw_text = soup.get_text(separator='\n', strip=True)
                text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
                log_debug(f"bs4.get_text() fallback gave {len(text)} chars")
            except Exception as e:
                log_debug(f"bs4.get_text() fallback failed: {e}")

    # 5. Обрезаем при необходимости
    if len(text) > max_length:
        text = text[:max_length] + "\n\n... (truncated)"
        log_debug(f"Truncated to {max_length} characters")

    log_debug(f"Final length: {len(text)} characters")
    log_debug(f"Preview: {text[:500]}...")
    log_debug("=== fetch_url finished ===\n")

    # Короткое сообщение в stderr для мониторинга в реальном времени
    print(f"[MCP] fetch_url: {url} -> extracted {len(text)} chars", file=sys.stderr, flush=True)

    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")