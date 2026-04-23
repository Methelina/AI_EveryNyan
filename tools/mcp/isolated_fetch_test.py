#!/usr/bin/env python3
"""
isolated_fetch_test.py
Тест извлечения контента с URL с выбором метода:
  1) Комбинированный (trafilatura + bs4 fallback) → очистка bs4 перед trafilatura
  2) Только bs4 + markdownify (без trafilatura)
  3) Оригинальный (trafilatura на сыром HTML + markdownify fallback) – версия 0.1.0

Использование:
    python isolated_fetch_test.py <URL> [method]

    method: 1, 2 или 3 (по умолчанию 1)
    Если method не указан, скрипт предложит выбрать интерактивно.

Результат сохраняется в файл с именем, содержащим название метода.
"""

import os
import sys
import asyncio
import httpx
from datetime import datetime

# Попытка импорта необходимых библиотек
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[ERROR] BeautifulSoup4 не установлена. Установите: pip install beautifulsoup4")
    sys.exit(1)

try:
    from markdownify import markdownify as md
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    print("[ERROR] markdownify не установлена. Установите: pip install markdownify")
    sys.exit(1)


# ----------------------------------------------------------------------
# Общая вспомогательная функция очистки bs4
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
# Метод 1: Комбинированный (trafilatura + bs4 fallback)
# ----------------------------------------------------------------------
async def _fallback_bs4(cleaned_html: str) -> str:
    """Fallback: извлечение через bs4.get_text() + markdownify."""
    try:
        soup = BeautifulSoup(cleaned_html, 'lxml' if 'lxml' in sys.modules else 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        if text.strip():
            print(f"  ✅ Извлечено через bs4.get_text(): {len(text)} символов")
            return text
        else:
            raise ValueError("Пустой текст")
    except Exception as e:
        print(f"  ⚠️ bs4.get_text() не сработал ({e}), пробуем markdownify")
        text = md(cleaned_html, strip=["img", "script", "style", "nav", "footer", "header"])
        text = "\n".join(line for line in text.splitlines() if line.strip())
        if not text.strip():
            raise RuntimeError("Не удалось извлечь содержимое страницы.")
        print(f"  ✅ Извлечено через markdownify: {len(text)} символов")
        return text


async def fetch_method_combined(url: str) -> str:
    """Метод 1: trafilatura + bs4 fallback (с предварительной очисткой bs4)."""
    print(f"\n=== Загрузка: {url} (метод 1: комбинированный) ===")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    print(f"  Загружено HTML: {len(html)} символов")

    cleaned_html = clean_html_with_bs4(html)
    print(f"  Очищено HTML (bs4): {len(cleaned_html)} символов")

    if not TRAFILATURA_AVAILABLE:
        print("  ⚠️ trafilatura не установлена, сразу переходим к bs4 fallback")
        return await _fallback_bs4(cleaned_html)

    downloaded = trafilatura.bare_extraction(
        cleaned_html,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        favor_precision=True,
    )

    if downloaded and downloaded.text:
        text = downloaded.text
        meta = []
        if downloaded.title:
            meta.append(f"# {downloaded.title}")
        if downloaded.author:
            meta.append(f"Author: {downloaded.author}")
        if downloaded.date:
            meta.append(f"Date: {downloaded.date}")
        if meta:
            text = "\n".join(meta) + "\n\n" + text
        print(f"  ✅ Извлечено через trafilatura: {len(text)} символов")
        return text
    else:
        print("  ⚠️ trafilatura не дала результата, используем bs4 fallback")
        return await _fallback_bs4(cleaned_html)


# ----------------------------------------------------------------------
# Метод 2: Только bs4 + markdownify (без trafilatura)
# ----------------------------------------------------------------------
async def fetch_method_bs4_md(url: str) -> str:
    """Метод 2: только bs4 очистка и markdownify (без trafilatura)."""
    print(f"\n=== Загрузка: {url} (метод 2: bs4 + markdownify) ===")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    print(f"  Загружено HTML: {len(html)} символов")

    cleaned_html = clean_html_with_bs4(html)
    print(f"  Очищено HTML (bs4): {len(cleaned_html)} символов")

    text = md(cleaned_html, strip=["img", "script", "style", "nav", "footer", "header", "aside"])
    text = "\n".join(line for line in text.splitlines() if line.strip())
    if not text.strip():
        print("  ⚠️ markdownify вернул пустоту, пробуем bs4.get_text()")
        try:
            soup = BeautifulSoup(cleaned_html, 'lxml' if 'lxml' in sys.modules else 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            raise RuntimeError("Не удалось извлечь содержимое страницы.")

    print(f"  ✅ Извлечено: {len(text)} символов")
    return text


# ----------------------------------------------------------------------
# Метод 3: Оригинальный (trafilatura на сыром HTML + markdownify fallback)
# ----------------------------------------------------------------------
async def fetch_method_original(url: str) -> str:
    """Метод 3: оригинальная версия 0.1.0 – trafilatura на сыром HTML, без bs4 очистки."""
    print(f"\n=== Загрузка: {url} (метод 3: оригинальный, без bs4) ===")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    print(f"  Загружено HTML: {len(html)} символов")

    if not TRAFILATURA_AVAILABLE:
        print("  ⚠️ trafilatura не установлена, падаем на markdownify")
        text = md(html, strip=["img", "script", "style"])
        text = "\n".join(line for line in text.splitlines() if line.strip())
        if not text.strip():
            raise RuntimeError("Не удалось извлечь содержимое страницы.")
        print(f"  ✅ Извлечено через markdownify: {len(text)} символов")
        return text

    downloaded = trafilatura.bare_extraction(
        html,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        favor_precision=True,
    )

    if downloaded and downloaded.text:
        text = downloaded.text
        meta = []
        if downloaded.title:
            meta.append(f"# {downloaded.title}")
        if downloaded.author:
            meta.append(f"Author: {downloaded.author}")
        if downloaded.date:
            meta.append(f"Date: {downloaded.date}")
        if meta:
            text = "\n".join(meta) + "\n\n" + text
        print(f"  ✅ Извлечено через trafilatura: {len(text)} символов")
        return text
    else:
        print("  ⚠️ trafilatura не дала результата, падаем на markdownify")
        text = md(html, strip=["img", "script", "style"])
        text = "\n".join(line for line in text.splitlines() if line.strip())
        if not text.strip():
            raise RuntimeError("Не удалось извлечь содержимое страницы.")
        print(f"  ✅ Извлечено через markdownify: {len(text)} символов")
        return text


# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
async def main():
    if len(sys.argv) < 2:
        print("Использование: python isolated_fetch_test.py <URL> [method]")
        print("  method: 1 - комбинированный (trafilatura + bs4 fallback)")
        print("          2 - только bs4 + markdownify")
        print("          3 - оригинальный (trafilatura на сыром HTML + markdownify)")
        print("  Если method не указан, будет запрошен интерактивно.")
        sys.exit(1)

    url = sys.argv[1]

    if len(sys.argv) >= 3:
        method_choice = sys.argv[2]
    else:
        print("\nВыберите метод извлечения:")
        print("  1 - Комбинированный (trafilatura + bs4 fallback)")
        print("  2 - Только bs4 + markdownify")
        print("  3 - Оригинальный (trafilatura на сыром HTML + markdownify)")
        method_choice = input("Ваш выбор (1/2/3): ").strip()

    if method_choice == "1":
        full_text = await fetch_method_combined(url)
        method_name = "combined_trafilatura_bs4"
    elif method_choice == "2":
        full_text = await fetch_method_bs4_md(url)
        method_name = "bs4_markdownify_only"
    elif method_choice == "3":
        full_text = await fetch_method_original(url)
        method_name = "original_trafilatura_md"
    else:
        print("Неверный выбор. Используйте 1, 2 или 3.")
        sys.exit(1)

    # Сохраняем в файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"fetched_{method_name}_{timestamp}.txt"
    out_file = os.path.abspath(out_file)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Метод: {method_name}\n")
        f.write(f"Извлечено: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(full_text)

    print(f"\n✅ Результат сохранён в файл: {out_file}")
    print(f"   Размер файла: {len(full_text)} символов")
    print(f"   Просмотр первых 500 символов:\n{full_text[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())