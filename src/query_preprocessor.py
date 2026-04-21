#!/usr/bin/env python3
"""
Query Preprocessor for AI_EveryNyan.
Provides lemmatization for indexing (RAG) and optional stopword removal.

\src\query_preprocessor.py
Version:     0.2.0
Author:      Sorol L.'.L.'.
Updated:     2026-04-21

Patch Notes v0.2.0:
  [+] Added lemmatize_text() method for indexing (preserves stopwords by default).
  [+] process_query() now defaults remove_stopwords=False for indexing.
  [*] Better language detection and error handling.

v0.1.0: Initial implementation with lemmatization and stopword removal.
"""

import logging
import re
from typing import Optional, Callable, Tuple, List

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger("AI_EveryNyan.QueryPreprocessor")


class QueryPreprocessor:
    """
    Preprocess text for RAG indexing and search.
    - For indexing: use lemmatize_text() to create a normalized copy (preserves stopwords).
    - For search queries: use process_query() with remove_stopwords=False (or True if needed).
    """
    
    # Minimal stopword list (only most common noise words)
    # Used only when remove_stopwords=True
    RUSSIAN_STOPWORDS = {
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'а',
        'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
        'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
        'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
        'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'был', 'до', 'вас', 'ни',
        'быть', 'они', 'нее', 'со', 'дня', 'это', 'чем', 'тоже', 'себя',
        'вам', 'была', 'чем', 'этом', 'этот', 'через', 'нее', 'весь'
    }
    
    def __init__(self, 
                 add_thought_callback: Optional[Callable[[str, Tuple[int,int,int]], None]] = None):
        self.add_thought = add_thought_callback
        self.nlp_ru = None
        self.nlp_en = None
        
        if not SPACY_AVAILABLE:
            self._log("[PREPROC] WARNING: spaCy not installed. Lemmatization disabled.", (255,200,100))
            return
        
        try:
            self.nlp_ru = spacy.load("ru_core_news_sm")
            self._log("[PREPROC] Russian model 'ru_core_news_sm' loaded.", (150,255,150))
        except OSError:
            self._log("[PREPROC] Russian model not found. Run: python -m spacy download ru_core_news_sm", (255,150,150))
        except Exception as e:
            self._log(f"[PREPROC] Failed to load Russian model: {e}", (255,100,100))
        
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            self._log("[PREPROC] English model 'en_core_web_sm' loaded.", (150,255,150))
        except OSError:
            self._log("[PREPROC] English model not found. Run: python -m spacy download en_core_web_sm", (255,150,150))
        except Exception as e:
            self._log(f"[PREPROC] Failed to load English model: {e}", (255,100,100))
    
    def _log(self, text: str, color: Tuple[int, int, int] = (200,200,150)):
        logger.info(f"[PREPROC] {text}")
        if self.add_thought:
            try:
                self.add_thought(text, color)
            except Exception:
                pass
    
    def _detect_language(self, text: str) -> str:
        if not text:
            return 'unknown'
        if any('\u0400' <= ch <= '\u04FF' for ch in text):
            return 'ru'
        if any('a' <= ch.lower() <= 'z' for ch in text):
            return 'en'
        return 'unknown'
    
    def lemmatize_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Lemmatize text without altering its structure.
        Used for indexing (creates a normalized copy).
        
        Args:
            text: Input string
            remove_stopwords: If True, removes common stopwords (default False for indexing)
        """
        if not text or not isinstance(text, str):
            return ""
        
        lang = self._detect_language(text)
        nlp = None
        if lang == 'ru':
            nlp = self.nlp_ru
        elif lang == 'en':
            nlp = self.nlp_en
        
        if nlp is None:
            return text
        
        try:
            doc = nlp(text.lower())
            lemmas = []
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                if remove_stopwords:
                    is_stop = token.is_stop
                    if lang == 'ru' and token.text in self.RUSSIAN_STOPWORDS:
                        is_stop = True
                    if is_stop:
                        continue
                lemma = token.lemma_.strip()
                if lemma:
                    lemmas.append(lemma)
            result = " ".join(lemmas)
            return result if result else text.lower()
        except Exception as e:
            self._log(f"[PREPROC] Lemmatization error: {e}", (255,100,100))
            return text
    
    def process_query(self, query: str, remove_stopwords: bool = False) -> str:
        """
        Preprocess a user query for RAG search.
        By default, does NOT remove stopwords to preserve context.
        """
        if not query:
            return ""
        processed = self.lemmatize_text(query, remove_stopwords=remove_stopwords)
        if processed != query:
            self._log(f"[PREPROC] Query: '{query[:50]}...' -> '{processed[:50]}...'", (180,220,200))
        return processed


if __name__ == "__main__":
    def test_callback(text, color):
        print(f"[TEST] {text}")
    pp = QueryPreprocessor(add_thought_callback=test_callback)
    test = "Так, я вернулась, твоя Линда! :) На чем мы остановились?"
    print(f"Original: {test}")
    print(f"Lemmatized (no stopwords): {pp.lemmatize_text(test, remove_stopwords=True)}")
    print(f"Lemmatized (preserve stopwords): {pp.lemmatize_text(test, remove_stopwords=False)}")