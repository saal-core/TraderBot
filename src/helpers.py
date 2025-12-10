"""Utility functions for text processing, language detection, and query operations."""
import re
import logging
import unicodedata
from typing import List, Dict
import ollama
from config.settings import get_settings
from config.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT,
    STANDALONE_QUESTION_SYSTEM_PROMPT,
    STANDALONE_QUESTION_USER_PROMPT,
)

settings = get_settings()


def detect_language(text: str) -> str:
    """
    Detect if text is Arabic or English.

    Args:
        text: Input text to analyze

    Returns:
        str: "Arabic" or "English"
    """
    is_arabic = any('\u0600' <= ch <= '\u06FF' for ch in text)
    return "Arabic" if is_arabic else "English"


def clean_text(text: str) -> str:
    """
    Cleans and normalizes text output from an LLM to fix common formatting issues
    like special Unicode characters and spacing errors.

    Args:
        text: Raw text from LLM

    Returns:
        str: Cleaned text
    """
    # 1. Normalize compatibility characters (turns '𝑎' into 'a', '𝑠' into 's', etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix common run-on words that occur after normalization
    spacing_fixes = {
        r'asof': 'as of',
        r'theclose': 'the close',
        r'oftrading': 'of trading',
        r'onNovember': 'on November',
        r'\.Thestock': '. The stock',
        r'isdown': 'is down',
    }

    for pattern, replacement in spacing_fixes.items():
        def repl(match):
            # Keep capitalization if original was capitalized
            if match.group(0)[0].isupper():
                return replacement.capitalize()
            return replacement
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 3. Collapse any leftover spaced-out characters (e.g., "a s o f")
    def collapse_spaced_chars(match):
        return re.sub(r'\s+', '', match.group(0))
    text = re.sub(r'((?:\b\w\b\s){2,})', collapse_spaced_chars, text)

    # 4. General whitespace cleanup
    text = re.sub(r' +', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple newlines

    return text.strip()


def is_text_broken(text: str) -> bool:
    """
    Detects if text is likely broken due to Unicode issues.

    Args:
        text: Text to check

    Returns:
        bool: True if text appears broken
    """
    # Check for letters separated by spaces
    if re.search(r"(?<=\w)\s+(?=\w)", text):
        return True

    # Check for uncommon Unicode letters outside ASCII
    for c in text:
        if ord(c) > 127 and c.isalpha():
            return True

    return False


def clean_if_broken(text: str) -> str:
    """
    Cleans text only if it's broken. Otherwise returns original text.

    Args:
        text: Text to check and clean

    Returns:
        str: Cleaned text if broken, otherwise original
    """
    if not is_text_broken(text):
        return text

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove Markdown formatting
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)       # Bold
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)          # Italics
    text = re.sub(r"`(.*?)`", r"\1", text)                # Inline code
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text) # Links

    # Remove weird spacing between letters
    text = re.sub(r"(?<=\w)\s+(?=\w)", "", text)

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_citations(text: str) -> str:
    """
    Remove citation markers like [1], [2] from text.

    Args:
        text: Text containing citations

    Returns:
        str: Text without citations
    """
    return re.sub(r'\[\d+\]', '', text)


def replace_wallet_with_portfolio(text: str) -> str:
    """
    Replace 'wallet' with 'portfolio' (case-insensitive).

    Args:
        text: Text containing 'wallet'

    Returns:
        str: Text with 'wallet' replaced by 'portfolio'
    """
    return re.sub(r'\bwallet\b', 'portfolio', text, flags=re.IGNORECASE)


def classify_query_type(question: str) -> str:
    """
    Classifies a user's question into 'portfolio', 'general', or 'other'
    using a robust prompt structure with rules, examples, and fallbacks.

    Args:
        question: User's question

    Returns:
        str: Classification result ('portfolio', 'general', or 'other')
    """
    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": CLASSIFICATION_USER_PROMPT.format(question=question)}
    ]

    try:
        response_text = ollama.chat(
            model=settings.ollama_model,
            messages=messages,
            options={"num_predict": 2000, "temperature": 0.5}
        )
        raw_output = response_text.message.content
        cleaned = raw_output.strip().lower()

        if "portfolio" in cleaned:
            return "portfolio"
        if "general" in cleaned:
            return "general"
        if "other" in cleaned:
            return "other"

        # Fallback to 'other' to prevent incorrect web searches
        logging.warning(f"Unrecognized classification output: '{raw_output}'. Defaulting to 'other'.")
        return "other"

    except Exception as e:
        logging.error(f"Error during classification: {e}. Defaulting to 'other'.")
        return "other"


def create_standalone_question(question: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Rephrases a follow-up question to be a complete, standalone question
    using chat history for context.

    Args:
        question: Follow-up question
        chat_history: List of previous messages

    Returns:
        str: Standalone version of the question
    """
    if not chat_history:
        return question

    # Build history string from last 4 messages
    history_str = ""
    for message in chat_history[-4:]:
        history_str += f"{message['role'].capitalize()}: {message['content']}\n"

    user_prompt = STANDALONE_QUESTION_USER_PROMPT.format(
        history=history_str,
        question=question
    )

    messages = [
        {"role": "system", "content": STANDALONE_QUESTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=messages,
            options={"temperature": 0.0, "num_predict": 100}
        )
        standalone_question = response['message']['content'].strip()
        return standalone_question
    except Exception as e:
        logging.error(f"Error creating standalone question: {e}. Using original question.")
        return question
