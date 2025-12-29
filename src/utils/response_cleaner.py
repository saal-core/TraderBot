"""
Response cleaning utilities for LLM outputs.

Handles common formatting issues from various LLM models including:
- Thinking tokens (<think>...</think>) from Qwen3
- Raw LaTeX that doesn't render in Streamlit
- Special Unicode characters that may break rendering
- Excessive whitespace and formatting issues
"""

import re
from typing import Optional


def clean_llm_response(text: Optional[str]) -> str:
    """
    Clean LLM response for proper display in Streamlit.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Cleaned text suitable for Streamlit markdown rendering
    """
    if not text:
        return ""
    
    # 1. Remove thinking tokens (Qwen3 style: <think>...</think>)
    text = remove_thinking_tokens(text)
    
    # 2. Clean up LaTeX-style formatting that Streamlit doesn't handle well
    text = clean_latex_formatting(text)
    
    # 3. Fix markdown formatting issues
    text = fix_markdown_formatting(text)
    
    # 4. Normalize whitespace
    text = normalize_whitespace(text)
    
    # 5. Clean special characters that may cause rendering issues
    text = clean_special_characters(text)
    
    return text.strip()


def remove_thinking_tokens(text: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen3 and similar models.
    Also handles variations like [think], {think}, etc.
    """
    # Remove <think>...</think> blocks (multiline, including nested content)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove [think]...[/think] blocks
    text = re.sub(r'\[think\].*?\[/think\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove {think}...{/think} blocks  
    text = re.sub(r'\{think\}.*?\{/think\}', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any orphaned thinking markers
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?think\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{/?think\}', '', text, flags=re.IGNORECASE)
    
    return text


def clean_latex_formatting(text: str) -> str:
    """
    Convert LaTeX-style formatting to plain text or Streamlit-compatible markdown.
    """
    # Remove inline LaTeX markers that don't render: $...$ 
    # Keep simple math but remove complex LaTeX
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)  # Remove block LaTeX
    
    # Convert simple fractions \frac{a}{b} to a/b
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
    
    # Remove other LaTeX commands while keeping content
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)  # \textbf{x} -> **x**
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)    # \textit{x} -> *x*
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)        # \text{x} -> x
    
    # Remove remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Clean up leftover LaTeX artifacts
    text = re.sub(r'\$([^$]+)\$', r'\1', text)  # Remove inline math markers
    
    return text


def fix_markdown_formatting(text: str) -> str:
    """
    Fix common markdown formatting issues for Streamlit rendering.
    """
    # Fix inconsistent header spacing (ensure space after #)
    text = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', text, flags=re.MULTILINE)
    
    # Fix bullet points without proper spacing
    text = re.sub(r'^(\s*)[-*+]([^\s])', r'\1- \2', text, flags=re.MULTILINE)
    
    # Fix numbered lists without proper spacing
    text = re.sub(r'^(\s*)(\d+)\.\s*([^\s])', r'\1\2. \3', text, flags=re.MULTILINE)
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Ensure code blocks are properly formatted
    # Fix unclosed code blocks
    code_block_pattern = r'```(\w*)\n(.*?)(?=```|$)'
    matches = list(re.finditer(code_block_pattern, text, flags=re.DOTALL))
    
    # Count opening and closing code blocks
    opens = text.count('```') 
    if opens % 2 != 0:
        # Add missing closing block
        text = text + '\n```'
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving intentional formatting.
    """
    # Replace tabs with spaces
    text = text.replace('\t', '    ')
    
    # Remove trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    # Remove leading/trailing blank lines
    text = text.strip()
    
    return text


def clean_special_characters(text: str) -> str:
    """
    Clean special characters that may cause rendering issues.
    """
    # Replace problematic Unicode characters
    replacements = {
        '\u2028': '\n',      # Line separator
        '\u2029': '\n\n',    # Paragraph separator
        '\u00A0': ' ',       # Non-breaking space
        '\u200B': '',        # Zero-width space
        '\u200C': '',        # Zero-width non-joiner
        '\u200D': '',        # Zero-width joiner
        '\uFEFF': '',        # Zero-width no-break space (BOM)
        '→': '->',           # Arrow (keep readable)
        '←': '<-',
        '↔': '<->',
        '≥': '>=',
        '≤': '<=',
        '≠': '!=',
        '×': 'x',
        '÷': '/',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def format_for_streamlit(text: str, escape_html: bool = False) -> str:
    """
    Format text specifically for Streamlit markdown rendering.
    
    Args:
        text: Text to format
        escape_html: Whether to escape HTML tags
        
    Returns:
        Streamlit-ready text
    """
    text = clean_llm_response(text)
    
    if escape_html:
        # Escape HTML tags that might interfere with rendering
        # But preserve markdown formatting
        text = re.sub(r'<(?!/?(?:b|i|u|s|code|pre|br|hr)\b)[^>]+>', 
                      lambda m: m.group().replace('<', '&lt;').replace('>', '&gt;'), 
                      text)
    
    return text


def extract_clean_content(text: str, remove_sources: bool = False) -> str:
    """
    Extract the main content from LLM response, optionally removing source citations.
    
    Args:
        text: Raw response
        remove_sources: Whether to remove source/reference sections
        
    Returns:
        Clean main content
    """
    text = clean_llm_response(text)
    
    if remove_sources:
        # Remove common source/reference sections
        text = re.sub(r'\n\*\*Sources?:\*\*.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\n\*\*References?:\*\*.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\n---\s*\n.*$', '', text, flags=re.DOTALL)
    
    return text.strip()
