"""
Response cleaner utility for LLM responses.
Cleans and formats LLM output for proper display in the UI.
"""
import re
import html


def clean_llm_response(text: str) -> str:
    """
    Clean a complete LLM response for display.
    Removes thinking tokens, extra whitespace, and formatting artifacts.
    Preserves HTML styling tags for styled responses.
    Fixes common HTML malformation issues.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Cleaned text ready for display
    """
    if not text:
        return ""
    
    # Unescape HTML entities (e.g., &lt; -> <, &gt; -> >) 
    # This fixes cases where LLM outputs HTML entities instead of actual tags
    text = html.unescape(text)
    
    # Remove thinking tokens (e.g., <think>...</think> from Qwen3)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove partial thinking tags at boundaries
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)
    
    # Fix malformed HTML tags - LLM sometimes generates invalid patterns
    # Fix <tag< pattern (should be <tag>)
    text = re.sub(r'<(p|div|span|ul|li|ol)<', r'<\1>', text)
    text = re.sub(r'</(p|div|span|ul|li|ol)<', r'</\1>', text)
    
    # Fix double angle brackets <<tag or tag>>
    text = re.sub(r'<<', '<', text)
    text = re.sub(r'>>', '>', text)
    
    # Fix missing closing bracket in tags like <p , <div , etc
    text = re.sub(r'<(p|div|span|ul|li|ol|br)\s+([^>])', r'<\1>\2', text)
    
    # Fix malformed HTML - remove orphan closing tags at start of lines
    # Pattern: line starts with text followed by </span> or </li> without opening tag
    text = re.sub(r'^[^<]*</span>', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[^<]*</li>', '', text, flags=re.MULTILINE)
    
    # Remove lines that are just orphan closing tags
    text = re.sub(r'^\s*</span>\s*</li>\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*number</span></li>\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up empty list items
    text = re.sub(r'<li>\s*</li>', '', text)
    
    # Remove excessive whitespace while preserving paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # CRITICAL FIX: Remove 4-space indentation which Markdown treats as code blocks
    # This prevents HTML from being rendered as raw text in code blocks
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    # Clean up leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_llm_chunk(chunk: str) -> str:
    """
    Clean an individual streaming chunk from LLM.
    Lighter cleaning than full response to preserve streaming continuity.
    
    Args:
        chunk: A streaming text chunk
        
    Returns:
        Cleaned chunk
    """
    if not chunk:
        return ""
    
    # Unescape HTML entities in chunks too
    chunk = html.unescape(chunk)
    
    # Remove thinking tokens that might appear in chunks
    chunk = re.sub(r'<think>.*?</think>', '', chunk, flags=re.DOTALL)
    
    # Remove partial thinking tags at chunk boundaries
    chunk = re.sub(r'<think>.*$', '', chunk)
    chunk = re.sub(r'^.*?</think>', '', chunk)
    
    return chunk

