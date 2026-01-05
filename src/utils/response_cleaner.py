"""
Response cleaner utility for LLM responses.
Cleans and formats LLM output for proper display in the UI.
"""
import re


def clean_llm_response(text: str) -> str:
    """
    Clean a complete LLM response for display.
    Removes thinking tokens, extra whitespace, and formatting artifacts.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Cleaned text ready for display
    """
    if not text:
        return ""
    
    # Remove thinking tokens (e.g., <think>...</think> from Qwen3)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any remaining XML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace while preserving paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
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
    
    # Remove thinking tokens that might appear in chunks
    chunk = re.sub(r'<think>.*?</think>', '', chunk, flags=re.DOTALL)
    
    # Remove partial thinking tags at chunk boundaries
    chunk = re.sub(r'<think>.*$', '', chunk)
    chunk = re.sub(r'^.*?</think>', '', chunk)
    
    return chunk
