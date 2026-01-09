"""
TOON (Token-Oriented Object Notation) Formatter

A token-efficient data format for LLM prompts that reduces token usage by 30-60%
compared to JSON while maintaining clarity and structure.

Key Features:
- Indentation-based structure (like YAML) instead of braces/brackets
- Tabular format for arrays of uniform objects (CSV-like with headers)
- Smart quoting only when necessary
- Array length declarations for structure clarity

References:
- https://github.com/phdowellphd/TOON
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd


def _needs_quoting(value: str) -> bool:
    """
    Check if a string value needs quoting in TOON format.
    
    Quoting is needed for:
    - Strings containing commas
    - Strings with leading/trailing spaces
    - Empty strings
    - Strings that could be misinterpreted as numbers/booleans
    """
    if not value:
        return True
    if ',' in value or value != value.strip():
        return True
    # Check if it looks like a number or boolean
    if value.lower() in ('true', 'false', 'null', 'none'):
        return True
    try:
        float(value)
        return True  # It's a number, quote to preserve as string
    except ValueError:
        return False


def _format_value(value: Any) -> str:
    """Format a single value for TOON output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if _needs_quoting(value):
            # Escape quotes within the string
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        return value
    # For other types, convert to string
    return str(value)


def list_to_toon(items: List[Any], name: Optional[str] = None) -> str:
    """
    Convert a simple list to TOON format.
    
    Args:
        items: List of items (strings, numbers, etc.)
        name: Optional name for the list
        
    Returns:
        TOON formatted string
        
    Example:
        Input: ["AAPL", "MSFT", "GOOGL"]
        Output:
            symbols[3]
            AAPL
            MSFT
            GOOGL
    """
    if not items:
        return f"{name}[0]" if name else ""
    
    lines = []
    if name:
        lines.append(f"{name}[{len(items)}]")
    
    for item in items:
        lines.append(_format_value(item))
    
    return "\n".join(lines)


def dict_to_toon(data: Dict[str, Any], indent: int = 0) -> str:
    """
    Convert a dictionary to TOON format using indentation.
    
    Args:
        data: Dictionary to convert
        indent: Current indentation level
        
    Returns:
        TOON formatted string
        
    Example:
        Input: {"name": "A-Balanced", "ytd_return": 12.5, "value": 1500000}
        Output:
            name: A-Balanced
            ytd_return: 12.5
            value: 1500000
    """
    if not data:
        return ""
    
    indent_str = "  " * indent
    lines = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.append(dict_to_toon(value, indent + 1))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # List of dicts - use tabular format
                lines.append(f"{indent_str}{key}[{len(value)}]")
                lines.append(_list_of_dicts_to_toon(value, indent + 1))
            else:
                # Simple list
                lines.append(f"{indent_str}{key}[{len(value)}]")
                for item in value:
                    lines.append(f"{indent_str}  {_format_value(item)}")
        else:
            lines.append(f"{indent_str}{key}: {_format_value(value)}")
    
    return "\n".join(lines)


def _list_of_dicts_to_toon(items: List[Dict], indent: int = 0) -> str:
    """
    Convert a list of uniform dictionaries to TOON tabular format.
    
    This is where TOON really shines - instead of repeating keys for each object,
    we declare headers once and list values in CSV-like rows.
    
    Args:
        items: List of dictionaries with the same keys
        indent: Indentation level
        
    Returns:
        TOON tabular format string
        
    Example:
        Input: [{"symbol": "AAPL", "price": 150}, {"symbol": "MSFT", "price": 380}]
        Output:
            {symbol,price}
            AAPL,150
            MSFT,380
    """
    if not items:
        return ""
    
    indent_str = "  " * indent
    
    # Get all keys from first item (assuming uniform structure)
    keys = list(items[0].keys())
    
    lines = []
    # Header declaration
    lines.append(f"{indent_str}{{{','.join(keys)}}}")
    
    # Data rows
    for item in items:
        row_values = [_format_value(item.get(k, "")) for k in keys]
        lines.append(f"{indent_str}{','.join(row_values)}")
    
    return "\n".join(lines)


def dataframe_to_toon(df: pd.DataFrame, name: Optional[str] = None) -> str:
    """
    Convert a pandas DataFrame to TOON tabular format.
    
    This provides significant token savings over df.to_string() or JSON.
    
    Args:
        df: Pandas DataFrame to convert
        name: Optional name for the data (e.g., "results", "stocks")
        
    Returns:
        TOON formatted string
        
    Example:
        Input DataFrame:
            symbol  price   change
            AAPL    150.00  +2.5%
            MSFT    380.00  -1.2%
            
        Output:
            results[2]
            {symbol,price,change}
            AAPL,150.0,+2.5%
            MSFT,380.0,-1.2%
    """
    if df is None or df.empty:
        return f"{name}[0]" if name else "No data"
    
    lines = []
    
    # Array length declaration
    if name:
        lines.append(f"{name}[{len(df)}]")
    
    # Column headers
    columns = list(df.columns)
    lines.append(f"{{{','.join(columns)}}}")
    
    # Data rows
    for _, row in df.iterrows():
        row_values = [_format_value(row[col]) for col in columns]
        lines.append(",".join(row_values))
    
    return "\n".join(lines)


def to_toon(data: Any, name: Optional[str] = None) -> str:
    """
    Universal TOON converter - automatically detects data type and converts.
    
    Args:
        data: Any data structure (dict, list, DataFrame, or primitive)
        name: Optional name for the data structure
        
    Returns:
        TOON formatted string
        
    This is the main entry point for TOON conversion.
    """
    if data is None:
        return "No data"
    
    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        return dataframe_to_toon(data, name)
    
    # Handle list
    if isinstance(data, list):
        if not data:
            return f"{name}[0]" if name else ""
        # Check if it's a list of dicts (uniform objects)
        if isinstance(data[0], dict):
            header = f"{name}[{len(data)}]\n" if name else ""
            return header + _list_of_dicts_to_toon(data)
        # Simple list
        return list_to_toon(data, name)
    
    # Handle dict
    if isinstance(data, dict):
        if name:
            return f"{name}:\n{dict_to_toon(data, indent=1)}"
        return dict_to_toon(data)
    
    # Handle primitives
    return _format_value(data)


# Convenience function for common use case: formatting query results
def format_query_results(df: pd.DataFrame) -> str:
    """
    Format database query results for LLM consumption.
    
    Args:
        df: Query results DataFrame
        
    Returns:
        TOON formatted results ready for prompt injection
    """
    if df is None or df.empty:
        return "No results found"
    
    return dataframe_to_toon(df, "results")


def format_symbol_list(symbols: List[str]) -> str:
    """
    Format a list of stock symbols for LLM consumption.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        TOON formatted symbol list
    """
    return list_to_toon(symbols, "symbols")


def format_entity_list(portfolios: List[str] = None, accounts: List[str] = None) -> str:
    """
    Format portfolio names and account IDs for entity extraction.
    
    Args:
        portfolios: List of portfolio names
        accounts: List of account IDs
        
    Returns:
        TOON formatted entity lists
    """
    parts = []
    
    if portfolios:
        parts.append(list_to_toon(portfolios, "portfolios"))
    
    if accounts:
        parts.append(list_to_toon(accounts, "accounts"))
    
    return "\n\n".join(parts) if parts else "No entities available"
