"""
Schema Graph Service - SGAM-inspired schema knowledge representation.

This module provides a graph-based representation of the database schema
to assist with table selection, join path discovery, and column retrieval
for multi-table question answering.

Based on concepts from: "SGAM: Schema Graph-Assisted Multi-table QA"
"""

from typing import Dict, List, Set, Optional, Tuple


# =============================================================================
# SCHEMA GRAPH DATA STRUCTURE
# =============================================================================

SCHEMA_GRAPH = {
    # ---------------------------------------------------------------------
    # NODES: Table definitions with primary keys and semantic descriptions
    # ---------------------------------------------------------------------
    "nodes": {
        "portfolio_summary": {
            "type": "table",
            "schema": "ai_trading",
            "primary_keys": ["datetime", "portfolio_name"],
            "description": "Portfolio-level performance metrics, returns, and benchmarks",
            "use_cases": [
                "Portfolio performance and returns (YTD, MTD, QTD, all-time)",
                "Benchmark/index comparisons",
                "Portfolio metadata (description, group, cost model)",
                "Net liquidity and capital allocation"
            ]
        },
        "portfolio_holdings": {
            "type": "table",
            "schema": "ai_trading",
            "primary_keys": ["datetime", "portfolio_name", "symbol"],
            "description": "Individual stock holdings with position sizes and values over time",
            "use_cases": [
                "Current holdings and positions",
                "Historical trends of specific stocks",
                "Market value changes over time"
            ]
        },
        "portfolio_holdings_realized_pnl": {
            "type": "table",
            "schema": "ai_trading",
            "primary_keys": ["datetime", "portfolio_name", "symbol"],
            "description": "Detailed P&L breakdown per stock with realized and unrealized profits",
            "use_cases": [
                "Profit/Loss analysis (realized, unrealized, total)",
                "Top/bottom performing stocks",
                "Stock-level profitability"
            ]
        }
    },
    
    # ---------------------------------------------------------------------
    # EDGES: Join relationships between tables
    # ---------------------------------------------------------------------
    "edges": [
        {
            "from_table": "portfolio_holdings",
            "to_table": "portfolio_summary",
            "join_columns": [("portfolio_name", "portfolio_name")],
            "optional_join_columns": [("datetime", "datetime")],
            "relationship": "many-to-one",
            "description": "Holdings belong to a portfolio"
        },
        {
            "from_table": "portfolio_holdings_realized_pnl",
            "to_table": "portfolio_summary",
            "join_columns": [("portfolio_name", "portfolio_name")],
            "optional_join_columns": [("datetime", "datetime")],
            "relationship": "many-to-one",
            "description": "P&L records belong to a portfolio"
        },
        {
            "from_table": "portfolio_holdings_realized_pnl",
            "to_table": "portfolio_holdings",
            "join_columns": [
                ("portfolio_name", "portfolio_name"),
                ("symbol", "symbol")
            ],
            "optional_join_columns": [("datetime", "datetime")],
            "relationship": "one-to-one",
            "description": "P&L extends holdings with profit data"
        }
    ],
    
    # ---------------------------------------------------------------------
    # COLUMN SEMANTICS: Semantic groupings for attribute-level retrieval
    # ---------------------------------------------------------------------
    "column_semantics": {
        "profit_loss": {
            "keywords": ["profit", "loss", "pnl", "p&l", "gain", "money made", "earnings", "realized", "unrealized"],
            "columns": {
                "portfolio_holdings_realized_pnl": [
                    "ytd_realized_pnl", "ytd_unrealized_pnl", "ytd_total_pnl", "daily_realized_pnl"
                ],
                "portfolio_summary": [
                    "ytd_profit", "mtd_profit", "qtd_profit", "daily_profit", "all_profit"
                ]
            }
        },
        "returns": {
            "keywords": ["return", "performance", "growth", "gain percentage", "yield"],
            "columns": {
                "portfolio_summary": [
                    "ytd_return", "mtd_return", "qtd_return", "daily_return", 
                    "wtd_return", "all_return"
                ]
            }
        },
        "benchmark": {
            "keywords": ["benchmark", "index", "comparison", "s&p", "nasdaq", "market", "beat", "outperform"],
            "columns": {
                "portfolio_summary": [
                    "default_index", "ytd_index_return", "mtd_index_return",
                    "qtd_index_return", "daily_index_return", "all_index_return",
                    "index_annual_return"
                ]
            }
        },
        "positions": {
            "keywords": ["holdings", "shares", "quantity", "positions", "stocks owned", "own", "have"],
            "columns": {
                "portfolio_holdings": ["positions", "symbol", "market_value"],
                "portfolio_holdings_realized_pnl": ["positions", "symbol"]
            }
        },
        "value": {
            "keywords": ["value", "worth", "balance", "liquidity", "capital", "cash", "aum"],
            "columns": {
                "portfolio_summary": ["net_liquidity", "allocated_amount", "utilized_amount", "unrealized_pl"],
                "portfolio_holdings": ["market_value"],
                "portfolio_holdings_realized_pnl": ["market_value"]
            }
        },
        "identity": {
            "keywords": ["portfolio", "wallet", "account", "name", "group", "id"],
            "columns": {
                "portfolio_summary": ["portfolio_name", "account_id", "group_name", "user_id", "is_active", "cost_model"],
                "portfolio_holdings": ["portfolio_name", "symbol"],
                "portfolio_holdings_realized_pnl": ["portfolio_name", "symbol", "group_name"]
            }
        }
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_optimal_path(tables: List[str]) -> List[Dict]:
    """
    Find the optimal join path to connect a set of tables.
    Uses a simple BFS approach since the graph is small.
    
    Args:
        tables: List of table names to connect
        
    Returns:
        List of join edge definitions (dictionaries) to form the path.
        Returns empty list if single table or no path found.
    """
    if not tables or len(tables) < 2:
        return []

    # Sort tables to ensure deterministic ordering (e.g. summary first if present)
    # This helps with WHERE clause consistency
    sorted_tables = sorted(tables)
    
    # We only need to connect everything to the "primary" table (usually portfolio_summary or the largest hub)
    # The schema is currently star-like with portfolio_summary potentially as a hub, 
    # OR linear: pnl <-> summary <-> holdings
    
    # Let's try to connect each subsequent table to the first one defined in the list
    # For N tables, we need N-1 joins (minimum spanning tree)
    
    edges = []
    connected = {sorted_tables[0]}
    remaining = set(sorted_tables[1:])
    
    # Greedy approach: find direct connection from any connected node to any remaining node
    while remaining:
        progress = False
        for target in list(remaining):
            for source in list(connected):
                join = get_join_path(source, target)
                if join:
                    # Found a path!
                    if join["type"] in ["direct", "reverse"]:
                        edges.append(join["join"])
                    elif join["type"] == "two_hop":
                        # If it's a two hop, we might be introducing an intermediate table
                        # For simplicity in this v1, checking if we can just return the edges
                        edges.extend(join["path"])
                        
                    connected.add(target)
                    remaining.remove(target)
                    progress = True
                    break
            if progress:
                break
        
        if not progress:
            # Graph might be disconnected or complex path not supported yet
            print(f"⚠️ Warning: Could not connect tables {remaining} to {connected}")
            break
            
    return edges

def get_table_for_query(query: str) -> Tuple[str, str]:
    """
    Determine the best table to use based on query keywords.
    
    Args:
        query: Natural language query
        
    Returns:
        Tuple of (table_name, reason)
    """
    query_lower = query.lower()
    
    # Priority 1: P&L-specific queries → portfolio_holdings_realized_pnl
    pnl_keywords = ["profit", "loss", "pnl", "p&l", "realized", "unrealized", 
                    "top stock", "bottom stock", "best stock", "worst stock",
                    "most profitable", "losing stock", "performing stock", "performing asset"]
    if any(kw in query_lower for kw in pnl_keywords):
        return ("portfolio_holdings_realized_pnl", 
                "Query mentions profit/loss metrics - using P&L table")
    
    # Priority 2: Holdings/positions queries → portfolio_holdings
    holdings_keywords = ["holding", "position", "shares", "how many", "quantity", "own", "have"]
    if any(kw in query_lower for kw in holdings_keywords):
        # But check if it also needs P&L
        if "profit" in query_lower or "loss" in query_lower:
            return ("portfolio_holdings_realized_pnl",
                    "Query needs both holdings and P&L - using P&L table")
        return ("portfolio_holdings", 
                "Query about holdings/positions - using holdings table")
    
    # Priority 3: Portfolio performance → portfolio_summary
    performance_keywords = ["return", "performance", "benchmark", "index", 
                           "ytd", "mtd", "liquidity", "portfolio value", "balance", "worth"]
    if any(kw in query_lower for kw in performance_keywords):
        return ("portfolio_summary",
                "Query about portfolio performance - using summary table")
    
    # Default: portfolio_summary (safest default for general queries)
    return ("portfolio_summary", 
            "Default to summary table for general portfolio queries")


def get_join_path(source_table: str, target_table: str) -> Optional[Dict]:
    """
    Find the join path between two tables.
    
    Args:
        source_table: Starting table name
        target_table: Target table name
        
    Returns:
        Dict with join information or None if no path exists
    """
    for edge in SCHEMA_GRAPH["edges"]:
        # Check direct connection
        if edge["from_table"] == source_table and edge["to_table"] == target_table:
            return {
                "type": "direct",
                "join": edge
            }
        if edge["from_table"] == target_table and edge["to_table"] == source_table:
            return {
                "type": "reverse",
                "join": edge
            }
    
    # Check for path through intermediate table
    for edge1 in SCHEMA_GRAPH["edges"]:
        if edge1["from_table"] == source_table:
            intermediate = edge1["to_table"]
            for edge2 in SCHEMA_GRAPH["edges"]:
                if edge2["from_table"] == intermediate and edge2["to_table"] == target_table:
                    return {
                        "type": "two_hop",
                        "path": [edge1, edge2]
                    }
    
    return None


def get_columns_for_semantic(semantic_category: str, table_name: str = None) -> List[str]:
    """
    Get column names for a semantic category.
    
    Args:
        semantic_category: Category like "profit_loss", "returns", etc.
        table_name: Optional filter by table
        
    Returns:
        List of column names
    """
    if semantic_category not in SCHEMA_GRAPH["column_semantics"]:
        return []
    
    category = SCHEMA_GRAPH["column_semantics"][semantic_category]
    columns = category.get("columns", {})
    
    if table_name:
        return columns.get(table_name, [])
    
    # Return all columns across tables
    all_columns = []
    for table_cols in columns.values():
        all_columns.extend(table_cols)
    return list(set(all_columns))


def match_query_to_semantics(query: str) -> List[str]:
    """
    Match a query to semantic categories based on keywords.
    
    Args:
        query: Natural language query
        
    Returns:
        List of matching semantic categories
    """
    query_lower = query.lower()
    matches = []
    
    for category, info in SCHEMA_GRAPH["column_semantics"].items():
        keywords = info.get("keywords", [])
        if any(kw in query_lower for kw in keywords):
            matches.append(category)
    
    return matches


def generate_join_sql(source_table: str, target_table: str) -> Optional[str]:
    """
    Generate SQL JOIN clause between two tables.
    
    Args:
        source_table: Source table name
        target_table: Target table name
        
    Returns:
        SQL JOIN clause string or None
    """
    join_path = get_join_path(source_table, target_table)
    if not join_path:
        return None
    
    schema = "ai_trading"
    
    if join_path["type"] == "direct":
        edge = join_path["join"]
        conditions = [
            f"{schema}.{source_table}.{src} = {schema}.{target_table}.{tgt}"
            for src, tgt in edge["join_columns"]
        ]
        return f"JOIN {schema}.{target_table} ON " + " AND ".join(conditions)
    
    elif join_path["type"] == "reverse":
        edge = join_path["join"]
        conditions = [
            f"{schema}.{source_table}.{tgt} = {schema}.{target_table}.{src}"
            for src, tgt in edge["join_columns"]
        ]
        return f"JOIN {schema}.{target_table} ON " + " AND ".join(conditions)
    
    return None


# =============================================================================
# TABLE SELECTION DECISION TREE (for prompt injection)
# =============================================================================

TABLE_SELECTION_GUIDANCE = """
### Table Selection Decision Tree

Follow this logic to select the correct table:

1. **Does the query mention "profit", "loss", "P&L", "realized", "unrealized", or ask about stock-level profitability?**
   → Use `ai_trading.portfolio_holdings_realized_pnl`
   - Contains: ytd_total_pnl, ytd_realized_pnl, ytd_unrealized_pnl, daily_realized_pnl
   - Best for: "most profitable stock", "biggest losses", "top gainers"

2. **Does the query ask about portfolio returns, performance, or benchmarks?**
   → Use `ai_trading.portfolio_summary`
   - Contains: ytd_return, mtd_return, all_return, default_index, *_index_return
   - Best for: "portfolio performance", "compare to benchmark", "YTD returns"

3. **Does the query need BOTH portfolio metrics AND stock-level P&L?**
   → JOIN: portfolio_summary ↔ portfolio_holdings_realized_pnl
   → ON: portfolio_name (AND datetime for latest snapshot)
   - Example: "P&L breakdown for portfolios with positive returns"

4. **Does the query ask about historical trends of specific holdings?**
   → Use `ai_trading.portfolio_holdings` with ORDER BY datetime
   - Contains: datetime, portfolio_name, symbol, positions, market_value
   - Best for: "AAPL value over time", "position history"
"""


if __name__ == "__main__":
    # Quick test
    test_queries = [
        "What is my top profitable stock?",
        "Show my portfolio returns YTD",
        "How many shares of AAPL do I have?",
        "Compare my portfolio to S&P 500"
    ]
    
    print("Schema Graph Table Selection Test:")
    print("=" * 50)
    for q in test_queries:
        table, reason = get_table_for_query(q)
        print(f"\nQuery: {q}")
        print(f"  → Table: {table}")
        print(f"  → Reason: {reason}")
