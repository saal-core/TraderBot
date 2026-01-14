"""
Query Decomposition Service - SGAM Phase 2

Decomposes natural language queries into structured components:
- Focus: What the user wants to know (SELECT targets)
- Constraints: Filters and conditions (WHERE clauses)
- Aggregations: SUM, AVG, COUNT, etc.
- Complexity: Single-table vs multi-table detection

Based on SGAM paper concepts for improved SQL generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QueryDecomposition:
    """Structured representation of a decomposed query."""
    
    # What to SELECT
    focus_columns: List[str] = field(default_factory=list)
    focus_description: str = ""
    
    # WHERE conditions
    constraints: Dict[str, any] = field(default_factory=dict)
    
    # Aggregation functions needed
    aggregations: List[str] = field(default_factory=list)
    
    # Sorting
    sort_direction: Optional[str] = None  # "ASC" or "DESC"
    sort_column: Optional[str] = None
    
    # Limit
    limit: Optional[int] = None
    
    # Table recommendation
    primary_table: Optional[str] = None
    secondary_tables: List[str] = field(default_factory=list)
    
    # Complexity classification
    complexity: str = "single"  # "single", "multi", "complex"
    
    # Original query
    original_query: str = ""
    
    # --- NEW SGAM FIELDS ---
    # Required tables identified from schema graph
    required_tables: List[str] = field(default_factory=list)
    
    # Join path edges (list of join definitions from schema_graph)
    join_path: List[Dict] = field(default_factory=list)
    
    # Query pattern classification for optimal response structure
    query_pattern: str = "snapshot"  # "snapshot", "trend", "aggregate", "lookup"
    
    # LLM generation guidance - specific instructions for SQL construction
    generation_guidance: str = ""
    
    def __str__(self) -> str:
        return (
            f"QueryDecomposition(\n"
            f"  focus: {self.focus_description}\n"
            f"  columns: {self.focus_columns}\n"
            f"  constraints: {self.constraints}\n"
            f"  aggregations: {self.aggregations}\n"
            f"  tables: {self.primary_table} + {self.secondary_tables}\n"
            f"  complexity: {self.complexity}\n"
            f"  pattern: {self.query_pattern}\n"
            f"  guidance: {self.generation_guidance[:100]}...\n"
            f")"
        )


# =============================================================================
# KEYWORD PATTERNS
# =============================================================================

# Patterns that indicate specific constraints
CONSTRAINT_PATTERNS = {
    # Temporal constraints
    "temporal": {
        "ytd": "year-to-date",
        "mtd": "month-to-date", 
        "qtd": "quarter-to-date",
        "daily": "daily",
        "today": "daily",
        "this year": "ytd",
        "this month": "mtd",
        "this quarter": "qtd",
        "since inception": "all-time",
        "all-time": "all-time",
        "all time": "all-time"
    },
    
    # Ranking constraints
    "ranking": {
        "top": ("DESC", "LIMIT"),
        "bottom": ("ASC", "LIMIT"),
        "highest": ("DESC", "LIMIT 1"),
        "lowest": ("ASC", "LIMIT 1"),
        "best": ("DESC", "LIMIT"),
        "worst": ("ASC", "LIMIT"),
        "most": ("DESC", "LIMIT"),
        "least": ("ASC", "LIMIT")
    },
    
    # Value filters
    "value_filter": {
        "positive": "> 0",
        "negative": "< 0",
        "above": ">",
        "below": "<",
        "greater than": ">",
        "less than": "<",
        "more than": ">",
        "at least": ">=",
        "exceeding": ">"
    },
    
    # Status filters
    "status": {
        "active": {"is_active": 1},
        "inactive": {"is_active": 0}
    },
    
    # Asset group filters - DISABLED (group_name is Portfolio Group, not Asset Class)
    # "asset_group": {
    #     "crypto": {"group_name": "Crypto"},
    #     "equity": {"group_name": "Equity"},
    #     "etf": {"group_name": "ETF"},
    #     "stock": {"group_name": "Equity"}
    # }
}

# Focus keywords - what type of data is requested
FOCUS_KEYWORDS = {
    # P&L focus → portfolio_holdings_realized_pnl
    "profit_loss": [
        "profit", "loss", "pnl", "p&l", "profitable", 
        "realized", "unrealized", "gains", "losses",
        "earning", "money made", "losing money"
    ],
    
    # Returns focus → portfolio_summary
    "returns": [
        "return", "performance", "performing", "growth",
        "ytd return", "mtd return", "all-time return"
    ],
    
    # Benchmark focus → portfolio_summary
    "benchmark": [
        "benchmark", "index", "compare", "comparison",
        "outperforming", "underperforming", "beat", "beating",
        "s&p", "nasdaq", "qqq"
    ],
    
    # Holdings focus → portfolio_holdings
    "holdings": [
        "holdings", "positions", "shares", "quantity",
        "how many", "stocks held", "own"
    ],
    
    # Value focus → portfolio_summary or portfolio_holdings
    "value": [
        "value", "worth", "balance", "liquidity",
        "capital", "aum", "net liquidity", "market value"
    ],
    
    # Portfolio identity → portfolio_summary
    "identity": [
        "portfolio", "portfolios", "wallet", "wallets",
        "account", "accounts", "group"
    ]
}

# Aggregation keywords
AGGREGATION_KEYWORDS = {
    "total": "SUM",
    "sum": "SUM",
    "average": "AVG",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "how many": "COUNT",
    "number of": "COUNT",
    "maximum": "MAX",
    "max": "MAX",
    "minimum": "MIN",
    "min": "MIN"
}

# Table mapping based on focus
FOCUS_TO_TABLE = {
    "profit_loss": "portfolio_holdings_realized_pnl",
    "returns": "portfolio_summary",
    "benchmark": "portfolio_summary",
    "holdings": "portfolio_holdings",
    "value": "portfolio_summary",
    "identity": "portfolio_summary"
}

# =============================================================================
# QUERY PATTERNS - Semantic patterns for optimal SQL structure
# =============================================================================

QUERY_PATTERNS = {
    "trend": {
        "keywords": ["over time", "trend", "history", "historical", "growth", "change", "evolution"],
        "guidance": "SELECT datetime along with the requested metrics. Do NOT filter for MAX(datetime). ORDER BY datetime ASC for chronological order.",
        "sql_template": "SELECT datetime, {columns} FROM {table} WHERE portfolio_name = ? ORDER BY datetime"
    },
    "snapshot": {
        "keywords": ["current", "now", "today", "latest", "what is", "show me", "list"],
        "guidance": "Filter for the latest snapshot using: WHERE datetime = (SELECT MAX(datetime) FROM {table} WHERE is_active = 1). This ensures current data.",
        "sql_template": "SELECT {columns} FROM {table} WHERE datetime = (SELECT MAX(datetime) FROM {table})"
    },
    "aggregate": {
        "keywords": ["total", "sum", "average", "count", "how many", "overall"],
        "guidance": "Use aggregation functions (SUM, AVG, COUNT). Filter for latest snapshot first, then aggregate. GROUP BY if comparing across entities.",
        "sql_template": "SELECT SUM({column}) FROM {table} WHERE datetime = (SELECT MAX(datetime) FROM {table})"
    },
    "comparison": {
        "keywords": ["compare", "versus", "vs", "difference", "better", "worse", "outperform", "beat"],
        "guidance": "Include comparison columns side-by-side (e.g., ytd_return AND ytd_index_return). Filter for latest snapshot.",
        "sql_template": "SELECT portfolio_name, {metric}, {benchmark_metric} FROM {table} WHERE datetime = (SELECT MAX(datetime) FROM {table})"
    },
    "ranking": {
        "keywords": ["top", "bottom", "best", "worst", "highest", "lowest", "most", "least"],
        "guidance": "ORDER BY the relevant metric (DESC for top/best/highest, ASC for bottom/worst/lowest). Use LIMIT to get the requested count.",
        "sql_template": "SELECT {columns} FROM {table} WHERE datetime = (SELECT MAX(datetime) FROM {table}) ORDER BY {metric} {direction} LIMIT {n}"
    }
}


# =============================================================================
# QUERY DECOMPOSER CLASS
# =============================================================================

class QueryDecomposer:
    """
    Decomposes natural language queries into structured components.
    
    This helps with:
    - Table selection (P&L vs Summary vs Holdings)
    - Column suggestion based on focus
    - Constraint extraction for WHERE clauses
    """
    
    def __init__(self):
        self.constraint_patterns = CONSTRAINT_PATTERNS
        self.focus_keywords = FOCUS_KEYWORDS
        self.aggregation_keywords = AGGREGATION_KEYWORDS
    
    def decompose(self, query: str) -> QueryDecomposition:
        """
        Main entry point - decompose a query into structured components.
        
        Args:
            query: Natural language query
            
        Returns:
            QueryDecomposition with extracted components
        """
        decomposition = QueryDecomposition(original_query=query)
        query_lower = query.lower()
        
        print(f"🔍 Decomposing query: '{query}'")
        
        # 1. Extract focus (what data is requested)
        focus_categories = self._extract_focus(query_lower)
        decomposition.focus_description = ", ".join(focus_categories) if focus_categories else "general"
        
        # 2. Determine primary table based on focus
        decomposition.primary_table = self._determine_primary_table(focus_categories)
        
        # 3. Extract constraints
        decomposition.constraints = self._extract_constraints(query_lower)
        
        # 4. Extract aggregations
        decomposition.aggregations = self._extract_aggregations(query_lower)
        
        # 5. Extract sorting/ranking
        sort_dir, limit = self._extract_ranking(query_lower)
        decomposition.sort_direction = sort_dir
        decomposition.limit = limit
        
        # 6. Determine complexity
        decomposition.complexity = self._classify_complexity(query_lower, focus_categories)
        
        # 7. Identify secondary tables if needed
        if decomposition.complexity in ["multi", "complex"]:
            decomposition.secondary_tables = self._identify_secondary_tables(
                decomposition.primary_table, focus_categories
            )
        
        # 8. NEW: Detect query pattern and generate guidance
        decomposition.query_pattern = self._detect_query_pattern(query_lower)
        decomposition.generation_guidance = self._generate_guidance(decomposition)
        
        # 9. NEW: Populate required_tables (primary + secondary)
        decomposition.required_tables = [decomposition.primary_table] + decomposition.secondary_tables
        
        # 10. NEW: Populate join_path if multi-table (import from schema_graph)
        if len(decomposition.required_tables) > 1:
            try:
                from src.services.schema_graph import get_optimal_path
                decomposition.join_path = get_optimal_path(decomposition.required_tables)
            except ImportError:
                decomposition.join_path = []
        
        print(f"✅ Decomposition Result: Pattern={decomposition.query_pattern}, Table={decomposition.primary_table}, Complexity={decomposition.complexity}")
        return decomposition

    
    def _extract_focus(self, query: str) -> List[str]:
        """Extract what type of data the query is asking for."""
        matches = []
        
        for category, keywords in self.focus_keywords.items():
            if any(kw in query for kw in keywords):
                matches.append(category)
        
        return matches if matches else ["identity"]  # Default to portfolio-level
    
    def _determine_primary_table(self, focus_categories: List[str]) -> str:
        """Determine the primary table based on focus categories."""
        # Priority order for table selection
        priority = ["profit_loss", "holdings", "benchmark", "returns", "value", "identity"]
        
        for focus in priority:
            if focus in focus_categories:
                return FOCUS_TO_TABLE[focus]
        
        return "portfolio_summary"  # Default
    
    def _extract_constraints(self, query: str) -> Dict[str, any]:
        """Extract filter constraints from the query."""
        constraints = {}
        
        # Check temporal constraints
        for pattern, value in self.constraint_patterns["temporal"].items():
            if pattern in query:
                constraints["temporal_period"] = value
                break
        
        # Check status filters
        for pattern, filter_dict in self.constraint_patterns["status"].items():
            if pattern in query:
                constraints.update(filter_dict)
        
        # Check asset group filters (DISABLED)
        # if "asset_group" in self.constraint_patterns:
        #     for pattern, filter_dict in self.constraint_patterns["asset_group"].items():
        #         if pattern in query:
        #             constraints.update(filter_dict)
        
        # Check value filters (with number extraction)
        value_match = re.search(r'(above|below|greater than|more than|less than|at least|exceeding)\s+\$?([\d,]+(?:\.\d+)?)', query)
        if value_match:
            operator_word = value_match.group(1)
            value = float(value_match.group(2).replace(',', ''))
            operator = self.constraint_patterns["value_filter"].get(operator_word, ">")
            constraints["value_filter"] = {"operator": operator, "value": value}
        
        # Check for specific portfolio mentions
        portfolio_match = re.search(r"(?:for|in|portfolio)\s+['\"]?([A-Za-z0-9\s]+)['\"]?", query)
        if portfolio_match:
            constraints["portfolio_name"] = portfolio_match.group(1).strip()
        
        return constraints
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """Extract aggregation functions needed."""
        aggregations = []
        
        for keyword, agg_func in self.aggregation_keywords.items():
            if keyword in query:
                if agg_func not in aggregations:
                    aggregations.append(agg_func)
        
        return aggregations
    
    def _extract_ranking(self, query: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract sorting direction and limit from ranking keywords."""
        sort_direction = None
        limit = None
        
        for keyword, (direction, limit_type) in self.constraint_patterns["ranking"].items():
            if keyword in query:
                sort_direction = direction
                
                # Try to extract number for limit
                number_match = re.search(rf'{keyword}\s+(\d+)', query)
                if number_match:
                    limit = int(number_match.group(1))
                elif "LIMIT 1" in limit_type:
                    limit = 1
                
                break
        
        return sort_direction, limit
    
    def _classify_complexity(self, query: str, focus_categories: List[str]) -> str:
        """Classify query complexity: single, multi, or complex."""
        
        # Multiple focus categories often mean multi-table
        if len(focus_categories) >= 3:
            return "complex"
        
        # Specific patterns for multi-table queries
        multi_table_patterns = [
            "for each portfolio",
            "along with",
            "and their",
            "breakdown by",
            "grouped by asset",
            "with their holdings",
            "with p&l"
        ]
        
        if any(p in query for p in multi_table_patterns):
            return "multi" if len(focus_categories) == 2 else "complex"
        
        # Check for explicit join needs
        if "profit_loss" in focus_categories and "returns" in focus_categories:
            return "multi"
        
        if "holdings" in focus_categories and "profit_loss" in focus_categories:
            return "multi"
        
        return "single"
    
    def _identify_secondary_tables(self, primary_table: str, focus_categories: List[str]) -> List[str]:
        """Identify secondary tables needed for joins."""
        all_tables = set()
        
        for focus in focus_categories:
            table = FOCUS_TO_TABLE.get(focus)
            if table and table != primary_table:
                all_tables.add(table)
        
        return list(all_tables)
    
    def _detect_query_pattern(self, query: str) -> str:
        """
        Detect the semantic query pattern to determine optimal SQL structure.
        
        Patterns:
        - trend: Historical analysis over time
        - snapshot: Current state query
        - aggregate: SUM/AVG/COUNT operations
        - comparison: Compare entities or benchmarks
        - ranking: Top/bottom N queries
        """
        # Priority order: ranking > comparison > aggregate > trend > snapshot (default)
        for pattern_name in ["ranking", "comparison", "aggregate", "trend"]:
            pattern_info = QUERY_PATTERNS.get(pattern_name, {})
            keywords = pattern_info.get("keywords", [])
            if any(kw in query for kw in keywords):
                return pattern_name
        
        return "snapshot"  # Default pattern
    
    def _generate_guidance(self, decomposition: 'QueryDecomposition') -> str:
        """
        Generate specific LLM guidance based on detected pattern and focus.
        
        This guidance will be injected into the SQL prompt to steer the LLM
        towards the optimal query structure.
        """
        pattern = decomposition.query_pattern
        pattern_info = QUERY_PATTERNS.get(pattern, {})
        base_guidance = pattern_info.get("guidance", "")
        
        # Build comprehensive guidance
        guidance_parts = [
            f"### SQL Generation Guidance",
            f"",
            f"**Query Pattern:** {pattern.upper()}",
            f"**Instruction:** {base_guidance}",
        ]
        
        # Add table-specific guidance
        table = decomposition.primary_table
        guidance_parts.append(f"")
        guidance_parts.append(f"**Target Table:** `ai_trading.{table}`")
        
        # Add join guidance if multi-table
        if decomposition.secondary_tables:
            guidance_parts.append(f"**Join Required:** You must JOIN the following tables:")
            for sec_table in decomposition.secondary_tables:
                guidance_parts.append(f"  - `ai_trading.{sec_table}` on `portfolio_name`")
        
        # Add sorting/limit guidance for ranking pattern
        if pattern == "ranking" and decomposition.sort_direction:
            direction_word = "highest" if decomposition.sort_direction == "DESC" else "lowest"
            limit_str = f"LIMIT {decomposition.limit}" if decomposition.limit else ""
            guidance_parts.append(f"")
            guidance_parts.append(f"**Sorting:** ORDER BY the relevant metric {decomposition.sort_direction} to get {direction_word} values. {limit_str}")
        
        # Add aggregation reminders
        if decomposition.aggregations:
            agg_str = ", ".join(decomposition.aggregations)
            guidance_parts.append(f"")
            guidance_parts.append(f"**Aggregations:** Use {agg_str}. Remember to GROUP BY non-aggregated columns if needed.")
        
        # Add focus-specific column suggestions
        focus = decomposition.focus_description
        if "profit_loss" in focus:
            guidance_parts.append(f"")
            guidance_parts.append(f"**Suggested Columns:** `symbol`, `ytd_total_pnl`, `ytd_realized_pnl`, `ytd_unrealized_pnl`")
        elif "returns" in focus or "benchmark" in focus:
            guidance_parts.append(f"")
            guidance_parts.append(f"**Suggested Columns:** `portfolio_name`, `ytd_return`, `ytd_index_return`, `default_index`")
        elif "holdings" in focus:
            guidance_parts.append(f"")
            guidance_parts.append(f"**Suggested Columns:** `symbol`, `positions`, `market_value`")
        elif "value" in focus:
            guidance_parts.append(f"")
            guidance_parts.append(f"**Suggested Columns:** `portfolio_name`, `net_liquidity`, `allocated_amount`")
        
        return "\n".join(guidance_parts)
    

    def get_decomposition_summary(self, decomposition: QueryDecomposition) -> str:
        """
        Generate a human-readable summary for injection into the SQL prompt.
        Now includes generation_guidance for specific LLM instructions.
        """
        lines = [
            "### Query Analysis",
            f"**Focus:** {decomposition.focus_description}",
            f"**Query Pattern:** {decomposition.query_pattern.upper()}",
            f"**Recommended Table:** `ai_trading.{decomposition.primary_table}`",
        ]
        
        if decomposition.secondary_tables:
            secondary = ", ".join([f"`ai_trading.{t}`" for t in decomposition.secondary_tables])
            lines.append(f"**Secondary Tables:** {secondary}")
        
        if decomposition.constraints:
            constraint_str = ", ".join([f"{k}={v}" for k, v in decomposition.constraints.items()])
            lines.append(f"**Detected Constraints:** {constraint_str}")
        
        if decomposition.aggregations:
            lines.append(f"**Aggregations:** {', '.join(decomposition.aggregations)}")
        
        if decomposition.sort_direction:
            lines.append(f"**Sorting:** {decomposition.sort_direction}")
            if decomposition.limit:
                lines.append(f"**Limit:** {decomposition.limit}")
        
        lines.append(f"**Complexity:** {decomposition.complexity}")
        
        # Append the generation guidance (the key new feature)
        if decomposition.generation_guidance:
            lines.append("")
            lines.append(decomposition.generation_guidance)
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def decompose_query(query: str) -> QueryDecomposition:
    """Convenience function to decompose a query."""
    decomposer = QueryDecomposer()
    return decomposer.decompose(query)


def get_query_analysis(query: str) -> str:
    """Get a formatted analysis summary for prompt injection."""
    decomposer = QueryDecomposer()
    decomposition = decomposer.decompose(query)
    return decomposer.get_decomposition_summary(decomposition)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from questions import ALL_QUESTIONS
    
    decomposer = QueryDecomposer()
    
    print("=" * 60)
    print("Query Decomposition Test")
    print("=" * 60)
    
    # Test a sample of questions
    test_questions = ALL_QUESTIONS[:10]
    
    for q in test_questions:
        print(f"\n{'─' * 60}")
        print(f"Query: {q}")
        
        decomposition = decomposer.decompose(q)
        print(decomposition)
        print(f"\nSummary for prompt:\n{decomposer.get_decomposition_summary(decomposition)}")
