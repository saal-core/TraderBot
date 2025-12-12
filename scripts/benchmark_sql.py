import time
import pandas as pd
import logging
from typing import List, Dict
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.query_router import QueryRouter
from src.services.stock_matcher import StockMatcher
from src.services.vanna_service import VannaService
from src.services.database_handler import DatabaseQueryHandler
from src.services.sql_utilities import PostgreSQLExecutor
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def benchmark_query(query: str, llm_type: str, use_vanna: bool, router: QueryRouter, stock_matcher: StockMatcher, vanna_service: VannaService, db_handlers: Dict) -> Dict:
    """
    Run benchmark for a single query and LLM type.
    
    Args:
        db_handlers: Dictionary with keys "gpt-oss", "claude", "gpt-4.1"
    """
    results = {
        "query": query,
        "llm_type": llm_type,
        "use_vanna": use_vanna,
        "routing_time": 0,
        "stock_matching_time": 0,
        "sql_generation_time": 0,
        "execution_time": 0,
        "explanation_time": 0,
        "total_time": 0,
        "sql_query": None,
        "success": False,
        "error": None
    }
    
    start_total = time.time()

    # 1. Routing
    start_time = time.time()
    query_type = router.classify_query(query)
    results["routing_time"] = time.time() - start_time
    logging.info(f"[{llm_type}|Vanna:{use_vanna}] Routing: {results['routing_time']:.4f}s ({query_type})")

    if query_type != "database":
        logging.warning(f"Query classified as {query_type}, skipping SQL generation.")
        return results

    # 2. Stock Matching
    start_time = time.time()
    stock_mentions = stock_matcher.extract_stock_mentions(query)
    matched_symbols = stock_matcher.fuzzy_match_symbols(stock_mentions)
    results["stock_matching_time"] = time.time() - start_time
    logging.info(f"[{llm_type}|Vanna:{use_vanna}] Stock Matching: {results['stock_matching_time']:.4f}s")

    # 3. SQL Generation
    start_time = time.time()
    sql = ""
    
    if use_vanna:
        # Switch LLM in Vanna
        vanna_service.set_llm(llm_type)
        enhanced_query = f"{query}\nContext: {matched_symbols}"
        sql = vanna_service.generate_sql(enhanced_query)
    else:
        # Use DatabaseQueryHandler
        handler = db_handlers.get(llm_type)
        if not handler:
            raise ValueError(f"No handler for llm_type: {llm_type}")
        
        # Get schema (cached or fresh)
        schema = handler.sql_executor.get_schema_info()

        sql = handler.generate_sql(query, schema)

    results["sql_generation_time"] = time.time() - start_time
    results["sql_query"] = sql
    logging.info(f"[{llm_type}|Vanna:{use_vanna}] SQL Generation: {results['sql_generation_time']:.4f}s")


    if not sql or "ERROR" in sql:
        results["error"] = f"Failed to generate SQL: {sql}"
        return results

    # 4. Execution
    start_time = time.time()
    # Execute using sql_executor
    if use_vanna:
        df = vanna_service.execute_sql(sql)
    else:
        success, df, msg = db_handlers["gpt-oss"].sql_executor.execute_query(sql)
        if not success:
            df = None
            
    results["execution_time"] = time.time() - start_time
    logging.info(f"[{llm_type}|Vanna:{use_vanna}] Execution: {results['execution_time']:.4f}s")

    # 5. Explanation (Always uses gpt-oss)
    start_time = time.time()
    
    if use_vanna:
        chat_history = [{"role": "user", "content": query}]
        explanation = vanna_service.explain_results(query, df, chat_history)
    else:
        # Use gpt-oss handler for explanation
        explanation = db_handlers["gpt-oss"].explain_results(query, df, sql)
        
    results["explanation_time"] = time.time() - start_time
    logging.info(f"[{llm_type}|Vanna:{use_vanna}] Explanation: {results['explanation_time']:.4f}s")

    results["total_time"] = time.time() - start_total
    results["success"] = True


    return results

def main():
    print("Starting Benchmark... (Testing Ollama, Claude, and GPT-4.1)")
    
    # Initialize components
    router = QueryRouter()
    sql_executor = PostgreSQLExecutor()
    stock_matcher = StockMatcher(sql_executor=sql_executor)
    vanna_service = VannaService()
    
    # Initialize DatabaseQueryHandlers for all three models
    print("\nInitializing handlers...")
    handlers = {
        "gpt-oss": DatabaseQueryHandler(sql_executor=sql_executor, llm_type="gpt-oss"),
        "claude": DatabaseQueryHandler(sql_executor=sql_executor, llm_type="claude"),
        "gpt-4.1": DatabaseQueryHandler(sql_executor=sql_executor, llm_type="gpt-4.1"),
        "gpt-5.1": DatabaseQueryHandler(sql_executor=sql_executor, llm_type="gpt-5.1")
    }
    
    if not vanna_service.connect():
        print("Warning: Failed to connect Vanna Service to DB. Vanna benchmarks will be skipped.")

    # Test Queries
    queries = [
    # # --- Level 1: Single-Table Scenarios (Basic Filtering & Aggregation) ---
    # "What is the total net liquidity across all my active portfolios?",
    # "Which portfolio has the highest Year-to-Date return percentage?",
    # "List all portfolios that have over $50,000 in unutilized (cash) funds.",
    # "Show me the top 3 worst-performing portfolios by all-time return.",
    # "How many shares of Apple (AAPL) do I hold across all portfolios?",
    # "What is the total market value of all my positions created in the last 7 days?",
    # "Which individual stock holding has the highest unrealized paper loss right now?",
    # "Which asset group (e.g., Equity, Crypto) has generated the most realized profit this year?",
    # "Show me the daily realized P&L for Microsoft (MSFT) for the last month.",
    # "What is the total combined profit (realized + unrealized) for all 'Equity' trades?",

    # --- Level 2: Two-Table Scenarios (Joins & Cross-Reference) ---
    "Show me the current holdings for Apple, but only for portfolios that are currently active.",
    "What is the total market value of positions in the 'Aggressive Growth' portfolio? (Verify it is active first).",
    "What percentage of the 'Tech Fund' portfolio's Net Liquidity is currently invested in NVDA?",
    "Which active portfolios have more than 50% of their allocated capital currently utilized?",
    "For my largest portfolio by Net Liquidity, what is the top profitable stock YTD?",
    "List the total realized profit for all portfolios that started with an allocated amount greater than $100,000.",

    # --- Level 3: Three-Table Scenarios (Complex Joins & Logic) ---
    "List all stocks in active portfolios where my current market value is higher than the total profit I've realized on them this year.",
    "For every active portfolio, show the Portfolio Name, Total Net Liquidity, and the stock symbol that has contributed the most to its YTD Total PnL.",
    "Generate a report for 'TSLA' showing: which active portfolios hold it, how many shares are held, and how much realized profit it has generated in those specific portfolios.",

    # # --- Level 4: Python/Pandas Post-Processing (Requires Raw Data Fetch) ---
    # "Calculate the standard deviation of daily returns for the 'A-Balanced' portfolio over the last 90 days.",
    # "How does the daily return of 'Tech Growth' correlate with 'Conservative Income' over the last year?",
    # "What is the Sharpe Ratio of my best-performing portfolio YTD (assuming a risk-free rate of 2%)?"
]
    # Scenarios: Test all three models
    scenarios = [
        {"llm": "gpt-oss", "vanna": False},
        {"llm": "claude", "vanna": False},
        {"llm": "gpt-4.1", "vanna": False},
        {"llm": "gpt-5.1", "vanna": False},
    ]
    
    all_results = []

    for query in queries:
        print(f"\n--- Benchmarking Query: {query} ---")
        for scenario in scenarios:
            llm = scenario["llm"]
            use_vanna = scenario["vanna"]
            print(f"  Testing with LLM={llm}, Vanna={use_vanna}...")
            
            result = benchmark_query(
                query, llm, use_vanna, 
                router, stock_matcher, vanna_service, 
                handlers
            )
            all_results.append(result)

    # Output Summary
    print("\n\n=== Benchmark Results ===")
    df_results = pd.DataFrame(all_results)
    
    # Display simplified table
    cols = ["query", "llm_type", "use_vanna", "routing_time", "stock_matching_time", "sql_generation_time", "explanation_time", "total_time", "success"]
    print(df_results[cols].to_string())
    
    # Summary Statistics
    print("\n\n=== Summary Statistics ===")
    summary = df_results[df_results["success"] == True].groupby("llm_type").agg({
        "sql_generation_time": ["mean", "median", "min", "max"],
        "execution_time": ["mean", "median"],
        "total_time": ["mean", "median"],
        "success": "sum"
    })
    print(summary)
    
    # Model Comparison
    print("\n\n=== Model Comparison ===")
    model_stats = df_results[df_results["success"] == True].groupby("llm_type").agg({
        "sql_generation_time": "mean",
        "total_time": "mean",
        "success": "count"
    }).round(4)
    model_stats.columns = ["Avg SQL Gen Time", "Avg Total Time", "Successful Queries"]
    print(model_stats)
    
    # Save to CSV
    df_results.to_csv("benchmark_results.csv", index=False)
    print("\nDetailed results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
