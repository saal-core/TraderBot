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

def benchmark_query(query: str, llm_type: str, use_vanna: bool, router: QueryRouter, stock_matcher: StockMatcher, vanna_service: VannaService, db_handler_gpt: DatabaseQueryHandler, db_handler_claude: DatabaseQueryHandler) -> Dict:
    """
    Run benchmark for a single query and LLM type.
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
    
    try:
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
            # We need schema info for DatabaseQueryHandler
            # Assuming db_handler has access to schema via sql_executor or we pass it
            # DatabaseQueryHandler.generate_sql takes (query, schema)
            
            # Select appropriate handler
            handler = db_handler_claude if llm_type == "claude" else db_handler_gpt
            
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
        # Execute using Vanna service's executor or direct sql_executor
        # VannaService.execute_sql returns DF. 
        # PostgreSQLExecutor.execute_query returns (success, df, msg)
        
        if use_vanna:
            df = vanna_service.execute_sql(sql)
        else:
            success, df, msg = db_handler_gpt.sql_executor.execute_query(sql) # Use any executor
            if not success:
                df = None
                
        results["execution_time"] = time.time() - start_time
        logging.info(f"[{llm_type}|Vanna:{use_vanna}] Execution: {results['execution_time']:.4f}s")

        # 5. Explanation (Always uses GPT-OSS)
        start_time = time.time()
        
        if use_vanna:
            chat_history = [{"role": "user", "content": query}]
            explanation = vanna_service.explain_results(query, df, chat_history)
        else:
            # Use DatabaseQueryHandler's explain_results (which uses Ollama/GPT-OSS by default if initialized that way)
            # We should use db_handler_gpt for explanation to ensure GPT-OSS is used
            explanation = db_handler_gpt.explain_results(query, df, sql)
            
        results["explanation_time"] = time.time() - start_time
        logging.info(f"[{llm_type}|Vanna:{use_vanna}] Explanation: {results['explanation_time']:.4f}s")

        results["total_time"] = time.time() - start_total
        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        logging.error(f"Error benchmarking {query} with {llm_type}|Vanna:{use_vanna}: {e}")

    return results

def main():
    print("Starting Benchmark...")
    
    # Initialize components
    router = QueryRouter()
    sql_executor = PostgreSQLExecutor()
    stock_matcher = StockMatcher(sql_executor=sql_executor)
    vanna_service = VannaService()
    
    # Initialize DatabaseQueryHandlers
    db_handler_gpt = DatabaseQueryHandler(sql_executor=sql_executor, llm_type="gpt-oss")
    db_handler_claude = DatabaseQueryHandler(sql_executor=sql_executor, llm_type="claude")
    
    if not vanna_service.connect():
        print("Failed to connect Vanna Service to DB. Exiting.")
        return



    # Test Queries
    queries = [
    "Which portfolio outperformed its YTD index return the most?",
    "Provide a summary of my entire investment portfolio.",
    "what is the performance in A-Balanced",
    "What is the description for \"Balanced Fund\"?",
    "What is the best performing portfolio by MTD return?",
    "What is the best performing portfolio by all-time return?",
    "What is the total net liquidity across all portfolios for account U17032472?",
    "which portfolios have negative returns",
    "What is the best performing portfolio by WTD return?",
    "What is the best performing portfolio by YTD return?",
    "Give me a summary of my portfolio.",
    "What is the best performing portfolio by QTD return?",
    "What is the highest overall profit any investment generated YTD?",
    "What investment has the best performance in terms of YTD profit?",
    "Which asset group generated the highest total profit YTD?",
    "Which portfolio has the highest YTD profit?",
    "What is the YTD unrealized profit/loss for MSFT?",
    "top 10 profit stocks in the current database",
    "List the top 10 most profitable stocks overall.",
    "Which portfolio has the highest overall profit this year?",
    "List all investments that have a negative total profit.",
    "bottom 10 stocks in the current database",
    "what is the ytd profit for A-balanced",
    "Which portfolio has the highest QTD profit?",
    "Compare performance between different portfolio groups.",
    "Show the total quantity held for each stock symbol, sorted by the highest quantity first.",
    "Show me any portfolios that are losing money overall.",
    "How do portfolios perform across groups?",
    "compare ABalanced performance against QQQ"
]

    scenarios = [
        {"llm": "gpt-oss", "vanna": False},
        {"llm": "claude", "vanna": False},
        # {"llm": "gpt-oss", "vanna": True},
        # {"llm": "claude", "vanna": True}
    ]
    
    all_results = []

    for query in queries:
        print(f"\n--- Benchmarking Query: {query} ---")
        for scenario in scenarios:
            llm = scenario["llm"]
            use_vanna = scenario["vanna"]
            print(f"Testing with LLM={llm}, Vanna={use_vanna}...")
            
            result = benchmark_query(
                query, llm, use_vanna, 
                router, stock_matcher, vanna_service, 
                db_handler_gpt, db_handler_claude
            )
            all_results.append(result)

    # Output Summary
    print("\n\n=== Benchmark Results ===")
    df_results = pd.DataFrame(all_results)
    
    # Display simplified table
    cols = ["query", "llm_type", "use_vanna", "routing_time", "stock_matching_time", "sql_generation_time", "explanation_time", "total_time", "success"]
    print(df_results[cols].to_string())
    
    # Save to CSV
    df_results.to_csv("benchmark_results.csv", index=False)
    print("\nDetailed results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
