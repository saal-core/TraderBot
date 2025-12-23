#!/usr/bin/env python3
"""
Model Comparison Benchmark Script

Compares explanation generation time between:
- gpt-oss:20b on RTX 4090 (via Ollama)
- QWEN3:30B-3B on H100 (via OpenAI-compatible API)

Usage:
    python benchmark_explanation_models.py
"""

import sys
import os
import time
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.qwen_explanation_service import QwenExplanationService
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config

from dotenv import load_dotenv
load_dotenv()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_name: str
    test_case: str
    elapsed_time: float
    success: bool
    response_length: int
    error: str = ""


class OllamaExplanationService:
    """Ollama-based explanation service for benchmark comparison."""
    
    def __init__(self):
        ollama_config = get_ollama_config()
        self.model_name = ollama_config["model_name"]
        
        self.llm = Ollama(
            model=self.model_name,
            base_url=ollama_config["base_url"],
            temperature=float(ollama_config.get("temperature_greeting", 0.3))
        )
        
        self.explain_prompt = PromptTemplate(
            input_variables=["query", "results", "sql_query"],
            template="""You are a financial portfolio assistant interpreting data for users.

**User Question:** {query}

**Context (SQL Query Used):**
{sql_query}

**Retrieved Data:**
{results}

**Your Role:**
Interpret and explain the data **from the user's perspective**. Your job is to answer their question directly, not describe the data structure.

**Rules:**
1. **Answer the question directly** - Focus on what the user asked, not on how the data is structured
2. **Never mention data rows, columns, or table structures** - Speak as if you're a financial advisor explaining insights
3. **Use specific numbers and names** - Reference actual values from the data (portfolio names, amounts, percentages)
4. **Be conversational and helpful** - The user doesn't need to know about databases or queries
5. **If no results found** - Simply say the information wasn't found, don't suggest technical solutions
6. **Format nicely** - Use bullet points or brief paragraphs for clarity when appropriate
7. **No code or SQL** - Never include code, SQL, or technical syntax in your response

**Response:**"""
        )
        
        self.explain_chain = self.explain_prompt | self.llm | StrOutputParser()
    
    def explain_results(self, query: str, results_df, sql_query: str) -> Tuple[str, float]:
        """Generate explanation and return (explanation, elapsed_time)."""
        results_text = results_df.to_string(index=False) if results_df is not None and not results_df.empty else "No results found"
        
        try:
            start_time = time.time()
            print(f"⏱️  [Ollama RTX4090] Starting: Results Explanation Generation...")
            
            explanation = self.explain_chain.invoke({
                "query": query,
                "results": results_text,
                "sql_query": sql_query
            })
            
            elapsed = time.time() - start_time
            print(f"✅ [Ollama RTX4090] Completed: Results Explanation in {elapsed:.2f}s")
            
            return explanation.strip(), elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ [Ollama RTX4090] Error: {e}")
            return f"Error: {e}", elapsed


# ============================================================================
# Test Cases
# ============================================================================

TEST_CASES = [
    {
        "name": "simple_portfolio_list",
        "query": "Show me all my portfolios",
        "sql_query": "SELECT portfolio_name, total_value FROM ai_trading.portfolio_summary",
        "results": pd.DataFrame({
            "portfolio_name": ["A-Balanced", "A-Growth", "A-Income"],
            "total_value": [1500000.50, 2300000.75, 980000.25]
        })
    },
    {
        "name": "ytd_returns",
        "query": "What is the YTD return for A-Balanced?",
        "sql_query": "SELECT portfolio_name, ytd_return FROM ai_trading.portfolio_summary WHERE portfolio_name = 'A-Balanced'",
        "results": pd.DataFrame({
            "portfolio_name": ["A-Balanced"],
            "ytd_return": [12.5]
        })
    },
    {
        "name": "top_holdings",
        "query": "What are my top 5 holdings by value?",
        "sql_query": "SELECT symbol, quantity, market_value FROM ai_trading.portfolio_holdings ORDER BY market_value DESC LIMIT 5",
        "results": pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "quantity": [500, 300, 100, 150, 200],
            "market_value": [125000.00, 98000.00, 85000.00, 72000.00, 68000.00]
        })
    },
    {
        "name": "complex_comparison",
        "query": "Compare the performance of A-Balanced vs A-Growth",
        "sql_query": "SELECT portfolio_name, ytd_return, mtd_return, total_value FROM ai_trading.portfolio_summary WHERE portfolio_name IN ('A-Balanced', 'A-Growth')",
        "results": pd.DataFrame({
            "portfolio_name": ["A-Balanced", "A-Growth"],
            "ytd_return": [12.5, 18.3],
            "mtd_return": [2.1, 3.8],
            "total_value": [1500000.50, 2300000.75]
        })
    },
    {
        "name": "large_data",
        "query": "Show me all holdings with their profit/loss",
        "sql_query": "SELECT symbol, quantity, market_value, unrealized_pnl FROM ai_trading.portfolio_holdings",
        "results": pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "NFLX"],
            "quantity": [500, 300, 100, 150, 200, 250, 180, 400, 600, 120],
            "market_value": [125000, 98000, 85000, 72000, 68000, 55000, 48000, 42000, 38000, 35000],
            "unrealized_pnl": [15000, 12000, -3000, 8000, 22000, 4500, -8000, 6000, -5000, 3000]
        })
    },
]


def run_benchmark(num_iterations: int = 3) -> List[BenchmarkResult]:
    """
    Run benchmark comparing both models.
    
    Args:
        num_iterations: Number of times to run each test case
        
    Returns:
        List of BenchmarkResult objects
    """
    results: List[BenchmarkResult] = []
    
    print("\n" + "=" * 80)
    print("🚀 EXPLANATION MODEL BENCHMARK")
    print("=" * 80)
    print(f"Comparing: gpt-oss:20b (RTX 4090) vs QWEN3:30B-3B (H100)")
    print(f"Iterations per test: {num_iterations}")
    print("=" * 80 + "\n")
    
    # Initialize services
    print("📦 Initializing services...")
    
    try:
        ollama_service = OllamaExplanationService()
        print(f"  ✅ Ollama ({ollama_service.model_name}) initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize Ollama: {e}")
        ollama_service = None
    
    try:
        qwen_service = QwenExplanationService()
        success, msg, _ = qwen_service.test_connection()
        if success:
            print(f"  ✅ QWEN ({qwen_service.model_name}) initialized - {msg}")
        else:
            print(f"  ⚠️ QWEN initialized but connection test failed: {msg}")
    except Exception as e:
        print(f"  ❌ Failed to initialize QWEN: {e}")
        qwen_service = None
    
    print("\n" + "-" * 80 + "\n")
    
    # Run benchmarks
    for test_case in TEST_CASES:
        print(f"\n📝 Test Case: {test_case['name']}")
        print(f"   Query: {test_case['query'][:60]}...")
        print("-" * 40)
        
        for iteration in range(num_iterations):
            print(f"\n  Iteration {iteration + 1}/{num_iterations}")
            
            # Test Ollama (RTX 4090)
            if ollama_service:
                try:
                    explanation, elapsed = ollama_service.explain_results(
                        test_case["query"],
                        test_case["results"],
                        test_case["sql_query"]
                    )
                    results.append(BenchmarkResult(
                        model_name="gpt-oss:20b (RTX 4090)",
                        test_case=test_case["name"],
                        elapsed_time=elapsed,
                        success=not explanation.startswith("Error"),
                        response_length=len(explanation)
                    ))
                except Exception as e:
                    results.append(BenchmarkResult(
                        model_name="gpt-oss:20b (RTX 4090)",
                        test_case=test_case["name"],
                        elapsed_time=0,
                        success=False,
                        response_length=0,
                        error=str(e)
                    ))
            
            # Test QWEN (H100)
            if qwen_service:
                try:
                    explanation, elapsed = qwen_service.explain_results(
                        test_case["query"],
                        test_case["results"],
                        test_case["sql_query"]
                    )
                    results.append(BenchmarkResult(
                        model_name="QWEN3:30B-3B (H100)",
                        test_case=test_case["name"],
                        elapsed_time=elapsed,
                        success=not explanation.startswith("Error"),
                        response_length=len(explanation)
                    ))
                except Exception as e:
                    results.append(BenchmarkResult(
                        model_name="QWEN3:30B-3B (H100)",
                        test_case=test_case["name"],
                        elapsed_time=0,
                        success=False,
                        response_length=0,
                        error=str(e)
                    ))
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print a summary of benchmark results."""
    
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 80 + "\n")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame([
        {
            "Model": r.model_name,
            "Test Case": r.test_case,
            "Time (s)": r.elapsed_time,
            "Success": r.success,
            "Response Length": r.response_length
        }
        for r in results
    ])
    
    # Filter successful runs only
    df_success = df[df["Success"] == True]
    
    if df_success.empty:
        print("❌ No successful runs to analyze")
        return
    
    # Per-model summary
    print("📈 Per-Model Performance:\n")
    model_summary = df_success.groupby("Model").agg({
        "Time (s)": ["mean", "min", "max", "std"],
        "Response Length": "mean"
    }).round(3)
    print(model_summary.to_string())
    
    # Per-test-case comparison
    print("\n\n📋 Per-Test-Case Comparison:\n")
    test_summary = df_success.pivot_table(
        index="Test Case",
        columns="Model",
        values="Time (s)",
        aggfunc="mean"
    ).round(3)
    print(test_summary.to_string())
    
    # Speed comparison
    print("\n\n🏆 Speed Comparison:\n")
    
    ollama_mean = df_success[df_success["Model"].str.contains("RTX")]["Time (s)"].mean()
    qwen_mean = df_success[df_success["Model"].str.contains("H100")]["Time (s)"].mean()
    
    if ollama_mean > 0 and qwen_mean > 0:
        faster = "QWEN (H100)" if qwen_mean < ollama_mean else "Ollama (RTX 4090)"
        speedup = max(ollama_mean, qwen_mean) / min(ollama_mean, qwen_mean)
        print(f"  Average Ollama (RTX 4090): {ollama_mean:.3f}s")
        print(f"  Average QWEN (H100):       {qwen_mean:.3f}s")
        print(f"  Winner: {faster} ({speedup:.2f}x faster)")
    
    # Success rate
    print("\n\n✅ Success Rate:\n")
    success_rate = df.groupby("Model")["Success"].mean() * 100
    print(success_rate.to_string())


def save_results(results: List[BenchmarkResult], filename: str = "benchmark_results.csv"):
    """Save results to CSV file."""
    df = pd.DataFrame([
        {
            "model": r.model_name,
            "test_case": r.test_case,
            "elapsed_time_s": r.elapsed_time,
            "success": r.success,
            "response_length": r.response_length,
            "error": r.error
        }
        for r in results
    ])
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")


def main():
    """Run the benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark explanation models")
    parser.add_argument("-n", "--iterations", type=int, default=3, help="Number of iterations per test")
    parser.add_argument("-o", "--output", type=str, default="benchmark_results.csv", help="Output CSV file")
    args = parser.parse_args()
    
    results = run_benchmark(num_iterations=args.iterations)
    print_summary(results)
    save_results(results, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Benchmark complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
