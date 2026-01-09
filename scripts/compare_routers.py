#!/usr/bin/env python3
"""
Router Comparison Script

Compares the accuracy and performance of:
1. GPTOSSQueryRouter (pure LLM-based classification) - v1
2. OptimizedQueryRouter (tiered pattern + LLM fallback) - v2

Usage:
    cd /home/dev/Hussein/TraderBot
    source my_env/bin/activate
    python scripts/compare_routers.py
"""
import sys
import os
import time
import csv
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.gpt_oss_query_router import GPTOSSQueryRouter
from src.services.gpt_oss_query_router_v2 import OptimizedQueryRouter


# ==============================================================================
# TEST QUESTIONS WITH GROUND TRUTH
# ==============================================================================
# Format: (question, expected_category)
# Categories: "database", "internet_data", "greeting"
#
# ADD YOUR TEST QUESTIONS HERE:
# ==============================================================================

TEST_QUESTIONS = [
    # Database Questions
    ("Which portfolio outperformed its YTD index return the most?", "database"),
    ("Provide a summary of my entire investment portfolio.", "database"),
    ("What is the performance in A-Balanced?", "database"),
    ("What is the description for \"Balanced Fund\"?", "database"),
    ("What is the best performing portfolio by MTD return?", "database"),
    ("What is the best performing portfolio by all-time return?", "database"),
    ("What is the total net liquidity across all portfolios for account U17032472?", "database"),
    ("Which portfolios have negative returns?", "database"),
    ("What is the best performing portfolio by WTD return?", "database"),
    ("What is the best performing portfolio by YTD return?", "database"),
    ("Give me a summary of my portfolio.", "database"),
    ("What is the best performing portfolio by QTD return?", "database"),
    ("What is the highest overall profit any investment generated YTD?", "database"),
    ("What investment has the best performance in terms of YTD profit?", "database"),
    ("Which asset group generated the highest total profit YTD?", "database"),
    ("Which portfolio has the highest YTD profit?", "database"),
    ("What is the YTD unrealized profit/loss for MSFT?", "database"),
    ("Top 10 profit stocks", "database"),
    ("List the top 10 most profitable stocks overall.", "database"),
    ("Which portfolio has the highest overall profit this year?", "database"),
    ("List all investments that have a negative total profit.", "database"),
    ("Bottom 10 stocks", "database"),
    ("What is the YTD profit for A-balanced?", "database"),
    ("Which portfolio has the highest QTD profit?", "database"),
    ("Compare performance between different portfolio groups.", "database"),
    ("Show the total quantity held for each stock symbol, sorted by the highest quantity first.", "database"),
    ("Show me any portfolios that are losing money overall.", "database"),
    ("How do portfolios perform across groups?", "database"),

    # Internet Questions
    ("What's the current price of Tesla?", "internet_data"),
    ("Give me the latest news on Apple.", "internet_data"),
    ("What's the S&P 500 performance today?", "internet_data"),
    ("Who are the top gainers in the Nasdaq this week?", "internet_data"),
    ("Is there any news about oil prices affecting markets?", "internet_data"),
    ("If I had invested AED 10,000 in NVIDIA in January, how much would it be worth now?", "internet_data"),
]



# ==============================================================================
# COMPARISON LOGIC
# ==============================================================================

def run_comparison(test_questions: list, sql_executor=None) -> dict:
    """
    Run both routers on the test questions and compare results.
    
    Args:
        test_questions: List of (question, expected_category) tuples
        sql_executor: Optional SQL executor for database entity context
        
    Returns:
        Dictionary with comparison results
    """
    if not test_questions:
        print("⚠️  No test questions provided!")
        print("   Please add questions to TEST_QUESTIONS in scripts/compare_routers.py")
        return {}

    print("=" * 80)
    print("ROUTER COMPARISON TEST: GPT-OSS v1 vs Optimized v2")
    print("=" * 80)
    print(f"Total questions: {len(test_questions)}")
    print()

    # Initialize routers
    print("Initializing routers...")
    v1_router = GPTOSSQueryRouter(sql_executor=sql_executor)
    v2_router = OptimizedQueryRouter(sql_executor=sql_executor)
    print("✅ Routers initialized\n")

    # Results storage
    results = []
    v1_correct = 0
    v2_correct = 0
    v1_times = []
    v2_times = []

    # Run tests
    for i, (question, expected) in enumerate(test_questions, 1):
        print(f"[{i}/{len(test_questions)}] Testing: {question[:60]}...")
        
        # Test v1 router (GPTOSSQueryRouter)
        start = time.time()
        v1_result = v1_router.classify_query(question)
        v1_time = time.time() - start
        v1_times.append(v1_time)
        v1_match = v1_result == expected
        if v1_match:
            v1_correct += 1

        # Sleep 5 seconds to give models time for better answers
        print("   ⏳ Waiting 5 seconds before v2 router...")
        time.sleep(5)

        # Test v2 router (OptimizedQueryRouter)
        start = time.time()
        v2_result = v2_router.classify_query(question)
        v2_time = time.time() - start
        v2_times.append(v2_time)
        v2_match = v2_result == expected
        if v2_match:
            v2_correct += 1

        # Store result
        results.append({
            "question": question,
            "expected": expected,
            "v1_result": v1_result,
            "v1_correct": v1_match,
            "v1_time_sec": round(v1_time, 3),
            "v2_result": v2_result,
            "v2_correct": v2_match,
            "v2_time_sec": round(v2_time, 3),
        })

        # Print inline result with timing
        v1_status = "✓" if v1_match else f"✗ ({v1_result})"
        v2_status = "✓" if v2_match else f"✗ ({v2_result})"
        print(f"   Expected: {expected}")
        print(f"   v1 (GPT-OSS):   {v1_status} | Time: {v1_time:.3f}s")
        print(f"   v2 (Optimized): {v2_status} | Time: {v2_time:.3f}s")
        print()

        # Sleep 5 seconds before next question
        if i < len(test_questions):
            print("   ⏳ Waiting 5 seconds before next question...")
            time.sleep(5)

    # Calculate metrics
    total = len(test_questions)
    v1_accuracy = (v1_correct / total) * 100 if total > 0 else 0
    v2_accuracy = (v2_correct / total) * 100 if total > 0 else 0
    
    avg_v1_time = sum(v1_times) / len(v1_times) if v1_times else 0
    avg_v2_time = sum(v2_times) / len(v2_times) if v2_times else 0
    
    min_v1_time = min(v1_times) if v1_times else 0
    max_v1_time = max(v1_times) if v1_times else 0
    min_v2_time = min(v2_times) if v2_times else 0
    max_v2_time = max(v2_times) if v2_times else 0

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'GPT-OSS v1':<20} {'Optimized v2':<20}")
    print("-" * 65)
    print(f"{'Correct':<25} {v1_correct}/{total:<18} {v2_correct}/{total:<18}")
    print(f"{'Accuracy':<25} {v1_accuracy:.1f}%{'':<17} {v2_accuracy:.1f}%{'':<17}")
    print(f"{'Avg Time (sec)':<25} {avg_v1_time:.3f}{'':<17} {avg_v2_time:.3f}{'':<17}")
    print(f"{'Min Time (sec)':<25} {min_v1_time:.3f}{'':<17} {min_v2_time:.3f}{'':<17}")
    print(f"{'Max Time (sec)':<25} {max_v1_time:.3f}{'':<17} {max_v2_time:.3f}{'':<17}")
    print(f"{'Total Time (sec)':<25} {sum(v1_times):.2f}{'':<17} {sum(v2_times):.2f}{'':<17}")
    print()

    # Winner announcement
    print("=" * 80)
    print("WINNER DETERMINATION")
    print("=" * 80)
    
    if v1_accuracy > v2_accuracy:
        print("🏆 Winner: GPT-OSS v1 (higher accuracy)")
    elif v2_accuracy > v1_accuracy:
        print("🏆 Winner: Optimized v2 (higher accuracy)")
    else:
        if avg_v1_time < avg_v2_time:
            print("🏆 Winner: GPT-OSS v1 (same accuracy, faster)")
        elif avg_v2_time < avg_v1_time:
            print("🏆 Winner: Optimized v2 (same accuracy, faster)")
        else:
            print("🏆 Tie: Both routers performed equally")
    
    # Speed comparison
    if avg_v2_time < avg_v1_time:
        speedup = avg_v1_time / avg_v2_time if avg_v2_time > 0 else 0
        print(f"\n⚡ Speed: Optimized v2 is {speedup:.1f}x faster than GPT-OSS v1")
    elif avg_v1_time < avg_v2_time:
        speedup = avg_v2_time / avg_v1_time if avg_v1_time > 0 else 0
        print(f"\n⚡ Speed: GPT-OSS v1 is {speedup:.1f}x faster than Optimized v2")

    print()

    return {
        "results": results,
        "v1_accuracy": v1_accuracy,
        "v2_accuracy": v2_accuracy,
        "v1_avg_time": avg_v1_time,
        "v2_avg_time": avg_v2_time,
        "v1_min_time": min_v1_time,
        "v2_min_time": min_v2_time,
        "v1_max_time": max_v1_time,
        "v2_max_time": max_v2_time,
    }


def save_results_to_csv(comparison_data: dict, output_path: str = None):
    """Save detailed results to CSV file"""
    if not comparison_data or "results" not in comparison_data:
        print("No results to save")
        return

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "router_comparison_results.csv"
        )

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "expected", 
            "v1_result", "v1_correct", "v1_time_sec",
            "v2_result", "v2_correct", "v2_time_sec"
        ])
        writer.writeheader()
        writer.writerows(comparison_data["results"])

    print(f"📄 CSV results saved to: {output_path}")


def save_results_to_text(comparison_data: dict, output_path: str = None):
    """Save detailed text report with misclassified questions for each method"""
    if not comparison_data or "results" not in comparison_data:
        print("No results to save")
        return

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "router_comparison_report.txt"
        )

    results = comparison_data["results"]
    
    # Separate misclassified questions by method
    v1_wrong = [r for r in results if not r["v1_correct"]]
    v2_wrong = [r for r in results if not r["v2_correct"]]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("QUERY ROUTER COMPARISON REPORT: GPT-OSS v1 vs Optimized v2\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Summary
        total = len(results)
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Questions: {total}\n\n")
        
        f.write(f"GPT-OSS v1 (Pure LLM):\n")
        f.write(f"  - Accuracy: {total - len(v1_wrong)}/{total} correct ({comparison_data['v1_accuracy']:.1f}%)\n")
        f.write(f"  - Avg Time: {comparison_data['v1_avg_time']:.3f}s\n")
        f.write(f"  - Min Time: {comparison_data['v1_min_time']:.3f}s\n")
        f.write(f"  - Max Time: {comparison_data['v1_max_time']:.3f}s\n\n")
        
        f.write(f"Optimized v2 (Tiered Pattern + LLM):\n")
        f.write(f"  - Accuracy: {total - len(v2_wrong)}/{total} correct ({comparison_data['v2_accuracy']:.1f}%)\n")
        f.write(f"  - Avg Time: {comparison_data['v2_avg_time']:.3f}s\n")
        f.write(f"  - Min Time: {comparison_data['v2_min_time']:.3f}s\n")
        f.write(f"  - Max Time: {comparison_data['v2_max_time']:.3f}s\n")
        f.write("\n")

        # v1 Misclassifications
        f.write("=" * 80 + "\n")
        f.write(f"GPT-OSS v1 - WRONG CLASSIFICATIONS ({len(v1_wrong)} errors)\n")
        f.write("=" * 80 + "\n")
        if v1_wrong:
            for i, r in enumerate(v1_wrong, 1):
                f.write(f"\n{i}. Question: {r['question']}\n")
                f.write(f"   Expected: {r['expected']}\n")
                f.write(f"   Got:      {r['v1_result']}\n")
                f.write(f"   Time:     {r['v1_time_sec']:.3f}s\n")
        else:
            f.write("\nNo misclassifications! ✓\n")
        f.write("\n")

        # v2 Misclassifications
        f.write("=" * 80 + "\n")
        f.write(f"Optimized v2 - WRONG CLASSIFICATIONS ({len(v2_wrong)} errors)\n")
        f.write("=" * 80 + "\n")
        if v2_wrong:
            for i, r in enumerate(v2_wrong, 1):
                f.write(f"\n{i}. Question: {r['question']}\n")
                f.write(f"   Expected: {r['expected']}\n")
                f.write(f"   Got:      {r['v2_result']}\n")
                f.write(f"   Time:     {r['v2_time_sec']:.3f}s\n")
        else:
            f.write("\nNo misclassifications! ✓\n")
        f.write("\n")

        # Full Results Table
        f.write("=" * 80 + "\n")
        f.write("FULL RESULTS WITH TIMING\n")
        f.write("=" * 80 + "\n\n")
        for i, r in enumerate(results, 1):
            v1_mark = "✓" if r["v1_correct"] else "✗"
            v2_mark = "✓" if r["v2_correct"] else "✗"
            f.write(f"{i}. {r['question'][:70]}...\n")
            f.write(f"   Expected: {r['expected']}\n")
            f.write(f"   v1: {v1_mark} {r['v1_result']} ({r['v1_time_sec']:.3f}s)\n")
            f.write(f"   v2: {v2_mark} {r['v2_result']} ({r['v2_time_sec']:.3f}s)\n\n")

    print(f"📄 Text report saved to: {output_path}")


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("QUERY ROUTER COMPARISON TOOL")
    print("GPT-OSS v1 (Pure LLM) vs Optimized v2 (Tiered Pattern + LLM)")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Check if test questions exist
    if not TEST_QUESTIONS:
        print("=" * 80)
        print("NO TEST QUESTIONS CONFIGURED")
        print("=" * 80)
        print("""
Please add your test questions to the TEST_QUESTIONS list in this file.

Format:
    TEST_QUESTIONS = [
        ("Your question here", "expected_category"),
        # Categories: "database", "internet_data", "greeting"
    ]

Example:
    TEST_QUESTIONS = [
        ("Hello", "greeting"),
        ("What are my holdings?", "database"),
        ("What is Apple stock price today?", "internet_data"),
    ]
""")
        return

    # Optional: Connect to database for entity context
    # Uncomment the following to enable database connectivity:
    # -----------------------------------------------------------
    # from src.services.sql_utilities import SQLExecutor
    # from src.config.settings import get_postgres_config
    # 
    # pg_config = get_postgres_config()
    # sql_executor = SQLExecutor(
    #     host=pg_config["host"],
    #     port=pg_config["port"],
    #     database=pg_config["database"],
    #     user=pg_config["user"],
    #     password=pg_config["password"]
    # )
    # results = run_comparison(TEST_QUESTIONS, sql_executor=sql_executor)
    # -----------------------------------------------------------

    # Run without database context
    results = run_comparison(TEST_QUESTIONS, sql_executor=None)

    # Save results
    if results:
        save_results_to_csv(results)
        save_results_to_text(results)


if __name__ == "__main__":
    main()
