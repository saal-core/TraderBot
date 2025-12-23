#!/usr/bin/env python3
"""
Router Comparison Script

Compares the accuracy and performance of:
1. QueryRouter (pattern-matching + LLM fallback)
2. GPTOSSQueryRouter (pure LLM-based classification)

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

from src.services.query_router import QueryRouter
from src.services.gpt_oss_query_router import GPTOSSQueryRouter


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

    print("=" * 70)
    print("ROUTER COMPARISON TEST")
    print("=" * 70)
    print(f"Total questions: {len(test_questions)}")
    print()

    # Initialize routers
    print("Initializing routers...")
    original_router = QueryRouter(sql_executor=sql_executor)
    gpt_oss_router = GPTOSSQueryRouter(sql_executor=sql_executor)
    print("✅ Routers initialized\n")

    # Results storage
    results = []
    original_correct = 0
    gpt_oss_correct = 0
    original_times = []
    gpt_oss_times = []

    # Run tests
    for i, (question, expected) in enumerate(test_questions, 1):
        print(f"[{i}/{len(test_questions)}] Testing: {question[:60]}...")
        
        # Test original router
        start = time.time()
        original_result = original_router.classify_query(question)
        original_time = time.time() - start
        original_times.append(original_time)
        original_match = original_result == expected
        if original_match:
            original_correct += 1

        # Sleep 5 seconds to give models time for better answers
        print("   ⏳ Waiting 5 seconds before GPT-OSS...")
        time.sleep(5)

        # Test GPT-OSS router
        start = time.time()
        gpt_oss_result = gpt_oss_router.classify_query(question)
        gpt_oss_time = time.time() - start
        gpt_oss_times.append(gpt_oss_time)
        gpt_oss_match = gpt_oss_result == expected
        if gpt_oss_match:
            gpt_oss_correct += 1

        # Store result
        results.append({
            "question": question,
            "expected": expected,
            "original_result": original_result,
            "original_correct": original_match,
            "original_time": original_time,
            "gpt_oss_result": gpt_oss_result,
            "gpt_oss_correct": gpt_oss_match,
            "gpt_oss_time": gpt_oss_time,
        })

        # Print inline result
        orig_status = "✓" if original_match else f"✗ ({original_result})"
        gpt_status = "✓" if gpt_oss_match else f"✗ ({gpt_oss_result})"
        print(f"   Expected: {expected} | Original: {orig_status} | GPT-OSS: {gpt_status}")
        print()

        # Sleep 5 seconds before next question
        if i < len(test_questions):
            print("   ⏳ Waiting 5 seconds before next question...")
            time.sleep(5)

    # Calculate metrics
    total = len(test_questions)
    original_accuracy = (original_correct / total) * 100 if total > 0 else 0
    gpt_oss_accuracy = (gpt_oss_correct / total) * 100 if total > 0 else 0
    
    avg_original_time = sum(original_times) / len(original_times) if original_times else 0
    avg_gpt_oss_time = sum(gpt_oss_times) / len(gpt_oss_times) if gpt_oss_times else 0

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Original Router':<20} {'GPT-OSS Router':<20}")
    print("-" * 65)
    print(f"{'Correct':<25} {original_correct}/{total:<18} {gpt_oss_correct}/{total:<18}")
    print(f"{'Accuracy':<25} {original_accuracy:.1f}%{'':<16} {gpt_oss_accuracy:.1f}%{'':<16}")
    print(f"{'Avg Time (sec)':<25} {avg_original_time:.3f}{'':<16} {avg_gpt_oss_time:.3f}{'':<16}")
    print(f"{'Total Time (sec)':<25} {sum(original_times):.2f}{'':<16} {sum(gpt_oss_times):.2f}{'':<16}")
    print()

    # Winner announcement
    if original_accuracy > gpt_oss_accuracy:
        print("🏆 Winner: Original Router (higher accuracy)")
    elif gpt_oss_accuracy > original_accuracy:
        print("🏆 Winner: GPT-OSS Router (higher accuracy)")
    else:
        if avg_original_time < avg_gpt_oss_time:
            print("🏆 Winner: Original Router (same accuracy, faster)")
        elif avg_gpt_oss_time < avg_original_time:
            print("🏆 Winner: GPT-OSS Router (same accuracy, faster)")
        else:
            print("🏆 Tie: Both routers performed equally")

    print()

    return {
        "results": results,
        "original_accuracy": original_accuracy,
        "gpt_oss_accuracy": gpt_oss_accuracy,
        "original_avg_time": avg_original_time,
        "gpt_oss_avg_time": avg_gpt_oss_time,
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
            "original_result", "original_correct", "original_time",
            "gpt_oss_result", "gpt_oss_correct", "gpt_oss_time"
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
    original_wrong = [r for r in results if not r["original_correct"]]
    gpt_oss_wrong = [r for r in results if not r["gpt_oss_correct"]]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("QUERY ROUTER COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Summary
        total = len(results)
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Questions: {total}\n")
        f.write(f"Original Router: {total - len(original_wrong)}/{total} correct ({comparison_data['original_accuracy']:.1f}%)\n")
        f.write(f"GPT-OSS Router:  {total - len(gpt_oss_wrong)}/{total} correct ({comparison_data['gpt_oss_accuracy']:.1f}%)\n")
        f.write(f"Original Avg Time: {comparison_data['original_avg_time']:.3f}s\n")
        f.write(f"GPT-OSS Avg Time:  {comparison_data['gpt_oss_avg_time']:.3f}s\n")
        f.write("\n")

        # Original Router Misclassifications
        f.write("=" * 70 + "\n")
        f.write(f"ORIGINAL ROUTER - WRONG CLASSIFICATIONS ({len(original_wrong)} errors)\n")
        f.write("=" * 70 + "\n")
        if original_wrong:
            for i, r in enumerate(original_wrong, 1):
                f.write(f"\n{i}. Question: {r['question']}\n")
                f.write(f"   Expected: {r['expected']}\n")
                f.write(f"   Got:      {r['original_result']}\n")
        else:
            f.write("\nNo misclassifications! ✓\n")
        f.write("\n")

        # GPT-OSS Router Misclassifications
        f.write("=" * 70 + "\n")
        f.write(f"GPT-OSS ROUTER - WRONG CLASSIFICATIONS ({len(gpt_oss_wrong)} errors)\n")
        f.write("=" * 70 + "\n")
        if gpt_oss_wrong:
            for i, r in enumerate(gpt_oss_wrong, 1):
                f.write(f"\n{i}. Question: {r['question']}\n")
                f.write(f"   Expected: {r['expected']}\n")
                f.write(f"   Got:      {r['gpt_oss_result']}\n")
        else:
            f.write("\nNo misclassifications! ✓\n")
        f.write("\n")

        # Full Results Table
        f.write("=" * 70 + "\n")
        f.write("FULL RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for i, r in enumerate(results, 1):
            orig_mark = "✓" if r["original_correct"] else "✗"
            gpt_mark = "✓" if r["gpt_oss_correct"] else "✗"
            f.write(f"{i}. {r['question'][:60]}...\n")
            f.write(f"   Expected: {r['expected']} | Original: {orig_mark} {r['original_result']} | GPT-OSS: {gpt_mark} {r['gpt_oss_result']}\n\n")

    print(f"📄 Text report saved to: {output_path}")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("QUERY ROUTER COMPARISON TOOL")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Check if test questions exist
    if not TEST_QUESTIONS:
        print("=" * 70)
        print("NO TEST QUESTIONS CONFIGURED")
        print("=" * 70)
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
