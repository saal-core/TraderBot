"""
Script to fetch distinct group names from the database.
"""
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
load_dotenv()

from query_executor import execute_query

def check_group_names():
    print("🚀 Fetching distinct group names from database...")
    
    print("\n--- Portfolio Summary Groups ---")
    execute_query("SELECT DISTINCT group_name FROM ai_trading.portfolio_summary")

    print("\n--- Realized P&L Groups ---")
    execute_query("SELECT DISTINCT group_name FROM ai_trading.portfolio_holdings_realized_pnl")

if __name__ == "__main__":
    check_group_names()
