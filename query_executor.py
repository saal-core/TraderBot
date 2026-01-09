import os
import psycopg2
from tabulate import tabulate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# PUT YOUR SQL QUERY HERE
# ============================================
sql_query = """
SELECT ph.symbol, ph.portfolio_name, ph.ytd_total_pnl,
       CASE WHEN ph.ytd_total_pnl > 0 THEN 'Profitable' ELSE 'Losing' END AS pl_status
FROM ai_trading.portfolio_holdings ph
JOIN (
    SELECT portfolio_name, SUM(market_value) AS total_value
    FROM ai_trading.portfolio_holdings
    WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_holdings)
    GROUP BY portfolio_name
) p ON ph.portfolio_name = p.portfolio_name
WHERE ph.datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_holdings)
  AND ph.market_value > 0.1 * p.total_value
ORDER BY ph.ytd_total_pnl DESC;
"""
# ============================================


def get_connection():
    """Create a database connection using .env credentials"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        options=f"-c search_path={os.getenv('DB_SCHEMA', 'public')}"
    )


def execute_query(query: str):
    """Execute a SQL query and print the results"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(query)
    
    # Fetch results
    if cursor.description:  # SELECT query
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        if rows:
            print(tabulate(rows, headers=columns, tablefmt="psql"))
            print(f"\n({len(rows)} rows)")
        else:
            print("No results returned.")
    else:
        conn.commit()
        print(f"Query executed. Rows affected: {cursor.rowcount}")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    execute_query(sql_query)
