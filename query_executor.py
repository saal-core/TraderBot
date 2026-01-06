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
SELECT *
FROM ai_trading.portfolio_summary ps
LIMIT 10;
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
