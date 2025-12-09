# train_vanna.py

import os
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# --- 1. Vanna Setup ---
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        OLLAMA_MODEL = "gpt-oss:20b"
        ChromaDB_VectorStore.__init__(self, config={'persist_directory': './vanna_chromadb_store_tues'})
        Ollama.__init__(self, config={'model': OLLAMA_MODEL})

vn = MyVanna()

# --- 2. Database Connection ---
try:
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = "40.172.107.136", "AI-Trading-QNA", "aitradinguser", "UpIEvFTyAp51", 31937
    vn.connect_to_postgres(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    print("Successfully connected to the database.")
except Exception as e:
    print(f"Database connection failed: {e}. Training may be inaccurate.")

# ==============================================================================
# --- 3. VANNA TRAINING DATA ---
# ==============================================================================
print("Starting Vanna training... This may take a moment.")

# ==============================================================================
# == Part 1: Comprehensive Schema Documentation
# ==============================================================================

# ------------------------------------------------------------------------------
# -- Table: portfolio_holdings
# ------------------------------------------------------------------------------
vn.add_documentation(
    table_name='portfolio_holdings',
    schema_name='ai_trading',
    documentation="Contains historical, time-series data of individual asset holdings within each portfolio. Use this table for analyzing trends of a specific stock's value or position over time."
)
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='datetime', documentation='The exact timestamp of the data snapshot. Essential for historical analysis. Data Type: timestamp.')
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='portfolio_name', documentation="The name of the investment portfolio. Can also be called a 'wallet'. Use single quotes in WHERE clauses (e.g., `portfolio_name = 'Tech Growth'`). Data Type: character varying.")
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='symbol', documentation="The stock ticker symbol (e.g., 'AAPL'). Use single quotes in WHERE clauses. Data Type: character varying.")
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='positions', documentation='The number of shares held. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='market_value', documentation='The total market value of the positions at that datetime. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='ytd_unrealized_pl', documentation='Year-To-Date unrealized (on paper) profit or loss. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='created_timestamp', documentation='Timestamp when the record was created. Data Type: timestamp.')
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='updated_timestamp', documentation='Timestamp when the record was last updated. Data Type: timestamp.')
vn.add_documentation(table_name='portfolio_holdings', schema_name='ai_trading', column_name='last_updated_time', documentation='A more granular timestamp of the last modification. Data Type: timestamp.')

# ------------------------------------------------------------------------------
# -- Table: portfolio_holdings_realized_pnl
# ------------------------------------------------------------------------------
vn.add_documentation(
    table_name='portfolio_holdings_realized_pnl',
    schema_name='ai_trading',
    documentation='The most detailed table for profit and loss (P&L) analysis. Contains both realized (from sales) and unrealized profits. Use this for specific profit/loss questions.'
)
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='datetime', documentation='The date for which the P&L data is recorded. Data Type: timestamp.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='group_name', documentation="A high-level category for the portfolio (e.g., 'Equity'). Data Type: character varying.")
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='portfolio_name', documentation='The name of the investment portfolio. Data Type: character varying.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='symbol', documentation='The stock ticker symbol. Data Type: character varying.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='positions', documentation='Number of shares held. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='market_value', documentation='Total market value of the holding. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='ytd_unrealized_pnl', documentation='Year-To-Date unrealized (paper) profit or loss. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='ytd_realized_pnl', documentation='Year-To-Date realized (locked-in) profit or loss from sold positions. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='ytd_total_pnl', documentation='The sum of realized and unrealized P&L for the year. Best metric for overall profitability. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='created_timestamp', documentation='Timestamp when this record was created. Data Type: timestamp.')
vn.add_documentation(table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='daily_realized_pnl', documentation='Profit or loss realized on this specific day. Data Type: numeric.')
vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='utilized_amount',
    documentation="""The amount of the allocated capital that is currently invested. 
    **Hint**: To find 'funds available for new investments' or 'unutilized capital', you must calculate the difference: (`allocated_amount` - `utilized_amount`).
    Data Type: numeric."""
)
# ------------------------------------------------------------------------------
# -- Table: portfolio_summary
# ------------------------------------------------------------------------------
vn.add_documentation(
    table_name='portfolio_summary',
    schema_name='ai_trading',
    documentation='Aggregated summary table with high-level performance metrics for each portfolio. Best source for overall portfolio value, returns, and performance across different timeframes (YTD, QTD, MTD, etc.).'
)
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='datetime', documentation='The date for which the summary is calculated. Use MAX(datetime) to get the latest data. Data Type: timestamp.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='portfolio_name', documentation='The unique name of the investment portfolio. Data Type: character varying.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='net_liquidity', documentation='The total current value of the portfolio (cash + holdings). Data Type: numeric.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='ytd_return', documentation='The Year-To-Date return of the portfolio as a percentage. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='ytd_profit', documentation="The Year-To-Date profit in absolute monetary terms. Use this for questions about 'profit since the start of the year'. Data Type: numeric.")
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='all_return', documentation='The total return of the portfolio since its inception as a percentage. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='all_profit', documentation='The total profit of the portfolio since its inception in absolute monetary terms. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='allocated_amount', documentation='The total capital allocated to this portfolio. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='utilized_amount', documentation="The amount of allocated capital currently invested. **Hint**: 'Funds available for new investments' is calculated as (`allocated_amount` - `utilized_amount`). Data Type: numeric.")
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='is_active', documentation="A numeric flag indicating if the portfolio is active. **IMPORTANT**: In all SQL WHERE clauses, you MUST use `is_active = 1` for active portfolios. DO NOT use boolean values like TRUE. Data Type: numeric.")
# (Adding other key summary columns for completeness)
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='mtd_return', documentation='The Month-To-Date return as a percentage. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='qtd_return', documentation='The Quarter-To-Date return as a percentage. Data Type: numeric.')
vn.add_documentation(table_name='portfolio_summary', schema_name='ai_trading', column_name='daily_return', documentation='The daily return as a percentage. Data Type: numeric.')

# --- Find and REPLACE these two documentation blocks in train_vanna.py ---

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='allocated_amount',
    documentation="""The total capital allocated to this portfolio. 
    **CRITICAL**: This column is used to calculate 'available funds'.
    The formula is (`allocated_amount` - `utilized_amount`).
    Data Type: numeric."""
)

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='utilized_amount',
    documentation="""The amount of allocated capital that is currently invested. 
    **CRITICAL**: To answer any question about 'funds available for new investments', 'unutilized capital', or 'cash available', you MUST generate a SQL query that calculates (`allocated_amount` - `utilized_amount`). 
    Do NOT look for a separate 'cash' column.
    Data Type: numeric."""
)

print("Schema documentation added.")

# ==============================================================================
# == Part 2: Rich Question-SQL Examples
# ==============================================================================

# --- Basic Portfolio & Holding Information ---
vn.add_question_sql(question="How many portfolios are there?", sql="SELECT COUNT(DISTINCT portfolio_name) FROM ai_trading.portfolio_summary WHERE is_active = 1;")
vn.add_question_sql(question="List all active portfolios.", sql="SELECT DISTINCT portfolio_name FROM ai_trading.portfolio_summary WHERE is_active = 1 ORDER BY portfolio_name;")
vn.add_question_sql(question="Show me the holdings in the 'F-R&D-600K' portfolio.", sql="SELECT symbol, positions, market_value FROM ai_trading.portfolio_holdings WHERE portfolio_name = 'F-R&D-600K';")
vn.add_question_sql(question="What is the total value of all portfolios combined?", sql="SELECT SUM(net_liquidity) FROM ai_trading.portfolio_summary WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);")
vn.add_question_sql(question="What are the different portfolio groups?", sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary WHERE group_name IS NOT NULL;")

# --- Performance & Profit/Loss (using portfolio_summary) ---
vn.add_question_sql(question="ما هو ربح السنة حتى اليوم لكل محفظة", sql="SELECT portfolio_name, ytd_profit FROM ai_trading.portfolio_summary WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary) ORDER BY ytd_profit DESC;")
vn.add_question_sql(question="What is the Year-To-Date profit for each portfolio?", sql="SELECT portfolio_name, ytd_profit FROM ai_trading.portfolio_summary WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary) ORDER BY ytd_profit DESC;")
vn.add_question_sql(question="Which portfolio has the highest all-time return?", sql="SELECT portfolio_name, all_return FROM ai_trading.portfolio_summary ORDER BY all_return DESC LIMIT 1;")
vn.add_question_sql(question="List portfolios by their month-to-date return.", sql="SELECT portfolio_name, mtd_return FROM ai_trading.portfolio_summary ORDER BY mtd_return DESC;")
vn.add_question_sql(question="What is the daily return for the 'Tech Growth' portfolio?", sql="SELECT daily_return FROM ai_trading.portfolio_summary WHERE portfolio_name = 'Tech Growth' ORDER BY datetime DESC LIMIT 1;")

# --- Detailed Profit/Loss (using portfolio_holdings_realized_pnl) ---
vn.add_question_sql(question="What is the highest realized profit from a single stock?", sql="SELECT symbol, portfolio_name, ytd_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl ORDER BY ytd_realized_pnl DESC LIMIT 1;")
vn.add_question_sql(question="List the top 10 most profitable stocks overall.", sql="SELECT symbol, SUM(ytd_total_pnl) AS total_profit FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY symbol ORDER BY total_profit DESC LIMIT 10;")
vn.add_question_sql(question="Show me the 10 biggest losing stocks.", sql="SELECT symbol, SUM(ytd_total_pnl) AS total_profit FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY symbol ORDER BY total_profit ASC LIMIT 10;")
vn.add_question_sql(question="Which holdings are currently at a loss?", sql="SELECT symbol, portfolio_name, ytd_total_pnl FROM ai_trading.portfolio_holdings_realized_pnl WHERE ytd_total_pnl < 0 ORDER BY ytd_total_pnl ASC;")

# --- Capital & Funds Availability ---
vn.add_question_sql(question="What is the total amount of investment that was used?", sql="SELECT SUM(utilized_amount) AS total_investment_used FROM ai_trading.portfolio_summary WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);")
vn.add_question_sql(question="Which portfolio has the most cash available to invest?", sql="SELECT portfolio_name, (allocated_amount - utilized_amount) AS available_funds FROM ai_trading.portfolio_summary WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1) ORDER BY available_funds DESC LIMIT 10;")
vn.add_question_sql(question="List portfolios by their unutilized capital.", sql="SELECT portfolio_name, (allocated_amount - utilized_amount) AS unutilized_capital FROM ai_trading.portfolio_summary WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1) ORDER BY unutilized_capital DESC;")

# --- Historical Trend Analysis ---
vn.add_question_sql(question="What is the historical trend of the FDX holding’s market value in the F‑R&D‑600K portfolio?", sql="SELECT datetime, market_value FROM ai_trading.portfolio_holdings WHERE portfolio_name = 'F-R&D-600K' AND symbol = 'FDX' ORDER BY datetime ASC;")
vn.add_question_sql(question="Show me the market value history for AAPL in the 'Tech Growth' portfolio over time.", sql="SELECT datetime, market_value FROM ai_trading.portfolio_holdings WHERE portfolio_name = 'Tech Growth' AND symbol = 'AAPL' ORDER BY datetime ASC;")
vn.add_question_sql(question="Plot the trend of GOOG positions in the Alpha fund.", sql="SELECT datetime, positions FROM ai_trading.portfolio_holdings WHERE portfolio_name = 'Alpha fund' AND symbol = 'GOOG' ORDER BY datetime ASC;")

# --- Synonym Handling ('wallet') ---
vn.add_question_sql(question="How many wallets are there?", sql="SELECT COUNT(DISTINCT portfolio_name) FROM ai_trading.portfolio_summary WHERE is_active = 1;")
vn.add_question_sql(question="Show me the profit since the start of the year for every wallet.", sql="SELECT portfolio_name, ytd_profit FROM ai_trading.portfolio_summary WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary) ORDER BY ytd_profit DESC;")

# --- Add these to your Vanna setup section ---

vn.add_question_sql(
    question="Which portfolios contain the largest amount of funds available for new investments?",
    sql="""
    SELECT portfolio_name, (allocated_amount - utilized_amount) AS available_funds
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY available_funds DESC;
    """
)

vn.add_question_sql(
    question="Which portfolio has the most cash available to invest?",
    sql="""
    SELECT portfolio_name, (allocated_amount - utilized_amount) AS available_funds
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY available_funds DESC
    LIMIT 10;
    """
)

vn.add_question_sql(
    question="List all portfolios by their unutilized capital.",
    sql="""
    SELECT portfolio_name, (allocated_amount - utilized_amount) AS unutilized_capital
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY unutilized_capital DESC;
    """
)

# --- Add these new examples to the Question-SQL section of train_vanna.py ---

vn.add_question_sql(
    question="Which wallet has the most money left to spend?",
    sql="""
    SELECT portfolio_name, (allocated_amount - utilized_amount) AS available_funds
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY available_funds DESC;
    """
)

vn.add_question_sql(
    question="Calculate the available funds for each portfolio.",
    sql="""
    SELECT portfolio_name, (allocated_amount - utilized_amount) AS available_funds
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY portfolio_name;
    """
)


vn.add_question_sql(
    question="Show the total quantity held for each stock symbol, sorted by the highest quantity first.",
    sql="""
    SELECT symbol, sum(positions) AS qty FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY symbol ORDER BY qty DESC
    """
)


# --- Add this new example to train_vanna.py ---

vn.add_question_sql(
    question="How do portfolios perform across groups?",
    sql="""
    SELECT 
        group_name,
        COUNT(portfolio_name) AS number_of_portfolios,
        AVG(ytd_return) AS average_ytd_return,
        SUM(ytd_profit) AS total_ytd_profit,
        AVG(all_return) AS average_all_time_return,
        SUM(all_profit) AS total_all_time_profit
    FROM 
        ai_trading.portfolio_summary
    WHERE 
        is_active = 1 
        AND group_name IS NOT NULL
        AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    GROUP BY 
        group_name
    ORDER BY 
        average_ytd_return DESC;
    """
)

vn.add_question_sql(
    question="Compare performance between different portfolio groups.",
    sql="""
    SELECT 
        group_name,
        COUNT(portfolio_name) AS number_of_portfolios,
        AVG(ytd_return) AS average_ytd_return,
        SUM(ytd_profit) AS total_ytd_profit,
        AVG(all_return) AS average_all_time_return,
        SUM(all_profit) AS total_all_time_profit
    FROM 
        ai_trading.portfolio_summary
    WHERE 
        is_active = 1 
        AND group_name IS NOT NULL
        AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    GROUP BY 
        group_name
    ORDER BY 
        average_ytd_return DESC;
    """
)

# --- Update this documentation in train_vanna.py ---

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='group_name',
    documentation="""A category the portfolio belongs to (e.g., 'Balanced', 'Aggressive').
    **Hint**: To compare performance across groups, you should GROUP BY this column and calculate aggregate metrics like AVG(ytd_return), SUM(ytd_profit), and COUNT(portfolio_name).
    Data Type: character varying."""
)


# --- Add these new examples to train_vanna.py ---

vn.add_question_sql(
    question="When was the last update for the F-R&D-600K portfolio?",
    sql="""
    SELECT last_updated_time
    FROM ai_trading.portfolio_summary
    WHERE portfolio_name = 'F-R&D-600K'
    ORDER BY datetime DESC
    LIMIT 1;
    """
)

vn.add_question_sql(
    question="What is the latest update timestamp for the 'Tech Growth' wallet?",
    sql="""
    SELECT last_updated_time
    FROM ai_trading.portfolio_summary
    WHERE portfolio_name = 'Tech Growth'
    ORDER BY datetime DESC
    LIMIT 1;
    """
)

vn.add_question_sql(
    question="Show me the last time the portfolio summaries were updated.",
    sql="""
    SELECT MAX(last_updated_time) AS latest_update
    FROM ai_trading.portfolio_summary;
    """
)

# --- Update this documentation in train_vanna.py ---

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='last_updated_time',
    documentation="""The precise timestamp of the last data update for the portfolio summary.
    **CRITICAL**: This column is named `last_updated_time`. When asked for the 'last update' or 'latest update time', you MUST use this column. 
    Do NOT use `last_updated` or `updated_at` for this table.
    Data Type: timestamp without time zone."""
)


# --- Add these new examples to train_vanna.py ---

vn.add_question_sql(
    question="what is the total investment amount across all portfolios",
    sql="""
    SELECT SUM(allocated_amount) AS total_investment_across_all_portfolios 
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);
    """
)

vn.add_question_sql(
    question="What is the total capital allocated across all active portfolios?",
    sql="""
    SELECT SUM(allocated_amount) AS total_capital_allocated
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);
    """
)

vn.add_question_sql(
    question="Sum up the allocated amounts for every portfolio.",
    sql="""
    SELECT SUM(allocated_amount) AS total_allocated_amount
    FROM ai_trading.portfolio_summary
    WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);
    """
)

# --- Update this documentation in train_vanna.py ---

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='allocated_amount',
    documentation="""The total capital allocated to this portfolio. 
    **CRITICAL**: When asked for 'total investment amount' or 'total capital allocated', you MUST use this column. 
    The column name is `allocated_amount`. Do NOT use `total_capital_allocated`.
    Data Type: numeric."""
)

# --- Update this documentation in train_vanna.py ---

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='portfolio_description',
    documentation="""This column contains a text description of the portfolio's strategy.
    **CRITICAL**: The column name is `portfolio_description`. 
    When asked for a portfolio's 'description' or 'strategy', you MUST use this column. 
    Do NOT use a column named `strategy`.
    Data Type: character varying."""
)


# --- Add this new example to train_vanna.py ---

vn.add_question_sql(
    question="What is the description of each portfolio?",
    sql="""
    SELECT portfolio_name, portfolio_description
    FROM ai_trading.portfolio_summary
    WHERE portfolio_description IS NOT NULL;
    """
)

vn.add_question_sql(
    question="Show me the strategy for every portfolio.",
    sql="""
    SELECT portfolio_name, portfolio_description
    FROM ai_trading.portfolio_summary
    WHERE portfolio_description IS NOT NULL;
    """
)

# --- Add this block to train_vanna.py ---

# --- Set the Default Behavior for Ambiguous "Highest Returns" Questions ---
# --- Update this documentation in train_vanna.py ---

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='all_return',
    documentation="""The total return of the portfolio since its inception, as a percentage.
    **CRITICAL**: This column is named `all_return`. When a user asks for 'total return', 'all-time return', or 'return since inception', you MUST use the `all_return` column.
    Do NOT use a column named `total_return`.
    Data Type: numeric."""
)
# --- Add this new example to train_vanna.py ---

vn.add_question_sql(
    question="what is total return each portfolio since inception",
    sql="""
    SELECT 
        portfolio_name, 
        all_return
    FROM 
        ai_trading.portfolio_summary
    WHERE 
        is_active = 1 
        AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY 
        all_return DESC;
    """
)

vn.add_question_sql(
    question="Show me the all-time return for every portfolio.",
    sql="""
    SELECT 
        portfolio_name, 
        all_return
    FROM 
        ai_trading.portfolio_summary
    WHERE 
        is_active = 1 
        AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY 
        all_return DESC;
    """
)

# --- Add this block to train_vanna.py ---

# --- Enforce the "all_return" Default for Negative Returns Queries ---

vn.add_question_sql(
    question="which portfolios have negative returns",
    sql="""
    -- Defaulting to all-time return for the general question 'negative returns'
    SELECT 
        portfolio_name, 
        all_return
    FROM 
        ai_trading.portfolio_summary
    WHERE 
        is_active = 1 
        AND all_return < 0
        AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY 
        all_return ASC;
    """
)

vn.add_question_sql(
    question="Show me any portfolios that are losing money overall.",
    sql="""
    -- Defaulting to all-time return for 'losing money overall'
    SELECT 
        portfolio_name, 
        all_return,
        all_profit
    FROM 
        ai_trading.portfolio_summary
    WHERE 
        is_active = 1 
        AND all_return < 0
        AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
    ORDER BY 
        all_return ASC;
    """
)

# --- Update this documentation in train_vanna.py ---



# --- Add these new examples to train_vanna.py ---

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='portfolio_startdate',
    documentation="""The inception date of the portfolio (when it was created).
    **CRITICAL**: This column is named `portfolio_startdate`. When asked about 'new portfolios', their 'inception date', or 'creation date', you MUST use this column.
    Do NOT use a column named `inception_date`.
    Data Type: date."""
)

vn.add_documentation(
    table_name='portfolio_summary', 
    schema_name='ai_trading', 
    column_name='cost_model',
    documentation="""The cost basis accounting method used (e.g., 'FIFO').
    **CRITICAL**: This column is named `cost_model`. When asked about the 'cost model', 'cost basis', or 'accounting method', you MUST use this column.
    Do NOT use a column named `cost_basis_method`.
    Data Type: character varying."""
)

# --- Add these new examples to train_vanna.py ---

vn.add_question_sql(
    question="What cost models are being used for different portfolios?",
    sql="""
    SELECT DISTINCT portfolio_name, cost_model
    FROM ai_trading.portfolio_summary
    WHERE cost_model IS NOT NULL
    ORDER BY portfolio_name;
    """
)

vn.add_question_sql(
    question="Show me the cost basis method for each portfolio.",
    sql="""
    SELECT portfolio_name, cost_model
    FROM ai_trading.portfolio_summary
    WHERE cost_model IS NOT NULL
    ORDER BY portfolio_name;
    """
)

vn.add_question_sql(
    question="List the accounting methods for our wallets.",
    sql="""
    SELECT portfolio_name, cost_model
    FROM ai_trading.portfolio_summary
    WHERE cost_model IS NOT NULL;
    """
)

# --- Add this new example to train_vanna.py ---

vn.add_question_sql(
    question="What was my best-performing stock last quarter?",
    sql="""
    -- NOTE: Quarter-to-date data is not available at the individual stock level.
    -- This query provides the best-performing stock based on Year-To-Date (YTD) total profit as the closest approximation.
    SELECT 
        symbol,
        SUM(ytd_total_pnl) AS year_to_date_total_profit
    FROM 
        ai_trading.portfolio_holdings_realized_pnl
    GROUP BY
        symbol
    ORDER BY 
        year_to_date_total_profit DESC
    LIMIT 1;
    """
)

vn.add_question_sql(
    question="Which stock had the highest returns last month?",
    sql="""
    -- NOTE: Month-to-date data is not available at the individual stock level.
    -- This query provides the top 5 performing stocks based on Year-To-Date (YTD) total profit as the closest approximation.
    SELECT 
        symbol,
        SUM(ytd_total_pnl) AS year_to_date_total_profit
    FROM 
        ai_trading.portfolio_holdings_realized_pnl
    GROUP BY
        symbol
    ORDER BY 
        year_to_date_total_profit DESC
    LIMIT 5;
    """
)




print("Question-SQL examples added.")
print("\n--- Vanna training complete! The index is ready in './vanna_chromadb_store' ---")
