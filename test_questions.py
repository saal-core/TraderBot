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

questions = [
    "Which portfolio outperformed its YTD index return the most?",
    "Provide a summary of my entire investment portfolio?",
    "What is the performance in A-Balanced?",
    "What is the description for \"Balanced Fund\"?",
    "What is the best performing portfolio by MTD return?",
    "What is the best performing portfolio by all-time return?",
    "What is the total net liquidity across all portfolios for account U17032472?",
    "Which portfolios have negative returns?",
    "What is the best performing portfolio by WTD return?",
    "What is the best performing portfolio by YTD return?",
    "Give me a summary of my portfolio.",
    "What is the best performing portfolio by QTD return?",
    "What is the highest overall profit any investment generated YTD?",
    "What investment has the best performance in terms of YTD profit?",
    "Which asset group generated the highest total profit YTD?",
    "Which portfolio has the highest YTD profit?",
    "What is the YTD unrealized profit/loss for MSFT?",
    "Top 10 profit stocks",
    "List the top 10 most profitable stocks overall.",
    "Which portfolio has the highest overall profit this year?",
    "List all investments that have a negative total profit.",
    "Bottom 10 stocks",
    "What is the YTD profit for A-balanced?",
    "Which portfolio has the highest QTD profit?",
    "Compare performance between different portfolio groups.",
    "Show the total quantity held for each stock symbol, sorted by the highest quantity first.",
    "Show me any portfolios that are losing money overall.",
    "How do portfolios perform across groups?",
    "Compare ABalanced performance against QQQ."
]
