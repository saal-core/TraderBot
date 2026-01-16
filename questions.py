
ALL_QUESTIONS = [
    # Single Table - Portfolio Summary
    "How many active portfolios do we have?",
    "What is the total AUM (net liquidity) across all portfolios?",
    "Which portfolio has the highest YTD return?",
    "List all portfolios that are underperforming their benchmark YTD.",
    "What is the average MTD return across all active portfolios?",
    "Show me portfolios with negative QTD returns.",
    "Which portfolio has the most available cash (allocated minus utilized)?",
    "What is the total unrealized P&L across all portfolios?",
    "List portfolios sorted by utilization percentage (highest first).",
    "Which portfolios were started in 2023 or later?",
    "What is the default benchmark index for each portfolio?",
    "Show me the top 5 portfolios by all-time return since inception.",
    "Which portfolios have not been updated in the last 7 days?",
    "What cost models are being used across portfolios?",
    "Which portfolio group has the highest average YTD return?",
    
    # Single Table - Holdings
    "How many distinct stocks are held across all portfolios?",
    "What are the top 10 holdings by market value?",
    "Which portfolio has the most positions?",
    "Show all holdings for portfolio 'Growth Fund'.",
    "What is the total market value of all holdings in each portfolio?",
    "Which stock appears in the most portfolios?",
    "What was the most recently added holding across all portfolios?",
    "List holdings with YTD unrealized P&L greater than $10,000.",
    "What is the average position size (market value) per portfolio?",
    "Show me all AAPL positions across all portfolios.",
    
    # Single Table - Realized P&L
    "What are the top 5 most profitable stocks this year?",
    "Which stocks have the highest realized P&L YTD?",
    "Show me all holdings with negative total P&L.",
    "What is the total realized P&L by asset group (Equity, Crypto, ETF)?",
    "Which portfolio has the highest total realized gains?",
    "List the bottom 10 performing stocks by YTD total P&L.",
    "What is the split between realized and unrealized P&L for each asset group?",
    "Show holdings where unrealized P&L exceeds realized P&L.",
    "What is the total P&L for crypto holdings across all portfolios?",
    "Which symbols had realized P&L today (daily_realized_pnl > 0)?",
    
    # Two Table - Summary + Holdings
    "For each active portfolio, show the total number of holdings and total market value.",
    "Which portfolios have more than 20 distinct holdings?",
    "Show the top 3 holdings by market value for the best performing portfolio (by YTD return).",
    "List all holdings for portfolios that are outperforming their benchmark.",
    "What is the average holding size for portfolios with YTD return above 10%?",
    "For portfolios with negative MTD return, show their largest holding.",
    "Which active portfolio has the highest concentration in a single stock (max holding as % of net liquidity)?",
    "Show holdings with unrealized P&L exceeding 5% of the portfolio's net liquidity.",
    "List portfolios where the sum of holdings market value differs from net liquidity by more than 10%.",
    "For each portfolio group, what is the average number of holdings?",
    
    # Two Table - Summary + Realized P&L
    "Show total realized and unrealized P&L for each active portfolio.",
    "Which portfolios have realized more than $50,000 in gains this year?",
    "For portfolios beating their benchmark, what is the breakdown of P&L by asset group?",
    "Show the total P&L (realized + unrealized) as a percentage of allocated amount for each portfolio.",
    "Which portfolio has the highest ratio of realized to unrealized P&L?",
    "For each active portfolio, show how much of the total P&L comes from Equity vs Crypto vs ETF.",
    "List portfolios where unrealized losses exceed realized gains.",
    "What is the average YTD return for portfolios grouped by their dominant asset class?",
    "Show portfolios where daily realized P&L exceeds the portfolio's daily profit.",
    "For portfolios started before 2022, what is the total P&L by asset group?",
    
    # Two Table - Holdings + Realized P&L
    "For each holding, show positions from both tables and flag any discrepancies.",
    "Show holdings where market value differs between the two tables by more than 1%.",
    "List all holdings with their realized P&L, sorted by market value.",
    "Which stocks have positions in holdings table but no matching record in realized P&L table?",
    "For each symbol, show total positions, market value, and YTD total P&L across all portfolios.",
    
    # Three Table - Complex
    "For each active portfolio, show YTD return, top 3 holdings by market value, and the realized P&L for those holdings.",
    "Show portfolios beating their benchmark along with their asset group breakdown and total P&L per group.",
    "For the top 5 portfolios by net liquidity, show all holdings with their realized and unrealized P&L.",
    "Which portfolios have crypto holdings that represent more than 20% of the portfolio's net liquidity? Show the crypto symbols and their P&L.",
    "For each portfolio group, show the total AUM, number of holdings, and total P&L broken down by asset type.",
    "Show all portfolios where the top holding (by market value) has negative total P&L.",
    "For portfolios with utilization above 80%, show their holdings sorted by P&L percentage (P&L / market value).",
    "List portfolios where more than 50% of holdings (by count) have negative unrealized P&L.",
    "For each active portfolio, calculate the weighted average P&L percentage across all holdings.",
    "Show portfolios where the sum of individual holding P&L differs from the portfolio-level unrealized P&L by more than 5%.",
    
    # Follow-up Sequences
    "What is my total AUM and how is it distributed across portfolio groups?",
    "For the largest portfolio group by AUM, which individual portfolios are in it and what are their YTD returns?",
    "For the best performing portfolio in that group, what are its top 5 holdings?",
    "What is the realized vs unrealized P&L breakdown for those top 5 holdings?",
    #----------------------------------------------------------------
    #---------------------------------------------------------------
    "How many of my portfolios are outperforming their benchmark YTD?",
    "For these portfolios , what asset groups are driving the returns?",
    "Within the asset group, which specific stocks have the highest total P&L?",
    "What is the realized vs unrealized P&L breakdown for those top 5 holdings?",
    #----------------------------------------------------------------
    #---------------------------------------------------------------
    "Which holdings represent more than 30% of any single portfolio's net liquidity?",
    "For those holdings, what is their current P&L status?",
    "For the most  symbol across all portfolios, what is the total exposure and combined P&L?",
    #----------------------------------------------------------------
    #---------------------------------------------------------------
    "Which portfolios have negative YTD returns while their benchmark is positive?",
    "For the worst underperforming portfolio, what is the P&L breakdown by asset group?",
    "Within the worst performing asset group, which specific holdings are dragging performance?",
    #----------------------------------------------------------------
    "What percentage of the portfolio do these losing positions represent, and when were they added?",
    "What is the total available cash across all portfolios (allocated minus utilized)?",
    "Which portfolios have the lowest utilization percentage and highest available cash?",
    "For those under-deployed portfolios, what is their current performance vs benchmark?",
    #----------------------------------------------------------------
    "What asset groups do those portfolios currently hold, and what is the P&L for each group?",
    "What is the total realized P&L across all portfolios this year?",
    "Which asset group has generated the most realized gains?",
    "Within that asset group, what are the top 10 stocks by realized P&L?",
    #----------------------------------------------------------------
    "Which portfolios hold those winning stocks, and what is the position size in each?",
    "What is the total daily profit across all active portfolios?",
    "Which portfolios had positive daily returns today?",
    "For those portfolios, which holdings had realized P&L today?",
    #----------------------------------------------------------------
    "What is the breakdown of today's realized P&L by asset group?",
    "Compare the YTD return of all portfolios in the 'Aggressive' group vs 'Conservative' group.",
    "For each group, what is the average number of holdings and average position size?",
    "What is the asset group allocation (Equity/Crypto/ETF) for each portfolio group?",
    "Which portfolio group has the better risk-adjusted return (return per dollar of unrealized P&L)?",
    
    # Edge Cases
    "Show portfolios where allocated_amount is zero or null.",
    "Are there any holdings without a matching portfolio in portfolio_summary?",
    "List symbols that appear in holdings but not in realized_pnl table.",
    "Show portfolios where the number of holdings differs between holdings and realized_pnl tables.",
    "What is the data freshness - when was each table last updated?",
    
    # Aggregate Summaries
    "Give me a complete portfolio summary: count, total AUM, average return, total P&L.",
    "What percentage of portfolios are beating their benchmark?",
    "What is the total P&L split by asset group across all portfolios?",
    "Show the distribution of portfolios by YTD return ranges (negative, 0-5%, 5-10%, 10%+).",
    "What is the total exposure to each symbol across all portfolios?",
]
