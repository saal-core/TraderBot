"""Handler for database/portfolio queries.

Processes queries that require access to the portfolio database,
using Vanna AI for text-to-SQL conversion and result explanation.
"""
from typing import Set, Optional
import logging

from domain.entities import QueryContext, QueryResult
from domain.interfaces import LLMProvider, PortfolioRepository
from infrastructure.observability import get_logger, get_metrics
from .query_handler import QueryHandler


logger = get_logger(__name__)
metrics = get_metrics()


class DatabaseQueryHandler(QueryHandler):
    """
    Handler for portfolio/database queries.
    
    Workflow:
    1. Generate SQL from natural language using Vanna
    2. Execute the SQL query
    3. Generate natural language explanation of results
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        repository: PortfolioRepository,
        explanation_prompt: str = ""
    ):
        """
        Initialize database handler.
        
        Args:
            llm: LLM provider for explanations
            repository: Portfolio repository for database access
            explanation_prompt: System prompt for result explanation
        """
        self._llm = llm
        self._repository = repository
        self._explanation_prompt = explanation_prompt or self._default_explanation_prompt()
    
    @property
    def supported_classifications(self) -> Set[str]:
        return {"portfolio", "database"}
    
    def handle(self, context: QueryContext) -> QueryResult:
        """Process a database query."""
        logger.info("Processing database query", query=context.query[:50])
        
        with metrics.timer("database_query"):
            try:
                # Check connection
                if not self._repository.connect():
                    return QueryResult(
                        content="I'm sorry, the database connection is not available.",
                        sources=["database"]
                    )
                
                # Generate SQL
                sql = self._repository.generate_sql(context.query)
                if not sql or "SELECT" not in sql.upper():
                    metrics.increment("database_queries", tags={"status": "sql_gen_failed"})
                    return QueryResult(
                        content="I couldn't generate a valid query for that question. Please try rephrasing.",
                        sources=["database"]
                    )
                
                # Execute query
                result = self._repository.execute_query(sql)
                if result.is_failure:
                    metrics.increment("database_queries", tags={"status": "execution_failed"})
                    return QueryResult(
                        content=f"I couldn't retrieve that information: {result.error.message}",
                        sql_query=sql,
                        sources=["database"]
                    )
                
                df = result.value
                if df.empty:
                    metrics.increment("database_queries", tags={"status": "no_results"})
                    return QueryResult(
                        content="I found no data in the database matching your query.",
                        sql_query=sql,
                        sources=["database"]
                    )
                
                # Explain results
                explanation = self._explain_results(context, sql, df)
                
                metrics.increment("database_queries", tags={"status": "success"})
                return QueryResult(
                    content=f"### 🧠 Answer (From Database)\n{explanation}",
                    sql_query=sql,
                    data_frame=df,
                    sources=["portfolio_database"],
                    metadata={"row_count": len(df)}
                )
                
            except Exception as e:
                logger.error("Database query failed", exception=e)
                metrics.increment("database_queries", tags={"status": "error"})
                return QueryResult(
                    content="Sorry, I encountered an error processing your database query.",
                    sources=["database"]
                )
    
    def _explain_results(self, context: QueryContext, sql: str, df) -> str:
        """Generate natural language explanation of query results."""
        result_text = df.to_string(index=False)
        
        prompt = (
            f"My original question was: \"{context.query}\"\n\n"
            f"The database returned the following data:\n---\n{result_text}\n---\n\n"
            "Based on this data, please provide a clear explanation."
        )
        
        messages = [
            {"role": "system", "content": self._explanation_prompt},
        ]
        
        # Add recent chat history for context
        for msg in context.recent_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": prompt})
        
        return self._llm.chat(messages, temperature=0.3)
    
    def _default_explanation_prompt(self) -> str:
        """Return default explanation prompt."""
        return (
            "You are a financial assistant explaining database query results. "
            "Provide clear, concise explanations focused on the most important insights. "
            "Use bullet points for multiple data points. "
            "Do not include raw data tables - summarize the key information."
        )
