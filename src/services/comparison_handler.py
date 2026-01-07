"""
Comparison and Hybrid Query Handler for orchestrating local database and internet data.
Handles queries that require both local portfolio data and external market data.
Supports both comparison queries AND general hybrid queries.
"""
import time
import json
import re
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.llm_provider import get_llm, get_streaming_llm, get_active_provider, get_provider_config
from src.services.chat_memory import ChatMemory
from src.services.database_handler import DatabaseQueryHandler
from src.services.internet_data_handler import InternetDataHandler
from src.config.prompts import (
    COMPARISON_PLAN_PROMPT, COMPARISON_EXPLANATION_PROMPT, PARTIAL_COMPARISON_PROMPT,
    HYBRID_PLAN_PROMPT, HYBRID_EXPLANATION_PROMPT
)
from dotenv import load_dotenv
load_dotenv()

# TOON formatter for token-efficient data formatting
from src.utils.toon_formatter import dataframe_to_toon

class ComparisonHandler:
    """
    Orchestrates comparison queries that require both local database 
    and external internet data.
    """

    def __init__(
        self,
        db_handler: DatabaseQueryHandler = None,
        internet_handler: InternetDataHandler = None,
        sql_executor=None,
        use_openai: bool = True,
        memory_max_pairs: int = 5
    ):
        """
        Initialize the comparison handler.

        Args:
            db_handler: Existing DatabaseQueryHandler instance (or creates new one)
            internet_handler: Existing InternetDataHandler instance (or creates new one)
            sql_executor: SQL executor for database queries
            use_openai: Deprecated - now uses LLM_PROVIDER env var
            memory_max_pairs: Maximum number of Q&A pairs to remember
        """
        # Initialize or use existing handlers
        self.db_handler = db_handler or DatabaseQueryHandler(sql_executor=sql_executor)
        self.internet_handler = internet_handler or InternetDataHandler(
            memory_max_pairs=memory_max_pairs
        )
        self.sql_executor = sql_executor
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

        # Get LLM from provider configuration
        provider = get_active_provider()
        
        # LLM for planning (low temperature for structured output)
        self.planning_llm = get_llm(temperature=0.1)
        
        # LLM for explanation (higher temperature for natural text)
        self.explanation_llm = get_streaming_llm(temperature=0.7)

        # Create planning chain
        self.plan_prompt = PromptTemplate(
            input_variables=["query"],
            template=COMPARISON_PLAN_PROMPT
        )
        self.plan_chain = self.plan_prompt | self.planning_llm | StrOutputParser()

        # Create explanation chain
        self.explain_prompt = PromptTemplate(
            input_variables=["query", "comparison_type", "local_data", "external_data"],
            template=COMPARISON_EXPLANATION_PROMPT
        )
        self.explain_chain = self.explain_prompt | self.explanation_llm | StrOutputParser()

        # Create partial comparison chain
        self.partial_prompt = PromptTemplate(
            input_variables=["query", "available_data", "missing_data"],
            template=PARTIAL_COMPARISON_PROMPT
        )
        self.partial_chain = self.partial_prompt | self.explanation_llm | StrOutputParser()

        # Create hybrid query planning chain
        self.hybrid_plan_prompt = PromptTemplate(
            input_variables=["query"],
            template=HYBRID_PLAN_PROMPT
        )
        self.hybrid_plan_chain = self.hybrid_plan_prompt | self.planning_llm | StrOutputParser()

        # Create hybrid explanation chain
        self.hybrid_explain_prompt = PromptTemplate(
            input_variables=["query", "query_type", "database_data", "internet_data"],
            template=HYBRID_EXPLANATION_PROMPT
        )
        self.hybrid_explain_chain = self.hybrid_explain_prompt | self.explanation_llm | StrOutputParser()

        print(f"✅ ComparisonHandler initialized with {provider.upper()} (supports comparison + hybrid queries)")

    def _plan_comparison(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to extract what needs to be compared from the query.

        Args:
            query: User's comparison question

        Returns:
            Dictionary with comparison plan
        """
        start_time = time.time()
        print(f"⏱️  Starting: Comparison Planning...")

        try:
            result = self.plan_chain.invoke({"query": query})
            
            # Clean up the result - remove markdown code blocks if present
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join([
                    line for line in lines 
                    if not line.startswith("```") and "json" not in line.lower()
                ])
                result = result.strip()

            # Parse JSON
            plan = json.loads(result)
            
            elapsed = time.time() - start_time
            print(f"✅ Completed: Comparison Planning in {elapsed:.2f}s")
            print(f"  → Comparison Type: {plan.get('comparison_type', 'unknown')}")
            print(f"  → Local Entity: {plan.get('local_entity', 'N/A')}")
            print(f"  → External Entity: {plan.get('external_entity', 'N/A')}")

            return plan

        except json.JSONDecodeError as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: JSON parsing error in comparison planning after {elapsed:.2f}s - {e}")
            # Return a default plan
            return {
                "comparison_type": "general_comparison",
                "local_entity": "portfolio data",
                "local_query_hint": query,
                "external_entity": "market data",
                "external_query_hint": query,
                "comparison_metrics": ["performance", "value"],
                "time_period": "current"
            }
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: Comparison planning after {elapsed:.2f}s - Error: {e}")
            return {
                "comparison_type": "general_comparison",
                "local_entity": "portfolio data",
                "local_query_hint": query,
                "external_entity": "market data", 
                "external_query_hint": query,
                "comparison_metrics": ["performance"],
                "time_period": "current"
            }

    def _fetch_local_data(
        self, 
        query_hint: str, 
        chat_history: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str], Optional[Any], Optional[str]]:
        """
        Fetch data from local database.

        Args:
            query_hint: Natural language query for database
            chat_history: Conversation history for context

        Returns:
            Tuple of (success, formatted_data_string, results_df, sql_query)
        """
        start_time = time.time()
        print(f"⏱️  Starting: Local Database Query...")
        print(f"  → Query: {query_hint}")

        try:
            # Generate SQL query
            sql_query = self.db_handler.generate_sql(query_hint, chat_history)
            
            if sql_query.startswith("ERROR"):
                print(f"❌ SQL generation failed: {sql_query}")
                return False, None, None, sql_query

            # Execute query
            success, results_df, message = self.sql_executor.execute_query(sql_query)
            
            if not success:
                print(f"❌ Query execution failed: {message}")
                return False, None, None, sql_query

            # Format results for LLM consumption using TOON format
            if results_df is not None and not results_df.empty:
                # Use TOON format for token efficiency (~45% reduction)
                formatted_data = dataframe_to_toon(results_df, "local_data")
                
                # Also create a summary
                row_count = len(results_df)
                col_names = list(results_df.columns)
                summary = f"Retrieved {row_count} rows with columns: {', '.join(col_names)}\n\n{formatted_data}"
                
                elapsed = time.time() - start_time
                print(f"✅ Completed: Local Database Query in {elapsed:.2f}s ({row_count} rows)")
                
                return True, summary, results_df, sql_query
            else:
                elapsed = time.time() - start_time
                print(f"⚠️ Completed: Local Database Query in {elapsed:.2f}s (no results)")
                return True, "No data found in local database for this query.", None, sql_query

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: Local Database Query after {elapsed:.2f}s - Error: {e}")
            return False, f"Error querying local database: {str(e)}", None, None

    def _fetch_external_data(
        self, 
        query_hint: str, 
        chat_history: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Fetch data from external internet sources.

        Args:
            query_hint: Natural language query for internet search
            chat_history: Conversation history for context

        Returns:
            Tuple of (success, formatted_data_string)
        """
        start_time = time.time()
        print(f"⏱️  Starting: External Internet Query...")
        print(f"  → Query: {query_hint}")

        try:
            # Use fetch_raw_data to avoid double-explanation
            # (ComparisonHandler will explain the combined data at the end)
            response = self.internet_handler.fetch_raw_data(query_hint, chat_history)
            
            if response and not response.startswith("Error"):
                elapsed = time.time() - start_time
                print(f"✅ Completed: External Internet Query in {elapsed:.2f}s")
                return True, response
            else:
                elapsed = time.time() - start_time
                print(f"⚠️ Completed: External Internet Query in {elapsed:.2f}s (limited data)")
                return True, response or "No external data available."

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: External Internet Query after {elapsed:.2f}s - Error: {e}")
            return False, f"Error fetching external data: {str(e)}"

    def _generate_comparison(
        self,
        query: str,
        comparison_type: str,
        local_data: Optional[str],
        external_data: Optional[str],
        local_success: bool,
        external_success: bool
    ) -> str:
        """
        Generate the comparison explanation using LLM.

        Args:
            query: Original user question
            comparison_type: Type of comparison being made
            local_data: Data from local database
            external_data: Data from internet
            local_success: Whether local fetch was successful
            external_success: Whether external fetch was successful

        Returns:
            Natural language comparison explanation
        """
        start_time = time.time()
        print(f"⏱️  Starting: Comparison Explanation Generation...")

        try:
            # Check if we have both data sources
            if local_success and external_success and local_data and external_data:
                # Full comparison
                explanation = self.explain_chain.invoke({
                    "query": query,
                    "comparison_type": comparison_type,
                    "local_data": local_data,
                    "external_data": external_data
                })
            else:
                # Partial comparison - handle missing data gracefully
                available_parts = []
                missing_parts = []

                if local_success and local_data:
                    available_parts.append(f"Local Portfolio Data:\n{local_data}")
                else:
                    missing_parts.append("Local portfolio data (could not retrieve from database)")

                if external_success and external_data:
                    available_parts.append(f"External Market Data:\n{external_data}")
                else:
                    missing_parts.append("External market data (could not fetch from internet)")

                available_str = "\n\n".join(available_parts) if available_parts else "No data available"
                missing_str = "\n".join(missing_parts) if missing_parts else "None"

                explanation = self.partial_chain.invoke({
                    "query": query,
                    "available_data": available_str,
                    "missing_data": missing_str
                })

            elapsed = time.time() - start_time
            print(f"✅ Completed: Comparison Explanation in {elapsed:.2f}s")

            return explanation.strip()

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: Comparison Explanation after {elapsed:.2f}s - Error: {e}")
            return f"I was able to gather some data but encountered an error generating the comparison: {str(e)}"

    def process(
        self, 
        query: str, 
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a comparison query by orchestrating local and external data fetching.

        Args:
            query: User's comparison question
            chat_history: Previous conversation history

        Returns:
            Dictionary with:
                - content: The comparison explanation
                - sql_query: SQL query used for local data (if any)
                - results_df: DataFrame from local query (if any)
                - local_data: Raw local data string
                - external_data: Raw external data string
                - comparison_plan: The plan used for comparison
        """
        if chat_history is None:
            chat_history = []

        print("\n" + "="*60)
        print("🔄 COMPARISON HANDLER - Processing Request")
        print("="*60)

        response = {
            "content": "",
            "sql_query": None,
            "results_df": None,
            "local_data": None,
            "external_data": None,
            "comparison_plan": None
        }

        try:
            # Step 1: Plan the comparison
            print("\n📋 Step 1: Planning Comparison...")
            plan = self._plan_comparison(query)
            response["comparison_plan"] = plan

            # Step 2: Fetch local data
            print("\n💾 Step 2: Fetching Local Portfolio Data...")
            local_success, local_data, results_df, sql_query = self._fetch_local_data(
                plan.get("local_query_hint", query),
                chat_history
            )
            response["local_data"] = local_data
            response["results_df"] = results_df
            response["sql_query"] = sql_query

            # Step 3: Fetch external data
            print("\n🌐 Step 3: Fetching External Market Data...")
            external_success, external_data = self._fetch_external_data(
                plan.get("external_query_hint", query),
                chat_history
            )
            response["external_data"] = external_data

            # Step 4: Generate comparison explanation
            print("\n📝 Step 4: Generating Comparison Analysis...")
            explanation = self._generate_comparison(
                query=query,
                comparison_type=plan.get("comparison_type", "general_comparison"),
                local_data=local_data,
                external_data=external_data,
                local_success=local_success,
                external_success=external_success
            )

            # Build final response content
            response["content"] = explanation

            # Add metadata about data sources
            sources_used = []
            if local_success and local_data:
                sources_used.append("📊 Local Portfolio Database")
            if external_success and external_data:
                sources_used.append("🌐 Real-time Market Data")

            if sources_used:
                response["content"] += f"\n\n---\n*Data Sources: {', '.join(sources_used)}*"

            print("\n" + "="*60)
            print("✅ COMPARISON HANDLER - Processing Complete")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n❌ COMPARISON HANDLER - Error: {e}")
            response["content"] = f"❌ Error processing comparison query: {str(e)}"

        return response

    def is_comparison_query(self, query: str) -> bool:
        """
        Only use ComparisonHandler when comparing LOCAL data to EXTERNAL benchmarks.
        """
        comparison_indicators = [
            # Pattern: portfolio + comparison word + external index
            r"compare.*portfolio.*(?:s&p|nasdaq|dow|market|benchmark|index)",
            r"(?:outperform|underperform|beat).*(?:market|index|benchmark)",
            r"portfolio.*(?:vs|versus|against).*(?:s&p|nasdaq|market)",
            r"benchmark.*(?:return|performance).*(?:since|ytd|current)",
        ]
    
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in comparison_indicators)

    def is_hybrid_query(self, query: str) -> bool:
        """
        Detect if query needs BOTH portfolio data AND market data (without comparison intent).
        """
        query_lower = query.lower()
        
        # Portfolio/database indicators
        portfolio_keywords = ['my portfolio', 'my holdings', 'my stocks', 'in my portfolio', 'i have', 'i own', 'my position']
        has_portfolio = any(kw in query_lower for kw in portfolio_keywords)
        
        # Market/internet indicators
        market_keywords = ['current price', 'market price', 'top gainers', 'top losers', 'market cap', 'today', 'current market', 'market movers']
        has_market = any(kw in query_lower for kw in market_keywords)
        
        return has_portfolio and has_market

    def _plan_hybrid(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to plan a hybrid query - what to fetch from database AND internet.
        
        Args:
            query: User's hybrid question
            
        Returns:
            Dictionary with hybrid query plan
        """
        start_time = time.time()
        print(f"⏱️  Starting: Hybrid Query Planning...")

        try:
            result = self.hybrid_plan_chain.invoke({"query": query})
            
            # Clean up the result - remove markdown code blocks if present
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join([
                    line for line in lines 
                    if not line.startswith("```") and "json" not in line.lower()
                ])
                result = result.strip()

            # Parse JSON
            plan = json.loads(result)
            
            elapsed = time.time() - start_time
            print(f"✅ Completed: Hybrid Query Planning in {elapsed:.2f}s")
            print(f"  → Query Type: {plan.get('query_type', 'unknown')}")
            print(f"  → Database Need: {plan.get('database_need', 'N/A')}")
            print(f"  → Internet Need: {plan.get('internet_need', 'N/A')}")

            return plan

        except json.JSONDecodeError as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: JSON parsing error in hybrid planning after {elapsed:.2f}s - {e}")
            # Return a default plan
            return {
                "query_type": "general_hybrid",
                "database_need": "portfolio data",
                "database_query_hint": query,
                "internet_need": "market data",
                "internet_query_hint": query,
                "combine_strategy": "combine and present both data sources"
            }
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: Hybrid planning after {elapsed:.2f}s - Error: {e}")
            return {
                "query_type": "general_hybrid",
                "database_need": "portfolio data",
                "database_query_hint": query,
                "internet_need": "market data", 
                "internet_query_hint": query,
                "combine_strategy": "combine and present both data sources"
            }

    def _generate_hybrid_response(
        self,
        query: str,
        query_type: str,
        database_data: Optional[str],
        internet_data: Optional[str],
        database_success: bool,
        internet_success: bool
    ) -> str:
        """
        Generate a combined response using both data sources.
        """
        start_time = time.time()
        print(f"⏱️  Starting: Hybrid Response Generation...")

        try:
            if database_success and internet_success and database_data and internet_data:
                # Full hybrid response
                explanation = self.hybrid_explain_chain.invoke({
                    "query": query,
                    "query_type": query_type,
                    "database_data": database_data,
                    "internet_data": internet_data
                })
            else:
                # Partial data available
                available_parts = []
                missing_parts = []

                if database_success and database_data:
                    available_parts.append(f"Portfolio Data:\n{database_data}")
                else:
                    missing_parts.append("Portfolio data (could not retrieve)")

                if internet_success and internet_data:
                    available_parts.append(f"Market Data:\n{internet_data}")
                else:
                    missing_parts.append("Market data (could not fetch)")

                available_str = "\n\n".join(available_parts) if available_parts else "No data available"
                missing_str = "\n".join(missing_parts) if missing_parts else "None"

                explanation = self.partial_chain.invoke({
                    "query": query,
                    "available_data": available_str,
                    "missing_data": missing_str
                })

            elapsed = time.time() - start_time
            print(f"✅ Completed: Hybrid Response in {elapsed:.2f}s")

            return explanation.strip()

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: Hybrid Response after {elapsed:.2f}s - Error: {e}")
            return f"I gathered data but encountered an error generating the response: {str(e)}"

    def process_hybrid(
        self, 
        query: str, 
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a hybrid query by fetching data from BOTH database AND internet.
        
        Args:
            query: User's hybrid question
            chat_history: Previous conversation history
            
        Returns:
            Dictionary with combined results from both sources
        """
        if chat_history is None:
            chat_history = []

        print("\n" + "="*60)
        print("🔀 HYBRID HANDLER - Processing Request")
        print("="*60)

        response = {
            "content": "",
            "sql_query": None,
            "results_df": None,
            "database_data": None,
            "internet_data": None,
            "hybrid_plan": None
        }

        try:
            # Step 1: Plan the hybrid query
            print("\n📋 Step 1: Planning Hybrid Query...")
            plan = self._plan_hybrid(query)
            response["hybrid_plan"] = plan

            # Step 2: Fetch database data
            print("\n💾 Step 2: Fetching Portfolio Data...")
            database_success, database_data, results_df, sql_query = self._fetch_local_data(
                plan.get("database_query_hint", query),
                chat_history
            )
            response["database_data"] = database_data
            response["results_df"] = results_df
            response["sql_query"] = sql_query

            # Step 3: Fetch internet data
            print("\n🌐 Step 3: Fetching Market Data...")
            internet_success, internet_data = self._fetch_external_data(
                plan.get("internet_query_hint", query),
                chat_history
            )
            response["internet_data"] = internet_data

            # Step 4: Generate combined response
            print("\n📝 Step 4: Generating Combined Response...")
            explanation = self._generate_hybrid_response(
                query=query,
                query_type=plan.get("query_type", "general_hybrid"),
                database_data=database_data,
                internet_data=internet_data,
                database_success=database_success,
                internet_success=internet_success
            )

            response["content"] = explanation

            # Add metadata about data sources
            sources_used = []
            if database_success and database_data:
                sources_used.append("📊 Portfolio Database")
            if internet_success and internet_data:
                sources_used.append("🌐 Real-time Market Data")

            if sources_used:
                response["content"] += f"\n\n---\n*Data Sources: {', '.join(sources_used)}*"

            print("\n" + "="*60)
            print("✅ HYBRID HANDLER - Processing Complete")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n❌ HYBRID HANDLER - Error: {e}")
            response["content"] = f"❌ Error processing hybrid query: {str(e)}"

        return response

    async def stream_analysis(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Stream an analysis based on the provided query and context data.
        """
        # Format context into a readable string
        context_str = ""
        for step_id, result in context.items():
            context_str += f"\n--- Data from Step {step_id} ---\n"
            if isinstance(result, list) and result:
                # It's likely database rows
                try:
                    import pandas as pd
                    df = pd.DataFrame(result)
                    context_str += dataframe_to_toon(df, f"Step_{step_id}_Data")
                except:
                    context_str += str(result)
            else:
                context_str += str(result)

        prompt = f"""You are a financial analyst.
User Query: {query}

Available Data:
{context_str}

Analyze the data and answer the user's question directly. 
Verify if the data supports the answer.
If comparing, highlight key differences.
Use HTML formatting just like other explanations (span class="currency", etc).

Analysis:"""

        # Stream response
        async for chunk in self.explanation_llm.astream(prompt):
             yield chunk.content