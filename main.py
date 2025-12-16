"""
TraderBot Main Application
A natural language interface for querying financial databases with 
comparison capabilities between local portfolio data and external market data.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict
import sys
import os
import json
from datetime import datetime

# Import configuration and custom modules
from src.config.settings import get_app_config, get_ollama_config, get_postgres_config
from src.services.sql_utilities import PostgreSQLExecutor
from src.services.query_router import QueryRouter
from src.services.database_handler import DatabaseQueryHandler
from src.services.greating_handler import GreetingHandler
from src.services.chat_memory import ChatMemory
from src.services.internet_data_handler import InternetDataHandler
from src.services.comparison_handler import ComparisonHandler

from dotenv import load_dotenv
load_dotenv()

# Page configuration
app_config = get_app_config()
st.set_page_config(
    page_title=app_config["title"],
    page_icon=app_config["icon"],
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sql-code {
        background-color: #282c34;
        color: #abb2bf;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .comparison-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False

    if "router" not in st.session_state:
        st.session_state.router = None

    if "db_handler" not in st.session_state:
        st.session_state.db_handler = None

    if "greeting_handler" not in st.session_state:
        st.session_state.greeting_handler = None

    if "internet_data_handler" not in st.session_state:
        st.session_state.internet_data_handler = None

    if "comparison_handler" not in st.session_state:
        st.session_state.comparison_handler = None

    if "sql_executor" not in st.session_state:
        st.session_state.sql_executor = None

    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ChatMemory(max_pairs=5)

    if "query_stats" not in st.session_state:
        st.session_state.query_stats = {
            "database": 0, 
            "greeting": 0, 
            "internet_data": 0,
            "comparison": 0
        }


def export_chat_history(format_type: str = "json") -> str:
    """
    Export chat history to JSON or TXT format

    Args:
        format_type: Either 'json' or 'txt'

    Returns:
        Formatted string of chat history
    """
    if format_type == "json":
        export_data = []
        for msg in st.session_state.messages:
            clean_msg = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.now().isoformat()
            }
            if "sql_query" in msg and msg["sql_query"]:
                clean_msg["sql_query"] = msg["sql_query"]
            if "query_type" in msg:
                clean_msg["query_type"] = msg["query_type"]
            export_data.append(clean_msg)
        return json.dumps(export_data, indent=2)
    else:  # txt format
        txt_lines = [f"TraderBot Chat History - Exported at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        txt_lines.append("=" * 80 + "\n\n")

        for i, msg in enumerate(st.session_state.messages, 1):
            txt_lines.append(f"Message {i} - {msg['role'].upper()}\n")
            txt_lines.append("-" * 40 + "\n")
            txt_lines.append(f"{msg['content']}\n")
            if "sql_query" in msg and msg["sql_query"]:
                txt_lines.append(f"\nSQL Query:\n{msg['sql_query']}\n")
            if "query_type" in msg:
                txt_lines.append(f"\nQuery Type: {msg['query_type']}\n")
            txt_lines.append("\n" + "=" * 80 + "\n\n")

        return "".join(txt_lines)


def get_query_statistics() -> Dict[str, int]:
    """
    Calculate statistics from chat history

    Returns:
        Dictionary with query type counts
    """
    stats = {"database": 0, "greeting": 0, "internet_data": 0, "comparison": 0, "total": 0}

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            stats["total"] += 1

    stats.update(st.session_state.query_stats)
    return stats


def initialize_handlers():
    """Initialize all handlers and database connection"""
    try:
        with st.spinner("Initializing chatbot components..."):
            ollama_config = get_ollama_config()

            # Initialize PostgreSQL executor
            st.session_state.sql_executor = PostgreSQLExecutor()

            # Test database connection
            success, message = st.session_state.sql_executor.test_connection()
            if not success:
                st.error(f"❌ {message}")
                return False

            st.success(message)

            # Initialize handlers
            st.session_state.router = QueryRouter(sql_executor=st.session_state.sql_executor)
            st.session_state.db_handler = DatabaseQueryHandler(sql_executor=st.session_state.sql_executor)
            st.session_state.greeting_handler = GreetingHandler()
            st.session_state.internet_data_handler = InternetDataHandler()
            
            # Initialize comparison handler with existing handlers
            st.session_state.comparison_handler = ComparisonHandler(
                db_handler=st.session_state.db_handler,
                internet_handler=st.session_state.internet_data_handler,
                sql_executor=st.session_state.sql_executor,
                use_openai=True  # Use OpenAI for better comparison planning
            )

            st.session_state.db_initialized = True
            st.success("✅ All components initialized successfully!")
            return True

    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        st.session_state.db_initialized = False
        return False


def display_chat_history():
    """Display chat message history"""
    max_display = int(os.getenv("MAX_DISPLAY_MESSAGES", 10))
    messages_to_display = st.session_state.messages[-max_display:]

    for message in messages_to_display:
        with st.chat_message(message["role"]):
            # Display timestamp if available
            if "timestamp" in message:
                st.caption(f"🕒 {message['timestamp']}")

            st.markdown(message["content"])

            # Display SQL query if present
            if "sql_query" in message and message["sql_query"]:
                with st.expander("🔍 View SQL Query"):
                    st.code(message["sql_query"], language="sql")

            # Display results dataframe if present
            if "results_df" in message and message["results_df"] is not None:
                with st.expander("📊 View Results Table"):
                    st.dataframe(message["results_df"], use_container_width=True)

            # Display comparison details if present
            if "comparison_plan" in message and message["comparison_plan"]:
                with st.expander("🔄 View Comparison Details"):
                    plan = message["comparison_plan"]
                    st.json(plan)


def process_database_query(user_query: str) -> Dict:
    """Process a database query with conversation context"""
    response = {
        "content": "",
        "sql_query": "",
        "results_df": None
    }

    try:
        # Get database schema
        schema = st.session_state.sql_executor.get_schema_info()

        if "Error" in schema:
            response["content"] = f"❌ Failed to retrieve database schema: {schema}"
            return response

        # Get chat history for follow-up questions
        chat_history = st.session_state.messages

        # Generate SQL with conversation context
        with st.spinner("🤖 Generating SQL query..."):
            sql_query = st.session_state.db_handler.generate_sql(user_query, schema, chat_history)

        if sql_query.startswith("ERROR"):
            response["content"] = f"❌ {sql_query}"
            return response

        response["sql_query"] = sql_query

        # Validate and execute query
        with st.spinner("✅ Validating and executing query..."):
            success, results_df, message = st.session_state.sql_executor.execute_query(sql_query)

        if not success:
            response["content"] = f"❌ Query execution failed: {message}"
            return response

        response["results_df"] = results_df

        # Generate natural language explanation
        with st.spinner("📝 Generating explanation..."):
            explanation = st.session_state.db_handler.explain_results(user_query, results_df, sql_query)

        response["content"] = f"✅ {explanation}\n\n💡 {message}"

    except Exception as e:
        response["content"] = f"❌ Error processing query: {str(e)}"

    return response


def process_greeting(user_query: str) -> Dict:
    """Process a greeting or chitchat with memory"""
    response = {
        "content": "",
        "sql_query": None,
        "results_df": None
    }

    try:
        with st.spinner("💬 Generating response..."):
            chat_history = st.session_state.messages
            greeting_response = st.session_state.greeting_handler.respond(user_query, chat_history)
            response["content"] = f"{greeting_response}"

    except Exception as e:
        response["content"] = f"❌ Error: {str(e)}"

    return response


def process_internet_data(user_query: str) -> Dict:
    """Process an internet data query using Perplexity API"""
    response = {
        "content": "",
        "sql_query": None,
        "results_df": None
    }

    try:
        with st.spinner("🌐 Fetching real-time data from the internet..."):
            chat_history = st.session_state.messages
            data_response = st.session_state.internet_data_handler.fetch_data(user_query, chat_history)
            response["content"] = f"🌐 {data_response}"

    except Exception as e:
        response["content"] = f"❌ Error: {str(e)}"

    return response


def process_comparison_query(user_query: str) -> Dict:
    """
    Process a comparison query that requires both local and external data.
    
    Args:
        user_query: User's comparison question
        
    Returns:
        Dictionary with comparison results
    """
    response = {
        "content": "",
        "sql_query": None,
        "results_df": None,
        "comparison_plan": None,
        "local_data": None,
        "external_data": None
    }

    try:
        with st.spinner("🔄 Processing comparison query..."):
            # Show sub-steps
            status_container = st.empty()
            
            status_container.info("📋 Step 1/4: Planning comparison...")
            
            chat_history = st.session_state.messages
            
            # Process comparison through the handler
            comparison_result = st.session_state.comparison_handler.process(
                user_query, 
                chat_history
            )
            
            # Extract results
            response["content"] = comparison_result.get("content", "")
            response["sql_query"] = comparison_result.get("sql_query")
            response["results_df"] = comparison_result.get("results_df")
            response["comparison_plan"] = comparison_result.get("comparison_plan")
            response["local_data"] = comparison_result.get("local_data")
            response["external_data"] = comparison_result.get("external_data")
            
            status_container.empty()

    except Exception as e:
        response["content"] = f"❌ Error processing comparison: {str(e)}"

    return response


def main():
    """Main application function"""

    # Header
    st.markdown(f'<div class="main-header">{app_config["icon"]} {app_config["title"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your PostgreSQL database in natural language</div>', unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Display current configuration
        ollama_config = get_ollama_config()
        postgres_config = get_postgres_config()

        st.subheader("🤖 Ollama Settings")
        st.text(f"Model: {ollama_config['model_name']}")
        st.text(f"URL: {ollama_config['base_url']}")

        st.subheader("🗄️ PostgreSQL Settings")
        if "connection_string" in postgres_config:
            st.text("Using connection string")
        else:
            st.text(f"Host: {postgres_config['host']}")
            st.text(f"Database: {postgres_config['database']}")
            st.text(f"User: {postgres_config['user']}")
            st.text(f"Port: {postgres_config['port']}")

        st.divider()

        if st.button("🔄 Initialize/Reinitialize", use_container_width=True):
            initialize_handlers()

        st.divider()

        # Display database schema
        if st.session_state.db_initialized and st.session_state.sql_executor:
            with st.expander("📋 View Database Schema"):
                schema_info = st.session_state.sql_executor.get_schema_info()
                st.text(schema_info)

        st.divider()

        # Quick Actions Panel
        st.subheader("⚡ Quick Actions")

        # Database queries
        with st.expander("💾 Database Queries", expanded=False):
            db_queries = [
                "Show me all portfolios",
                "Which stock has the highest profit?",
                "What are my top 5 holdings?",
            ]
            for query in db_queries:
                if st.button(query, key=f"db_{query}", use_container_width=True):
                    st.session_state.example_query = query

        # Comparison queries (NEW)
        with st.expander("🔄 Comparison Queries", expanded=False):
            comparison_queries = [
                "Compare my portfolio to S&P 500",
                "How does my portfolio perform against NASDAQ?",
                "Compare A-Balanced returns vs market benchmark",
                "Is my portfolio outperforming the market this year?",
            ]
            for query in comparison_queries:
                if st.button(query, key=f"comp_{query}", use_container_width=True):
                    st.session_state.example_query = query

        # Internet data queries
        with st.expander("🌐 Real-Time Data", expanded=False):
            internet_queries = [
                "What's the current price of Tesla?",
                "Give me the latest news on Apple",
                "What's the S&P 500 performance today?",
                "What's Bitcoin's price?",
            ]
            for query in internet_queries:
                if st.button(query, key=f"internet_{query}", use_container_width=True):
                    st.session_state.example_query = query

        # General queries
        with st.expander("💬 General", expanded=False):
            general_queries = [
                "Hello!",
                "What can you help me with?",
            ]
            for query in general_queries:
                if st.button(query, key=f"general_{query}", use_container_width=True):
                    st.session_state.example_query = query

        st.divider()

        # Configuration info
        with st.expander("ℹ️ Configuration Details"):
            st.json({
                "Ollama Model": ollama_config['model_name'],
                "Max Result Rows": os.getenv("MAX_RESULT_ROWS", 100),
                "Query Timeout": f"{os.getenv('QUERY_TIMEOUT', 30)} seconds",
                "SQL Injection Protection": os.getenv("ENABLE_SQL_INJECTION_PROTECTION", "True"),
            })

        st.divider()

        # Chat Management Section
        st.subheader("💬 Chat Management")

        # Display chat statistics
        if st.session_state.messages:
            stats = get_query_statistics()
            st.metric("Total Queries", stats["total"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("💾 DB", stats["database"])
                st.metric("🌐 Web", stats["internet_data"])
            with col2:
                st.metric("🔄 Compare", stats["comparison"])
                st.metric("💬 Chat", stats["greeting"])

            # Chat memory info
            if st.session_state.chat_memory:
                pairs_count = st.session_state.chat_memory.count_pairs(st.session_state.messages)
                st.caption(f"📝 Context Memory: {pairs_count}/5 pairs")

            st.divider()

            # Export options
            st.caption("📥 Export Chat History:")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Export TXT", use_container_width=True):
                    txt_content = export_chat_history("txt")
                    st.download_button(
                        label="⬇️ Download TXT",
                        data=txt_content,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            with col2:
                if st.button("📋 Export JSON", use_container_width=True):
                    json_content = export_chat_history("json")
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_content,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_stats = {
                "database": 0, 
                "greeting": 0, 
                "internet_data": 0,
                "comparison": 0
            }
            st.rerun()

    # Check if initialized
    if not st.session_state.db_initialized:
        st.warning("⚠️ Please initialize the chatbot using the sidebar.")
        st.info("Click the '🔄 Initialize/Reinitialize' button to start.")

        st.markdown("### 📋 Before You Start:")
        st.markdown("""
        1. Make sure PostgreSQL is running
        2. Update [config.py](config.py) with your database credentials:
           - `POSTGRES_HOST`
           - `POSTGRES_DATABASE`
           - `POSTGRES_USER`
           - `POSTGRES_PASSWORD`
        3. Make sure Ollama is running with the specified model
        4. Click 'Initialize/Reinitialize' in the sidebar
        """)
        
        st.markdown("### 🆕 New Feature: Comparison Queries")
        st.markdown("""
        You can now compare your local portfolio data with external market data!
        
        **Example queries:**
        - "Compare my portfolio to S&P 500"
        - "How does my portfolio perform against the market?"
        - "Is my portfolio outperforming NASDAQ this year?"
        """)
        return

    # Display chat history
    display_chat_history()

    # Handle example query selection
    if "example_query" in st.session_state:
        user_input = st.session_state.example_query
        del st.session_state.example_query
    else:
        user_input = st.chat_input("Ask a question about your data...")

    # Process user input
    if user_input:
        # Add user message to chat with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        with st.chat_message("user"):
            st.caption(f"🕒 {timestamp}")
            st.markdown(user_input)

        # Route the query
        with st.spinner("🤔 Understanding your question..."):
            query_type = st.session_state.router.classify_query(user_input)

        # Track query statistics
        if query_type in st.session_state.query_stats:
            st.session_state.query_stats[query_type] += 1

        # Show query type with appropriate icon
        type_icons = {
            "database": "💾",
            "greeting": "💬",
            "internet_data": "🌐",
            "comparison": "🔄"
        }
        icon = type_icons.get(query_type, "❓")
        st.info(f"{icon} Query Type: **{query_type}**")

        # Process based on type
        with st.chat_message("assistant"):
            if query_type == "database":
                response = process_database_query(user_input)
            elif query_type == "greeting":
                response = process_greeting(user_input)
            elif query_type == "internet_data":
                response = process_internet_data(user_input)
            elif query_type == "comparison":
                response = process_comparison_query(user_input)
            else:
                response = {
                    "content": "❌ Sorry, I couldn't understand your question.",
                    "sql_query": None,
                    "results_df": None
                }

            # Display response
            st.markdown(response["content"])

            # Display SQL query if present
            if response.get("sql_query"):
                with st.expander("🔍 View SQL Query"):
                    st.code(response["sql_query"], language="sql")

            # Display results dataframe if present
            if response.get("results_df") is not None:
                with st.expander("📊 View Results Table"):
                    st.dataframe(response["results_df"], use_container_width=True)

            # Display comparison details if present
            if response.get("comparison_plan"):
                with st.expander("🔄 View Comparison Details"):
                    st.json(response["comparison_plan"])

        # Add assistant response to history with timestamp
        assistant_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_entry = {
            "role": "assistant",
            "content": response["content"],
            "sql_query": response.get("sql_query"),
            "results_df": response.get("results_df"),
            "timestamp": assistant_timestamp,
            "query_type": query_type
        }
        
        # Add comparison-specific data if present
        if query_type == "comparison":
            message_entry["comparison_plan"] = response.get("comparison_plan")
            message_entry["local_data"] = response.get("local_data")
            message_entry["external_data"] = response.get("external_data")
        
        st.session_state.messages.append(message_entry)


if __name__ == "__main__":
    main()