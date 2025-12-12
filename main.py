import streamlit as st
import pandas as pd
from typing import List, Dict
import sys
import os


# Import configuration and custom modules
from src.config.settings import get_app_config, get_ollama_config, get_postgres_config
from src.services.sql_utilities import PostgreSQLExecutor
from src.services.query_router import QueryRouter
from src.services.database_handler import DatabaseQueryHandler
from src.services.greating_handler import GreetingHandler
from src.services.chat_memory import ChatMemory
# from src.services.perplexity_service import InternetComparisonHandler
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

    if "comparison_handler" not in st.session_state:
        st.session_state.comparison_handler = None

    if "sql_executor" not in st.session_state:
        st.session_state.sql_executor = None

    if "chat_memory" not in st.session_state:
        # Initialize chat memory with 5 Q&A pairs
        st.session_state.chat_memory = ChatMemory(max_pairs=5)


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

            # Initialize LLM handlers (pass sql_executor to both router and db_handler for dynamic data)
            st.session_state.router = QueryRouter(sql_executor=st.session_state.sql_executor)
            st.session_state.db_handler = DatabaseQueryHandler(sql_executor=st.session_state.sql_executor)
            st.session_state.greeting_handler = GreetingHandler()
            # st.session_state.comparison_handler = InternetComparisonHandler()

            st.session_state.db_initialized = True
            st.success("✅ All components initialized successfully!")
            return True

    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        st.session_state.db_initialized = False
        return False


def display_chat_history():
    """Display chat message history"""
    max_display = os.getenv("MAX_DISPLAY_MESSAGES", 10)
    messages_to_display = st.session_state.messages[-max_display:]

    for message in messages_to_display:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display SQL query if present
            if "sql_query" in message and message["sql_query"]:
                with st.expander("🔍 View SQL Query"):
                    st.code(message["sql_query"], language="sql")

            # Display results dataframe if present
            if "results_df" in message and message["results_df"] is not None:
                with st.expander("📊 View Results Table"):
                    st.dataframe(message["results_df"], use_container_width=True)


def process_database_query(user_query: str) -> Dict:
    """Process a database query"""
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

        # Generate SQL
        with st.spinner("🤖 Generating SQL query..."):
            sql_query = st.session_state.db_handler.generate_sql(user_query, schema)

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
            # Use chat memory to get context (automatically handles last N pairs)
            chat_history = st.session_state.messages

            greeting_response = st.session_state.greeting_handler.respond(user_query, chat_history)
            response["content"] = f"{greeting_response}"

    except Exception as e:
        response["content"] = f"❌ Error: {str(e)}"

    return response


def process_internet_comparison(user_query: str) -> Dict:
    """Process an internet comparison query (placeholder)"""
    response = {
        "content": "",
        "sql_query": None,
        "results_df": None
    }

    try:
        with st.spinner("🌐 Processing comparison request..."):
            comparison_response = st.session_state.comparison_handler.handle_comparison(user_query)
            response["content"] = f"🌐 {comparison_response}"

    except Exception as e:
        response["content"] = f"❌ Error: {str(e)}"

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

        # Example queries
        st.subheader("💡 Example Queries")
        st.caption("Click to use these example queries:")

        example_queries = [
            "Show me all records from the first table",
            "Count total rows in each table",
            "Hello!",
            "What tables are available?",
        ]

        for example in example_queries:
            if st.button(f"📝 {example}", key=f"example_{example}", use_container_width=True):
                st.session_state.example_query = example

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

        # Display chat memory stats
        if st.session_state.chat_memory and st.session_state.messages:
            pairs_count = st.session_state.chat_memory.count_pairs(st.session_state.messages)
            st.info(f"💬 Chat Memory: {pairs_count}/5 Q&A pairs")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
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
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Route the query
        with st.spinner("🤔 Understanding your question..."):
            query_type = st.session_state.router.classify_query(user_input)

        st.info(f"📌 Query Type: **{query_type}**")

        # Process based on type
        with st.chat_message("assistant"):
            if query_type == "database":
                response = process_database_query(user_input)
            elif query_type == "greeting":
                response = process_greeting(user_input)
            elif query_type == "internet_comparison":
                response = process_internet_comparison(user_input)
            else:
                response = {
                    "content": "❌ Sorry, I couldn't understand your question.",
                    "sql_query": None,
                    "results_df": None
                }

            # Display response
            st.markdown(response["content"])

            # Display SQL query if present
            if response["sql_query"]:
                with st.expander("🔍 View SQL Query"):
                    st.code(response["sql_query"], language="sql")

            # Display results dataframe if present
            if response["results_df"] is not None:
                with st.expander("📊 View Results Table"):
                    st.dataframe(response["results_df"], use_container_width=True)

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["content"],
            "sql_query": response.get("sql_query"),
            "results_df": response.get("results_df")
        })


if __name__ == "__main__":
    main()
