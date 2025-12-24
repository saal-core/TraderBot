"""
TraderBot Main Application (Streamlit Frontend)
A natural language interface for querying financial databases with 
comparison capabilities between local portfolio data and external market data.

This version uses the FastAPI backend for all processing.
"""
import streamlit as st
import pandas as pd
import requests
from typing import List, Dict
import os
import json
from datetime import datetime

# Import configuration
from src.config.settings import get_app_config, get_ollama_config, get_postgres_config

import warnings 
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

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


# ============================================================================
# API Client Functions
# ============================================================================

def api_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make an API request to the FastAPI backend"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=120)
        else:
            response = requests.post(url, json=data, timeout=120)
        
        # Try to parse JSON response
        try:
            result = response.json()
        except ValueError:
            return {"error": f"Invalid JSON response from API: {response.text[:200]}"}
        
        # Check for HTTP errors
        if response.status_code >= 400:
            error_detail = result.get("detail") or result.get("message") or result.get("error") or f"HTTP {response.status_code}"
            return {"error": error_detail, "success": False}
        
        return result
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Make sure the FastAPI backend is running.", "success": False}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server took too long to respond.", "success": False}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}", "success": False}


def check_api_health() -> bool:
    """Check if the API is healthy"""
    result = api_request("/health")
    return result.get("status") == "healthy"


def api_initialize() -> dict:
    """Initialize the API handlers"""
    return api_request("/initialize", method="POST")


def api_classify_query(query: str) -> str:
    """Classify a query using the API"""
    result = api_request("/query/classify", method="POST", data={"query": query})
    if "error" in result:
        return "unknown"
    return result.get("query_type", "unknown")


def api_process_query(query_type: str, query: str, chat_history: list) -> dict:
    """Process a query through the appropriate API endpoint"""
    endpoint_map = {
        "database": "/query/database",
        "greeting": "/query/greeting",
        "internet_data": "/query/internet",
        "comparison": "/query/comparison"
    }
    
    endpoint = endpoint_map.get(query_type, "/query/database")
    
    # Convert chat history to API format
    formatted_history = []
    for msg in chat_history:
        formatted_history.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
            "timestamp": msg.get("timestamp"),
            "sql_query": msg.get("sql_query"),
            "query_type": msg.get("query_type")
        })
    
    return api_request(endpoint, method="POST", data={
        "query": query,
        "chat_history": formatted_history
    })


def api_get_schema() -> str:
    """Get database schema from API"""
    result = api_request("/schema")
    return result.get("schema_info", "Schema not available")


def api_get_stats() -> dict:
    """Get query statistics from API"""
    return api_request("/stats")


def api_export_chat(messages: list, format_type: str) -> dict:
    """Export chat history via API"""
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
            "timestamp": msg.get("timestamp"),
            "sql_query": msg.get("sql_query"),
            "query_type": msg.get("query_type")
        })
    
    return api_request("/chat/export", method="POST", data={
        "format": format_type,
        "messages": formatted_messages
    })


# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_initialized" not in st.session_state:
        st.session_state.api_initialized = False

    if "query_stats" not in st.session_state:
        st.session_state.query_stats = {
            "database": 0, 
            "greeting": 0, 
            "internet_data": 0,
            "comparison": 0
        }


# ============================================================================
# Display Functions
# ============================================================================

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
                    st.json(message["comparison_plan"])


# ============================================================================
# Query Processing
# ============================================================================

def process_query(query_type: str, user_query: str) -> Dict:
    """Process a query through the API and return formatted response"""
    response = {
        "content": "",
        "sql_query": None,
        "results_df": None,
        "comparison_plan": None,
        "local_data": None,
        "external_data": None
    }
    
    # Get appropriate spinner message
    spinner_messages = {
        "database": "🤖 Processing database query...",
        "greeting": "💬 Generating response...",
        "internet_data": "🌐 Fetching real-time data...",
        "comparison": "🔄 Processing comparison query..."
    }
    
    with st.spinner(spinner_messages.get(query_type, "Processing...")):
        result = api_process_query(query_type, user_query, st.session_state.messages)
    
    # Check for errors (API returns error:null on success, so check value not just key)
    if result.get("error"):
        response["content"] = f"❌ {result['error']}"
        return response
    
    response["content"] = result.get("content", "")
    response["sql_query"] = result.get("sql_query")
    
    # Convert results list back to DataFrame for display
    if result.get("results"):
        response["results_df"] = pd.DataFrame(result["results"])
    
    # Handle comparison-specific data
    if query_type == "comparison":
        response["comparison_plan"] = result.get("comparison_plan")
        response["local_data"] = result.get("local_data")
        response["external_data"] = result.get("external_data")
    
    return response


# ============================================================================
# Main Application
# ============================================================================

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
        
        st.subheader("🌐 API Settings")
        st.text(f"URL: {API_BASE_URL}")
        
        # Check API health
        if check_api_health():
            st.success("API: Connected ✅")
        else:
            st.error("API: Not Connected ❌")

        st.divider()

        if st.button("🔄 Initialize/Reinitialize", use_container_width=True):
            with st.spinner("Initializing API components..."):
                result = api_initialize()
            
            if result.get("success"):
                st.session_state.api_initialized = True
                st.success(f"✅ {result.get('message', 'Initialized!')}")
            else:
                st.session_state.api_initialized = False
                error_msg = result.get("error") or result.get("message") or "Unknown error occurred"
                st.error(f"❌ {error_msg}")

        st.divider()

        # Display database schema
        if st.session_state.api_initialized:
            with st.expander("📋 View Database Schema"):
                schema_info = api_get_schema()
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

        # Comparison queries
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
                "API URL": API_BASE_URL,
                "Max Result Rows": os.getenv("MAX_RESULT_ROWS", 100),
                "Query Timeout": f"{os.getenv('QUERY_TIMEOUT', 30)} seconds",
            })

        st.divider()

        # Chat Management Section
        st.subheader("💬 Chat Management")

        # Display chat statistics from API
        if st.session_state.messages:
            stats = api_get_stats()
            if "error" not in stats:
                st.metric("Total Queries", stats.get("total", 0))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("💾 DB", stats.get("database", 0))
                    st.metric("🌐 Web", stats.get("internet_data", 0))
                with col2:
                    st.metric("🔄 Compare", stats.get("comparison", 0))
                    st.metric("💬 Chat", stats.get("greeting", 0))

            st.divider()

            # Export options
            st.caption("📥 Export Chat History:")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Export TXT", use_container_width=True):
                    result = api_export_chat(st.session_state.messages, "txt")
                    if result.get("success"):
                        st.download_button(
                            label="⬇️ Download TXT",
                            data=result["content"],
                            file_name=result["filename"],
                            mime="text/plain",
                            use_container_width=True
                        )

            with col2:
                if st.button("📋 Export JSON", use_container_width=True):
                    result = api_export_chat(st.session_state.messages, "json")
                    if result.get("success"):
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=result["content"],
                            file_name=result["filename"],
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
            # Reset API stats
            api_request("/stats/reset", method="POST")
            st.rerun()

    # Check if API is initialized
    if not st.session_state.api_initialized:
        # Check if API is already initialized on the backend
        health = api_request("/health")
        if health.get("initialized"):
            st.session_state.api_initialized = True
        else:
            st.warning("⚠️ Please initialize the chatbot using the sidebar.")
            st.info("Click the '🔄 Initialize/Reinitialize' button to start.")

            st.markdown("### 📋 Before You Start:")
            st.markdown("""
            1. Make sure the FastAPI backend is running: `uvicorn api:app --reload`
            2. Make sure PostgreSQL is running
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

        # Route the query via API
        with st.spinner("🤔 Understanding your question..."):
            query_type = api_classify_query(user_input)

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
            response = process_query(query_type, user_input)

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