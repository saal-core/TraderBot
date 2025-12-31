"""
TraderBot Main Application (Streamlit Frontend)
Natural language interface for querying financial databases with streaming support.
"""
import streamlit as st
import pandas as pd
import requests
from typing import Dict, Generator
import os
import json
from datetime import datetime

from src.config.settings import get_app_config, get_ollama_config, get_postgres_config

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
app_config = get_app_config()

st.set_page_config(page_title=app_config["title"], page_icon=app_config["icon"], layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


def api_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, timeout=120) if method == "GET" else requests.post(url, json=data, timeout=120)
        result = response.json()
        if response.status_code >= 400:
            return {"error": result.get("detail") or result.get("message") or "Error", "success": False}
        return result
    except Exception as e:
        return {"error": str(e), "success": False}


def stream_from_api(endpoint: str, query: str, chat_history: list) -> Generator[Dict, None, None]:
    url = f"{API_BASE_URL}{endpoint}"
    formatted_history = [{"role": m.get("role"), "content": m.get("content"), "timestamp": m.get("timestamp"),
                         "sql_query": m.get("sql_query"), "query_type": m.get("query_type")} for m in chat_history]
    try:
        response = requests.post(url, json={"query": query, "chat_history": formatted_history}, stream=True, timeout=120)
        for line in response.iter_lines():
            if line and line.decode('utf-8').startswith('data: '):
                try:
                    yield json.loads(line.decode('utf-8')[6:])
                except:
                    continue
    except Exception as e:
        yield {"type": "error", "content": str(e)}


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_initialized" not in st.session_state:
        st.session_state.api_initialized = False


def display_chat_history():
    for message in st.session_state.messages[-10:]:
        with st.chat_message(message["role"]):
            if message.get("timestamp"):
                st.caption(f"🕒 {message['timestamp']}")
            st.markdown(message["content"])
            if message.get("sql_query"):
                with st.expander("🔍 SQL Query"):
                    st.code(message["sql_query"], language="sql")
            if message.get("results_df") is not None:
                with st.expander("📊 Results"):
                    st.dataframe(message["results_df"], use_container_width=True)


def process_streaming_query(query_type: str, user_query: str) -> Dict:
    endpoint = {
        "database": "/query/database/stream",
        "internet_data": "/query/internet/stream"
    }.get(query_type)
    
    if not endpoint:
        return {"content": f"Streaming not supported for: {query_type}"}

    response = {"content": "", "sql_query": None, "results_df": None, "generation_time": None}
    status_ph = st.empty()
    content_ph = st.empty()
    accumulated = ""
    sql_query = results_data = generation_time = None

    for event in stream_from_api(endpoint, user_query, st.session_state.messages):
        t = event.get("type")
        if t == "error":
            status_ph.empty()
            content_ph.error(f"❌ {event.get('content')}")
            response["content"] = event.get("content")
            return response
        elif t == "status":
            status_ph.info(event.get("content", "Processing..."))
        elif t == "sql":
            sql_query = event.get("content")
            status_ph.success("✅ SQL generated")
        elif t == "results":
            results_data = event.get("content")
            status_ph.success(f"✅ {event.get('message', 'Done')}")
        elif t == "raw_data":
            status_ph.success("✅ Data fetched")
        elif t == "chunk":
            status_ph.empty()
            chunk = event.get("content", "")
            if chunk:
                accumulated += chunk
                content_ph.markdown(accumulated)
        elif t == "metadata":
            generation_time = event.get("elapsed_time")
        elif t == "done":
            status_ph.empty()

    status_ph.empty()
    response.update({"content": accumulated, "sql_query": sql_query, "generation_time": generation_time})
    if results_data:
        response["results_df"] = pd.DataFrame(results_data)
    
    if sql_query:
        with st.expander("🔍 SQL Query"):
            st.code(sql_query, language="sql")
    if results_data:
        with st.expander("📊 Results"):
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    if generation_time:
        st.caption(f"⚡ {generation_time:.2f}s")
    
    return response


def process_non_streaming_query(query_type: str, user_query: str) -> Dict:
    endpoint = {"greeting": "/query/greeting", "comparison": "/query/comparison"}.get(query_type)
    if not endpoint:
        return {"content": f"Unknown query type: {query_type}"}
    
    formatted_history = [{"role": m.get("role"), "content": m.get("content")} for m in st.session_state.messages]
    
    with st.spinner("Processing..."):
        result = api_request(endpoint, "POST", {"query": user_query, "chat_history": formatted_history})
    
    if result.get("error"):
        return {"content": f"❌ {result['error']}"}
    
    response = {"content": result.get("content", ""), "sql_query": result.get("sql_query"), "comparison_plan": result.get("comparison_plan")}
    if result.get("results"):
        response["results_df"] = pd.DataFrame(result["results"])
    return response


def main():
    st.markdown(f'<div class="main-header">{app_config["icon"]} {app_config["title"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your database in natural language</div>', unsafe_allow_html=True)

    initialize_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration")
        ollama_config = get_ollama_config()
        postgres_config = get_postgres_config()

        st.text(f"Model: {ollama_config['model_name']}")
        st.text(f"DB: {postgres_config.get('database', 'N/A')}")
        st.text(f"API: {API_BASE_URL}")

        health = api_request("/health")
        st.success("API: Connected ✅") if health.get("status") == "healthy" else st.error("API: Not Connected ❌")

        st.divider()

        if st.button("🔄 Initialize", use_container_width=True):
            with st.spinner("Initializing..."):
                result = api_request("/initialize", "POST")
            if result.get("success"):
                st.session_state.api_initialized = True
                st.success("✅ Initialized!")
            else:
                st.error(f"❌ {result.get('error', 'Failed')}")

        st.divider()

        st.subheader("⚡ Quick Actions")
        for label, queries in [
            ("💾 Database", ["Show all portfolios", "Top 5 holdings", "Best performing stock"]),
            ("🌐 Internet", ["Tesla price", "Apple news", "Bitcoin price"]),
            ("🔄 Comparison", ["Compare portfolio to S&P 500"]),
            ("💬 General", ["Hello!", "What can you do?"])
        ]:
            with st.expander(label):
                for q in queries:
                    if st.button(q, key=f"q_{q}", use_container_width=True):
                        st.session_state.example_query = q

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            api_request("/stats/reset", "POST")
            st.rerun()

    if not st.session_state.api_initialized:
        health = api_request("/health")
        if health.get("initialized"):
            st.session_state.api_initialized = True
        else:
            st.warning("⚠️ Click 'Initialize' in sidebar to start.")
            return

    display_chat_history()

    user_input = st.session_state.pop("example_query", None) or st.chat_input("Ask a question...")

    if user_input:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": timestamp})

        with st.chat_message("user"):
            st.caption(f"🕒 {timestamp}")
            st.markdown(user_input)

        with st.spinner("🤔 Classifying..."):
            result = api_request("/query/classify", "POST", {"query": user_input})
            query_type = result.get("query_type", "database")

        icons = {"database": "💾", "greeting": "💬", "internet_data": "🌐", "comparison": "🔄"}
        st.info(f"{icons.get(query_type, '❓')} Query Type: **{query_type}**")

        with st.chat_message("assistant"):
            if query_type in ["database", "internet_data"]:
                response = process_streaming_query(query_type, user_input)
            else:
                response = process_non_streaming_query(query_type, user_input)
                st.markdown(response["content"])
                if response.get("sql_query"):
                    with st.expander("🔍 SQL"):
                        st.code(response["sql_query"], language="sql")
                if response.get("results_df") is not None:
                    with st.expander("📊 Results"):
                        st.dataframe(response["results_df"], use_container_width=True)
                if response.get("comparison_plan"):
                    with st.expander("🔄 Details"):
                        st.json(response["comparison_plan"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["content"],
            "sql_query": response.get("sql_query"),
            "results_df": response.get("results_df"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_type": query_type,
            "comparison_plan": response.get("comparison_plan")
        })


if __name__ == "__main__":
    main()