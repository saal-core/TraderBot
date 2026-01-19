"""
TraderBot FastAPI Backend
REST API for natural language database queries with comparison capabilities.
Streaming support for real-time response generation.
"""
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd

# Import models
from src.api.models import (
    QueryRequest, InitializeResponse, HealthResponse, SchemaResponse,
    ExportRequest, ExportResponse, ChatMessage, StatsResponse
)

# Import services
from src.config.settings import get_app_config, get_ollama_config, get_postgres_config
from src.services.sql_utilities import PostgreSQLExecutor
from src.services.llm_query_router import LLMQueryRouter
from src.services.query_planner import QueryPlanner
from src.services.task_executor import TaskExecutor
from src.services.database_handler import DatabaseQueryHandler
from src.services.internet_data_handler import InternetDataHandler
from src.services.unified_response_generator import get_response_generator
# ComparisonHandler removed - comparison queries now handled by QueryPlanner + TaskExecutor
from src.services.chat_memory import ChatMemory
from src.services.context_manager import ContextAwareManager
from src.utils.response_cleaner import clean_llm_response, clean_llm_chunk

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Global State (initialized on startup or via /initialize endpoint)
# ============================================================================

class AppState:
    """Global application state holding all handlers"""
    def __init__(self):
        self.initialized = False
        self.sql_executor: Optional[PostgreSQLExecutor] = None
        self.router: Optional[LLMQueryRouter] = None
        self.planner: Optional[QueryPlanner] = None
        self.task_executor: Optional[TaskExecutor] = None
        self.db_handler: Optional[DatabaseQueryHandler] = None
        self.internet_data_handler: Optional[InternetDataHandler] = None
        self.context_manager: Optional[ContextAwareManager] = None
        # comparison_handler removed - now handled by planner/executor
        self.chat_memory: Optional[ChatMemory] = None
        self.query_stats = {
            "database": 0,
            "greeting": 0,
            "internet_data": 0,
            "hybrid": 0
        }

    def reset_stats(self):
        self.query_stats = {
            "database": 0,
            "greeting": 0,
            "internet_data": 0,
            "hybrid": 0
        }


app_state = AppState()


# ============================================================================
# FastAPI App Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    print("🚀 TraderBot API starting up...")
    yield
    print("👋 TraderBot API shutting down...")
    if app_state.sql_executor:
        app_state.sql_executor.close()


app = FastAPI(
    title="TraderBot API",
    description="Natural language interface for querying financial databases with streaming support",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def serialize_value(value: Any) -> Any:
    """Serialize a value to be JSON-safe."""
    if value is None:
        return None
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def convert_chat_history(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert ChatMessage models to dict format expected by handlers"""
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": serialize_value(msg.timestamp),
            "sql_query": msg.sql_query,
            "results": msg.results,  # Include results for follow-up question context
            "query_type": msg.query_type
        }
        for msg in messages
    ]


def dataframe_to_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of dicts for JSON serialization"""
    if df is None:
        return None
    result = df.to_dict(orient="records")
    for row in result:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
            elif hasattr(value, 'isoformat'):
                row[key] = value.isoformat()
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                row[key] = str(value)
            elif hasattr(value, 'item'):  # Handle numpy types
                row[key] = value.item()
    return result


def clean_response_content(content: str) -> str:
    """Clean LLM response content for proper frontend display"""
    if not content:
        return content
    return clean_llm_response(content)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        initialized=app_state.initialized
    )


@app.post("/initialize", response_model=InitializeResponse)
async def initialize():
    """Initialize all handlers and database connection"""
    try:
        app_state.sql_executor = PostgreSQLExecutor()
        success, message = app_state.sql_executor.test_connection()
        if not success:
            return InitializeResponse(success=False, message=message)
        
        app_state.router = LLMQueryRouter()
        app_state.db_handler = DatabaseQueryHandler(sql_executor=app_state.sql_executor)
        app_state.internet_data_handler = InternetDataHandler()
        app_state.chat_memory = ChatMemory(max_pairs=5)
        app_state.context_manager = ContextAwareManager()
        
        app_state.planner = QueryPlanner()
        app_state.task_executor = TaskExecutor(
            db_handler=app_state.db_handler,
            internet_handler=app_state.internet_data_handler
        )
        
        app_state.initialized = True
        app_state.reset_stats()
        
        return InitializeResponse(
            success=True,
            message="All components initialized successfully!"
        )
        
    except Exception as e:
        return InitializeResponse(
            success=False,
            message=f"Error initializing components: {str(e)}"
        )


# /query/classify endpoint removed - classification is done internally by /query/stream


# ============================================================================
# Streaming Endpoints (Legacy endpoints removed - use /query/stream)
# ============================================================================

# /query/database/stream endpoint removed - use /query/stream instead
# /query/internet/stream endpoint removed - use /query/stream instead


# ============================================================================
# Non-Streaming Endpoints
# ============================================================================

# /query/greeting endpoint removed - use /query/stream instead
# /query/comparison endpoint removed - use /query/stream instead


@app.get("/schema", response_model=SchemaResponse)
async def get_schema():
    """Get database schema information"""
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Call /initialize first."
        )
    
    try:
        schema_info = app_state.sql_executor.get_schema_info()
        return SchemaResponse(success=True, schema_info=schema_info)
    except Exception as e:
        return SchemaResponse(
            success=False,
            schema_info=f"Error retrieving schema: {str(e)}"
        )


@app.post("/chat/export", response_model=ExportResponse)
async def export_chat(request: ExportRequest):
    """Export chat history to JSON or TXT format"""
    try:
        if request.format == "json":
            export_data = []
            for msg in request.messages:
                clean_msg = {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or datetime.now().isoformat()
                }
                if msg.sql_query:
                    clean_msg["sql_query"] = msg.sql_query
                if msg.query_type:
                    clean_msg["query_type"] = msg.query_type
                export_data.append(clean_msg)
            
            content = json.dumps(export_data, indent=2)
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            txt_lines = [
                f"TraderBot Chat History - Exported at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            ]
            txt_lines.append("=" * 80 + "\n\n")
            
            for i, msg in enumerate(request.messages, 1):
                txt_lines.append(f"Message {i} - {msg.role.upper()}\n")
                txt_lines.append("-" * 40 + "\n")
                txt_lines.append(f"{msg.content}\n")
                if msg.sql_query:
                    txt_lines.append(f"\nSQL Query:\n{msg.sql_query}\n")
                if msg.query_type:
                    txt_lines.append(f"\nQuery Type: {msg.query_type}\n")
                txt_lines.append("\n" + "=" * 80 + "\n\n")
            
            content = "".join(txt_lines)
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        return ExportResponse(success=True, content=content, filename=filename)
        
    except Exception as e:
        return ExportResponse(
            success=False,
            content=f"Error exporting chat: {str(e)}",
            filename=""
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get query statistics"""
    total = sum(app_state.query_stats.values())
    return StatsResponse(
        database=app_state.query_stats["database"],
        greeting=app_state.query_stats["greeting"],
        internet_data=app_state.query_stats["internet_data"],
        hybrid=app_state.query_stats["hybrid"],
        total=total
    )


@app.post("/stats/reset")
async def reset_stats():
    """Reset query statistics"""
    app_state.reset_stats()
    return {"success": True, "message": "Statistics reset"}


# ============================================================================
# Streaming SSE Endpoint
# ============================================================================

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas Timestamp and datetime objects."""
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        return super().default(obj)


def generate_sse_event(event_type: str, data: Any) -> str:
    """Generate a Server-Sent Event formatted string."""
    event_data = {"type": event_type, "data": data}
    if event_type == "content":
        event_data = {"type": event_type, "content": data}
    return f"data:{json.dumps(event_data, cls=CustomJSONEncoder)}\n\n"


async def _stream_database_query(query: str, chat_history: List[Dict[str, str]]):
    """Internal generator for database queries"""
    try:
        # Step 1: Generate SQL
        yield generate_sse_event("status", {"message": "🔍 Generating SQL query..."})
        sql_query = app_state.db_handler.generate_sql(query, chat_history)

        if sql_query.startswith("ERROR"):
            yield generate_sse_event("error", {"error": sql_query})
            return

        yield generate_sse_event("sql", {"content": sql_query})

        # Step 2: Execute Query
        yield generate_sse_event("status", {"message": "▶️ Executing query..."})
        success, results_df, message = app_state.sql_executor.execute_query(sql_query)

        if not success:
            yield generate_sse_event("error", {"error": f"Query execution failed: {message}"})
            return

        # Step 3: Send Results
        results_data = dataframe_to_list(results_df)
        yield generate_sse_event("results", {"content": results_data, "message": message})
        
        # Step 4: Explain Results
        yield generate_sse_event("status", {"message": "✨ Generating explanation..."})
        
        # Explain results using stream
        # Note: stream_explain_results yields plain strings, so we wrap them
        for chunk in app_state.db_handler.stream_explain_results(query, results_df, sql_query):
            yield generate_sse_event("content", chunk)
            await asyncio.sleep(0)  # Force flush for real-time streaming
                
        app_state.query_stats["database"] += 1

        # Signal completion with SQL and Results for UI to display
        yield generate_sse_event("assistant_message_complete", {
            "query_type": "database",
            "sql_query": sql_query,
            "results": results_data
        })

    except Exception as e:
        yield generate_sse_event("error", {"error": f"Error processing database query: {str(e)}"})


async def _stream_internet_query(query: str, chat_history: List[Dict[str, str]]):
    """Internal generator for internet data queries"""
    try:
        # Step 1: Fetch Data
        yield generate_sse_event("status", {"message": "🌐 Fetching financial data..."})
        raw_response = app_state.internet_data_handler.fetch_raw_data(query, chat_history)

        if raw_response.startswith("Error"):
            yield generate_sse_event("error", {"error": raw_response})
            return

        yield generate_sse_event("raw_data", {"content": raw_response})

        # Step 2: Explain Data
        yield generate_sse_event("status", {"message": "✨ Generating explanation..."})
        
        for chunk_data in app_state.internet_data_handler.explain_internet_data_streaming(query, raw_response):
            chunk_type = chunk_data.get("type")
            if chunk_type == "chunk":
                yield generate_sse_event("content", chunk_data.get("content", ""))
                await asyncio.sleep(0)  # Force flush for real-time streaming
            elif chunk_type == "error":
                yield generate_sse_event("error", {"error": chunk_data.get("content", "Unknown error")})

        app_state.query_stats["internet_data"] += 1

    except Exception as e:
        yield generate_sse_event("error", {"error": f"Error processing internet query: {str(e)}"})

async def stream_query_response(query: str, chat_history: List[Dict[str, str]]):
    """
    Generator function that streams query processing updates.
    """
    async for event in _stream_query_generator(query, chat_history):
        yield event

async def _stream_query_generator(query: str, chat_history: List[Dict[str, str]]):
    """Internal generator to handle the async stream logic cleanly"""
    try:
        if not app_state.initialized:
            yield generate_sse_event("error", {"error": "Service not initialized. Call /initialize first."})
            yield generate_sse_event("stream_end", {})
            return

        # Step 0: Check Context Context Relevance (Smart Reset)
        # If the user switches topics, we should reset the history context
        if chat_history and len(chat_history) > 0:
            # We already have context_manager initialized, use it
            if app_state.context_manager:
                is_relevant = app_state.context_manager.check_relevance(query, chat_history)
                if not is_relevant:
                    print(f"🔄 Logic Break: Switching topics. Resetting context for query: {query}")
                    chat_history = [] # Reset history for this request
                    yield generate_sse_event("status", {"message": "🔄 New topic detected - Context reset..."})
            else:
                 print("⚠️ Context Manager not initialized, skipping check")

        # Step 1: Classify Query
        yield generate_sse_event("status", {"message": "🤔 Classifying query..."})
        query_type = app_state.router.classify_query(query, chat_history)
        
        if query_type == "greeting":
            # Use UnifiedResponseGenerator for streaming greeting responses
            generator = get_response_generator()
            for chunk in generator.stream_response(
                query=query,
                context_type="greeting",
                chat_history=chat_history
            ):
                yield generate_sse_event("content", chunk)
                await asyncio.sleep(0)  # Force flush for real-time streaming
        
        elif query_type == "database":
            async for event in _stream_database_query(query, chat_history):
                yield event
                
        elif query_type == "internet_data":
            async for event in _stream_internet_query(query, chat_history):
                yield event
                
        else:
            # Step 2: Plan Execution for complex queries (hybrid, comparison, etc.)
            yield generate_sse_event("status", {"message": f"🧠 Planning execution for {query_type} query..."})
            
            plan_result = app_state.planner.generate_plan(query, chat_history)
            
            async for event_data in app_state.task_executor.execute_plan_streaming(plan_result, chat_history):
                yield event_data
            
        yield generate_sse_event("stream_end", {})

    except Exception as e:
        yield generate_sse_event("error", {"error": str(e)})
        yield generate_sse_event("stream_end", {})


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Stream query processing with Server-Sent Events.
    
    This endpoint classifies the query, routes it to the appropriate handler,
    and streams progress updates and the final response back to the client.
    """
    chat_history = convert_chat_history(request.chat_history)
    
    return StreamingResponse(
        stream_query_response(request.query, chat_history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)