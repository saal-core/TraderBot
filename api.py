"""
TraderBot FastAPI Backend
REST API for natural language database queries with comparison capabilities.
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Import models
from src.api.models import (
    QueryRequest, QueryResponse, ClassifyRequest, ClassifyResponse,
    InitializeResponse, HealthResponse, SchemaResponse,
    ExportRequest, ExportResponse, ChatMessage, StatsResponse
)

# Import services
from src.config.settings import get_app_config, get_ollama_config, get_postgres_config
from src.services.sql_utilities import PostgreSQLExecutor
from src.services.gpt_oss_query_router_v2 import OptimizedQueryRouter
from src.services.database_handler import DatabaseQueryHandler
from src.services.greating_handler import GreetingHandler
from src.services.internet_data_handler import InternetDataHandler
from src.services.comparison_handler import ComparisonHandler
from src.services.chat_memory import ChatMemory

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
        self.router: Optional[OptimizedQueryRouter] = None
        self.db_handler: Optional[DatabaseQueryHandler] = None
        self.greeting_handler: Optional[GreetingHandler] = None
        self.internet_data_handler: Optional[InternetDataHandler] = None
        self.comparison_handler: Optional[ComparisonHandler] = None
        self.chat_memory: Optional[ChatMemory] = None
        self.query_stats = {
            "database": 0,
            "greeting": 0,
            "internet_data": 0,
            "comparison": 0
        }

    def reset_stats(self):
        self.query_stats = {
            "database": 0,
            "greeting": 0,
            "internet_data": 0,
            "comparison": 0
        }


app_state = AppState()


# ============================================================================
# FastAPI App Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    print("🚀 TraderBot API starting up...")
    yield
    # Shutdown
    print("👋 TraderBot API shutting down...")
    if app_state.sql_executor:
        app_state.sql_executor.close()


app = FastAPI(
    title="TraderBot API",
    description="Natural language interface for querying financial databases",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def convert_chat_history(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert ChatMessage models to dict format expected by handlers"""
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "sql_query": msg.sql_query,
            "query_type": msg.query_type
        }
        for msg in messages
    ]


def dataframe_to_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of dicts for JSON serialization"""
    if df is None:
        return None
    return df.to_dict(orient="records")


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
        # Initialize PostgreSQL executor
        app_state.sql_executor = PostgreSQLExecutor()
        
        # Test database connection
        success, message = app_state.sql_executor.test_connection()
        if not success:
            return InitializeResponse(success=False, message=message)
        
        # Initialize handlers
        app_state.router = OptimizedQueryRouter(sql_executor=app_state.sql_executor)
        app_state.db_handler = DatabaseQueryHandler(sql_executor=app_state.sql_executor)
        app_state.greeting_handler = GreetingHandler()
        app_state.internet_data_handler = InternetDataHandler()
        app_state.chat_memory = ChatMemory(max_pairs=5)
        
        # Initialize comparison handler with existing handlers
        app_state.comparison_handler = ComparisonHandler(
            db_handler=app_state.db_handler,
            internet_handler=app_state.internet_data_handler,
            sql_executor=app_state.sql_executor,
            use_openai=True
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


@app.post("/query/classify", response_model=ClassifyResponse)
async def classify_query(request: ClassifyRequest):
    """Classify a query into its type"""
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Call /initialize first."
        )
    
    try:
        query_type = app_state.router.classify_query(request.query)
        return ClassifyResponse(query_type=query_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying query: {str(e)}"
        )


@app.post("/query/database", response_model=QueryResponse)
async def process_database_query(request: QueryRequest):
    """Process a database query with SQL generation"""
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Call /initialize first."
        )
    
    try:
        chat_history = convert_chat_history(request.chat_history)
        
        # Get database schema
        schema = app_state.sql_executor.get_schema_info()
        if "Error" in schema:
            return QueryResponse(
                success=False,
                content=f"Failed to retrieve database schema: {schema}",
                query_type="database",
                error=schema
            )
        
        # Generate SQL
        sql_query = app_state.db_handler.generate_sql(
            request.query, schema, chat_history
        )
        
        if sql_query.startswith("ERROR"):
            return QueryResponse(
                success=False,
                content=sql_query,
                query_type="database",
                error=sql_query
            )
        
        # Execute query
        success, results_df, message = app_state.sql_executor.execute_query(sql_query)
        
        if not success:
            return QueryResponse(
                success=False,
                content=f"Query execution failed: {message}",
                sql_query=sql_query,
                query_type="database",
                error=message
            )
        
        # Generate explanation
        explanation = app_state.db_handler.explain_results(
            request.query, results_df, sql_query
        )
        
        app_state.query_stats["database"] += 1
        
        return QueryResponse(
            success=True,
            content=f"✅ {explanation}\n\n💡 {message}",
            sql_query=sql_query,
            results=dataframe_to_list(results_df),
            query_type="database"
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            content=f"Error processing query: {str(e)}",
            query_type="database",
            error=str(e)
        )


@app.post("/query/greeting", response_model=QueryResponse)
async def process_greeting(request: QueryRequest):
    """Process a greeting or chitchat"""
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Call /initialize first."
        )
    
    try:
        chat_history = convert_chat_history(request.chat_history)
        response = app_state.greeting_handler.respond(request.query, chat_history)
        
        app_state.query_stats["greeting"] += 1
        
        return QueryResponse(
            success=True,
            content=response,
            query_type="greeting"
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            content=f"Error: {str(e)}",
            query_type="greeting",
            error=str(e)
        )


@app.post("/query/internet", response_model=QueryResponse)
async def process_internet_query(request: QueryRequest):
    """Process an internet data query"""
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Call /initialize first."
        )
    
    try:
        chat_history = convert_chat_history(request.chat_history)
        response = app_state.internet_data_handler.fetch_data(
            request.query, chat_history
        )
        
        app_state.query_stats["internet_data"] += 1
        
        return QueryResponse(
            success=True,
            content=f"🌐 {response}",
            query_type="internet_data"
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            content=f"Error: {str(e)}",
            query_type="internet_data",
            error=str(e)
        )


@app.post("/query/comparison", response_model=QueryResponse)
async def process_comparison_query(request: QueryRequest):
    """Process a comparison query"""
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Call /initialize first."
        )
    
    try:
        chat_history = convert_chat_history(request.chat_history)
        
        result = app_state.comparison_handler.process(
            request.query, chat_history
        )
        
        app_state.query_stats["comparison"] += 1
        
        return QueryResponse(
            success=True,
            content=result.get("content", ""),
            sql_query=result.get("sql_query"),
            results=dataframe_to_list(result.get("results_df")),
            query_type="comparison",
            comparison_plan=result.get("comparison_plan"),
            local_data=result.get("local_data"),
            external_data=result.get("external_data")
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            content=f"Error processing comparison: {str(e)}",
            query_type="comparison",
            error=str(e)
        )


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
            # TXT format
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
        comparison=app_state.query_stats["comparison"],
        total=total
    )


@app.post("/stats/reset")
async def reset_stats():
    """Reset query statistics"""
    app_state.reset_stats()
    return {"success": True, "message": "Statistics reset"}


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
