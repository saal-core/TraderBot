"""
Pydantic models for FastAPI request/response validation
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Request Models
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    sql_query: Optional[str] = Field(None, description="SQL query if applicable")
    query_type: Optional[str] = Field(None, description="Type of query")


class QueryRequest(BaseModel):
    """Request for processing a query"""
    query: str = Field(..., description="User's natural language query")
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous chat messages for context"
    )


class ClassifyRequest(BaseModel):
    """Request for classifying a query"""
    query: str = Field(..., description="Query to classify")


class ExportRequest(BaseModel):
    """Request for exporting chat history"""
    format: str = Field("json", description="Export format: 'json' or 'txt'")
    messages: List[ChatMessage] = Field(..., description="Messages to export")


# ============================================================================
# Response Models
# ============================================================================

class QueryResponse(BaseModel):
    """Response from query processing"""
    success: bool = Field(..., description="Whether the query was processed successfully")
    content: str = Field(..., description="Natural language response")
    sql_query: Optional[str] = Field(None, description="Generated SQL query if applicable")
    results: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Query results as list of dicts"
    )
    query_type: str = Field(..., description="Type of query processed")
    comparison_plan: Optional[Dict[str, Any]] = Field(
        None,
        description="Comparison plan details if applicable"
    )
    local_data: Optional[str] = Field(None, description="Local data for comparisons")
    external_data: Optional[str] = Field(None, description="External data for comparisons")
    error: Optional[str] = Field(None, description="Error message if failed")


class ClassifyResponse(BaseModel):
    """Response from query classification"""
    query_type: str = Field(..., description="Classified query type")


class InitializeResponse(BaseModel):
    """Response from initialization"""
    success: bool = Field(..., description="Whether initialization succeeded")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Current timestamp")
    initialized: bool = Field(..., description="Whether handlers are initialized")


class SchemaResponse(BaseModel):
    """Database schema response"""
    success: bool = Field(..., description="Whether schema was retrieved")
    schema_info: str = Field(..., description="Database schema information")


class ExportResponse(BaseModel):
    """Chat export response"""
    success: bool = Field(..., description="Whether export succeeded")
    content: str = Field(..., description="Exported content")
    filename: str = Field(..., description="Suggested filename")


class StatsResponse(BaseModel):
    """Query statistics response"""
    database: int = Field(default=0, description="Database query count")
    greeting: int = Field(default=0, description="Greeting query count")
    internet_data: int = Field(default=0, description="Internet data query count")
    comparison: int = Field(default=0, description="Comparison query count")
    total: int = Field(default=0, description="Total query count")
