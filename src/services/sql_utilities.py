"""
SQL Utilities Module for PostgreSQL
Handles SQL query execution and validation for text-to-SQL chatbot with PostgreSQL
"""

import re
from typing import Tuple, Optional, Any, Dict
from contextlib import contextmanager
import pandas as pd
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
from src.config.settings import get_postgres_config
import os
import threading
from dotenv import load_dotenv
load_dotenv()


class ConnectionPoolManager:
    """
    Singleton connection pool manager for multi-user applications.
    Ensures only ONE connection pool is created and shared across all sessions.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, connection_config: Dict[str, Any] = None):
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, connection_config: Dict[str, Any] = None):
        """Initialize the shared connection pool (only once)"""
        if self._initialized:
            return

        self.connection_config = connection_config or get_postgres_config()
        self.connection_pool = None
        self._initialize_connection_pool()
        self._initialized = True

    def _initialize_connection_pool(self):
        """Initialize the shared connection pool"""
        try:
            # Get pool size from environment with proper int casting
            min_connections = int(os.getenv("POSTGRES_MIN_CONNECTIONS", "2"))
            max_connections = int(os.getenv("POSTGRES_MAX_CONNECTIONS", "30"))

            print(f"🔄 Initializing SHARED connection pool (min={min_connections}, max={max_connections})...")

            if "connection_string" in self.connection_config:
                self.connection_pool = pool.ThreadedConnectionPool(
                    min_connections,
                    max_connections,
                    self.connection_config["connection_string"],
                    connect_timeout=10,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5
                )
            else:
                self.connection_pool = pool.ThreadedConnectionPool(
                    min_connections,
                    max_connections,
                    host=self.connection_config["host"],
                    port=self.connection_config["port"],
                    database=self.connection_config["database"],
                    user=self.connection_config["user"],
                    password=self.connection_config["password"],
                    options=self.connection_config.get("options", ""),
                    connect_timeout=10,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5
                )
            print(f"✅ SHARED PostgreSQL connection pool initialized successfully")
            print(f"   This pool will be used by ALL user sessions")
        except Exception as e:
            print(f"❌ Error initializing shared connection pool: {e}")
            self.connection_pool = None

    def get_connection(self):
        """Get a connection from the shared pool (thread-safe)"""
        if self.connection_pool:
            try:
                conn = self.connection_pool.getconn()
                if conn:
                    # Reset the connection to ensure clean state
                    conn.rollback()
                return conn
            except pool.PoolError as e:
                print(f"❌ Connection pool error: {e}")
                print(f"   Pool status: {self.get_pool_status()}")
                raise
        return None

    def return_connection(self, conn):
        """Return a connection to the shared pool (thread-safe)"""
        if self.connection_pool and conn:
            try:
                # Ensure any pending transaction is closed before returning
                if not conn.closed:
                    conn.rollback()
                self.connection_pool.putconn(conn)
            except Exception as e:
                print(f"⚠️ Error returning connection to pool: {e}")
                # Try to close the connection if we can't return it
                try:
                    conn.close()
                except:
                    pass

    def get_pool_status(self) -> dict:
        """Get current connection pool status for debugging"""
        if not self.connection_pool:
            return {"status": "not_initialized"}

        try:
            pool_obj = self.connection_pool
            return {
                "status": "active",
                "pool_type": "ThreadedConnectionPool (multi-user safe)",
                "min_connections": pool_obj.minconn,
                "max_connections": pool_obj.maxconn,
                "closed": pool_obj.closed
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        """Close all connections in the shared pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("✅ Shared connection pool closed")

class SQLValidator:
    """Validates SQL queries to ensure they are safe SELECT queries only"""

    @staticmethod
    def is_select_query(query: str) -> bool:
        """
        Validate that the query is a SELECT statement only

        Args:
            query: SQL query string to validate

        Returns:
            True if query is a valid SELECT statement, False otherwise
        """
        if not query or not isinstance(query, str):
            return False

        # Remove comments and normalize whitespace
        query_clean = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)
        query_clean = query_clean.strip().upper()

        # Check if query starts with SELECT or WITH (for CTEs)
        if not (query_clean.startswith('SELECT') or query_clean.startswith('WITH')):
            return False

        # Forbidden keywords that could modify data
        forbidden_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
            'COPY', 'VACUUM', 'ANALYZE', 'REINDEX'
        ]

        for keyword in forbidden_keywords:
            if re.search(r'\b' + keyword + r'\b', query_clean):
                return False

        return True

    @staticmethod
    def validate_query(query: str) -> Tuple[bool, str]:
        """
        Comprehensive validation of SQL query

        Args:
            query: SQL query string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Query is empty"

        if not SQLValidator.is_select_query(query):
            return False, "Only SELECT queries are allowed for security reasons"

        # Check for potential SQL injection patterns
        suspicious_patterns = [
            r';\s*SELECT',  # Multiple queries
            r';\s*DROP',
            r';\s*DELETE',
            r';\s*UPDATE',
            r';\s*INSERT',
            r';\s*CREATE',
            r'UNION\s+SELECT.*FROM.*information_schema',  # Schema extraction
            r'UNION\s+SELECT.*FROM.*pg_catalog',  # PostgreSQL system catalog
        ]

        query_upper = query.upper()
        for pattern in suspicious_patterns:
            if re.search(pattern, query_upper):
                return False, "Query contains suspicious patterns"

        return True, "Query is valid"


class PostgreSQLExecutor:
    """
    Executes SQL queries against a PostgreSQL database.
    Uses a SHARED connection pool (singleton) for multi-user applications.
    """

    def __init__(self, connection_config: Dict[str, Any] = None):
        """
        Initialize PostgreSQL executor with shared connection pool

        Args:
            connection_config: PostgreSQL connection configuration
                             If None, uses config from config.py
        """
        self.validator = SQLValidator()
        # Use the shared singleton connection pool manager
        self.pool_manager = ConnectionPoolManager(connection_config)
        print(f"📌 PostgreSQLExecutor initialized (using shared pool)")

    def get_connection(self):
        """Get a connection from the shared pool (thread-safe)"""
        return self.pool_manager.get_connection()

    def return_connection(self, conn):
        """Return a connection to the shared pool (thread-safe)"""
        self.pool_manager.return_connection(conn)

    def get_pool_status(self) -> dict:
        """
        Get current connection pool status for debugging

        Returns:
            Dictionary with pool statistics
        """
        return self.pool_manager.get_pool_status()

    @contextmanager
    def get_connection_context(self):
        """
        Context manager for safe connection handling
        Ensures connections are always returned to the pool

        Usage:
            with executor.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = None
        try:
            conn = self.get_connection()
            if not conn:
                raise RuntimeError("Failed to get database connection")
            yield conn
        finally:
            if conn:
                self.return_connection(conn)

    def execute_query(self, query: str, params: tuple = None) -> Tuple[bool, Any, str]:
        """
        Execute a SQL query after validation

        Args:
            query: SQL query to execute
            params: Optional query parameters for parameterized queries

        Returns:
            Tuple of (success, result_dataframe, message)
        """
        # Validate query first
        is_valid, error_msg = self.validator.validate_query(query)
        if not is_valid:
            return False, None, error_msg

        conn = None
        try:
            conn = self.get_connection()
            if not conn:
                return False, None, "Failed to get database connection"

            # Execute query with pandas
            df = pd.read_sql_query(query, conn, params=params)

            # Check for max rows limit
            if len(df) > 1000:
                return True, df.head(), f"Query returned {len(df)} rows"

            if df.empty:
                return True, df, "Query executed successfully but returned no results"

            return True, df, f"Query executed successfully. Returned {len(df)} rows"

        except Exception as e:
            return False, None, f"Query execution error: {str(e)}"
        finally:
            # CRITICAL: Always return connection to pool
            if conn:
                self.return_connection(conn)


    def get_schema_info(self, schema: str = None) -> str:
        """
        Get database schema information for PostgreSQL

        Args:
            schema: Schema name (defaults to public or from config)

        Returns:
            String containing schema information
        """
        if schema is None:
            schema = os.getenv("DB_SCHEMA", "public")

        try:
            with self.get_connection_context() as conn:
                cursor = conn.cursor()

                # Get all tables in the schema
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """, (schema,))

                tables = cursor.fetchall()

                if not tables:
                    return f"No tables found in schema '{schema}'"

                schema_info = f"Database Schema (Schema: {schema}):\n\n"

                for (table_name,) in tables:
                    schema_info += f"Table: {table_name}\n"

                    # Get column information
                    cursor.execute("""
                        SELECT
                            column_name,
                            data_type,
                            is_nullable,
                            column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s
                        AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (schema, table_name))

                    columns = cursor.fetchall()

                    for col_name, col_type, is_nullable, col_default in columns:
                        nullable = "" if is_nullable == "YES" else "NOT NULL"
                        default = f"DEFAULT {col_default}" if col_default else ""
                        schema_info += f"  - {col_name} ({col_type}) {nullable} {default}\n".strip() + "\n"

                    # Get primary key information
                    cursor.execute("""
                        SELECT a.attname
                        FROM pg_index i
                        JOIN pg_attribute a ON a.attrelid = i.indrelid
                            AND a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = %s::regclass
                        AND i.indisprimary;
                    """, (f"{schema}.{table_name}",))

                    pk_columns = cursor.fetchall()
                    if pk_columns:
                        pk_list = ", ".join([col[0] for col in pk_columns])
                        schema_info += f"  PRIMARY KEY: ({pk_list})\n"

                    schema_info += "\n"

                cursor.close()
                return schema_info

        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the database connection

        Returns:
            Tuple of (success, message)
        """
        try:
            with self.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                cursor.close()

                return True, f"✅ Connected successfully. PostgreSQL version: {version}"

        except Exception as e:
            return False, f"❌ Connection failed: {str(e)}"

    def close(self):
        """
        Close all connections in the shared pool.
        Note: In multi-user apps, this should only be called on app shutdown,
        not when individual sessions end.
        """
        print("⚠️ Warning: Closing shared connection pool (affects all users)")
        self.pool_manager.close()

    def __del__(self):
        """
        Destructor - does NOT close the shared pool.
        The shared pool persists across all sessions.
        """
        pass  # Don't close the shared pool when individual executor instances are destroyed

# For backward compatibility, create an alias
SQLExecutor = PostgreSQLExecutor

