"""
SQL Utilities Module for PostgreSQL
Handles SQL query execution and validation for text-to-SQL chatbot with PostgreSQL
"""

import re
from typing import Tuple, Optional, Any, Dict
import pandas as pd
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
from src.config.settings import get_postgres_config
import os 
from dotenv import load_dotenv
load_dotenv()

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
    """Executes SQL queries against a PostgreSQL database"""

    def __init__(self, connection_config: Dict[str, Any] = None):
        """
        Initialize PostgreSQL executor

        Args:
            connection_config: PostgreSQL connection configuration
                             If None, uses config from config.py
        """
        self.connection_config = connection_config or get_postgres_config()
        self.validator = SQLValidator()
        self.connection_pool = None
        self._initialize_connection_pool()

    def _initialize_connection_pool(self):
        """Initialize connection pool for better performance"""
        try:
            if "connection_string" in self.connection_config:
                self.connection_pool = pool.SimpleConnectionPool(
                    os.getenv("POSTGRES_MIN_CONNECTIONS", 1),
                    os.getenv("POSTGRES_MAX_CONNECTIONS", 10),
                    self.connection_config["connection_string"]
                )
            else:
                self.connection_pool = pool.SimpleConnectionPool(
                    os.getenv("POSTGRES_MIN_CONNECTIONS", 1),
                    os.getenv("POSTGRES_MAX_CONNECTIONS", 10),
                    host=self.connection_config["host"],
                    port=self.connection_config["port"],
                    database=self.connection_config["database"],
                    user=self.connection_config["user"],
                    password=self.connection_config["password"],
                    options=self.connection_config.get("options", "")
                )
            print("✅ PostgreSQL connection pool initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing connection pool: {e}")
            self.connection_pool = None

    def get_connection(self):
        """Get a connection from the pool"""
        if self.connection_pool:
            return self.connection_pool.getconn()
        return None

    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self.connection_pool and conn:
            self.connection_pool.putconn(conn)

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

        conn = None
        try:
            conn = self.get_connection()
            if not conn:
                return "Error: Failed to get database connection"

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
        finally:
            if conn:
                self.return_connection(conn)

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the database connection

        Returns:
            Tuple of (success, message)
        """
        conn = None
        try:
            conn = self.get_connection()
            if not conn:
                return False, "Failed to get database connection"

            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()

            return True, f"✅ Connected successfully. PostgreSQL version: {version}"

        except Exception as e:
            return False, f"❌ Connection failed: {str(e)}"
        finally:
            if conn:
                self.return_connection(conn)

    def close(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("✅ All database connections closed")

    def __del__(self):
        """Destructor to ensure connections are closed"""
        self.close()

# For backward compatibility, create an alias
SQLExecutor = PostgreSQLExecutor

