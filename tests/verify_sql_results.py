import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import app, app_state

client = TestClient(app)

# Mock everything
app_state.initialized = True
app_state.router = MagicMock()
app_state.db_handler = MagicMock()
app_state.sql_executor = MagicMock()

# Setup mocks
app_state.router.classify_query.return_value = "database"
app_state.db_handler.generate_sql.return_value = "SELECT * FROM portfolio"
app_state.sql_executor.execute_query.return_value = (True, None, "Success") # None dataframe for simplicity
app_state.db_handler.stream_explain_results.return_value = iter(["Explanation"])

def test_database_sql_results_event():
    print("\n--- Testing Database SQL/Results Event ---")
    
    response = client.post("/query/stream", json={"query": "test db", "chat_history": []})
    
    found_complete_event = False
    for line in response.iter_lines():
        if line:
            decoded_line = line # TestClient yields strings in recent versions
            if decoded_line.startswith("data:"):
                try:
                    data = json.loads(decoded_line[5:])
                    if data.get("type") == "assistant_message_complete":
                        found_complete_event = True
                        content = data.get("data", {})
                        if content.get("sql_query") == "SELECT * FROM portfolio":
                            print("✅ PASS: assistant_message_complete contained correct SQL.")
                        else:
                            print(f"❌ FAIL: SQL in event: {content.get('sql_query')}")
                            
                        if "results" in content:
                            print("✅ PASS: assistant_message_complete contained results field.")
                        else:
                            print("❌ FAIL: results field missing.")
                except:
                    pass
                    
    if found_complete_event:
        print("✅ PASS: assistant_message_complete event FOUND.")
    else:
        print("❌ FAIL: assistant_message_complete event MISSING.")

if __name__ == "__main__":
    test_database_sql_results_event()
