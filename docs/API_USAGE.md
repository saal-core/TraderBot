# TraderBot API Usage Guide

This document provides examples on how to invoke the TraderBot API.

---

## Quick Start

### 1. Initialize the Service (Required First)

```bash
curl -X POST http://localhost:8001/initialize
```

---

## Two Ways to Query

| Endpoint | Use Case |
|----------|----------|
| `POST /query` | **Recommended for backend integration** - Returns complete JSON response |
| `POST /query/stream` | For real-time UI streaming via SSE |

---

## Non-Streaming Endpoint (Recommended)

**`POST /query`** - Simplest integration, no SSE parsing needed.

### Basic Query

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the top 5 stocks in my portfolio?"}'
```

### Response Format

```json
{
  "success": true,
  "query_type": "database",
  "full_response": "Here are your top 5 stocks...",
  "sql_query": "SELECT * FROM holdings ORDER BY value DESC LIMIT 5",
  "results": [{"symbol": "AAPL", "value": 18550.00}, ...],
  "error": null
}
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8001/query",
    json={"query": "What are the top 5 stocks in my portfolio?"}
).json()

print(response["full_response"])
print(response["results"])
```

---

## Streaming Endpoint (For Real-Time UI)

**`POST /query/stream`** - Returns Server-Sent Events for real-time streaming.

```bash
curl -X POST http://localhost:8001/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the top 5 stocks in my portfolio?"}'
```

---

## Using Chat History

Chat history enables **follow-up questions** by providing context from previous exchanges.

### Chat History Structure

```json
{
  "role": "user" | "assistant",
  "content": "Message text",
  "timestamp": "2026-01-27T10:00:00",       // optional
  "sql_query": "SELECT ...",                
  "results": [{"col": "value"}, ...],        
  "query_type": "database"                  
}
```

> **Tip for Backend Teams:** Only `role` and `content` are required. the sql query and results are important for follow-up questions.

---

### Example 1: Simple Follow-Up

```bash
curl -X POST http://localhost:8001/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "And what about MSFT?",
    "chat_history": [
      {"role": "user", "content": "What is the price of AAPL?"},
      {"role": "assistant", "content": "Apple (AAPL) is currently trading at $185.50."}
    ]
  }'
```

---

### Example 2: Database Query with Results Context

When follow-up queries reference previous results, include `results` for accuracy:
The results are the response of the sql query executed 

```bash
curl -X POST http://localhost:8001/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total value of these stocks?",
    "chat_history": [
      {
        "role": "user",
        "content": "Show me all tech stocks"
      },
      {
        "role": "assistant",
        "content": "Here are your tech stocks...",
        "query_type": "database",
        "sql_query": "SELECT * FROM holdings WHERE sector = '\''Technology'\''",
        "results": [ 
          {"symbol": "AAPL", "shares": 100, "price": 185.50},
          {"symbol": "MSFT", "shares": 50, "price": 420.00}
        ]
      }
    ]
  }'
```

---

### Example 3: Multi-Turn Conversation

> **Important:** Always include `sql_query` and `results` in assistant messages - they provide critical context for the LLM to understand follow-up questions.

```bash
curl -X POST http://localhost:8001/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare that to the S&P 500",
    "chat_history": [
      {
        "role": "user",
        "content": "What is my portfolio performance?"
      },
      {
        "role": "assistant",
        "content": "Your portfolio gained 12.5% this year.",
        "query_type": "database",
        "sql_query": "SELECT SUM(gain_pct) as total_gain FROM portfolio_performance WHERE year = 2026",
        "results": [{"total_gain": 12.5}]
      },
      {
        "role": "user",
        "content": "How does that compare to last year?"
      },
      {
        "role": "assistant",
        "content": "Last year your portfolio gained 8.2%.",
        "query_type": "database",
        "sql_query": "SELECT SUM(gain_pct) as total_gain FROM portfolio_performance WHERE year = 2025",
        "results": [{"total_gain": 8.2}]
      }
    ]
  }'
```

---

## SSE Response Events

The streaming endpoint returns Server-Sent Events:

| Event Type | Description |
|------------|-------------|
| `status` | Progress updates (e.g., "Generating SQL...") |
| `sql` | The generated SQL query |
| `results` | Query results as JSON array |
| `content` | Streamed response text chunks |
| `assistant_message_complete` | Final event with `full_response` key |
| `stream_end` | Stream finished |
| `error` | Error occurred |

### Handling the Complete Response

After streaming ends, `assistant_message_complete` contains the full response:

```json
{
  "type": "assistant_message_complete",
  "data": {
    "query_type": "database",
    "sql_query": "SELECT * FROM ...",
    "results": [...],
    "full_response": "Here is the complete LLM response text..."
  }
}
```

---

## Best Practices for Backend Integration

1. **Minimal History:** Only include the last 3-5 exchanges to avoid token limits
2. **Include Results:** For database follow-ups, always pass `results` from previous queries
3. **Topic Changes:** The API auto-detects topic changes and resets context
4. **Parse SSE:** Use an SSE parser library for your language

### Python SSE Client Example

```python
import requests
import json

def stream_query(query: str, chat_history: list = None):
    response = requests.post(
        "http://localhost:8001/query/stream",
        json={
            "query": query,
            "chat_history": chat_history or []
        },
        stream=True
    )
    
    full_response = ""
    for line in response.iter_lines():
        if line and line.startswith(b"data:"):
            event = json.loads(line[5:])
            
            if event["type"] == "content":
                print(event["content"], end="", flush=True)
            
            elif event["type"] == "assistant_message_complete":
                full_response = event["data"].get("full_response", "")
    
    return full_response
```

---

## JavaScript/Fetch Example

```javascript
async function streamQuery(query, chatHistory = []) {
  const response = await fetch('http://localhost:8001/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, chat_history: chatHistory })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (line.startsWith('data:')) {
        const event = JSON.parse(line.slice(5));
        
        if (event.type === 'content') {
          console.log(event.content);
        } else if (event.type === 'assistant_message_complete') {
          fullResponse = event.data.full_response;
        }
      }
    }
  }
  return fullResponse;
}
```
