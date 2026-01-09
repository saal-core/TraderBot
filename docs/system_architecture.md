# TraderBot System Architecture

## Overview

TraderBot is a natural language financial assistant that enables users to query portfolio databases, fetch real-time market data, and receive AI-powered explanations. The system uses a Streamlit frontend, FastAPI backend, and configurable LLM providers (Ollama or Qwen H100).

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Frontend["🖥️ Frontend"]
        UI["**Streamlit UI**<br/>main.py"]
    end

    subgraph Backend["⚙️ FastAPI Backend"]
        API["**API Server**<br/>api.py"]
    end

    subgraph QueryRouting["🧭 Query Routing"]
        Router["**LLM Query Router**<br/>Classifies user queries"]
        Planner["**Query Planner**<br/>Multi-step execution plans"]
    end

    subgraph Handlers["📦 Query Handlers"]
        DB["**Database Handler**<br/>SQL generation & execution"]
        Internet["**Internet Data Handler**<br/>Real-time market data"]
        Greeting["**Greeting Handler**<br/>Chitchat responses"]
        TaskExec["**Task Executor**<br/>Plan execution & streaming"]
    end

    subgraph ExternalServices["🌐 External Services"]
        FMP["**FMP API**<br/>Financial Modeling Prep"]
        LLM["**LLM Provider**<br/>Ollama / Qwen H100"]
    end

    subgraph DataStores["🗄️ Data Stores"]
        PostgreSQL["**PostgreSQL**<br/>Portfolio Database"]
        Memory["**Chat Memory**<br/>Conversation History"]
    end

    UI -->|"HTTP/SSE"| API
    API --> Router
    Router -->|"database"| DB
    Router -->|"internet_data"| Internet
    Router -->|"greeting"| Greeting
    Router -->|"hybrid/comparison"| Planner
    Planner --> TaskExec
    TaskExec --> DB
    TaskExec --> Internet

    DB -->|"Generate SQL"| LLM
    DB -->|"Execute Query"| PostgreSQL
    DB -->|"Explain Results"| LLM

    Internet -->|"Fetch Data"| FMP
    Internet -->|"Explain Data"| LLM

    Greeting -->|"Generate Response"| LLM

    API --> Memory
```

---

## Component Details

```mermaid
flowchart LR
    subgraph Config["⚙️ Configuration Layer"]
        direction TB
        LLMProvider["**llm_provider.py**<br/>Provider switching"]
        Settings["**settings.py**<br/>Environment config"]
        Prompts["**prompts.py**<br/>LLM prompt templates"]
    end

    subgraph Services["📦 Service Layer"]
        direction TB
        LLMRouter["**LLMQueryRouter**<br/>Query classification"]
        QueryPlanner["**QueryPlanner**<br/>Plan generation"]
        TaskExecutor["**TaskExecutor**<br/>Plan execution"]
        DBHandler["**DatabaseQueryHandler**<br/>SQL & explanations"]
        InternetHandler["**InternetDataHandler**<br/>Market data & news"]
        GreetHandler["**GreetingHandler**<br/>Conversation"]
        FMPService["**FMPService**<br/>FMP API wrapper"]
        ChatMemory["**ChatMemory**<br/>History management"]
        SQLUtils["**PostgreSQLExecutor**<br/>Connection pooling"]
    end

    LLMProvider --> LLMRouter
    LLMProvider --> QueryPlanner
    LLMProvider --> TaskExecutor
    LLMProvider --> DBHandler
    LLMProvider --> InternetHandler
    LLMProvider --> GreetHandler

    Settings --> LLMProvider
    Settings --> SQLUtils
    Prompts --> DBHandler
    Prompts --> InternetHandler
    Prompts --> QueryPlanner
```

---

## Request Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Streamlit as Streamlit UI
    participant FastAPI as FastAPI Server
    participant Router as LLM Query Router
    participant Planner as Query Planner
    participant Executor as Task Executor
    participant DBHandler as Database Handler
    participant InternetHandler as Internet Handler
    participant LLM as LLM Provider
    participant PostgreSQL
    participant FMP as FMP API

    User->>Streamlit: Enter natural language query
    Streamlit->>FastAPI: POST /query/stream
    FastAPI->>Router: classify_query(query, chat_history)
    Router->>LLM: Classification prompt
    LLM-->>Router: Category (database/internet_data/hybrid/greeting)

    alt database query
        FastAPI->>DBHandler: generate_sql()
        DBHandler->>LLM: SQL generation prompt
        LLM-->>DBHandler: SQL query
        DBHandler->>PostgreSQL: Execute SQL
        PostgreSQL-->>DBHandler: Query results
        DBHandler->>LLM: Explanation prompt (streaming)
        LLM-->>DBHandler: Explanation chunks
        DBHandler-->>FastAPI: SSE events
    else internet_data query
        FastAPI->>InternetHandler: fetch_raw_data()
        InternetHandler->>FMP: API requests
        FMP-->>InternetHandler: Market data
        InternetHandler->>LLM: Explanation prompt (streaming)
        LLM-->>InternetHandler: Explanation chunks
        InternetHandler-->>FastAPI: SSE events
    else hybrid query
        FastAPI->>Planner: generate_plan()
        Planner->>LLM: Planning prompt
        LLM-->>Planner: Execution plan
        Planner-->>Executor: Plan steps
        loop For each step
            Executor->>DBHandler: If database step
            Executor->>InternetHandler: If internet step
        end
        Executor-->>FastAPI: Combined SSE events
    end

    FastAPI-->>Streamlit: SSE stream
    Streamlit-->>User: Display response
```

---

## Data Flow for Query Types

```mermaid
graph TD
    subgraph QueryTypes["Query Classification"]
        Q["User Query"]
        Q --> C{LLM Router}
        C -->|database| T1["Portfolio/Holdings Questions"]
        C -->|internet_data| T2["Market Data/News"]
        C -->|hybrid| T3["Portfolio + Market Data"]
        C -->|greeting| T4["Chitchat/Greetings"]
    end

    subgraph DatabaseFlow["Database Query Flow"]
        T1 --> SQL["Generate SQL via LLM"]
        SQL --> Exec["Execute on PostgreSQL"]
        Exec --> Explain["Stream Explanation via LLM"]
    end

    subgraph InternetFlow["Internet Query Flow"]
        T2 --> Classify["Classify Sub-type"]
        Classify --> Fetch["Fetch from FMP API"]
        Fetch --> Stream["Stream Explanation via LLM"]
    end

    subgraph HybridFlow["Hybrid Query Flow"]
        T3 --> Plan["Generate Multi-step Plan"]
        Plan --> Step1["Step 1: Database Query"]
        Plan --> Step2["Step 2: Internet Query"]
        Step1 --> Combine["Combine Results"]
        Step2 --> Combine
        Combine --> FinalExplain["Stream Combined Explanation"]
    end

    subgraph GreetingFlow["Greeting Flow"]
        T4 --> Respond["LLM Response with Context"]
    end
```

---

## LLM Provider Configuration

```mermaid
flowchart TB
    subgraph Providers["LLM Provider Options"]
        ENV["LLM_PROVIDER env var"]
        ENV -->|"ollama"| Ollama["🦙 **Ollama**<br/>Local deployment<br/>OpenAI-compatible API"]
        ENV -->|"qwen"| Qwen["🚀 **Qwen H100**<br/>Remote H100 GPU<br/>OpenAI-compatible API"]
    end

    subgraph Functions["Provider Functions"]
        GetLLM["get_llm()"]
        GetStreaming["get_streaming_llm()"]
        GetSQL["get_sql_llm()"]
        GetExplanation["get_explanation_llm()"]
        GetClassification["get_classification_llm()"]
    end

    Ollama --> GetLLM
    Qwen --> GetLLM
    GetLLM --> GetStreaming
    GetLLM --> GetSQL
    GetLLM --> GetExplanation
    GetLLM --> GetClassification
```

---

## File Structure Overview

```
TraderBot/
├── main.py                    # Streamlit UI frontend
├── api.py                     # FastAPI backend server
├── test2sql_prompt.md         # SQL generation prompt template
│
├── src/
│   ├── config/
│   │   ├── llm_provider.py    # LLM provider switching (Ollama/Qwen)
│   │   ├── settings.py        # Environment configuration
│   │   └── prompts.py         # LLM prompt templates
│   │
│   ├── services/
│   │   ├── llm_query_router.py      # Query classification
│   │   ├── query_planner.py         # Multi-step plan generation
│   │   ├── task_executor.py         # Plan execution with streaming
│   │   ├── database_handler.py      # SQL generation & explanations
│   │   ├── internet_data_handler.py # Real-time market data
│   │   ├── greating_handler.py      # Chitchat responses
│   │   ├── fmp_service.py           # FMP API wrapper
│   │   ├── chat_memory.py           # Conversation history
│   │   └── sql_utilities.py         # PostgreSQL connection pooling
│   │
│   └── utils/
│       ├── response_cleaner.py      # LLM response formatting
│       └── toon_formatter.py        # Token-efficient data formatting
│
└── docs/
    └── system_architecture.md       # This file
```

---

## Key Technologies

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Database | PostgreSQL |
| LLM Providers | Ollama (local), Qwen H100 (remote) |
| LLM Framework | LangChain |
| Market Data | Financial Modeling Prep API |
| Streaming | Server-Sent Events (SSE) |
| Connection Pool | psycopg2 ThreadedConnectionPool |

---

## Streaming Architecture

```mermaid
sequenceDiagram
    participant Client as Streamlit Client
    participant API as FastAPI Server
    participant Handler as Query Handler
    participant LLM as LLM Provider

    Client->>API: GET /query/stream (EventSource)
    activate API
    API->>Handler: Process query

    loop Streaming Response
        Handler->>LLM: Request chunk
        LLM-->>Handler: Response chunk
        Handler-->>API: Yield SSE event
        API-->>Client: data: {"type": "content", "data": "..."}
    end

    Handler-->>API: Final metadata
    API-->>Client: data: {"type": "stream_end"}
    deactivate API
```

---

*Generated: 2026-01-08*
