# TraderBot Solution Architecture

This document describes the high-level architecture of the TraderBot application, a financial assistant that combines local portfolio data with real-time internet market data using LLMs.

## Architecture Graph

```mermaid
graph TD
    %% Styling
    classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef api fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef router fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef handler fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef service fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef llm fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef db fill:#e0f7fa,stroke:#006064,stroke-width:2px;
    classDef external fill:#eeeeee,stroke:#616161,stroke-width:2px;

    %% Client Layer
    User([User])
    Frontend["Streamlit App\n(main.py)"]:::frontend

    %% API Layer
    API["FastAPI Backend\n(api.py)"]:::api
    
    %% Routing Layer
    Router{"Optimized\nQuery Router"}:::router

    %% Handler Layer
    DBHandler[DatabaseQueryHandler]:::handler
    NetHandler[InternetDataHandler]:::handler
    CompHandler[ComparisonHandler]:::handler
    GreetHandler[GreetingHandler]:::handler

    %% Service Layer
    SQLExec[PostgreSQLExecutor]:::service
    AliasRes[PortfolioAliasResolver]:::service
    FMP[FMPService]:::service
    Perplexity[PerplexityService]:::service
    Memory[ChatMemory]:::service

    %% Infrastructure / External
    Postgres[(PostgreSQL DB)]:::db
    FMP_API((FMP API)):::external
    Perplexity_API((Perplexity API)):::external

    %% LLMs
    Ollama(("Ollama\nLocal LLM")):::llm
    OpenAI(("OpenAI\nGPT-4o")):::llm
    Qwen(("Qwen\nExplanation")):::llm

    %% Connections
    User -->|Interacts| Frontend
    Frontend -->|HTTP / SSE| API
    API --> Router
    API --> Memory

    %% Routing Logic
    Router -->|Classify| Ollama
    Router -->|"DB Query"| DBHandler
    Router -->|"Internet Query"| NetHandler
    Router -->|"Comparison"| CompHandler
    Router -->|"Greeting"| GreetHandler

    %% Database Flow
    DBHandler -->|Resolve Entities| AliasRes
    DBHandler -->|Generate SQL| OpenAI
    DBHandler -->|Execute SQL| SQLExec
    SQLExec --> Postgres
    DBHandler -->|Explain Results| Qwen
    
    %% Alias Resolver Dependencies
    AliasRes --> SQLExec

    %% Internet Flow
    NetHandler -->|Fetch Data| FMP
    NetHandler -->|Fetch News| Perplexity
    FMP --> FMP_API
    Perplexity --> Perplexity_API
    NetHandler -->|Explain Data| Qwen

    %% Comparison Flow
    CompHandler -->|Plan| OpenAI
    CompHandler -->|Get Local| DBHandler
    CompHandler -->|Get External| NetHandler
    CompHandler -->|Synthesize| OpenAI

    %% Helper Connections
    Router -.->|Entity Check| SQLExec
```

## Component Overview

### 1. Frontend (Streamlit)
- **File**: `main.py`
- **Role**: Provides a chat interface for the user.
- **Features**:
  - Handles user input.
  - Displays streaming responses (SSE).
  - Renders Markdown tables and SQL queries.
  - Manages session state and chat history.

### 2. Backend API (FastAPI)
- **File**: `api.py`
- **Role**: Exposes REST endpoints for the frontend.
- **Key Endpoints**:
  - `/query/stream`: Unified streaming endpoint.
  - `/initialize`: Sets up database and service connections.
  - `/query/classify`: Routes queries to appropriate handlers.

### 3. Query Router
- **File**: `src/services/gpt_oss_query_router_v2.py`
- **Role**: Classifies user queries into categories to determine the processing path.
- **Mechanism**:
  1.  **Tier 1**: Regex patterns (Instant).
  2.  **Tier 2**: Keyword/Entity matching (Fast).
  3.  **Tier 3**: LLM (Ollama) fallback (Slow/Accurate).
- **Categories**: `database`, `internet_data`, `comparison`, `greeting`.

### 4. Handlers
Handlers contain the core business logic for each query type.

- **DatabaseQueryHandler** (`src/services/database_handler.py`):
  - Resolves privacy-preserving aliases (using `PortfolioAliasResolver`).
  - Generates SQL queries using **OpenAI**.
  - Executes queries against **PostgreSQL**.
  - Explains results using **Qwen** (via OpenAI-compatible API).

- **InternetDataHandler** (`src/services/internet_data_handler.py`):
  - Fetches real-time market data, crypto, forex, and news.
  - Integration with **Financial Modeling Prep (FMP)** for structured data.
  - Integration with **Perplexity** for news and research.
  - Explains findings using **Qwen**.

- **ComparisonHandler** (`src/services/comparison_handler.py`):
  - Orchestrates complex queries requiring both local and external data.
  - Uses **OpenAI** to plan the comparison.
  - Calls `DatabaseQueryHandler` and `InternetDataHandler` in parallel.
  - Synthesizes a comparative response using **OpenAI**.

### 5. Services & Utilities
- **PostgreSQLExecutor**: Manages database connections and query execution.
- **PortfolioAliasResolver**: Maps user-friendly names to database IDs (and anonymizes them for the LLM).
- **ChatMemory**: Manages conversation context for follow-up questions.
