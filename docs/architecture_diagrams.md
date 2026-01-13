# TraderBot Architecture Analysis

This document provides comprehensive Mermaid diagrams for the TraderBot financial assistant application.

---

## 1. High Overview

A simplified view of the entire system showing major components and their interactions.

```mermaid
flowchart TB
    subgraph Frontend["🖥️ Frontend"]
        UI[Streamlit UI<br/>main.py]
    end

    subgraph API["🔌 FastAPI Backend"]
        API_Layer[api.py<br/>REST API + SSE Streaming]
    end

    subgraph Services["⚙️ Core Services"]
        Router[LLM Query Router]
        Planner[Query Planner]
        Executor[Task Executor]
    end

    subgraph Handlers["📊 Data Handlers"]
        DBHandler[Database Handler]
        InternetHandler[Internet Data Handler]
        GreetingHandler[Greeting Handler]
    end

    subgraph Response["💬 Response Generation"]
        URG[Unified Response Generator]
    end

    subgraph External["🌐 External"]
        DB[(PostgreSQL<br/>Portfolio Data)]
        FMP[FMP API<br/>Market Data]
        LLM[LLM Provider<br/>Ollama/Qwen]
    end

    UI <-->|HTTP/SSE| API_Layer
    API_Layer --> Router
    API_Layer --> Planner
    Planner --> Executor
    Executor --> DBHandler
    Executor --> InternetHandler
    API_Layer --> GreetingHandler
    DBHandler --> URG
    InternetHandler --> URG
    DBHandler --> DB
    DBHandler --> LLM
    InternetHandler --> FMP
    URG --> LLM
```

---

## 2. Detailed Overview

Complete system architecture showing all components, services, and data flow.

```mermaid
flowchart TB
    subgraph Client["🖥️ Streamlit Frontend (main.py)"]
        ChatInput[Chat Input]
        ChatDisplay[Chat Display with HTML]
        SessionState[Session State<br/>Messages & History]
    end

    subgraph FastAPI["🔌 FastAPI Backend (api.py)"]
        subgraph Endpoints
            StreamEndpoint["/query/stream"<br/>Main SSE Endpoint]
            DBStream["/query/database/stream"]
            NetStream["/query/internet/stream"]
            GreetingEP["/query/greeting"]
            ClassifyEP["/query/classify"]
            InitEP["/initialize"]
        end
        
        subgraph AppState["Application State"]
            AS_Router[router: LLMQueryRouter]
            AS_Planner[planner: QueryPlanner]
            AS_TaskExec[task_executor: TaskExecutor]
            AS_DBHandler[db_handler: DatabaseQueryHandler]
            AS_NetHandler[internet_data_handler]
            AS_Greeting[greeting_handler]
            AS_Context[context_manager]
            AS_Memory[chat_memory]
        end
    end

    subgraph CoreServices["⚙️ Core Services"]
        LLMRouter[llm_query_router.py<br/>Query Classification]
        QPlanner[query_planner.py<br/>Multi-Step Planning]
        TExecutor[task_executor.py<br/>Plan Execution]
        CtxMgr[context_manager.py<br/>Topic Detection]
        ChatMem[chat_memory.py<br/>Conversation History]
    end

    subgraph DataHandlers["📊 Data Handlers"]
        DBH[database_handler.py<br/>SQL Generation & Execution]
        IDH[internet_data_handler.py<br/>FMP API Integration]
        GH[greeting_handler.py<br/>Conversational Responses]
    end

    subgraph ResponseGen["💬 Response Generation"]
        URG[unified_response_generator.py<br/>Streaming HTML Responses]
        Prompts[prompts.py<br/>Prompt Templates & Glossary]
        Cleaner[response_cleaner.py<br/>HTML Sanitization]
    end

    subgraph LLMLayer["🤖 LLM Provider (llm_provider.py)"]
        LLMConfig[get_llm / get_streaming_llm]
        OllamaLLM[Ollama]
        QwenLLM[Qwen H100]
    end

    subgraph DataSources["💾 Data Sources"]
        PSQL[(PostgreSQL<br/>portfolio_holdings<br/>portfolio_holdings_realized_pnl)]
        FMPAPI[Financial Modeling Prep API<br/>Quotes, News, Gainers/Losers]
    end

    ChatInput --> StreamEndpoint
    StreamEndpoint --> CtxMgr
    StreamEndpoint --> LLMRouter
    LLMRouter -->|database| DBH
    LLMRouter -->|internet_data| IDH
    LLMRouter -->|greeting| GH
    LLMRouter -->|hybrid| QPlanner
    QPlanner --> TExecutor
    TExecutor --> DBH
    TExecutor --> IDH
    DBH --> URG
    IDH --> URG
    URG --> Prompts
    URG --> LLMConfig
    LLMConfig --> OllamaLLM
    LLMConfig --> QwenLLM
    DBH --> PSQL
    IDH --> FMPAPI
    URG --> Cleaner
    Cleaner --> ChatDisplay
```

---

## 3. Query Routing

How user queries are classified and routed to appropriate handlers.

```mermaid
flowchart TB
    subgraph Input
        Query[User Query]
        History[Chat History]
    end

    subgraph ContextCheck["🔄 Context Manager"]
        CM{Is Query<br/>Relevant to<br/>History?}
        Reset[Reset History]
        Keep[Keep History]
    end

    subgraph Classification["🏷️ LLM Query Router"]
        LLM_Classify[LLM Classification<br/>with Conversation Context]
        
        subgraph Categories
            DB[database]
            NET[internet_data]
            HYB[hybrid]
            GREET[greeting]
        end
    end

    subgraph Routing["➡️ Route to Handler"]
        DBPath[Database Handler<br/>SQL Generation]
        NetPath[Internet Handler<br/>FMP API Fetch]
        HybPath[Query Planner<br/>Multi-Step Plan]
        GreetPath[Greeting Handler<br/>Conversational]
    end

    Query --> CM
    History --> CM
    CM -->|NEW_TOPIC| Reset
    CM -->|RELEVANT| Keep
    Reset --> LLM_Classify
    Keep --> LLM_Classify
    
    LLM_Classify --> DB
    LLM_Classify --> NET
    LLM_Classify --> HYB
    LLM_Classify --> GREET
    
    DB --> DBPath
    NET --> NetPath
    HYB --> HybPath
    GREET --> GreetPath
```

### Query Classification Rules

```mermaid
flowchart LR
    subgraph ClassificationLogic["Classification Decision Tree"]
        Q[Query]
        
        Q --> Check1{Mentions<br/>portfolio, holdings,<br/>returns, PnL?}
        Check1 -->|Yes| Check2{Needs LIVE<br/>market data?}
        Check1 -->|No| Check3{Asks about<br/>stock prices,<br/>news, market?}
        
        Check2 -->|Yes| HybridOut[hybrid]
        Check2 -->|No| DatabaseOut[database]
        
        Check3 -->|Yes| InternetOut[internet_data]
        Check3 -->|No| Check4{Is greeting<br/>or small talk?}
        
        Check4 -->|Yes| GreetingOut[greeting]
        Check4 -->|No| FallbackOut[database<br/>Safe Fallback]
    end
```

---

## 4. Response Unifying

How responses are generated consistently across all query types.

```mermaid
flowchart TB
    subgraph Input["📥 Input Sources"]
        DBData[Database Results<br/>SQL Query + DataFrame]
        NetData[Internet Data<br/>Raw Market Data]
        GrtData[Greeting Context<br/>Chat History]
        HybData[Hybrid Context<br/>Combined Data]
    end

    subgraph URG["🔄 Unified Response Generator"]
        Init[Initialize<br/>get_streaming_llm]
        
        subgraph FormatContext["Format Data Context"]
            FmtDB["SQL Query Used:<br/>```sql<br/>SELECT ...```<br/>Query Results: {...}"]
            FmtNet["Retrieved Data:<br/>{market_data}"]
            FmtGrt["Recent conversation:<br/>- User: ...<br/>- Assistant: ..."]
            FmtHyb["Combined Data:<br/>{db + internet}"]
        end
        
        subgraph BuildPrompt["Build Prompt"]
            DetectLang[detect_language]
            Template[UNIFIED_RESPONSE_PROMPT<br/>+ Arabic Glossary if AR]
        end
        
        Stream[stream_response<br/>Yield Chunks]
    end

    subgraph Output["📤 Output"]
        Clean[response_cleaner.py<br/>clean_llm_chunk]
        SSE[SSE Events<br/>type: content]
    end

    DBData --> FmtDB
    NetData --> FmtNet
    GrtData --> FmtGrt
    HybData --> FmtHyb
    
    FmtDB --> DetectLang
    FmtNet --> DetectLang
    FmtGrt --> DetectLang
    FmtHyb --> DetectLang
    
    DetectLang --> Template
    Template --> Stream
    Stream --> Clean
    Clean --> SSE
```

### Response Prompt Structure

```mermaid
flowchart LR
    subgraph PromptComponents["UNIFIED_RESPONSE_PROMPT"]
        Lang["{language}"]
        Date["{today_date}"]
        Query["{query}"]
        Type["{context_type}"]
        Data["{data_context}"]
        Glossary["{arabic_glossary}"]
    end
    
    subgraph HTMLRules["HTML Formatting Rules"]
        Currency[".currency → $1,234.56"]
        Percent[".percent → +12.5%"]
        Highlight[".highlight → Portfolio Name"]
        Positive[".positive → outperforming"]
        Negative[".negative → underperforming"]
    end
    
    PromptComponents --> HTMLRules
```

---

## 5. Arabic Language Handling

Complete flow for detecting and responding in Arabic with proper RTL support.

```mermaid
flowchart TB
    subgraph Detection["🔍 Language Detection (prompts.py)"]
        Input[User Query]
        Regex["Arabic Unicode Pattern<br/>[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+"]
        Check{Arabic<br/>Characters<br/>Found?}
        AR[Return: Arabic]
        EN[Return: English]
    end

    subgraph Glossary["📚 Arabic Financial Glossary"]
        Terms["Portfolio → المحفظة<br/>Investment → الاستثمار<br/>Return → العائد<br/>Profit → الربح<br/>Loss → الخسارة<br/>YTD → منذ بداية السنة الحالية<br/>Benchmark → المعيار المرجعي<br/>..."]
    end

    subgraph PromptBuilding["🔧 Prompt Configuration"]
        PromptTemplate[UNIFIED_RESPONSE_PROMPT]
        IncludeGlossary{Language<br/>== Arabic?}
        AddGlossary[Add ARABIC_FINANCIAL_GLOSSARY]
        NoGlossary[glossary = empty]
    end

    subgraph LLMInstructions["📝 LLM Instructions"]
        TranslateAll["Translate ALL English<br/>financial terms to Arabic"]
        NoMixing["DO NOT mix English words<br/>in Arabic responses"]
        DigitsOK["Numbers can remain as digits<br/>e.g., 1,234.56"]
    end

    subgraph HTMLFormatting["🎨 RTL HTML Formatting"]
        WrapRTL["Wrap ENTIRE response in:<br/>&lt;div class='rtl-content'&gt;...&lt;/div&gt;"]
        
        subgraph CSSStyles["Frontend CSS (main.py)"]
            RTLClass[".rtl-content {<br/>  direction: rtl;<br/>  text-align: right;<br/>  font-family: 'Noto Naskh Arabic';<br/>}"]
        end
    end

    subgraph Example["✅ Example Output"]
        ArabicResponse["&lt;div class='rtl-content'&gt;<br/>&lt;p&gt;قيمة &lt;span class='highlight'&gt;محفظتك&lt;/span&gt;<br/>هي &lt;span class='currency'&gt;$150,000&lt;/span&gt;،<br/>بزيادة &lt;span class='percent'&gt;+5.2%&lt;/span&gt;<br/>منذ بداية السنة.&lt;/p&gt;<br/>&lt;/div&gt;"]
    end

    Input --> Regex
    Regex --> Check
    Check -->|Yes| AR
    Check -->|No| EN
    
    AR --> IncludeGlossary
    EN --> IncludeGlossary
    
    IncludeGlossary -->|Yes| AddGlossary
    IncludeGlossary -->|No| NoGlossary
    
    AddGlossary --> Terms
    Terms --> PromptTemplate
    NoGlossary --> PromptTemplate
    
    PromptTemplate --> TranslateAll
    TranslateAll --> NoMixing
    NoMixing --> DigitsOK
    DigitsOK --> WrapRTL
    WrapRTL --> RTLClass
    RTLClass --> ArabicResponse
```

### Language Detection Function

```mermaid
flowchart LR
    subgraph detect_language["def detect_language(text: str)"]
        Step1["1. Compile Arabic Unicode regex"]
        Step2["2. Find all Arabic character matches"]
        Step3["3. Return 'Arabic' if matches > 0"]
        Step4["4. Else return 'English'"]
    end
    
    Step1 --> Step2 --> Step3 --> Step4
```

---

## Summary

| Component | File | Purpose |
|-----------|------|---------|
| **API Layer** | `api.py` | FastAPI endpoints, SSE streaming, query routing |
| **Query Router** | `src/services/llm_query_router.py` | LLM-based query classification |
| **Query Planner** | `src/services/query_planner.py` | Multi-step execution planning |
| **Task Executor** | `src/services/task_executor.py` | Plan execution with SSE |
| **Database Handler** | `src/services/database_handler.py` | SQL generation & execution |
| **Internet Handler** | `src/services/internet_data_handler.py` | FMP API integration |
| **Response Generator** | `src/services/unified_response_generator.py` | Unified streaming responses |
| **Prompts** | `src/config/prompts.py` | All prompt templates & Arabic glossary |
| **Frontend** | `main.py` | Streamlit chat UI with RTL support |
