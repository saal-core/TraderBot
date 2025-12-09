import streamlit as st
import pandas as pd
import os
import requests
import re
import traceback
from typing import List, Dict, Optional,Any
import unicodedata
import re
import requests
# Vanna specific imports
import ollama
from vanna.ollama import Ollama
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from openai import OpenAI

# --- 1. Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_API_URL = "http://localhost:11434"
PERPLEXITY_API_KEY = ""
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
#config = {'model': OLLAMA_MODEL, 'ollama_api_url': OLLAMA_API_URL, 'persist_directory': './vanna_chromadb_store_test'}
config = {'model': OLLAMA_MODEL, 'ollama_api_url': OLLAMA_API_URL, 'persist_directory': './vanna_chromadb_store_tues','temperature': 0.01}


MAX_UI_MESSAGES_TO_DISPLAY = 20

# --- 2. Prompts Section ---

# ENHANCEMENT 2: Define Arabic glossary for reuse
ARABIC_FINANCIAL_GLOSSARY = """
- Portfolio: المحفظة
- Investment: الاستثمار
- Index: المؤشر
- Benchmark: المعيار المرجعي
- Return: العائد
- Profit: الربح
- Loss: الخسارة
- Unrealized Profit/Loss: الربح/الخسارة غير المحققة
- Liquidity: السيولة
- Market Value: القيمة السوقية
- Holdings: الحيازات
- Position: المركز
- Inception: تاريخ البدء / التأسيس
- Quarter-to-Date (QTD): منذ بداية الربع الحالي
- Month-to-Date (MTD): منذ بداية الشهر الحالي
- Year-to-Date (YTD): منذ بداية السنة الحالية
- Trend: الاتجاه
- Utilization: نسبة الاستخدام
- Cost Model: نموذج التكلفة
- Default Index: المؤشر الافتراضي
- Net Liquidity: صافي السيولة
- Market Index: مؤشر السوق
"""


# --- UPDATED PROMPT ---
# This prompt is now the more detailed version from your code snippet.
vanna_explanation_prompt = """You are an experienced equity fund manager. Your primary goal is to assist non-financial stakeholders in understanding financial data.
Your response must only use plain ASCII text characters. Do NOT use any special Unicode symbols.
Under no circumstances should your response include tables or tabular data. Instead, convert all tabular data to plain English text and seamlessly integrate specific data points into your explanation.
Please explain the results clearly, concisely, and factually for non-financial stakeholders.
Do not include citations, footnotes, or separate 'Note' sections. Only provide the final answer.
Avoid LaTeX or complex calculations. Be precise and concise.
Based on the Input Question, generate the response in the respective language (English or Arabic).

"If the answer contains multiple distinct points, facts, or sentences, you MUST format them as a bulleted list using markdown hyphens (e.g., `- First point.`)."
"A single-sentence answer does not need a bullet point.\n"



"""


CUSTOM_ERROR_MESSAGE = "Sorry, I am currently unable to retrieve that information. Please try rephrasing your question or ask about a different topic."

perplexity_system_prompt = "You are a helpful and factual AI assistant..." # (Omitted for brevity, same as before)
other_handler_prompt = "You are a helpful and professional AI financial assistant...Answer only finance related questions, Small Talk and Casual conversation questions. Respond in professional For other topics respond like 'I'm Sorry, I'm Financial Bot. I can only help you with finance queries'. You're capabilities limited to answering financial questions and guidance. You cannot answer on insurance, taxes etc. Handle Offensive Questions Gracefully. Handle English and Arabic Questions." # (Omitted for brevity, same as before)

# --- 3. Core Functions ---
def call_ollama_chat(messages: List[Dict[str, str]], model: str = OLLAMA_MODEL, temperature: float = 0.7, max_tokens: int = 10000) -> str:
    # (Omitted for brevity, same as before)
    print("\n--- Prompt to Ollama ---")
    print(f"Model: {model}, Temp: {temperature}, Max Tokens: {max_tokens}")
    for msg in messages:
        print(f"{msg['role'].upper()}: {msg['content']}")
    print("------------------------")
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
            stream=False
        )
        raw_output = response['message']['content'].strip()
        print("\n--- Raw Ollama Response ---\n", raw_output, "\n--------------------------")
        return raw_output
    except Exception as e:
        print(f"Error calling Ollama chat: {e}")
        traceback.print_exc()
        return f"Error: Could not get response from Ollama. {e}"


def classify_query_type(question: str) -> str:
    """
    Classifies a user's question into 'portfolio', 'general', or 'other'
    using a robust prompt structure that combines rules, examples, and safe fallbacks.
    """
    
    # The System Prompt defines the AI's role and high-level goal.
    system_prompt = "You are an expert query classifier. Your job is to classify the user's query into one of three categories: portfolio, general, or other. You must respond with only the single category name."

    # The User Prompt provides the detailed instructions, rules, and examples for the task.
    user_prompt = """Here are the rules and examples to follow.

---
**RULES**
1. portfolio  
   - Any question that is about **the user’s personal portfolio** or **portfolio-related information**, including:
     - Individual portfolio data: "What are my dividends this year?", "What is my portfolio value?", "Show my holdings"
     - Aggregate or descriptive portfolio queries: "How many portfolios do we have?", "What is the total investment across all portfolios?", "Which portfolios have the highest returns?", "What is the total return of each portfolio since inception?", "How does portfolio performance compare across different groups?"  
   - **All questions about portfolio statistics, returns, holdings, trends, profit/loss, allocations, YTD, MTD,benchmarks, or comparisons should be labeled as portfolio**, even if not specific to a single personal account.
   -**Quantity, Position, Lot everyword means portfolio
     - Any Questions related for trading related actions like 'Can you trade bitcoin for me?'etc., should be categorized as **other**
   This includes **performance metrics** (like returns, profit, value) and **descriptive details** (like default index, start date, description, cost model
   If any questions directly related to stock like 'bottom 15 loss stocks', 'top 15 profit stocks' should be labelled as portfolio**
2.  **general**: For questions requiring a live web search for public, real-time data like current stock prices or latest news.
    **Includes**
    -If I had invested AED 10,000 in NVIDIA in January, how much would it be worth now
    -CAGR calculation
    - Returns Calculation
3. **other**: This category is for any query that has **NO specific financial data request**. Respond to these queries as a friendly bot appropriately. 
    - **Includes:**
        - **Small Talk:** "Hi", "Tell me a joke", "How are you?"
        - **Identity:** "Who are you?", "Can you act as my financial advisor?"
        - **Out of Context:** "What is the weather today?", "Medical diagnosis", "Legal counsel".
        - **Gibberish:** "asdfkj!", random characters.
        - **Offensive:** Hate speech, threats, illegal activities.



---
**TIE-BREAKER LOGIC (CRITICAL)**
-   If a query mixes conversation with a financial request, classify it by the financial request.
-   **When uncertain, you MUST prefer `portfolio`** if the question mentions portfolio-related keywords.

---
**EXAMPLES**
User: "Which portfolios have the highest returns"
Category: portfolio

User: "what is the current stock price of microsoft?"
Category: general

User: "What is my total portfolio value?"
Category: portfolio

User: "Hi"
Category: other

User: "How are you?"
Category: other

User: "Which portfolios have the highest returns"
Category: portfolio

User: "what is the current stock price of microsoft?"
Category: general

User: "What is my total portfolio value?"
Category: portfolio

# --- NEW EXAMPLE ADDED HERE ---
User: "What is the default index for each portfolio"
Category: portfolio
User: "list all my portfolios"
Category: portfolio

---
**YOUR TASK**
Now, classify the following user question. Respond with ONLY the single category name.

User: "{question}"
Category:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(question=question)}
    ]
    
    try:
        #response_text = call_ollama_chat(messages, temperature=0.01, max_tokens=10)
        response_text = ollama.chat(
            model="gpt-oss:20b",
            messages=messages,
            options={"num_predict": 2000, "temperature": 0.5}
        )
        print("sample response")
        print(response_text)

        raw_output = response_text.message.content   # Access as attribute, not dict key
        
        cleaned = raw_output.strip().lower().strip()

        print("*************cleaned****************")
        print(cleaned)

        if "portfolio" in cleaned:
            return "portfolio"
        if "general" in cleaned:
            return "general"
        if "other" in cleaned:
            return "other"
        
        # CRITICAL FIX: The fallback is 'other' to prevent incorrect web searches for greetings/failures.
        print(f"Warning: Unrecognized classification output: '{response_text}'. Defaulting to 'other'.")
        return "other"
        
    except Exception as e:
        st.error(f"Error during classification: {e}. Defaulting to 'other'.")
        traceback.print_exc()
        return "other"


perplexity_prompt_content_1 = (
    "You are an experienced equity fund manager, highly knowledgeable in stock, equity, and finance. "
    "Answer only finance related questions. For other topics , respond **I'm Sorry, I'm a Financial Bot. I can only help you with financial related questions**"
    "Your primary goal is to assist non-financial stakeholders in understanding financial data and market insights. "
    "When a user mentions a common noun that can also be a company name (e.g., 'Apple', 'Amazon', 'Tesla'), "
    "always interpret it as the respective stock or company, not the general noun. "
    "Answer the question concisely and factually, explaining the response in plain, simple English that can be easily understood by stakeholders. "
    "If the response includes specific financial data or symbols (e.g., stock tickers, price points, growth rates, index names, specific years, parenthetical values), "
    "integrate them seamlessly into your explanation rather than just listing them. "
    "For instance, if you have '(AAPL) 211.07 232.27 2025', explain it as 'Apple's stock price moved from $211.07 to $232.27 by the year 2025.' "
    "If you encounter a term like 'S&P 500', use it as is, as it's a globally recognized index and does not require translation or additional explanation about its non-translation. "
    "Do not include citations, footnotes, or any separate 'Note' sections in your final answer. "
    "Only provide the final answer, without showing intermediate steps or formulas. "
    "Avoid LaTeX or complex calculations unless specifically asked for a detailed numerical breakdown. "
    "Be precise and concise."
    "If a source is available, include it at the very end. Otherwise, do not mention it."
    "In case of summary of the investment portfolio related questions, Please provide a concise and factual summary of this investment portfolio for non-financial stakeholders with focus on key indicators like portfolio name, total net liquidity, YTD profit, All-Time Return (percentage), All-Time Profit (absolute amount),Any associated Group Name or Benchmark Index. Integrate these specific data points seamlessly into your explanation. Explain what each metric means in simple terms. Investment summary should be overall and not personalized or user specific."
    "User Question can be in English or Arabic. Generate Answer in Respective Language as per input."
    "Display answers with proper alignment and bullet points wherever required."
)

perplexity_prompt_content_2 = (
    "You are an experienced equity fund manager, highly knowledgeable in stock, equity, and finance. "
    "You are a bilingual financial assistant proficient in both English and Arabic. "
    "When the user's question is in Arabic, respond entirely in Arabic. "
    "When in English, respond in English. "
    "Answer only finance related questions. For other topics , respond **I'm Sorry, I'm a Financial Bot. I can only help you with financial related questions**"
    "Your primary goal is to assist non-financial stakeholders in understanding financial data and market insights. "
    "When a user mentions a common noun that can also be a company name (e.g., 'Apple', 'Amazon', 'Tesla'), "
    "always interpret it as the respective stock or company, not the general noun. "
    "Answer the question concisely and factually, explaining the response in plain, simple English that can be easily understood by stakeholders. "
    "If the response includes specific financial data or symbols (e.g., stock tickers, price points, growth rates, index names, specific years, parenthetical values), "
    "integrate them seamlessly into your explanation rather than just listing them. "
    "For instance, if you have '(AAPL) 211.07 232.27 2025', explain it as 'Apple's stock price moved from $211.07 to $232.27 by the year 2025.' "
    "If you encounter a term like 'S&P 500', use it as is, as it's a globally recognized index and does not require translation or additional explanation about its non-translation. "
    "Do not include citations, footnotes, or any separate 'Note' sections in your final answer. "
    "Only provide the final answer, without showing intermediate steps or formulas. "
    "Avoid LaTeX or complex calculations unless specifically asked for a detailed numerical breakdown. "
    "Be precise and concise."
    "If a source is available, include it at the very end. Otherwise, do not mention it."
    "In case of summary of the investment portfolio related questions, Please provide a concise and factual summary of this investment portfolio for non-financial stakeholders with focus on key indicators like portfolio name, total net liquidity, YTD profit, All-Time Return (percentage), All-Time Profit (absolute amount),Any associated Group Name or Benchmark Index. Integrate these specific data points seamlessly into your explanation. Explain what each metric means in simple terms. Investment summary should be overall and not personalized or user specific."
    "User Question can be in English or Arabic. Generate Answer in Respective Language as per input."
    "Display answers with proper alignment and bullet points wherever required."
    "Ensure there is always a space between words and numbers. "
    "Avoid using markdown formatting such as italics, bold, or inline code. "
    "Do not concatenate numbers and words without spacing (e.g., write '200 in January', not '200inJanuary'). "
    "Use plain text with proper spacing and punctuation throughout."
)

perplexity_prompt_content = (
    "You are an experienced equity fund manager, highly knowledgeable in stock, equity, and finance. "
    "You are a bilingual financial assistant proficient in both English and Arabic. "
    "When the user's question is in Arabic, respond entirely in Arabic. "
    "When in English, respond in English. "
    "Answer only finance related questions. For other topics , respond **I'm Sorry, I'm a Financial Bot. I can only help you with financial related questions**"
   
    "Only provide the final answer, without showing intermediate steps or formulas. "
  
    "Be precise and concise."
   
    "In case of summary of the investment portfolio related questions, Please provide a concise and factual summary of this investment portfolio for non-financial stakeholders with focus on key indicators like portfolio name, total net liquidity, YTD profit, All-Time Return (percentage), All-Time Profit (absolute amount),Any associated Group Name or Benchmark Index. Integrate these specific data points seamlessly into your explanation. Explain what each metric means in simple terms. Investment summary should be overall and not personalized or user specific."
    "User Question can be in English or Arabic. Generate Answer in Respective Language as per input."
    "Display answers with proper alignment and bullet points wherever required."
    "Ensure there is always a space between words and numbers. "
    "Avoid using markdown formatting such as italics, bold, or inline code. "
    "Do not concatenate numbers and words without spacing (e.g., write '200 in January', not '200inJanuary'). "
    "Use plain text with proper spacing and punctuation throughout."
)



# --- UPDATED REPHRASING FUNCTION ---
def create_standalone_question(question: str, chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return question

    # --- SLIGHTLY UPDATED PROMPT ---
    system_prompt = "You are an expert at rephrasing questions. Given a chat history and a follow-up question, rewrite the follow-up question to be a complete, standalone question. **Make the new question descriptive and clear, as if for a database query.** For example, rephrase 'list them' to 'list the names of all portfolios'."

    history_str = ""
    for message in chat_history[-4:]:
        history_str += f"{message['role'].capitalize()}: {message['content']}\n"
    user_prompt = f"""Here is the chat history:\n---\n{history_str}---\nNow, here is the follow-up question: "{question}"\n\nPlease rewrite this follow-up question as a complete, descriptive, standalone question.\nStandalone Question:"""
    
    # (The rest of the function remains the same)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    try:
        standalone_question = call_ollama_chat(messages, temperature=0.0, max_tokens=100)
        return standalone_question
    except Exception as e:
        print(f"Error creating standalone question: {e}. Defaulting to original question.")
        return question

def query_perplexity_1(query: str, chat_history: List[Dict[str, str]]) -> str:
    print("Input Query for Perplexity:", query)
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    # Filter chat history to include only roles allowed by Perplexity.
    #filtered_dialogue_history = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]]
    filtered_dialogue_history = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]]

    # Add the new user question as the most recent message
    max_history_dialogue_messages = 7  # Leave room for new query
    recent_dialogue = filtered_dialogue_history[-max_history_dialogue_messages:]
    #filtered_dialogue_history.append({"role": "user", "content": query})

    recent_dialogue.append({"role": "user", "content": query})

    # Keep only the most recent 8 user/assistant messages
  

    # 🧩 Ensure alternation between user and assistant
   # Ensure alternation between user and assistant messages
    cleaned_dialogue = []
    last_role = None
    for msg in recent_dialogue:
        if msg["role"] == last_role:
            print(f"⚠️ Skipping consecutive {msg['role']} message for alternation rule.")
            continue
        cleaned_dialogue.append(msg)
        last_role = msg["role"]

    # 🧠 Construct messages list with system + cleaned dialogue
    perplexity_messages = [{"role": "system", "content": perplexity_prompt_content}] + cleaned_dialogue

    print("\n--- Perplexity Messages Constructed ---")
    for msg in perplexity_messages:
        print(f"{msg['role'].upper()}: {msg['content']}")
    print("-------------------------------------")

    payload = {
        "model": "sonar",  # or "sonar-large-online" if you need web access
        "messages": perplexity_messages,
        "temperature": 0.3
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        resp_json = response.json()

        # Extract answer
        answer = resp_json['choices'][0]['message']['content']
        answer = re.sub(r'\[\d+\]', '', answer)
        answer = re.sub(r'[^\x00-\x7F]+', '', answer)
        answer = re.sub(r'\n+', '\n', answer).strip()

        # Extract sources if available
        source_links = []
        if 'sources' in resp_json:
            for source in resp_json['sources']:
                if isinstance(source, dict) and 'url' in source:
                    source_links.append(source['url'])
        elif 'text_outputs' in resp_json and 'url_citations' in resp_json['text_outputs']:
            for citation in resp_json['text_outputs']['url_citations']:
                source_links.append(citation['url'])

        if source_links:
            answer += "\n\n**Source:** " + " ".join(source_links)

        return answer

    except requests.HTTPError as e:
        print(f"HTTP error: {e} | Response: {response.text}")
        return f"Error: {response.text}"
    except Exception as ex:
        print(f"General error: {ex}")
        return f"Error: {str(ex)}"

def clean_model_text(text: str) -> str:
    """
    Cleans and normalizes text output from an LLM to fix common formatting issues
    like special Unicode characters and spacing errors.
    """
    # 1. CRITICAL FIX: Normalize compatibility characters.
    # This turns '𝑎' into 'a', '𝑠' into 's', etc., fixing the main issue.
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix common run-on words that occur after normalization.
    # This turns 'asoftheclose' into 'as of the close'.
    # We use regex to be more precise.
    spacing_fixes = {
        r'asof': 'as of',
        r'theclose': 'the close',
        r'oftrading': 'of trading',
        r'onNovember': 'on November',
        r'\.Thestock': '. The stock', # Add space after a period
        r'isdown': 'is down',
    }
    for pattern, replacement in spacing_fixes.items():
        # Using a function in re.sub to handle case-insensitivity gracefully
        def repl(match):
            # If the original match was capitalized (e.g., 'Thestock'), keep the replacement capitalized
            if match.group(0)[0].isupper():
                return replacement.capitalize()
            return replacement
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 3. Collapse any leftover spaced-out characters (e.g., "a s o f" that might have slipped through)
    # This finds sequences of single letters followed by a space and joins them.
    def collapse_spaced_chars(match):
        return re.sub(r'\s+', '', match.group(0))
    text = re.sub(r'((?:\b\w\b\s){2,})', collapse_spaced_chars, text)

    # 4. General whitespace cleanup
    text = re.sub(r' +', ' ', text) # Collapse multiple spaces into one
    text = re.sub(r'\n{3,}', '\n\n', text) # Collapse multiple newlines
    
    return text.strip()  

def is_text_broken(text: str) -> bool:
    """
    Detects if text is likely broken:
    - Has letters separated by spaces (like '𝑤 𝑖 𝑡 ℎ')
    - Contains unusual Unicode characters outside normal ASCII
    """
    # Check for letters separated by spaces
    if re.search(r"(?<=\w)\s+(?=\w)", text):
        return True
    # Check for uncommon Unicode letters
    for c in text:
        if ord(c) > 127 and c.isalpha():
            return True
    return False

def clean_text_if_broken(text: str) -> str:
    """
    Cleans text only if it's broken. Otherwise returns original text.
    """
    if not is_text_broken(text):
        return text  # skip cleaning
    
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove Markdown formatting
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)       # Bold
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)          # Italics
    text = re.sub(r"`(.*?)`", r"\1", text)                # Inline code
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text) # Links

    # Remove weird spacing between letters
    text = re.sub(r"(?<=\w)\s+(?=\w)", "", text)

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text).strip()

    return text

def query_perplexity_1(query: str, chat_history: List[Dict[str, str]]) -> str:
    print("Input Query for Perplexity:", query)
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    filtered_dialogue_history = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]]
    recent_dialogue = filtered_dialogue_history[-7:] + [{"role": "user", "content": query}]

    # Ensure alternation between user and assistant
    cleaned_dialogue, last_role = [], None
    for msg in recent_dialogue:
        if msg["role"] == last_role:
            continue
        cleaned_dialogue.append(msg)
        last_role = msg["role"]

    perplexity_messages = [{"role": "system", "content": perplexity_prompt_content}] + cleaned_dialogue

    payload = {
        "model": "sonar",  # better performance
        "messages": perplexity_messages,
        "temperature": 0.4,
        "max_tokens": 1024
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        resp_json = response.json()

        answer = resp_json['choices'][0]['message']['content']
        answer = unicodedata.normalize("NFKC", answer)
        answer = re.sub(r'\[\d+\]', '', answer)
        answer = re.sub(r'\n{2,}', '\n\n', answer).strip()

        # Extract source links
        source_links = []
        if 'sources' in resp_json:
            source_links = [s['url'] for s in resp_json['sources'] if isinstance(s, dict) and 'url' in s]
        elif 'text_outputs' in resp_json and 'url_citations' in resp_json['text_outputs']:
            source_links = [c['url'] for c in resp_json['text_outputs']['url_citations']]

        if source_links:
            answer += "\n\n**Source:** " + " ".join(source_links)

        return answer

    except requests.HTTPError as e:
        print(f"HTTP error: {e} | Response: {response.text}")
        return f"Error: {response.text}"
    except Exception as ex:
        print(f"General error: {ex}")
        return f"Error: {str(ex)}"


# --- Replace your existing query_perplexity function with this robust version ---
def clean_model_text_openai(text: str) -> str:
    # Basic cleanup for stray markdown, excessive whitespace, etc.
    text = text.replace('*', '')  # remove markdown italics/bold
    text = re.sub(r'\s+', ' ', text).strip()
    return text

OPENAI_API_KEY=""

openai_text_prompt=(
    "You are an experienced equity fund manager, highly knowledgeable in stock, equity, and finance. "
    "You are a bilingual financial assistant proficient in both English and Arabic. "
    "When the user's question is in Arabic, respond entirely in Arabic. "
    "When in English, respond in English. "
    "Answer only finance related questions. For other topics , respond **I'm Sorry, I'm a Financial Bot. I can only help you with financial related questions**"
    "Your primary goal is to assist non-financial stakeholders in understanding financial data and market insights. "
    "When a user mentions a common noun that can also be a company name (e.g., 'Apple', 'Amazon', 'Tesla'), "
    "always interpret it as the respective stock or company, not the general noun. "
    "Answer the question concisely and factually, explaining the response in plain, simple English that can be easily understood by stakeholders. "
    
    "Do not include citations, footnotes, or any separate 'Note' sections in your final answer. "
    "Use Internet or external knowledge to answer question like real time stock price"
    "Display answers with proper alignment and bullet points wherever required."
  
    "Ensure there is always a space between words and numbers. "
    "Avoid using markdown formatting such as italics, bold, or inline code. "
    "Do not concatenate numbers and words without spacing (e.g., write '200 in January', not '200inJanuary'). "
    "Use plain text with proper spacing and punctuation throughout."

)

simple_prompt=(
    "You are an experienced equity fund manager, highly knowledgeable in stock, equity, and finance. "
    "You are a bilingual financial assistant proficient in both English and Arabic. "
    "When the user's question is in Arabic, respond entirely in Arabic. "
    "When in English, respond in English. "

      "Ensure there is always a space between words and numbers. "
    "Avoid using markdown formatting such as italics, bold, or inline code. "
    "Do not concatenate numbers and words without spacing (e.g., write '200 in January', not '200inJanuary'). "
    "Use plain text with proper spacing and punctuation throughout."


)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
def query_openai(query: str, chat_history: List[Dict[str, str]]) -> str:
    print("Input Query for OpenAI:", query)

    # 1. Start with recent history (keep last 6 turns)
    full_history = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]][-6:]
    full_history.append({"role": "user", "content": query})

    # 2. Enforce strict alternation
    cleaned_dialogue = []
    last_role = None
    for msg in reversed(full_history):
        if msg["role"] == last_role:
            continue
        cleaned_dialogue.append(msg)
        last_role = msg["role"]
    cleaned_dialogue.reverse()

    # 3. Construct final message list
    openai_messages = [{"role": "system", "content": simple_prompt}] + cleaned_dialogue

    print("\n--- Final Messages Sent to OpenAI ---")
    for msg in openai_messages:
        print(f"- Role: {msg['role']}")
    print("-------------------------------------\n")

    try:
        # --- API call ---
        response = openai_client.chat.completions.create(
            model="gpt-4o",   # You can switch to "gpt-4o" or "gpt-3.5-turbo" if preferred
            messages=openai_messages,
            temperature=0.3,
        )

        # Extract and clean response
        answer = response.choices[0].message.content
        #answer_text = response.choices[0].message.content.strip()
        #answer = clean_model_text_openai(answer)

        # Optionally extract any URLs if OpenAI model outputs them
        urls = re.findall(r'https?://\S+', answer)
        if urls:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {u}" for u in urls)

        return answer

    except Exception as ex:
        print(f"Error during OpenAI API call: {ex}")
        traceback.print_exc()
        return f"Error: {str(ex)}"


def query_perplexity(query: str, chat_history: List[Dict[str, str]]) -> str:
    print("Input Query for Perplexity:", query)
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    # 1. Start with the full recent history, including the new query
    # We take the last 6 messages from history + the new one, making a total of 7 turns.
    full_history = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]][-6:]
    full_history.append({"role": "user", "content": query})

    # 2. --- CRITICAL FIX: Enforce strict alternation ---
    cleaned_dialogue = []
    last_role = None
    # Iterate through the history in reverse to keep the most recent messages
    for msg in reversed(full_history):
        # If the current message has the same role as the last one we added, skip it
        if msg['role'] == last_role:
            continue
        # Otherwise, add it to our new list and update the last_role
        cleaned_dialogue.append(msg)
        last_role = msg['role']
    
    # Reverse the list back to chronological order
    cleaned_dialogue.reverse()

    # 3. Construct the final messages list for the API
    perplexity_messages = [{"role": "system", "content": perplexity_prompt_content}] + cleaned_dialogue

    # --- Debugging: Print the final message structure ---
    print("\n--- Final Perplexity Messages Sent to API ---")
    for msg in perplexity_messages:
        print(f"- Role: {msg['role']}")
    print("-------------------------------------------\n")


    payload = {
        "model": "sonar",
        "messages": perplexity_messages,
        "temperature": 0.3,
    }

    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        resp_json = response.json()

        answer = resp_json['choices'][0]['message']['content']
        answer = clean_model_text(answer) # Assuming clean_model_text is defined
        #answer=clean_text_if_broken(answer)

        #answer = re.sub(r'\[\d+\]', '', answer)
        
        source_links = []
        if 'sources' in resp_json:
            source_links = [s.get('url') for s in resp_json['sources'] if s.get('url')]
        
        if source_links:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {link}" for link in source_links)

        return answer

    except requests.HTTPError as e:
        # Provide a more user-friendly error from the API response if possible
        error_detail = e.response.json().get('error', {}).get('message', e.response.text)
        print(f"HTTP error: {e} | Response: {error_detail}")
        return f"Error from Perplexity API: {error_detail}"
    except Exception as ex:
        print(f"General error: {ex}")
        traceback.print_exc()
        return f"Error: {str(ex)}"

import requests
import traceback
from typing import List, Dict

def query_perplexity_final(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Sends a query to the Perplexity API with cleaned chat history ensuring strict
    alternation between user and assistant messages to prevent errors.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    # 1️⃣ Keep only the last few messages to limit payload size
    recent_history = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]][-15:]
    
    # 2️⃣ Append the new user query
    recent_history.append({"role": "user", "content": query})

    # 3️⃣ Enforce strict chronological alternation
    cleaned_dialogue = []
    last_role = "system"  # system role precedes user/assistant
    for msg in recent_history:
        if msg["role"] == last_role:
            # Skip this message to enforce alternation
            continue
        cleaned_dialogue.append(msg)
        last_role = msg["role"]

    # 4️⃣ Construct final messages payload
    perplexity_messages = [{"role": "system", "content": perplexity_prompt_content}] + cleaned_dialogue

    # Optional: final safety check to remove any consecutive same-role messages
    final_messages = [perplexity_messages[0]]  # start with system message
    for msg in perplexity_messages[1:]:
        if msg["role"] != final_messages[-1]["role"]:
            final_messages.append(msg)
    
    # Debug: print final structure
    print("\n--- Final Perplexity Messages ---")
    for msg in final_messages:
        print(f"- Role: {msg['role']}, Content preview: {msg['content'][:60]}")
    print("--------------------------------\n")

    payload = {
        "model": "sonar",
        "messages": final_messages,
        "temperature": 0.3,
    }

    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        resp_json = response.json()

        answer = resp_json['choices'][0]['message']['content']
        answer = clean_model_text(answer)  # clean formatting issues

        # Include sources if provided
        source_links = []
        if 'sources' in resp_json:
            source_links = [s.get('url') for s in resp_json['sources'] if s.get('url')]
        
        if source_links:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {link}" for link in source_links)

        return answer

    except requests.HTTPError as e:
        error_detail = e.response.json().get('error', {}).get('message', e.response.text)
        print(f"HTTP error: {e} | Response: {error_detail}")
        return f"Error from Perplexity API: {error_detail}"
    except Exception as ex:
        print(f"General error: {ex}")
        traceback.print_exc()
        return f"Error: {str(ex)}"


def translate_1(text: str, target_language: str) -> str:
    # (Omitted for brevity, same as before)
    if not text: return ""
    prompt = f"Translate the following text to {target_language}..."
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
    try:
        return call_ollama_chat(messages, temperature=0.0)
    except Exception as e:
        return text

# --- Find and REPLACE your entire translate function ---

def translate(text: str, target_language: str) -> str:
    """
    Translates text to the target language using a direct, non-conversational prompt.
    """
    if not text: 
        return ""
        
    # --- START: IMPROVED, DIRECT PROMPT ---
    # This prompt is designed to prevent the LLM from refusing to translate.
    # It avoids conversational language and gives a direct command.
    prompt = f"""You are a machine translation service. Your only function is to translate the user's text to the target language.
    
    Target Language: {target_language}

   
    
    Translate the following text. Respond with ONLY the translated text, nothing else.
    """
    # --- END: IMPROVED, DIRECT PROMPT ---

    messages = [
        {"role": "system", "content": prompt}, 
        {"role": "user", "content": text}
    ]
    
    try:
        # We use a low temperature for translation to make it more deterministic
        return call_ollama_chat(messages, temperature=0.05)
    except Exception as e:
        print(f"Error during translation: {e}")
        # Fallback to the original text if translation fails
        return text

# --- 4. Initialization and Vanna Setup ---
try:
    ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}], stream=False)
    st.sidebar.success(f"Connected to Ollama ({OLLAMA_MODEL})")
except Exception as e:
    st.sidebar.error(f"Failed to connect to Ollama: {e}")
    st.stop()

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None: config = {}
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
vn = MyVanna(config=config)

DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = "40.172.107.136", "AI-Trading-QNA", "aitradinguser", "UpIEvFTyAp51", 31937
try:
    vn.connect_to_postgres(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    st.sidebar.success("Connected to Vanna database.")
except Exception as e:
    st.sidebar.error(f"Vanna DB connection failed: {e}.")
    vn = None

# --- Vanna Documentation (unchanged) ---
# if vn is not None:
#     vn.add_documentation(
#         table_name='portfolio_default_prompt',
#         schema_name='ai_trading',
#         documentation='This is a utility table that stores a list of predefined, default questions or prompts. These can be displayed on a user interface as suggestions for what the user can ask.'
#     )
#     # --- Columns for portfolio_default_prompt ---
#     vn.add_documentation(
#         table_name='portfolio_default_prompt', schema_name='ai_trading', column_name='id',
#         documentation='A unique numerical identifier for each prompt. Data Type: integer.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_default_prompt', schema_name='ai_trading', column_name='prompt_text',
#         documentation="The actual text of the suggested question (e.g., 'What are my top 10 holdings?'). Data Type: character varying."
#     )
#     vn.add_documentation(
#         table_name='portfolio_default_prompt', schema_name='ai_trading', column_name='created_at',
#         documentation='The timestamp when the prompt was first created. Data Type: timestamp without time zone.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_default_prompt', schema_name='ai_trading', column_name='updated_at',
#         documentation='The timestamp when the prompt was last modified. Data Type: timestamp without time zone.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_default_prompt', schema_name='ai_trading', column_name='order',
#         documentation='A number used to control the display order of the prompts on the screen. Data Type: integer.'
#     )

#     # ==============================================================================
#     # == Table: portfolio_holdings
#     # ==============================================================================
#     vn.add_documentation(
#         table_name='portfolio_holdings',
#         schema_name='ai_trading',
#         documentation="This table contains snapshot data of individual asset holdings within each portfolio at specific points in time. It is the primary source for analyzing the historical trend of a specific stock's market value or position size within a portfolio."
#     )
#     # --- Columns for portfolio_holdings ---
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='datetime',
#         documentation='The exact date and time when this snapshot of holding data was recorded. Essential for any time-series or historical analysis. Data Type: timestamp without time zone.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='portfolio_name',
#         documentation="The name of the investment portfolio holding the asset. In SQL WHERE clauses, always enclose values in single quotes (e.g., `portfolio_name = 'My Portfolio'`). Data Type: character varying."
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='symbol',
#         documentation="The stock ticker symbol for the asset (e.g., 'AAPL', 'MSFT'). In SQL WHERE clauses, always enclose values in single quotes. Data Type: character varying."
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='positions',
#         documentation='The quantity or number of shares held for that symbol. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='market_value',
#         documentation='The total monetary market value of the positions held at that specific `datetime`. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='ytd_unrealized_pl',
#         documentation='The Year-To-Date unrealized ("on paper") profit or loss for this specific holding. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='created_timestamp',
#         documentation='The timestamp when this record was first created in the database. Data Type: timestamp without time zone.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='updated_timestamp',
#         documentation='The timestamp of the last update to this record. Data Type: timestamp without time zone.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_holdings', schema_name='ai_trading', column_name='last_updated_time',
#         documentation='A more granular timestamp indicating the last modification time. Data Type: timestamp without time zone.'
#     )

    # ==============================================================================
    # == Table: portfolio_holdings_realized_pnl
    # ==============================================================================
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl',
    #     schema_name='ai_trading',
    #     documentation='This is the most detailed table for profit and loss (P&L) analysis. It contains comprehensive data on both realized (from sold assets) and unrealized (from current holdings) profits and losses. Use this table for any question about specific profit figures.'
    # )
    # # --- Columns for portfolio_holdings_realized_pnl ---
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='datetime',
    #     documentation='The date for which the P&L data is recorded. Data Type: timestamp without time zone.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='group_name',
    #     documentation="A high-level category or group that the portfolio belongs to (e.g., 'Equity', 'Growth'). Data Type: character varying."
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='portfolio_name',
    #     documentation='The name of the specific investment portfolio. Data Type: character varying.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='symbol',
    #     documentation='The stock ticker symbol of the investment. Data Type: character varying.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='positions',
    #     documentation='The number of shares held. Data Type: numeric.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='market_value',
    #     documentation='The total market value of the holding. Data Type: numeric.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='ytd_unrealized_pnl',
    #     documentation='Year-To-Date unrealized (paper) profit or loss. Data Type: numeric.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='ytd_realized_pnl',
    #     documentation='Year-To-Date realized profit or loss from positions that have been sold. This is actual, locked-in profit/loss. Data Type: numeric.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='ytd_total_pnl',
    #     documentation='The sum of both realized and unrealized P&L for the year. This is the best metric for the overall profitability of a holding. Data Type: numeric.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='created_timestamp',
    #     documentation='The timestamp when this P&L record was created. Data Type: timestamp without time zone.'
    # )
    # vn.add_documentation(
    #     table_name='portfolio_holdings_realized_pnl', schema_name='ai_trading', column_name='daily_realized_pnl',
    #     documentation='The profit or loss that was realized on this specific day. Data Type: numeric.'
    # )

    # ==============================================================================
    # == Table: portfolio_summary
    # ==============================================================================
#     vn.add_documentation(
#         table_name='portfolio_summary',
#         schema_name='ai_trading',
#         documentation='This is an aggregated summary table that provides high-level performance metrics for each portfolio. It is the best source for questions about overall portfolio value, returns, and performance across different timeframes (YTD, QTD, MTD, etc.).'
#     )
#     # --- Columns for portfolio_summary ---
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='datetime',
#         documentation='The date for which the summary is calculated. Data Type: timestamp without time zone.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='portfolio_name',
#         documentation='The unique name of the investment portfolio. Data Type: character varying.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='account_id',
#         documentation='The identifier for the trading account associated with the portfolio. Data Type: character varying.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='user_id',
#         documentation='The identifier for the user associated with the portfolio. Data Type: character varying.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='default_index',
#         documentation="The benchmark index (e.g., 'S&P 500') used for performance comparison. Data Type: character varying."
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='portfolio_startdate',
#         documentation='The date when the portfolio was created (its inception date). Data Type: date.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='net_liquidity',
#         documentation='The total current value of the portfolio, including cash and all holdings. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='unrealized_pl',
#         documentation='The total unrealized profit or loss for the entire portfolio. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='last_updated_time',
#         documentation='The precise timestamp of the last summary update. Data Type: timestamp without time zone.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='ytd_return',
#         documentation='The Year-To-Date return of the portfolio as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='allocated_amount',
#         documentation='The total amount of capital allocated to this portfolio. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='index_annual_return',
#         documentation='The annual return of the benchmark index as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='portfolio_description',
#         documentation="A text description of the portfolio's strategy. Data Type: character varying."
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='utilized_amount',
#         documentation='The amount of the allocated capital that is currently invested. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='group_name',
#         documentation="A category the portfolio belongs to (e.g., 'Balanced', 'Aggressive'). Data Type: character varying."
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='cost_model',
#         documentation="The accounting method used for cost basis (e.g., 'FIFO'). Data Type: character varying."
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='ytd_profit',
#         documentation='The Year-To-Date profit of the portfolio in absolute monetary terms. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='all_profit',
#         documentation='The total profit of the portfolio since its inception date. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='qtd_return',
#         documentation='The Quarter-To-Date return as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='qtd_profit',
#         documentation='The Quarter-To-Date profit in absolute monetary terms. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='mtd_return',
#         documentation='The Month-To-Date return as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='mtd_profit',
#         documentation='The Month-To-Date profit in absolute monetary terms. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='wtd_return',
#         documentation='The Week-To-Date return as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='wtd_profit',
#         documentation='The Week-To-Date profit in absolute monetary terms. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='ytd_index_return',
#         documentation='The YTD return of the benchmark index as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='qtd_index_return',
#         documentation='The QTD return of the benchmark index as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='mtd_index_return',
#         documentation='The MTD return of the benchmark index as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='wtd_index_return',
#         documentation='The WTD return of the benchmark index as a percentage. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='all_index_return',
#         documentation="The total return of the benchmark index since the portfolio's inception. Data Type: numeric."
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='all_return',
#         documentation='The total return of the portfolio since its inception. Data Type: numeric.'
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='daily_return',
#         documentation="The portfolio's return for a single day as a percentage. Data Type: numeric."
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='daily_profit',
#         documentation="The portfolio's profit for a single day in absolute monetary terms. Data Type: numeric."
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary', 
#     schema_name='ai_trading', 
#     column_name='is_active',
#     documentation="""A numeric flag indicating if the portfolio is active. 
#     Data Type: numeric. 
#     **IMPORTANT**: In all SQL WHERE clauses, you MUST use `is_active = 1` to filter for active portfolios and `is_active = 0` for inactive ones. 
#     DO NOT use boolean values like TRUE or FALSE. For example, to find active portfolios, the correct syntax is `WHERE is_active = 1`."""
#     )
#     vn.add_documentation(
#         table_name='portfolio_summary', schema_name='ai_trading', column_name='daily_index_return',
#         documentation="The daily return of the benchmark index as a percentage. Data Type: numeric."
#     )


#     # Direct mapping for the Arabic question
#     vn.add_question_sql(
#     question="ما هو ربح السنة حتى اليوم لكل محفظة", # "What is the YTD profit for each portfolio?"
#     sql="""
#     SELECT portfolio_name, ytd_profit 
#     FROM ai_trading.portfolio_summary 
#     WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary) 
#     ORDER BY ytd_profit DESC;
#     """
#     )

# # English variations to make the training more robust
#     vn.add_question_sql(
#     question="What is the Year-To-Date profit for each portfolio?",
#     sql="""
#     SELECT portfolio_name, ytd_profit 
#     FROM ai_trading.portfolio_summary 
#     WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary) 
#     ORDER BY ytd_profit DESC;
#     """
#     )

#     vn.add_question_sql(
#     question="Show me the profit since the start of the year for every wallet.",
#     sql="""
#     SELECT portfolio_name, ytd_profit 
#     FROM ai_trading.portfolio_summary 
#     WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary) 
#     ORDER BY ytd_profit DESC;
#     """
#     )

# # Also, let's reinforce the documentation for ytd_profit to help the LLM
#     vn.add_documentation(
#     table_name='portfolio_summary', 
#     schema_name='ai_trading', 
#     column_name='ytd_profit',
#     documentation="""The Year-To-Date monetary profit (P&L) of the portfolio. This column directly answers questions about 'profit since the start of the year' or 'YTD profit'. Data Type: numeric."""
#     )
#     st.sidebar.success("Vanna training complete.")

    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # documentation='This table tracks the individual holdings within various investment portfolios. It contains data on the quantity of assets, their market value, and year-to-date unrealized profit or loss. It is crucial for understanding specific investments and portfolio performance.'
    # )

    # Column-level documentation
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='datetime',
    # documentation='The date and time when the portfolio holding data was recorded. Useful for historical analysis.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='portfolio_name',
    # documentation='The name or identifier of the investment portfolio. Used to group individual holdings. Queries about portfolios should reference this column.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='symbol',
    # documentation='The stock ticker symbol or identifier for the specific investment (e.g., AAPL, MSFT). Queries about individual investments or assets should use this column.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='positions',
    # documentation='The number of shares or units held for a particular investment within the portfolio.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='market_value',
    # documentation='The current total market value of the positions held for this investment. This represents the current worth of the investment.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='ytd_unrealized_pl',
    # documentation='The Year-To-Date (YTD) unrealized Profit or Loss for the holding. This reflects the gain or loss on paper that has not yet been realized through a sale. This column is key for understanding investment returns and profits before they are locked in. Often referred to as "return" or "profit" in the context of current holdings.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='created_timestamp',
    # documentation='The timestamp when this record was initially created.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='updated_timestamp',
    # documentation='The timestamp when this record was last updated.'
    # )
    # vn.add_documentation(
    # table_name='portfolio_holdings',
    # schema_name='ai_trading',
    # column_name='last_updated_time',
    # documentation='A detailed timestamp indicating the very last time the holding data was modified. Use this for freshness checks.'
    # )



# --- Add Question-SQL Pairs ---

# Questions about "best investment" / "returns" / "profit" based on ytd_unrealized_pl
#     vn.add_question_sql(
#     question='What is the best investment that gives the best return?',
#     sql="SELECT symbol, ytd_unrealized_pl FROM ai_trading.portfolio_holdings ORDER BY ytd_unrealized_pl DESC LIMIT 1;"
#     )

#     vn.add_question_sql(
#     question='Show me my realized gains this month?',
#     sql="SELECT SUM(daily_realized_pnl) AS realized_gains_this_month FROM ai_trading.portfolio_holdings_realized_pnl WHERE datetime >= date_trunc('month', current_date) AND datetime < date_trunc('month', current_date) + interval '1 month';"
#     )

#     vn.add_question_sql(
#     question='Show me my unrealized gains this month?',
#     sql="SELECT portfolio_name, SUM(ytd_unrealized_pl) AS unrealized_gains_this_month FROM ai_trading.portfolio_holdings WHERE datetime >= date_trunc('month', current_date) AND datetime < date_trunc('month', current_date) + interval '1 month' GROUP BY portfolio_name ORDER BY unrealized_gains_this_month DESC"
#     )

#     vn.add_question_sql(
#     question='Which investment has the highest profit this year?',
#     sql="SELECT symbol, ytd_unrealized_pl FROM ai_trading.portfolio_holdings ORDER BY ytd_unrealized_pl DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='Show me the investment with the highest year-to-date unrealized profit.',
#     sql="SELECT symbol, ytd_unrealized_pl FROM ai_trading.portfolio_holdings ORDER BY ytd_unrealized_pl DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='What investment has the best performance in terms of YTD profit?',
#     sql="SELECT symbol, ytd_unrealized_pl FROM ai_trading.portfolio_holdings ORDER BY ytd_unrealized_pl DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='List the top 3 investments by their current returns.',
#     sql="SELECT symbol, ytd_unrealized_pl FROM ai_trading.portfolio_holdings ORDER BY ytd_unrealized_pl DESC LIMIT 3;"
#     )

# # Questions about "portfolio" performance
#     vn.add_question_sql(
#     question='Which portfolio has the highest overall profit this year?',
#     sql="SELECT portfolio_name, SUM(ytd_unrealized_pl) AS total_ytd_profit FROM ai_trading.portfolio_holdings GROUP BY portfolio_name ORDER BY total_ytd_profit DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='Show me the total year-to-date profit for each portfolio.',
#     sql="SELECT portfolio_name, SUM(ytd_unrealized_pl) AS total_ytd_profit FROM ai_trading.portfolio_holdings GROUP BY portfolio_name ORDER BY total_ytd_profit DESC;"
#     )
#     vn.add_question_sql(
#     question='List all portfolios and their current total market value.',
#     sql="SELECT portfolio_name, SUM(market_value) AS total_market_value FROM ai_trading.portfolio_holdings GROUP BY portfolio_name ORDER BY total_market_value DESC;"
#     )
#     vn.add_question_sql(
#     question='What is the best performing portfolio based on unrealized gains?',
#     sql="SELECT portfolio_name, SUM(ytd_unrealized_pl) AS total_unrealized_pl FROM ai_trading.portfolio_holdings GROUP BY portfolio_name ORDER BY total_unrealized_pl DESC LIMIT 1;"
#     )

# # Questions about specific investments
#     vn.add_question_sql(
#     question='What is the current market value of AAPL in all portfolios?',
#     sql="SELECT SUM(market_value) FROM ai_trading.portfolio_holdings WHERE symbol = 'AAPL';"
#     )
#     vn.add_question_sql(
#     question='How many positions do I have for TSLA?',
#     sql="SELECT SUM(positions) FROM ai_trading.portfolio_holdings WHERE symbol = 'TSLA';"
#     )
#     vn.add_question_sql(
#     question='Show me all holdings in the "Tech Growth" portfolio.',
#     sql="SELECT symbol, positions, market_value FROM ai_trading.portfolio_holdings WHERE portfolio_name = 'Tech Growth';"
#     )
#     vn.add_question_sql(
#     question='What is the YTD unrealized profit/loss for MSFT?',
#     sql="SELECT SUM(ytd_unrealized_pl) FROM ai_trading.portfolio_holdings WHERE symbol = 'MSFT';"
#     )

# # Questions involving datetime
#     vn.add_question_sql(
#     question='Show me the market value of AAPL at the latest available datetime.',
#     sql="SELECT market_value FROM ai_trading.portfolio_holdings WHERE symbol = 'AAPL' ORDER BY datetime DESC LIMIT 1;"
#     )

# # General questions
#     vn.add_question_sql(
#     question='Tell me about my portfolio holdings.',
#     sql="SELECT portfolio_name, symbol, positions, market_value, ytd_unrealized_pl FROM ai_trading.portfolio_holdings LIMIT 10;" # Limit for broad overview
#     )

# Table-level documentation
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     documentation='This table provides detailed profit and loss (P&L) information for portfolio holdings, including both unrealized (paper) and realized (closed positions) gains and losses, aggregated year-to-date and daily. It is critical for comprehensive performance analysis of investments and portfolios.'
#     )

# # Column-level documentation
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='datetime',
#     documentation='The specific date and time for which the P&L data was recorded. Important for time-series analysis.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='group_name',
#     documentation='A higher-level grouping or category for portfolios (e.g., "Equity", "Fixed Income"). Useful for aggregated analysis across types of portfolios.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='portfolio_name',
#     documentation='The name or identifier of the specific investment portfolio. Used for portfolio-level analysis and grouping.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='symbol',
#     documentation='The stock ticker symbol or identifier for the individual investment (e.g., MSFT, GOOG). This represents a single asset or security.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='positions',
#     documentation='The total number of shares or units held for a particular investment at the recorded datetime.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='market_value',
#     documentation='The total current market value of all positions for this investment. This indicates the current worth of the holding.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='ytd_unrealized_pnl',
#     documentation='The Year-To-Date (YTD) unrealized Profit or Loss. This is the paper gain or loss on open positions that has not yet been realized by selling. Often referred to as "current return" or "potential profit."'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='ytd_realized_pnl',
#     documentation='The Year-To-Date (YTD) realized Profit or Loss. This represents the actual gain or loss from positions that have been closed (sold) within the current year. This is a definitive measure of "actual profit" or "actual return."'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='ytd_total_pnl',
#     documentation='The Year-To-Date (YTD) total Profit or Loss, which is the sum of both realized and unrealized P&L. This column provides a comprehensive view of "overall profit" or "total return" for the year.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='created_timestamp',
#     documentation='The timestamp when the P&L record was initially created in the system.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='daily_realized_pnl',
#     documentation='The actual profit or loss realized for a specific investment on a particular day. Useful for tracking short-term performance and daily returns.'
#     )



# --- Add Question-SQL Pairs ---

# Questions about "best investment" / "returns" / "profit" (Leveraging realized/total P&L)
#     vn.add_question_sql(
#     question='What investment gave the best actual return this year?',
#     sql="SELECT symbol, ytd_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl ORDER BY ytd_realized_pnl DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='Which investment has the highest total profit year-to-date?',
#     sql="SELECT symbol, ytd_total_pnl FROM ai_trading.portfolio_holdings_realized_pnl ORDER BY ytd_total_pnl DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='Show me the top 3 investments by actual realized profit this year.',
#     sql="SELECT symbol, ytd_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl ORDER BY ytd_realized_pnl DESC LIMIT 3;"
#     )
#     vn.add_question_sql(
#     question='What is the highest overall profit any investment generated YTD?',
#     sql="SELECT MAX(ytd_total_pnl) FROM ai_trading.portfolio_holdings_realized_pnl;"
#     )
#     vn.add_question_sql(
#     question='Which investment lost the most actual money this year?',
#     sql="SELECT symbol, ytd_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl ORDER BY ytd_realized_pnl ASC LIMIT 1;"
#     )

# # Questions about "portfolio" performance
#     vn.add_question_sql(
#     question='Which portfolio has the best total year-to-date profit?',
#     sql="SELECT portfolio_name, SUM(ytd_total_pnl) AS total_portfolio_pnl FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY portfolio_name ORDER BY total_portfolio_pnl DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='Show me the actual realized profit for each portfolio this year.',
#     sql="SELECT portfolio_name, SUM(ytd_realized_pnl) AS total_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY portfolio_name ORDER BY total_realized_pnl DESC;"
#     )
#     vn.add_question_sql(
#     question='List portfolios by their total YTD returns (realized and unrealized).',
#     sql="SELECT portfolio_name, SUM(ytd_total_pnl) AS total_returns FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY portfolio_name ORDER BY total_returns DESC;"
#     )
#     vn.add_question_sql(
#     question='What is the daily realized profit for "Aggressive Growth" portfolio for the latest date?',
#     sql="SELECT daily_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl WHERE portfolio_name = 'Aggressive Growth' ORDER BY datetime DESC LIMIT 1;"
#     )

# # Questions about "group_name" performance
#     vn.add_question_sql(
#     question='Which asset group generated the highest total profit YTD?',
#     sql="SELECT group_name, SUM(ytd_total_pnl) AS total_group_pnl FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY group_name ORDER BY total_group_pnl DESC LIMIT 1;"
#     )
#     vn.add_question_sql(
#     question='Show me the realized profit for each asset group this year.',
#     sql="SELECT group_name, SUM(ytd_realized_pnl) AS total_group_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl GROUP BY group_name ORDER BY total_group_realized_pnl DESC;"
#     )

# # Questions about specific investments / symbols
#     vn.add_question_sql(
#     question='What was the daily realized P&L for GOOG yesterday?', # Requires knowing 'yesterday' or getting max(datetime)
#     sql="SELECT daily_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl WHERE symbol = 'GOOG' ORDER BY datetime DESC LIMIT 1;" # Assumes latest date is 'yesterday' if not specified
#     )
#     vn.add_question_sql(
#     question='Tell me the YTD total P&L for AAPL.',
#     sql="SELECT SUM(ytd_total_pnl) FROM ai_trading.portfolio_holdings_realized_pnl WHERE symbol = 'AAPL';"
#     )

# # General overview questions
#     vn.add_question_sql(
#     question='Give me an overview of all portfolio P&L figures.',
#     sql="SELECT datetime, group_name, portfolio_name, symbol, ytd_unrealized_pnl, ytd_realized_pnl, ytd_total_pnl, daily_realized_pnl FROM ai_trading.portfolio_holdings_realized_pnl LIMIT 10;"
#     )




# --- Add Documentation ---

# Table-level documentation
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     documentation='This table provides aggregated summary metrics for each investment portfolio, including various time-based returns (YTD, QTD, MTD, WTD, Daily, All-time), profit figures, net liquidity, and comparison against a default index. It is the primary source for overall portfolio performance analysis.'
#     )

# # Column-level documentation
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='datetime',
#     documentation='The date for which the portfolio summary data was recorded. Critical for time-series analysis of portfolio performance.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='portfolio_name',
#     documentation='The unique name or identifier for each investment portfolio. Use this for querying specific portfolios.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='account_id',
#     documentation='The ID of the trading account associated with the portfolio. Can be used for grouping portfolios by account.'
#     )
    
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='default_index',
#     documentation='The benchmark index (e.g., S&P 500, NASDAQ) against which the portfolio\'s performance is compared. Used for relative performance analysis.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='portfolio_startdate',
#     documentation='The inception date of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='net_liquidity',
#     documentation='The total cash and market value of all holdings in the portfolio. Represents the total current worth of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='unrealized_pl',
#     documentation='The total unrealized (paper) profit or loss for the entire portfolio. This is the gain/loss on open positions.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='last_updated_time',
#     documentation='The precise timestamp when the portfolio\'s summary data was last updated. Useful for freshness checks.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='ytd_return',
#     documentation='The Year-To-Date (YTD) percentage return of the portfolio. This is a key metric for annual performance. Often referred to as "annual return" or "return this year."'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='allocated_amount',
#     documentation='The total capital allocated to the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='index_annual_return',
#     documentation='The annual percentage return of the default benchmark index. For comparison with `ytd_return`.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='portfolio_description',
#     documentation='A brief description or strategy of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='utilized_amount',
#     documentation='The amount of allocated capital currently utilized by holdings.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='group_name',
#     documentation='A broader category or group to which the portfolio belongs (e.g., "Equity", "Balanced"). Useful for aggregated performance analysis across portfolio types.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='cost_model',
#     documentation='The cost basis accounting method used for the portfolio (e.g., FIFO, LIFO, Average Cost).'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='ytd_profit',
#     documentation='The Year-To-Date (YTD) monetary profit (P&L) of the portfolio. This represents the total gain or loss in absolute currency terms for the year. Often referred to as "profit this year" or "net profit YTD."'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='all_profit',
#     documentation='The total monetary profit (P&L) of the portfolio since its inception. This is the "all-time profit" or "total historical profit."'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='qtd_return',
#     documentation='The Quarter-To-Date (QTD) percentage return of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='qtd_profit',
#     documentation='The Quarter-To-Date (QTD) monetary profit of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='mtd_return',
#     documentation='The Month-To-Date (MTD) percentage return of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='mtd_profit',
#     documentation='The Month-To-Date (MTD) monetary profit of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='wtd_return',
#     documentation='The Week-To-Date (WTD) percentage return of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='wtd_profit',
#     documentation='The Week-To-Date (WTD) monetary profit of the portfolio.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='ytd_index_return',
#     documentation='The Year-To-Date (YTD) percentage return of the portfolio\'s default benchmark index.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='qtd_index_return',
#     documentation='The Quarter-To-Date (QTD) percentage return of the portfolio\'s default benchmark index.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='mtd_index_return',
#     documentation='The Month-To-Date (MTD) percentage return of the portfolio\'s default benchmark index.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='wtd_index_return',
#     documentation='The Week-To-Date (WTD) percentage return of the portfolio\'s default benchmark index.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='all_index_return',
#     documentation='The total percentage return of the portfolio\'s default benchmark index since the portfolio\'s inception.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='all_return',
#     documentation='The total percentage return of the portfolio since its inception. This is the "all-time return."'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='daily_return',
#     documentation='The daily percentage return of the portfolio. Useful for granular performance tracking.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='daily_profit',
#     documentation='The daily monetary profit or loss of the portfolio. Useful for granular P&L tracking.'
#     )
#     vn.add_documentation(
#     table_name='portfolio_summary', 
#     schema_name='ai_trading', 
#     column_name='is_active',
#     documentation="""A numeric flag indicating if the portfolio is active. 
#     Data Type: numeric. 
#     **IMPORTANT**: In all SQL WHERE clauses, you MUST use `is_active = 1` to filter for active portfolios and `is_active = 0` for inactive ones. 
#     DO NOT use boolean values like TRUE or FALSE. For example, to find active portfolios, the correct syntax is `WHERE is_active = 1`."""
# )
#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='daily_index_return',
#     documentation='The daily percentage return of the portfolio\'s default benchmark index.'
#     )

#     vn.add_documentation(
#     table_name='portfolio_summary',
#     schema_name='ai_trading',
#     column_name='portfolio_name',
#     documentation="The unique name or identifier for each investment portfolio. Users may also refer to this as a 'wallet' or 'account'."
#     )





# --- Add Question-SQL Pairs ---

# Questions about "best portfolio" / "returns" / "profit" across various timeframes
# YTD
# vn.add_question_sql(
#     question='What is the best performing portfolio by YTD return?',
#     sql="SELECT portfolio_name, ytd_return FROM ai_trading.portfolio_summary ORDER BY ytd_return DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Which portfolio has the highest YTD profit?',
#     sql="SELECT portfolio_name, ytd_profit FROM ai_trading.portfolio_summary ORDER BY ytd_profit DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Show me the YTD returns for all active portfolios.',
#     sql="SELECT portfolio_name, ytd_return FROM ai_trading.portfolio_summary WHERE is_active = TRUE ORDER BY ytd_return DESC;"
# )

# # QTD
# vn.add_question_sql(
#     question='What is the best performing portfolio by QTD return?',
#     sql="SELECT portfolio_name, qtd_return FROM ai_trading.portfolio_summary ORDER BY qtd_return DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Which portfolio has the highest QTD profit?',
#     sql="SELECT portfolio_name, qtd_profit FROM ai_trading.portfolio_summary ORDER BY qtd_profit DESC LIMIT 1;"
# )

# # MTD
# vn.add_question_sql(
#     question='What is the best performing portfolio by MTD return?',
#     sql="SELECT portfolio_name, mtd_return FROM ai_trading.portfolio_summary ORDER BY mtd_return DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Which portfolio has the highest MTD profit?',
#     sql="SELECT portfolio_name, mtd_profit FROM ai_trading.portfolio_summary ORDER BY mtd_profit DESC LIMIT 1;"
# )

# # WTD
# vn.add_question_sql(
#     question='What is the best performing portfolio by WTD return?',
#     sql="SELECT portfolio_name, wtd_return FROM ai_trading.portfolio_summary ORDER BY wtd_return DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Which portfolio has the highest WTD profit?',
#     sql="SELECT portfolio_name, wtd_profit FROM ai_trading.portfolio_summary ORDER BY wtd_profit DESC LIMIT 1;"
# )

# # Daily
# vn.add_question_sql(
#     question='What is the best performing portfolio by daily return?',
#     sql="SELECT portfolio_name, daily_return FROM ai_trading.portfolio_summary ORDER BY daily_return DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Which portfolio had the highest daily profit?',
#     sql="SELECT portfolio_name, daily_profit FROM ai_trading.portfolio_summary ORDER BY daily_profit DESC LIMIT 1;"
# )

# # All-time
# vn.add_question_sql(
#     question='What is the best performing portfolio by all-time return?',
#     sql="SELECT portfolio_name, all_return FROM ai_trading.portfolio_summary ORDER BY all_return DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Which portfolio has the highest all-time profit?',
#     sql="SELECT portfolio_name, all_profit FROM ai_trading.portfolio_summary ORDER BY all_profit DESC LIMIT 1;"
# )

# # Questions about net liquidity
# vn.add_question_sql(
#     question='Show me the net liquidity of "My Diversified Portfolio".',
#     sql="SELECT net_liquidity FROM ai_trading.portfolio_summary WHERE portfolio_name = 'My Diversified Portfolio';"
# )
# vn.add_question_sql(
#     question='List all portfolios by their current net liquidity.',
#     sql="SELECT portfolio_name, net_liquidity FROM ai_trading.portfolio_summary ORDER BY net_liquidity DESC;"
# )

# # Questions comparing to index
# vn.add_question_sql(
#     question='Which portfolio outperformed its YTD index return the most?',
#     sql="SELECT portfolio_name, (ytd_return - ytd_index_return) AS outperformance FROM ai_trading.portfolio_summary ORDER BY outperformance DESC LIMIT 1;"
# )
# vn.add_question_sql(
#     question='Show me the YTD return for "Growth Fund" and its benchmark index.',
#     sql="SELECT ytd_return, ytd_index_return, default_index FROM ai_trading.portfolio_summary WHERE portfolio_name = 'Growth Fund';"
# )

# # Questions involving user_id or account_id (assuming current_user_id is available in context)
# # IMPORTANT: Be careful with user_id and PII. Only use if safe and necessary.

# vn.add_question_sql(
#     question='What is the total net liquidity across all portfolios for account 456?',
#     sql="SELECT SUM(net_liquidity) FROM ai_trading.portfolio_summary WHERE account_id = 456;"
# )

# # Questions about portfolio description and status
# vn.add_question_sql(
#     question='What is the description for "Balanced Fund"?',
#     sql="SELECT portfolio_description FROM ai_trading.portfolio_summary WHERE portfolio_name = 'Balanced Fund';"
# )
# vn.add_question_sql(
#     question='List all active portfolios.',
#     sql="SELECT portfolio_name FROM ai_trading.portfolio_summary WHERE is_active = TRUE;"
# )

# # Questions involving group_name
# vn.add_question_sql(
#     question='What is the average YTD return for portfolios in the "Equity" group?',
#     sql="SELECT AVG(ytd_return) FROM ai_trading.portfolio_summary WHERE group_name = 'Equity';"
# )
# vn.add_question_sql(
#     question='Which portfolio group has the highest total all-time profit?',
#     sql="SELECT group_name, SUM(all_profit) AS total_group_profit FROM ai_trading.portfolio_summary GROUP BY group_name ORDER BY total_group_profit DESC LIMIT 1;"
# )

# # General overview questions
# vn.add_question_sql(
#     question='Give me a summary of all portfolios with their latest YTD returns and profits.',
#     sql="SELECT portfolio_name, ytd_return, ytd_profit, net_liquidity FROM ai_trading.portfolio_summary ORDER BY last_updated_time DESC LIMIT 10;"
# )
# vn.add_question_sql(
#     question='Show me the current state of my portfolios.',
#     sql="SELECT portfolio_name, net_liquidity, ytd_return, ytd_profit FROM ai_trading.portfolio_summary ORDER BY last_updated_time DESC;" # Assuming current_user_id is available
# )


# # --- 1. For "Highest Realized Profit" ---

# vn.add_question_sql(
#     question="what is my highest realized profit",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_realized_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     ORDER BY ytd_realized_pnl DESC
#     LIMIT 1;
#     """
# )

# vn.add_question_sql(
#     question="Which stock had the best actual profit from sales?",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_realized_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     ORDER BY ytd_realized_pnl DESC
#     LIMIT 1;
#     """
# )

# vn.add_question_sql(
#     question="Show me the single biggest realized gain this year.",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_realized_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     ORDER BY ytd_realizd_pnl DESC
#     LIMIT 1;
#     """
# )


# # --- 2. For "Highest Unrealized Profit" ---

# vn.add_question_sql(
#     question="what is my highest unrealized profit",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_unrealized_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     ORDER BY ytd_unrealized_pnl DESC
#     LIMIT 1;
#     """
# )

# vn.add_question_sql(
#     question="Which holding has the biggest 'on paper' gain?",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_unrealized_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     ORDER BY ytd_unrealized_pnl DESC
#     LIMIT 1;
#     """
# )

# vn.add_question_sql(
#     question="Show me the top unrealized profit.",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_unrealized_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     ORDER BY ytd_unrealized_pnl DESC
#     LIMIT 1;
#     """
# )


# # --- 3. For "Top 10 Profit Stocks" ---

# vn.add_question_sql(
#     question="top 10 profit stocks",
#     sql="""
#     SELECT symbol, SUM(ytd_total_pnl) AS total_profit
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     GROUP BY symbol
#     ORDER BY total_profit DESC
#     LIMIT 10;
#     """
# )

# vn.add_question_sql(
#     question="What are the 10 best performing stocks overall?",
#     sql="""
#     SELECT symbol, SUM(ytd_total_pnl) AS total_profit
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     GROUP BY symbol
#     ORDER BY total_profit DESC
#     LIMIT 10;
#     """
# )

# vn.add_question_sql(
#     question="List the top 10 stocks by total year-to-date profit.",
#     sql="""
#     SELECT symbol, SUM(ytd_total_pnl) AS total_profit
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     GROUP BY symbol
#     ORDER BY total_profit DESC
#     LIMIT 10;
#     """
# )


# # --- 4. For "Bottom 10 Stocks" ---

# vn.add_question_sql(
#     question="bottom 10 stocks",
#     sql="""
#     SELECT symbol, SUM(ytd_total_pnl) AS total_profit
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     GROUP BY symbol
#     ORDER BY total_profit ASC
#     LIMIT 10;
#     """
# )

# vn.add_question_sql(
#     question="What are the 10 worst performing stocks?",
#     sql="""
#     SELECT symbol, SUM(ytd_total_pnl) AS total_profit
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     GROUP BY symbol
#     ORDER BY total_profit ASC
#     LIMIT 10;
#     """
# )

# vn.add_question_sql(
#     question="Show me the biggest 10 losers in the portfolio.",
#     sql="""
#     SELECT symbol, SUM(ytd_total_pnl) AS total_profit
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     GROUP BY symbol
#     ORDER BY total_profit ASC
#     LIMIT 10;
#     """
# )


# # --- 5. For "Loss Stocks Across Portfolio" ---

# vn.add_question_sql(
#     question="loss stocks across portfolio",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_total_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     WHERE ytd_total_pnl < 0
#     ORDER BY ytd_total_pnl ASC;
#     """
# )

# vn.add_question_sql(
#     question="Which holdings are currently at a loss?",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_total_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     WHERE ytd_total_pnl < 0
#     ORDER BY ytd_total_pnl ASC;
#     """
# )

# vn.add_question_sql(
#     question="List all investments that have a negative total profit.",
#     sql="""
#     SELECT symbol, portfolio_name, ytd_total_pnl
#     FROM ai_trading.portfolio_holdings_realized_pnl
#     WHERE ytd_total_pnl < 0
#     ORDER BY ytd_total_pnl ASC;
#     """
# )

# # --- Add/Reinforce Documentation for group_name in portfolio_summary ---
# # This documentation should already exist if you added the previous block,
# # but it's good to ensure it's robust for 'groups'.

# # vn.add_documentation(
# #     table_name='portfolio_summary',
# #     schema_name='ai_trading',
# #     column_name='group_name',
# #     documentation='A broader category or group to which a portfolio belongs (e.g., "Equity", "Fixed Income", "Growth", "Value"). Users may ask "What are my groups?" to see these categories. This column helps in analyzing performance across different types of portfolio strategies or asset classes.'
# #     )

# # --- Add Question-SQL Pairs for "What are my groups?" ---
# # We want to retrieve unique group names.
# vn.add_question_sql(
#     question='What are my groups?',
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary;" # Assuming user_id context
#     )
# vn.add_question_sql(
#     question='List all portfolio groups I have.',
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary;"
#     )
# vn.add_question_sql(
#     question='What are the different investment categories I use?',
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary;"
#     )
# vn.add_question_sql(
#     question='Show me my portfolio groups.',
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary;"
#     )
# vn.add_question_sql(
#     question='What types of portfolios do I have?',
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary;"
#     )


#     # --- If you want to list ALL groups in the system (not just for a specific user) ---
# vn.add_question_sql(
#     question='List all available portfolio groups.',
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary ORDER BY group_name;"
#     )
# vn.add_question_sql(
#     question='What are the different group names in the system?',
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_summary ORDER BY group_name;"
#     )

# Explain that portfolio_holdings is the correct table for historical data
# vn.add_documentation(
#     table_name='portfolio_holdings',
#     schema_name='ai_trading',
#     documentation="""
#     This table is the primary source for historical, time-series data for individual holdings
#     within a portfolio. Use this table to answer any question about the 'trend', 'history',
#     or 'changes over time' of a specific stock's value, positions, or profit/loss.
#     """
# )


# -- Column-Level Documentation --

# Explain the role of the 'datetime' column for trends
# vn.add_documentation(
#     table_name='portfolio_holdings',
#     schema_name='ai_trading',
#     column_name='datetime',
#     documentation="""
#     The timestamp for each data point. This column is essential for any query about
#     historical trends, time-series analysis, or plotting data 'over time'.
#     Always use this column for the x-axis of a trend and ORDER BY it.
#     """
# )

# # Explain the role of the 'market_value' column
# vn.add_documentation(
#     table_name='portfolio_holdings',
#     schema_name='ai_trading',
#     column_name='market_value',
#     documentation="""
#     The total monetary worth of a holding at a specific point in time. Use this column
#     when asked about the 'value trend', 'historical value', or 'market value over time'.
#     """
# )

# # Reinforce the purpose of 'symbol' and 'portfolio_name' for filtering
# vn.add_documentation(
#     table_name='portfolio_holdings',
#     schema_name='ai_trading',
#     column_name='symbol',
#     documentation="The stock ticker for a specific holding. Use this in a WHERE clause to filter for a particular security like 'FDX' or 'AAPL'."
# )

# vn.add_documentation(
#     table_name='portfolio_holdings',
#     schema_name='ai_trading',
#     column_name='portfolio_name',
#     documentation="The name of the portfolio. Use this in a WHERE clause to scope the query to a specific portfolio like 'F-R&D-600K'."
# )
# # --- Optional: If `group_name` in `portfolio_holdings_realized_pnl` is also relevant for listing groups ---
# # You might want to also add similar Q&A pairs for portfolio_holdings_realized_pnl if
# # it's possible for groups to exist there that don't appear in portfolio_summary.
# # However, usually summary tables are better for "listing unique entities."
# vn.add_documentation(
#     table_name='portfolio_holdings_realized_pnl',
#     schema_name='ai_trading',
#     column_name='group_name',
#     documentation='A higher-level grouping or category for portfolios within this P&L detail. Used for aggregated analysis across types of portfolios. Can be used to list available portfolio groups.'
#     )
# vn.add_question_sql(
#     question='What are my groups (detailed P&L view)?', # Make question slightly different if you want to differentiate
#     sql="SELECT DISTINCT group_name FROM ai_trading.portfolio_holdings_realized_pnl  ORDER BY group_name;" # Assuming user_id
#     )

# # The specific question asked
# vn.add_question_sql(
#     question="What is the historical trend of the FDX holding’s market value in the F‑R&D‑600K portfolio",
#     sql="""
#     SELECT
#       datetime,
#       market_value
#     FROM
#       ai_trading.portfolio_holdings
#     WHERE
#       portfolio_name = 'F-R&D-600K' AND symbol = 'FDX'
#     ORDER BY
#       datetime ASC;
#     """
# )

# # A variation with different wording and symbols
# vn.add_question_sql(
#     question="Show me the market value history for AAPL in the Tech Growth portfolio over time.",
#     sql="""
#     SELECT
#       datetime,
#       market_value
#     FROM
#       ai_trading.portfolio_holdings
#     WHERE
#       portfolio_name = 'Tech Growth' AND symbol = 'AAPL'
#     ORDER BY
#       datetime ASC;
#     """
# )

# # A variation asking for a different metric (positions)
# vn.add_question_sql(
#     question="Plot the trend of my GOOG positions in the Alpha fund.",
#     sql="""
#     SELECT
#       datetime,
#       positions
#     FROM
#       ai_trading.portfolio_holdings
#     WHERE
#       portfolio_name = 'Alpha fund' AND symbol = 'GOOG'
#     ORDER BY
#       datetime ASC;
#     """
# )
# vn.add_question_sql(
#     question='Give me an overall summary of my investments.',
#     sql="""
#     SELECT
#         'Overall Summary' AS summary_type,
#         SUM(ps.net_liquidity) AS total_net_liquidity,
#         AVG(ps.ytd_return) AS average_ytd_return,
#         SUM(ps.ytd_profit) AS total_ytd_profit,
#         SUM(ps.all_profit) AS total_all_profit
#     FROM
#         ai_trading.portfolio_summary ps
#     WHERE
        
#         ps.is_active = TRUE
#         AND ps.datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE  is_active = TRUE)
#     ;
#     """
#     )
# vn.add_question_sql(
#     question='Provide a summary of my entire investment portfolio.',
#     sql="""
#     SELECT
#         'Overall Summary' AS summary_type,
#         SUM(ps.net_liquidity) AS total_net_liquidity,
#         AVG(ps.ytd_return) AS average_ytd_return,
#         SUM(ps.ytd_profit) AS total_ytd_profit,
#         SUM(ps.all_profit) AS total_all_profit
#     FROM
#         ai_trading.portfolio_summary ps
#     WHERE
        
#         ps.is_active = TRUE
#         AND ps.datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE  is_active = TRUE)
#     ;
#     """
#     )

#     # A more general summary if user_id is not implied or to list multiple portfolios
# vn.add_question_sql(
#     question='Give me a summary of my portfolio.', # Slightly different phrasing
#     sql="""
#     SELECT
#         ps.portfolio_name,
#         ps.net_liquidity,
#         ps.ytd_return,
#         ps.ytd_profit,
#         ps.group_name
#     FROM
#         ai_trading.portfolio_summary ps
    
#     ORDER BY
#         ps.portfolio_name;
#     """
#     )

# vn.add_question_sql(
#     question="What is the total amount of investment that was used",
#     sql="""
#     SELECT SUM(utilized_amount) AS total_investment_used
#     FROM ai_trading.portfolio_summary
#     WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);
#     """
# )

# vn.add_question_sql(
#     question="How much money is currently invested across all portfolios?",
#     sql="""
#     SELECT SUM(utilized_amount) AS total_currently_invested
#     FROM ai_trading.portfolio_summary
#     WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);
#     """
# )

# vn.add_question_sql(
#     question="Show me the total utilized amount for all active wallets.",
#     sql="""
#     SELECT SUM(utilized_amount) AS total_utilized_amount
#     FROM ai_trading.portfolio_summary
#     WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);
#     """
# )

# vn.add_question_sql(
#     question="What is the sum of capital put to use in the market?",
#     sql="""
#     SELECT SUM(utilized_amount) AS total_capital_in_use
#     FROM ai_trading.portfolio_summary
#     WHERE is_active = 1 AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1);
#     """
# )
import re
def replace_wallet_with_portfolio(text: str) -> str:
    # Replace 'wallet' as a standalone word, ignoring case
    return re.sub(r'\bwallet\b', 'portfolio', text, flags=re.IGNORECASE)

# --- 5. Streamlit User Interface ---
st.set_page_config(page_title="AI Financial Assistant", layout="centered")
st.title("📈 AI Financial Assistant")
st.caption("Ask about your portfolio data or general financial topics.")

st.sidebar.header("User Context")
#user_id_input = st.sidebar.text_input("Enter Your User ID (e.g., '123')", value="123")
#st.session_state.user_id = user_id_input

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages[-MAX_UI_MESSAGES_TO_DISPLAY:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question_input = st.chat_input("Ask your question...")

if question_input:
    st.session_state.messages.append({"role": "user", "content": question_input})
    with st.chat_message("user"):
        st.markdown(question_input)

    with st.spinner("Thinking..."):
        is_arabic = any('\u0600' <= ch <= '\u06FF' for ch in question_input)
        query_type = classify_query_type(question_input)
        st.info(f"Query classified as: **{query_type.upper()}**")
        response_content = ""

        if query_type == "portfolio":
            #if not st.session_state.get("user_id"):
                #response_content = "Please enter a User ID in the sidebar to ask questions about your portfolio."
            if vn is None:
                response_content = "I'm sorry, the database connection is not available."
            else:
                processing_question = translate(question_input, "English") if is_arabic else question_input
                if is_arabic: st.info(f"Translated for processing: '{processing_question}'")
                print("ProceSSING QUES")
                print(processing_question)
                processing_question=replace_wallet_with_portfolio(processing_question)
                #temp
                #result_text=''
                try:
                    sql = vn.generate_sql(processing_question)
                    if not sql or "SELECT" not in sql.upper():
                       raise ValueError("Failed to generate a valid SQL query.")
                    #sql = sql.replace("'{user_id}'", f"'{st.session_state.user_id}'")
                    st.code(sql, language="sql")
                    result_df = vn.run_sql(sql)
                    
                    # --- START: INTEGRATED ADVANCED EXPLANATION LOGIC ---
                    if result_df is not None and not result_df.empty:
                        st.dataframe(result_df)
                        result_text = result_df.to_string(index=False)
                        
                        filtered_dialogue_history = [msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]]
                        recent_dialogue = filtered_dialogue_history[-8:]
                        if recent_dialogue and recent_dialogue[0]["role"] == "assistant":
                           recent_dialogue = recent_dialogue[1:]

                        # Construct the detailed final prompt for explanation
                        final_user_instruction = (
                            f"My original question was: \"{processing_question}\"\n\n"
                            f"The database returned the following data:\n---\n{result_text}\n---\n\n"
                            "Based on our conversation and the data above, please provide a clear explanation."
                       )

                        messages_for_explanation = [
                           {"role": "system", "content": vanna_explanation_prompt}
                       ] + recent_dialogue + [
                           {"role": "user", "content": final_user_instruction}
                    ]

                        explanation = call_ollama_chat(messages_for_explanation, temperature=0.3)
                        response_content = "### 🧠 Answer (From Database)\n" + explanation
                    else:
                        st.warning("Query executed successfully but returned no data.")
                        response_content = "I found no data in the database that matches your query."
                    # Construct the final prompt for the explanation model
                    # This prompt works for both cases (data or no data)
                       
                    # --- END: INTEGRATED ADVANCED EXPLANATION LOGIC ---
                        
                except Exception as e:
                    #response_content = f"I encountered an error trying to get data for your portfolio: {e}"
                    
                    #traceback.print_exc()
                    #st.error(f"An unexpected error occurred: {e}")
                    #traceback.print_exc()

                        # Set the user-friendly, standardized response
                    print("Exception block in running query--")
                    #if result_df: result_text = result_df.to_string(index=False)

                    #if result_df is not None and not result_df.empty:
                        #result_text = result_df.to_string(index=False)
                    #else:
                        #result_text = ""
                    
                    

                    response_content = CUSTOM_ERROR_MESSAGE
                    

                    #st.error(response_content)
                    
        
        elif query_type == "general":
            px_result = query_perplexity_final(question_input, st.session_state.messages)
            #px_result = query_openai(question_input, st.session_state.messages)
            response_content = "### 🌐 Answer (Internet)\n" + px_result

        elif query_type == "other":
            # --- START: CORRECTED LOGIC ---
            recent_dialogue = st.session_state.messages[-8:]
            messages_for_other = [
                {"role": "system", "content": other_handler_prompt}
            ] + recent_dialogue
            response_content = "### 🤖 Answer (Conversational)\n" + call_ollama_chat(messages_for_other, temperature=0.7)
            # --- END: CORRECTED LOGIC ---

        # --- START: CORRECTED LOGIC ---
        if is_arabic and response_content:
             response_content = translate(response_content, "Arabic")
        # --- END: CORRECTED LOGIC ---

    st.session_state.messages.append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)

