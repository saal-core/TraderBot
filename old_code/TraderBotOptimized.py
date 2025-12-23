import streamlit as st
import pandas as pd
import os
import logging
import requests
import re
import traceback
from typing import List, Dict, Optional, Any
import unicodedata
# Vanna specific imports
import ollama
from config import get_vanna, get_db_params, OLLAMA_MODEL
from openai import OpenAI

# --- 1. Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
logging.basicConfig(level=logging.INFO)


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
    logging.debug("\n--- Prompt to Ollama ---")
    logging.debug(f"Model: {model}, Temp: {temperature}, Max Tokens: {max_tokens}")
    for msg in messages:
        logging.debug(f"{msg['role'].upper()}: {msg['content']}")
    logging.debug("------------------------")
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
            stream=False
        )
        raw_output = response['message']['content'].strip()
        logging.debug(raw_output)
        return raw_output
    except Exception as e:
        logging.error(f"Error calling Ollama chat: {e}")
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
        response_text = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"num_predict": 2000, "temperature": 0.5}
        )
        raw_output = response_text.message.content
        cleaned = raw_output.strip().lower().strip()

        if "portfolio" in cleaned:
            return "portfolio"
        if "general" in cleaned:
            return "general"
        if "other" in cleaned:
            return "other"
        
        # CRITICAL FIX: The fallback is 'other' to prevent incorrect web searches for greetings/failures.
        logging.warning(f"Unrecognized classification output: '{response_text}'. Defaulting to 'other'.")
        return "other"
        
    except Exception as e:
        st.error(f"Error during classification: {e}. Defaulting to 'other'.")
        traceback.print_exc()
        return "other"


# Perplexity system prompt for financial queries
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
        logging.error(f"Error creating standalone question: {e}. Defaulting to original question.")
        return question


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
    logging.info(f"Input Query for Perplexity: {query}")
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
        logging.error(f"HTTP error: {e} | Response: {response.text}")
        return f"Error: {response.text}"
    except Exception as ex:
        logging.error(f"General error: {ex}")
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
    logging.info(f"Input Query for OpenAI: {query}")

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

    logging.debug("\n--- Final Messages Sent to OpenAI ---")
    for msg in openai_messages:
        logging.debug(f"- Role: {msg['role']}")
    logging.debug("-------------------------------------\n")

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
        logging.error(f"Error during OpenAI API call: {ex}")
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
    logging.debug("\n--- Final Perplexity Messages ---")
    for msg in final_messages:
        logging.debug(f"- Role: {msg['role']}, Content preview: {msg['content'][:60]}")
    logging.debug("--------------------------------\n")

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
        logging.error(f"HTTP error: {e} | Response: {error_detail}")
        return f"Error from Perplexity API: {error_detail}"
    except Exception as ex:
        logging.error(f"General error: {ex}")
        traceback.print_exc()
        return f"Error: {str(ex)}"


 

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
        logging.error(f"Error during translation: {e}")
        # Fallback to the original text if translation fails
        return text

# --- 4. Initialization and Vanna Setup ---
try:
    ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}], stream=False)
    st.sidebar.success(f"Connected to Ollama ({OLLAMA_MODEL})")
except Exception as e:
    st.sidebar.error(f"Failed to connect to Ollama: {e}")
    st.stop()

vn = get_vanna()

try:
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = get_db_params()
    if all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
        vn.connect_to_postgres(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        st.sidebar.success("Connected to Vanna database.")
    else:
        st.sidebar.warning("Database environment variables not fully set; portfolio queries disabled")
except Exception as e:
    st.sidebar.error(f"Vanna DB connection failed: {e}.")
    vn = None

# --- Vanna Documentation removed (1384 lines of commented code) ---
# All Vanna training documentation has been moved to vanna_train.py

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
                logging.debug("Processing portfolio question")
                logging.debug(processing_question)
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
                    logging.error("Exception running portfolio query")
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
