from typing import Dict, List, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config

class StockMatcher:
    """
    Handles extraction and matching of stock symbols from user queries.
    """

    def __init__(self, llm=None, sql_executor=None):
        """
        Initialize the StockMatcher.

        Args:
            llm: The LLM instance to use for extraction and matching.
            sql_executor: SQL executor for fetching symbols from DB.
        """
        self.sql_executor = sql_executor
        self._symbols_cache = None

        if llm:
            self.llm = llm
        else:
            # Default to Ollama if no LLM provided
            ollama_config = get_ollama_config()
            self.llm = Ollama(
                model=ollama_config["model_name"],
                base_url=ollama_config["base_url"],
                temperature=0.1 # Low temp for extraction
            )

    def _get_all_symbols_dict(self) -> Dict[str, str]:
        """
        Fetch all symbols from database and return as dictionary
        Uses cache to avoid repeated queries
        """
        if self._symbols_cache is not None:
            return self._symbols_cache

        if not self.sql_executor:
            return {}

        try:
            query = """
            SELECT DISTINCT symbol
            FROM ai_trading.portfolio_holdings
            WHERE symbol IS NOT NULL
            ORDER BY symbol
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                # Create dictionary with index as id
                symbols_dict = {str(i): symbol for i, symbol in enumerate(df['symbol'].tolist())}
                self._symbols_cache = symbols_dict
                return symbols_dict
            return {}
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return {}

    def extract_stock_mentions(self, query: str) -> List[str]:
        """
        Extract potential stock mentions from query using LLM
        """
        extraction_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Extract any stock names, company names, or stock symbols mentioned in this question.
Return only the extracted terms, separated by commas. If none found, return "NONE".

Examples:
- "What is the price of Apple stock?" -> Apple
- "Show me MSFT and GOOGL performance" -> MSFT, GOOGL
- "How is Tesla doing?" -> Tesla
- "What are my portfolios?" -> NONE

Question: {query}

Extracted terms:"""
        )

        extraction_chain = extraction_prompt | self.llm | StrOutputParser()

        try:
            result = extraction_chain.invoke({"query": query})
            result = result.strip()

            if result.upper() == "NONE" or not result:
                return []

            # Split by comma and clean
            terms = [term.strip() for term in result.split(',') if term.strip()]
            return terms
        except Exception as e:
            print(f"Error extracting stock mentions: {e}")
            return []

    def _match_symbol_from_list(self, extracted_term: str, all_symbols_list: List[str]) -> Optional[str]:
        """
        Use LLM to find the best matching symbol from the list for the extracted term.
        """
        if not extracted_term or not all_symbols_list:
            return None
            
        # Create a string representation of the symbols list
        symbols_str = ", ".join(all_symbols_list)
        
        match_prompt = PromptTemplate(
            input_variables=["term", "symbols"],
            template="""You are an expert financial data assistant.
Your task is to identify the correct stock symbol from the provided list that corresponds to the company or term mentioned by the user.

User Term: "{term}"

Available Symbols List:
{symbols}

Instructions:
1. Find the symbol in the list that best matches the User Term.
2. Example: If User Term is "Apple", and list has "AAPL", return "AAPL".
3. Example: If User Term is "National Bank", and list has "NBK", return "NBK".
4. If the exact symbol is in the list, return it.
5. If a very strong match is found (e.g. company name to ticker), return the ticker.
6. If NO match is found, return "NONE".
7. Return ONLY the symbol name (or "NONE"). Do not add any explanation.

Matching Symbol:"""
        )
        
        match_chain = match_prompt | self.llm | StrOutputParser()
        
        try:
            result = match_chain.invoke({
                "term": extracted_term,
                "symbols": symbols_str
            })
            
            result = result.strip()
            if result == "NONE" or result not in all_symbols_list:
                return None
                
            return result
            
        except Exception as e:
            print(f"Error matching symbol: {e}")
            return None

    def fuzzy_match_symbols(self, mentioned_terms: List[str]) -> str:
        """
        Match mentioned terms to actual symbols using LLM selection from list
        """
        symbols_dict = self._get_all_symbols_dict()
        
        if not mentioned_terms or not symbols_dict:
            return "No stock symbols detected in the question"

        symbols_list = list(symbols_dict.values())
        matched_info = []

        for term in mentioned_terms:
            matched_symbol = self._match_symbol_from_list(term, symbols_list)
            
            if matched_symbol:
                matched_info.append(f"'{term}' -> '{matched_symbol}'")

        if matched_info:
            return "Detected stock mentions:\n" + "\n".join(matched_info)
        else:
            return f"No close matches found for: {', '.join(mentioned_terms)}"
