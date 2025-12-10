# TraderBot — AI Financial Assistant

A Streamlit-based assistant that answers portfolio questions via Vanna (text-to-SQL) and general finance queries via Perplexity, with translation handled by Ollama.

## Prerequisites
- Python 3.10+
- `uv` (optional): fast Python package manager (detected: `uv 0.9.4`)
- Ollama (LLM server) (detected: `ollama 0.12.3`)
- Streamlit (detected: `1.46.1`)
- PostgreSQL database with your portfolio data

## Environment
- Copy `.env.example` to `.env` and fill the values:
  - Database: `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`, `DB_SCHEMA`
  - Ollama: `OLLAMA_MODEL` (default `gpt-oss:20b`), `OLLAMA_API_URL` (default `http://localhost:11434`)
  - APIs: `PERPLEXITY_API_KEY` (required), `OPENAI_API_KEY` (optional)
  - Vanna: `VANNA_PERSIST_DIR` (vector store path)
  - App: `LOG_LEVEL`, `MAX_UI_MESSAGES_TO_DISPLAY`

## Install Dependencies
- With `uv`:
  ```bash
  uv pip install -r requirements.txt
  ```
- With `pip`:
  ```bash
  pip install -r requirements.txt
  ```

## Models
- Pull the Ollama model defined in `OLLAMA_MODEL` (default is `gpt-oss:20b`):
  ```bash
  ollama pull gpt-oss:20b
  ```

## Training Data (Phase 3)
Training data is externalized to JSON under `data/vanna_training/`:
- `schema_docs.json` (alias for table-level docs)
- `column_docs.json`
- `question_sql_pairs.json`

Generate JSON from the legacy script (one-time or when updating training):
```bash
python3 scripts/extract_training_data.py
# or
uv run scripts/extract_training_data.py
```

## Train Vanna
`vanna_train.py` now loads JSON and skips inline training when JSON is present.
```bash
python3 vanna_train.py
# or
uv run vanna_train.py
```
Notes:
- If DB env vars are missing, training proceeds without DB connection.
- Persistence directory: `VANNA_PERSIST_DIR`.

## Run the App
Start the Ollama server (GPU optional):
```bash
CUDA_VISIBLE_DEVICES=0 ollama serve
```

Run Streamlit (new entry point):
```bash
# Use the same interpreter that installed requirements
python3 -m streamlit run src/main.py
# or
uv run streamlit run src/main.py
```
If you see `ModuleNotFoundError` for any package, ensure you ran `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`) in the same environment used by `python3` above.

## Old vs New
- Old UI entry: `TraderBotOptimized.py`
- New UI entry: `src/main.py` (uses services in `src/services/*`)

## Troubleshooting
- Perplexity key required: app checks `PERPLEXITY_API_KEY` on startup.
- Ensure `OLLAMA_API_URL` matches your server (default `http://localhost:11434`).
- DB connection is required for portfolio queries; general queries still work without DB.
- If you see “database connection is not available”:
  - Verify `.env` contains correct `DB_*` values and `DB_SCHEMA` (if used).
  - Confirm the database is reachable from your machine and port `5432` is open.
  - Test credentials with `psql`:
    ```bash
    psql -h $DB_HOST -U $DB_USER -d $DB_NAME -p $DB_PORT -c "SELECT 1;"
    ```

## References
- Settings: `src/config/settings.py`
- Streamlit entry: `src/main.py`
- Services: `src/services/ollama_service.py`, `src/services/vanna_service.py`, `src/services/perplexity_service.py`, `src/services/translation_service.py`
