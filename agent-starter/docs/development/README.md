# Development Guide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
ollama pull llama3.1
python scripts/ingest_documents.py
uvicorn app.main:app --reload     # http://localhost:8000
pytest                            # no Ollama or network needed
python scripts/ai/validate_governance.py
docker build -t document-chat .
```

See the root `README.md` for configuration, Docker, and Railway details.
