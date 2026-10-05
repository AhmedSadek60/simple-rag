# AI Document Chat

Ask questions about your own PDF, DOCX, TXT and Markdown files. A small FastAPI app retrieves the most relevant passages from a local ChromaDB index and asks an [Ollama](https://ollama.com) model to answer **using only those passages**, with sources.

This repository also follows the [agent-starter](docs/agent-starter-template.md) engineering contract for AI coding agents (see [`AGENTS.md`](AGENTS.md)).

## Overview

- One service: FastAPI backend + static HTML/CSS/JS chat page.
- Local LLM through Ollama, local embeddings (`all-MiniLM-L6-v2`), local ChromaDB.
- Answers cite their sources (document and PDF page). If nothing relevant is found, the app says so instead of guessing.
- Ships with a Dockerfile and Railway config.

## Architecture

```
Browser (static/)  ->  POST /api/chat  ->  ChatService
                                              |-- Retriever -> VectorStore (ChromaDB) <- Embedder
                                              '-- LLMProvider (Ollama or Together AI)
```

| Path | Purpose |
|---|---|
| `app/main.py` | App factory, startup wiring, request-size limit, static files |
| `app/api/routes.py` | `GET /health`, `POST /api/chat`, error-to-HTTP mapping |
| `app/services/chat_service.py` | The RAG flow: retrieve, build prompt, call LLM, collect sources |
| `app/rag/` | Document loading + chunking, embeddings, ChromaDB store, retriever, ingestion |
| `app/llm/ollama_client.py` | `OllamaProvider`; `app/llm/together_client.py` has `TogetherProvider` |
| `app/prompts/rag_prompt.txt` | The grounding prompt |
| `scripts/ingest_documents.py` | Rebuilds the index from `documents/` |

**How a question is answered:** the question is embedded, the `TOP_K` nearest chunks are fetched from ChromaDB, chunks farther than `MAX_DISTANCE` are dropped, and the rest are placed in the prompt. If no chunk is close enough the LLM is not called and the "not found" message is returned. LangChain is used only for its text splitter; everything else is plain Python to keep the code small.

## Features

PDF / DOCX / TXT / MD ingestion · configurable chunking and retrieval · source citations · clear error messages · loading state, Enter-to-send, Shift+Enter for newline · request size limits · no login, no external API calls.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/download) (locally, or a reachable remote endpoint)
- Docker (optional)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # use requirements.txt for runtime only
cp .env.example .env
```

The first run downloads the embedding model (~90 MB) from Hugging Face.

## Installing Ollama

Install it from <https://ollama.com/download> (macOS, Windows, Linux) and make sure it is running (`ollama serve` if it is not started automatically). It listens on `http://localhost:11434`.

## Pulling an LLM

```bash
ollama pull llama3.1
```

Any model works; set `OLLAMA_MODEL` to its name (for example `OLLAMA_MODEL=llama3.2:3b` for a smaller one).

## Ingesting documents

Put your PDF/DOCX/TXT/MD files inside the `documents/` directory and run the ingestion command:

```
documents/
├── company_policy.pdf
├── employee_handbook.docx
└── faq.txt
```

```bash
python scripts/ingest_documents.py
```

This **deletes and rebuilds** the whole index, so it is safe to re-run after adding, changing or removing files. Sample documents about a fictional company are included; delete them and add your own.

If the index is empty when the app starts and `INGEST_ON_STARTUP=true`, the app ingests `documents/` by itself.

## Starting the application

```bash
uvicorn app.main:app --reload
```

## Using the chat interface

Open <http://localhost:8000> and ask, for example, "How many days of annual leave do employees get?".

API:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' \
     -d '{"message": "What is the vacation policy?"}'
# {"answer": "...", "sources": [{"document": "employee_handbook.docx", "page": null}]}
```

`page` is only set for PDFs. Errors return `{"detail": "<readable message>"}` with HTTP 4xx/5xx.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `together` |
| `TOGETHER_API_KEY` | _(empty)_ | Together AI API key. Required when `LLM_PROVIDER=together`. Set it as an environment variable only; never commit it |
| `TOGETHER_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` | Together model id. Check Together's current model list; free models change |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (when `LLM_PROVIDER=ollama`). **Must be changed in production.** |
| `OLLAMA_MODEL` | `llama3.1` | Model name as shown by `ollama list` |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | LLM request timeout |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model. Changing it requires re-ingesting |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | Where the index is stored |
| `DOCUMENTS_DIRECTORY` | `./documents` | Source documents |
| `TOP_K` | `5` | Chunks retrieved per question |
| `MAX_DISTANCE` | `0.9` | Cosine-distance cutoff; lower is stricter. With the default embedding model, related text scored about 0.4-0.85 and unrelated text 0.93+ on the sample data. Tune for your documents |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters per chunk / overlap. Re-ingest after changing |
| `INGEST_ON_STARTUP` | `true` | Build the index at startup when it is empty |
| `MAX_MESSAGE_CHARS` | `2000` | Maximum question length |
| `LOG_LEVEL` | `INFO` | Python log level |
| `PORT` | `8000` | Port to listen on (set automatically by Railway) |

## Running the tests

```bash
pytest
```

The tests need neither Ollama nor network access (a fake embedder and fake LLM are used).

## Docker

```bash
docker build -t document-chat .
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 document-chat
```

`host.docker.internal` reaches Ollama on your machine (on Linux add `--add-host=host.docker.internal:host-gateway`). The container honors `PORT`.

Or run the app and an Ollama container together:

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.1
```

## Railway Deployment

1. Push this repository to GitHub.
2. In [Railway](https://railway.com), create a **New Project -> Deploy from GitHub repo** and pick the repository. Railway builds from the `Dockerfile` (configured in `railway.toml`).
3. In the service **Variables** tab set the LLM variables: either `LLM_PROVIDER=together` + `TOGETHER_API_KEY` + `TOGETHER_MODEL` (simplest, see [Together AI](#together-ai-hosted-llm)), or `OLLAMA_BASE_URL` + `OLLAMA_MODEL` (see next section). Do not set `PORT`; Railway provides it.
4. Generate a public domain under **Settings -> Networking**.
5. Deploy. Check **Deployments -> View logs**: you should see `Vector store is empty; ingesting documents` followed by `Indexed N chunks`.
6. Test:
   ```bash
   curl https://<your-domain>/health
   curl -X POST https://<your-domain>/api/chat -H 'content-type: application/json' \
        -d '{"message": "What are the office hours?"}'
   ```

Memory: the image bundles CPU PyTorch and the embedding model, so give the service at least 1 GB of RAM.

**Persistence:** the Railway container filesystem is **ephemeral**; `data/chroma` is lost on every deploy or restart. That is why `INGEST_ON_STARTUP=true` rebuilds the index from the `documents/` folder baked into the image (a few seconds for small collections). To keep an index across restarts you can attach a Railway Volume mounted at `/app/data/chroma`, or later replace `app/rag/vector_store.py` with an external vector database. Documents are part of the repository/image, so adding documents means committing them and redeploying.

## Together AI (hosted LLM)

To use a hosted model instead of Ollama (no GPU or Ollama server needed), set:

```
LLM_PROVIDER=together
TOGETHER_API_KEY=<your key>
TOGETHER_MODEL=<model id from https://api.together.ai/models>
```

Create the key in the Together dashboard (Settings -> API keys). It is a long secret string shown once at creation; the short key *id* is not the key. Put it in Railway **Variables** or your shell environment, not in `.env.example` or any committed file. **Questions and retrieved document passages are sent to Together's API**, so use it only with documents you are allowed to share with that provider.

## Ollama in Production

**Railway cannot use the Ollama on your laptop**, and `http://localhost:11434` inside the Railway container points at the container itself, where no Ollama runs. Set `OLLAMA_BASE_URL` to an endpoint the service can reach:

- **A second Railway service running the `ollama/ollama` image** in the same project. Use its private address, e.g. `OLLAMA_BASE_URL=http://ollama.railway.internal:11434`, add a volume at `/root/.ollama` so models survive restarts, and pull the model once (`ollama pull llama3.1`, for example from a one-off shell on that service). Railway offers no GPUs, so use a small model (`llama3.2:3b` or similar) and expect slow answers on CPU.
- **A GPU machine you operate** (cloud VM or home server) running Ollama, exposed over a private network (Tailscale, WireGuard) or behind an authenticating reverse proxy.
- **A hosted Ollama-compatible service.** Note that this app currently sends no credentials; supporting an API key would be a small addition in `app/llm/ollama_client.py`.

Never expose a bare Ollama port to the public internet: it has no authentication.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "The LLM could not produce an answer" with Together | HTTP 401 in the logs means the API key is wrong; 404 means the model id is not available to your account; 429 means rate limited |
| "Unable to connect to the local LLM" | Ollama is not running or `OLLAMA_BASE_URL` is wrong. Check `curl $OLLAMA_BASE_URL/api/tags`. In Docker, `localhost` is the container, not your machine |
| "The LLM could not produce an answer" | Usually the model is not pulled (`model "x" not found` in the logs). Run `ollama pull <OLLAMA_MODEL>` |
| "No documents are currently available" / "knowledge base has not been initialized" | `documents/` is empty or the index was never built. Add files and run `python scripts/ingest_documents.py` |
| Answers say "I couldn't find this information" for things that are in the documents | Raise `MAX_DISTANCE` (e.g. `1.1`) or `TOP_K`; check the file was actually indexed (the ingest output lists each file) |
| Scanned PDFs return nothing | Only embedded text is extracted; there is no OCR |
| New documents not showing up | Re-run ingestion (the app only auto-ingests an *empty* index) |
| Railway deploy fails / restarts | Read the build/deploy logs. Out-of-memory kills need a bigger plan. The health check hits `/health` and allows 300 s for the first start |
| Port errors | The app must listen on `$PORT`; do not hard-code it in Railway variables |
| Index disappears after redeploy | Expected: the filesystem is ephemeral (see Railway Deployment) |

## Limitations

- Single-turn Q&A: no conversation history, no streaming, no authentication, no document upload through the UI.
- Source list shows every retrieved chunk that passed the distance cutoff, so it can include loosely related documents.
- Answer quality depends on the Ollama model you choose.
