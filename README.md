# Lecture Analyzer - LlamaIndex + Ollama starter

This starter app shows how to use LlamaIndex (llama-index) to chunk and index local text files and use a local Ollama server to produce embeddings.

Requirements

- Python 3.10+
- A running Ollama server (default: http://localhost:11434) that exposes an /embed endpoint.
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Quick start

Index a directory (creates an index directory `./index_store` by default):

```bash
python main.py index --data-dir ./data --index-dir ./index_store
```

Run an interactive query loop against the stored index:

```bash
python main.py query --index-dir ./index_store
```

Notes

- This is a minimal starter. Depending on your `llama-index` version you may need to adapt imports or API usage.
- Currently the loader only handles plain text and markdown files; add PDF/other loaders as needed.

Additional configuration notes

- Web UI / API: the web UI and HTTP endpoints are locked to the embedding and LLM models defined in `settings.py` (EMBEDDING_MODEL and LLM_MODEL). This prevents users of the HTTP UI from changing which Ollama model is used.
- CLI: the local command-line tool (`main.py`) still accepts a `--model` flag for convenience and local testing. Use that flag only on trusted local machines; it does not affect the web UI.
