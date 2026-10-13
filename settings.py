"""Central configuration for models and Ollama host."""
import os

# Ollama server URL
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Embedding model (locked; not changeable via UI)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Emb-4b")

# LLM model for synthesis (locked)
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2-3b-16k")
