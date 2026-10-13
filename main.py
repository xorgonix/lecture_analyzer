"""Simple CLI to index a directory and query a stored LlamaIndex using Ollama embeddings."""
import argparse
import os
from typing import List

"""Simple CLI to index a directory and query a stored LlamaIndex using Ollama embeddings."""
import argparse
import os
from typing import List

from ollama_embed import OllamaEmbedding
from settings import EMBEDDING_MODEL, LLM_MODEL

# Robust imports for llama-index (works with v0.13.2 layout)
try:
    # newer layout / core package
    from llama_index.core.readers.file.base import SimpleDirectoryReader
    from llama_index.core.indices.vector_store.base import GPTVectorStoreIndex
    from llama_index.core.service_context import ServiceContext
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.node_parser import SimpleNodeParser
except Exception:
    # fallback to legacy top-level imports
    try:
        from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext
        from llama_index.storage.storage_context import StorageContext
        from llama_index.node_parser import SimpleNodeParser
    except Exception:
        # last resort: import whatever is available; errors will be raised later if missing
        from llama_index import *


def build_index(data_dir: str, index_dir: str, model: str | None = None) -> None:
    """Load text files from data_dir, chunk, embed with Ollama, and store index locally.

    This implementation uses basic llama-index APIs but keeps imports lazy to avoid hard errors
    if the package isn't installed. It stores the index to disk.
    """
    # llama-index imports are handled at module top with robust fallbacks

    # Collect documents
    print(f"Reading files from: {data_dir}")
    reader = SimpleDirectoryReader(data_dir)
    docs = reader.load_data()

    # create an embedding wrapper that llama-index can call via a function
    model = model or EMBEDDING_MODEL
    emb = OllamaEmbedding(model=model)

    # Build a ServiceContext that uses our embedding function
    # Newer llama-index versions allow passing a custom embedding function; adapt if API differs
    def embed_fn(texts: List[str]) -> List[List[float]]:
        return emb.embed(texts)

    # Simple approach: use GPTVectorStoreIndex and pass embed fn using ServiceContext if supported
    try:
        service_context = ServiceContext.from_defaults(embed_model=embed_fn)
        index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    except Exception:
        # Fallback: try old API path
        index = GPTVectorStoreIndex.from_documents(docs)

    # Persist index
    os.makedirs(index_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=index_dir)
    print(f"Index saved to: {index_dir}")


def query_index(index_dir: str, model: str | None = None) -> None:
    try:
        from llama_index import StorageContext, load_index_from_storage
        from llama_index import ServiceContext
    except Exception:
        print("Error importing llama-index. Please install requirements from requirements.txt")
        raise

    model = model or EMBEDDING_MODEL
    emb = OllamaEmbedding(model=model)

    # load
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)

    # interactive loop
    print("Entering interactive query loop. Type 'exit' to quit.")
    while True:
        q = input("Query> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        try:
            # Use index.query; llama-index will call its stored embedder; if not, we can compute manually
            resp = index.query(q)
            print(resp)
        except Exception as e:
            print("Query failed:", e)


def parse_args():
    p = argparse.ArgumentParser(description="Lecture Analyzer: index and query using llama-index & Ollama")
    sub = p.add_subparsers(dest="cmd")

    ip = sub.add_parser("index", help="Create index from a data directory")
    ip.add_argument("--data-dir", required=True, help="Directory containing text files to index")
    ip.add_argument("--index-dir", default="./index_store", help="Directory to persist index")
    ip.add_argument("--model", default="llama2", help="Ollama model name to use for embeddings")

    qp = sub.add_parser("query", help="Query an existing index")
    qp.add_argument("--index-dir", default="./index_store", help="Directory where index is stored")
    qp.add_argument("--model", default="llama2", help="Ollama model name to use for embeddings")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "index":
        build_index(args.data_dir, args.index_dir, model=args.model)
    elif args.cmd == "query":
        query_index(args.index_dir, model=args.model)
    else:
        print("No command supplied. Use 'index' or 'query'.")


if __name__ == "__main__":
    main()
