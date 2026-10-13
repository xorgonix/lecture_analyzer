import requests
from typing import List
from settings import OLLAMA_HOST, EMBEDDING_MODEL


class OllamaEmbedding:
    """Simple embedding wrapper that calls a local Ollama server.

    It tries POST /embed with JSON {"model": model, "input": texts} and
    supports a couple of common response shapes returned by embedding APIs.

    Usage:
        emb = OllamaEmbedding(host="http://localhost:11434", model="llama2")
        vectors = emb.embed(["hello world", "second text"])
    """

    def __init__(self, host: str = None, model: str = None):
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.model = model or EMBEDDING_MODEL

    def _call_ollama(self, inputs: List[str]) -> dict:
        # try common endpoints used by different Ollama installs
        payload = {"model": self.model, "input": inputs}
        urls = [f"{self.host}/embed", f"{self.host}/api/embed"]
        last_exc = None
        for url in urls:
            try:
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                continue
        # if both failed, raise the last exception
        raise last_exc

    def _parse_response(self, resp: dict, n: int) -> List[List[float]]:
        # Try common shapes: {embedding: [...]} or {data:[{embedding:[...]}]} or {embeddings: [...]} 
        if isinstance(resp, dict):
            if "embedding" in resp and isinstance(resp["embedding"], list):
                # single embedding returned
                if n == 1:
                    return [resp["embedding"]]
            if "data" in resp and isinstance(resp["data"], list):
                out = []
                for item in resp["data"]:
                    if isinstance(item, dict) and "embedding" in item:
                        out.append(item["embedding"])
                if len(out) == n:
                    return out
            if "embeddings" in resp and isinstance(resp["embeddings"], list):
                return resp["embeddings"]
        # As a fallback, try if response itself is a list of embeddings
        if isinstance(resp, list):
            return resp

        raise ValueError(f"Unexpected embedding response shape: {resp}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return list of vectors."""
        if not texts:
            return []
        resp = self._call_ollama(texts)
        return self._parse_response(resp, len(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        out = self.embed([text])
        return out[0]
