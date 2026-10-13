from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from fastapi.staticfiles import StaticFiles
import threading
import os
import shutil

from main import build_index
from ollama_embed import OllamaEmbedding
import numpy as np
from sklearn.cluster import KMeans

app = FastAPI(title="Lecture Analyzer API")

# serve a simple static HTML UI
APP_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)


INDEX_HTML = os.path.join(TEMPLATES_DIR, "index.html")
if not os.path.exists(INDEX_HTML):
    # write a simple htmx-based UI
    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(r"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Lecture Analyzer</title>
    <script src="https://unpkg.com/htmx.org@1.10.0"></script>
    <style>body{font-family:system-ui,Segoe UI,Arial;margin:20px}textarea{width:100%;height:160px}</style>
  </head>
  <body>
    <h1>Lecture Analyzer — HTMX UI</h1>

    <h2>Add / Paste Text</h2>
    <form hx-post="/add_text" hx-target="#add-result" hx-swap="innerHTML">
      <input type="text" name="title" placeholder="Title (optional)" />
      <br/>
      <textarea name="text" placeholder="Paste lecture transcript or notes here"></textarea>
      <br/>
      <input type="text" name="index_dir" value="./index_store" />
      <button type="submit">Add Text</button>
    </form>
    <div id="add-result"></div>

    <h2>Upload File</h2>
    <form id="upload-form" hx-post="/upload" hx-target="#upload-result" hx-encoding="multipart/form-data">
      <input type="file" name="file" />
      <input type="text" name="index_dir" value="./index_store" />
      <button type="submit">Upload & Index</button>
    </form>
    <div id="upload-result"></div>

    <h2>Query</h2>
    <form hx-post="/query_form" hx-target="#results" hx-swap="innerHTML">
      <input type="text" name="index_dir" value="./index_store" />
      <input type="text" name="query" placeholder="Ask a question..." />
      <button type="submit">Query</button>
    </form>

    <div id="results"></div>
  </body>
</html>
""")


@app.get("/", response_class=HTMLResponse)
def ui_index():
    with open(INDEX_HTML, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


from settings import EMBEDDING_MODEL, LLM_MODEL


@app.post("/add_text")
async def add_text(title: Optional[str] = Form(None), text: str = Form(...), index_dir: str = Form("./index_store")):
    """Add pasted text as a new document and index it (create index if missing)."""
    try:
        # create a small temp file under index_dir/input_texts
        os.makedirs(index_dir, exist_ok=True)
        save_dir = os.path.join(index_dir, "_uploads")
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, (title or "paste").replace(" ", "_")[:100] + ".txt")
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write(text)

        # Use build_index to index the new file(s) in the provided directory
        # Run in background to avoid blocking the response
        def target():
            try:
                build_index(save_dir, index_dir, model=EMBEDDING_MODEL)
            except Exception as e:
                print("Indexing failed:", e)

        threading.Thread(target=target, daemon=True).start()
        return HTMLResponse(f"<div>Added and indexing started: {fname}</div>")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), index_dir: str = Form("./index_store")):
    """Upload a file and index it. Only basic text-like files are handled in the starter (txt, md)."""
    try:
        os.makedirs(index_dir, exist_ok=True)
        save_dir = os.path.join(index_dir, "_uploads")
        os.makedirs(save_dir, exist_ok=True)
        dest = os.path.join(save_dir, os.path.basename(file.filename))
        with open(dest, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        # If it's a text file, we can index it via build_index on the folder
        def target():
            try:
                build_index(save_dir, index_dir, model=EMBEDDING_MODEL)
            except Exception as e:
                print("Indexing failed:", e)

        threading.Thread(target=target, daemon=True).start()
        return HTMLResponse(f"<div>Uploaded and indexing started: {dest}</div>")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chunk_synthesize")
async def chunk_synthesize(title: Optional[str] = Form(None), text: str = Form(...), parts: int = Form(6), index_dir: str = Form("./index_store")):
    """Split a long lecture into `parts` semantic chunks, synthesize a short title and 1-paragraph summary for each, and save as documents for indexing."""
    try:
        # Best-effort use of llama-index splitter if available
        try:
            from llama_index import TextSplitter
            splitter = TextSplitter()
            chunks = splitter.split_text(text)
        except Exception:
            # fallback: naive paragraph split and group into `parts`
            paras = [p.strip() for p in text.split('\n\n') if p.strip()]
            if not paras:
                paras = [text]
            # group paras into roughly equal parts
            k = max(1, len(paras) // parts)
            chunks = []
            for i in range(0, len(paras), k):
                chunks.append('\n\n'.join(paras[i:i+k]))

        # If there are more chunks than requested parts, merge to get roughly `parts` pieces
        if len(chunks) > parts:
            merged = []
            per = max(1, len(chunks) // parts)
            for i in range(0, len(chunks), per):
                merged.append('\n\n'.join(chunks[i:i+per]))
            chunks = merged

        # Synthesize titles/summaries for each chunk using a tiny heuristic or llama-index LLM if available
        synthesized = []
        for idx, c in enumerate(chunks[:parts]):
            # simple heuristics: title = first sentence up to 8 words
            first_sent = c.split('.\n')[0].split('.')
            title_guess = (first_sent[0].strip().split()[:8])
            title_txt = ' '.join(title_guess) + ("..." if len(title_guess) >= 8 else "")
            summary = c.strip().replace('\n', ' ')[:400]
            # create a provocative "what-if" prompt
            what_if = f"What if... {summary.split('.')[0]}?"
            synthesized.append({"title": title_txt or f"Part {idx+1}", "summary": summary, "what_if": what_if, "text": c})

        # Save synthesized chunks
        os.makedirs(index_dir, exist_ok=True)
        chunk_dir = os.path.join(index_dir, "_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        saved = []
        for i, s in enumerate(synthesized):
            fname = os.path.join(chunk_dir, f"chunk_{i+1:02d}.txt")
            with open(fname, "w", encoding="utf-8") as fh:
                fh.write(f"Title: {s['title']}\n\nSummary: {s['summary']}\n\nWhat-if: {s['what_if']}\n\nText:\n{s['text']}")
            saved.append(fname)

        # Kick off indexing of the chunk_dir in background
        def target():
            try:
                build_index(chunk_dir, index_dir, model=EMBEDDING_MODEL)
            except Exception as e:
                print("Indexing chunks failed:", e)

        threading.Thread(target=target, daemon=True).start()

        # Return an HTML summary for the UI
        html = ["<h3>Chunks created</h3><ol>"]
        for s, f in zip(synthesized, saved):
            html.append(f"<li><strong>{s['title']}</strong><div><em>{s['summary']}</em></div><div>file: {f}</div></li>")
        html.append("</ol>")
        return HTMLResponse('\n'.join(html))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment_preview")
async def segment_preview(text: str = Form(...), parts: int = Form(6), window_size: int = Form(400), step: int = Form(200)):
    """Create candidate spans via sliding window, compute Ollama embeddings, cluster them into `parts` clusters, and return an HTML preview with offsets and sample text."""
    try:
        # produce sliding windows with char offsets
        spans = []
        for start in range(0, max(1, len(text) - 1), step):
            end = min(len(text), start + window_size)
            span = text[start:end].strip()
            if span:
                spans.append({"start": start, "end": end, "text": span})
            if end == len(text):
                break

        if not spans:
            return HTMLResponse("<div>No spans generated from text</div>")

        # compute embeddings in batches
        emb = OllamaEmbedding(model=EMBEDDING_MODEL)
        batch_size = 16
        vectors = []
        for i in range(0, len(spans), batch_size):
            batch_texts = [s["text"] for s in spans[i : i + batch_size]]
            vs = emb.embed(batch_texts)
            vectors.extend(vs)

        arr = np.array(vectors)
        k = max(1, min(parts, len(arr)))
        km = KMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(arr)

        clusters = {i: [] for i in range(k)}
        for idx, l in enumerate(labels):
            clusters[l].append(spans[idx])

        # Build HTML preview showing cluster membership and min/max offsets
        html = ["<h3>Segmentation preview</h3>", "<div>Generated spans: {} | Clusters: {}</div>".format(len(spans), k), "<ol>"]
        for cid, items in clusters.items():
            if not items:
                continue
            starts = [it["start"] for it in items]
            ends = [it["end"] for it in items]
            html.append(f"<li><strong>Cluster {cid}</strong> - span count: {len(items)} - offsets: {min(starts)}..{max(ends)}<ul>")
            for it in items[:6]:
                txt = it["text"].replace('<', '&lt;')[:300]
                html.append(f"<li>offsets {it['start']}..{it['end']}: <pre style='white-space:pre-wrap'>{txt}</pre></li>")
            html.append("</ul></li>")
        html.append("</ol>")
        html.append("<div>Use chunk & synthesize to persist chosen clusters as documents.</div>")
        return HTMLResponse('\n'.join(html))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _extract_citations(resp) -> List[dict]:
    """Best-effort extraction of citations from llama-index response objects."""
    out = []
    try:
        # try common helper
        if hasattr(resp, "get_formatted_sources"):
            src = resp.get_formatted_sources()
            return [{"source": s} for s in [src]]

        # source_nodes is common in many response objects
        if hasattr(resp, "source_nodes") and resp.source_nodes:
            for n in resp.source_nodes:
                try:
                    text = None
                    if hasattr(n, "node") and hasattr(n.node, "get_text"):
                        text = n.node.get_text()
                    elif hasattr(n, "text"):
                        text = n.text
                    else:
                        text = str(n)
                    out.append({"text": text, "source": getattr(n, "source", None)})
                except Exception:
                    out.append({"text": str(n)})
            return out
    except Exception:
        pass
    return out


@app.post("/query_form")
async def query_form(index_dir: str = Form("./index_store"), query: str = Form(...)):
    """HTMX-friendly query endpoint that returns HTML snippet with answer + citations."""
    try:
        from llama_index import StorageContext, load_index_from_storage

        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        resp = index.query(query)

        citations = _extract_citations(resp)
        html_parts = []
        html_parts.append(f"<div><strong>Answer:</strong><div>{str(resp)}</div></div>")
        if citations:
            html_parts.append("<h3>Citations</h3><ul>")
            for c in citations:
                text = c.get("text") or c.get("source") or ""
                html_parts.append(f"<li><pre style='white-space:pre-wrap'>{text}</pre></li>")
            html_parts.append("</ul>")
        return HTMLResponse("\n".join(html_parts))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

