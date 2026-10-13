from pathlib import Path
import json
import argparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

try:
    from ollama_embed import OllamaEmbedding
    OLLAMA_AVAILABLE = True
except Exception:
    OllamaEmbedding = None
    OLLAMA_AVAILABLE = False


def load_chunks(dirpath):
    p = Path(dirpath)
    files = sorted([f for f in p.glob('fine_chunk_*.txt')])
    texts = [f.read_text(encoding='utf-8') for f in files]
    names = [f.name for f in files]
    return names, texts


def load_index_json(dirpath):
    p = Path(dirpath) / 'index.json'
    if not p.exists():
        return None
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def embed_texts(texts):
    if OLLAMA_AVAILABLE:
        emb = OllamaEmbedding()
        return emb.embed(texts)
    # fallback: TF-IDF vectors
    vec = TfidfVectorizer(max_features=4096)
    X = vec.fit_transform(texts)
    return X.toarray()


def cluster_texts(vectors, k=6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(vectors)
    return labels, kmeans


def summarize_cluster(texts, labels, cluster_id, top_n=3):
    members = [t for t, l in zip(texts, labels) if l == cluster_id]
    # naive summary: return first lines of top_n members
    return [m.strip().splitlines()[0][:140] for m in members[:top_n]]


def map_cluster_to_offsets(cluster_members, index_json):
    # cluster_members: list of dicts with 'idx' and 'file'
    if not index_json:
        return None
    # index_json is list of entries; ensure we can index by file name
    file_map = {entry['file']: entry for entry in index_json}
    starts = []
    ends = []
    for m in cluster_members:
        ent = file_map.get(m['file'])
        if ent:
            starts.append(ent.get('start'))
            ends.append(ent.get('end'))
    if not starts:
        return None
    return min(starts), max(ends)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='chunks_fine_overlap')
    parser.add_argument('--k', type=int, default=6)
    args = parser.parse_args()

    names, texts = load_chunks(args.dir)
    print(f'Loaded {len(texts)} chunks from {args.dir}')
    vectors = embed_texts(texts)
    labels, model = cluster_texts(vectors, k=args.k)

    clusters = {}
    for i, (n, t, lab) in enumerate(zip(names, texts, labels)):
        clusters.setdefault(lab, []).append({'idx': i, 'file': n, 'preview': t.strip().splitlines()[0][:140]})

    index_json = load_index_json(args.dir)

    out = []
    for lab in sorted(clusters.keys()):
        members = clusters[lab]
        idxs = [m['idx'] for m in members]
        mapped = map_cluster_to_offsets(members, index_json)
        section = {
            'cluster': int(lab),
            'count': len(members),
            'min_idx': min(idxs),
            'max_idx': max(idxs),
            'previews': [m['preview'] for m in members[:3]]
        }
        if mapped:
            section['start_char'], section['end_char'] = mapped
        out.append(section)

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
