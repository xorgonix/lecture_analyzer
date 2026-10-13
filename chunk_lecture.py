"""Small helper to split a lecture text file into N balanced, paragraph-aware chunks.

Usage:
    python chunk_lecture.py path/to/lecture.txt --parts 6 --outdir ./chunks

This creates files chunk_01.txt ... chunk_N.txt in the output directory.
"""
import argparse
from pathlib import Path


def split_into_paragraphs(text: str):
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paras:
        # fallback: break on single newlines
        paras = [p.strip() for p in text.split('\n') if p.strip()]
    return paras


def chunk_paragraphs(paragraphs, parts: int):
    # Distribute paragraphs into `parts` buckets trying to balance character length
    totals = [0] * parts
    buckets = [[] for _ in range(parts)]
    for p in paragraphs:
        # put paragraph into the bucket with minimal total length
        idx = min(range(parts), key=lambda i: totals[i])
        buckets[idx].append(p)
        totals[idx] += len(p)
    return ['\n\n'.join(b) for b in buckets if b]


def chunk_by_size(text: str, parts: int):
    """Split text into `parts` contiguous character-balanced chunks, trying to cut at newline or space boundaries."""
    if parts <= 1:
        return [text]
    length = len(text)
    approx = max(1, length // parts)
    chunks = []
    start = 0
    for i in range(parts - 1):
        target = start + approx
        # try to find a nearby newline first, then a space, else use the character boundary
        cut = None
        for j in range(target, min(length, target + 500)):
            if text[j] == '\n':
                cut = j + 1
                break
        if cut is None:
            for j in range(target, min(length, target + 200)):
                if text[j].isspace():
                    cut = j + 1
                    break
        if cut is None:
            cut = min(length, target)
        chunks.append(text[start:cut].strip())
        start = cut
    # final
    last = text[start:].strip()
    if last:
        chunks.append(last)
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('infile', help='Path to lecture text file')
    p.add_argument('--parts', type=int, default=6, help='Number of chunks to split into')
    p.add_argument('--outdir', default='./chunks', help='Output directory for chunk files')
    args = p.parse_args()

    path = Path(args.infile)
    if not path.exists():
        print('Input file not found:', path)
        raise SystemExit(2)

    text = path.read_text(encoding='utf-8')
    paras = split_into_paragraphs(text)
    chunks = chunk_paragraphs(paras, args.parts)

    # If paragraph-aware splitting produced fewer chunks than requested,
    # fall back to a simple contiguous character split to reach the desired count.
    if len(chunks) < args.parts:
        chunks = chunk_by_size(text, args.parts)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(chunks, start=1):
        fname = out / f'chunk_{i:02d}.txt'
        fname.write_text(c, encoding='utf-8')
        print('Wrote', fname)

    print(f'Done — created {len(chunks)} chunk files in {out}')


if __name__ == '__main__':
    main()
