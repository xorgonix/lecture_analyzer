from pathlib import Path
import json
import argparse


def chunk_with_overlap(text, size=1000, overlap=200):
	if size <= overlap:
		raise ValueError('size must be greater than overlap')
	chunks = []
	i = 0
	n = len(text)
	while i < n:
		end = min(n, i + size)
		chunk = text[i:end]
		chunks.append((i, end, chunk))
		if end == n:
			break
		i = end - overlap
	return chunks


def write_chunks(index, outdir):
	out = Path(outdir)
	out.mkdir(parents=True, exist_ok=True)
	idx = []
	for j, (start, end, txt) in enumerate(index, start=1):
		name = f'fine_chunk_{j:04d}.txt'
		(out / name).write_text(txt, encoding='utf-8')
		preview = txt.strip().splitlines()[0][:140]
		idx.append({'file': name, 'start': start, 'end': end, 'chars': len(txt), 'preview': preview})
	(out / 'index.json').write_text(json.dumps(idx, indent=2), encoding='utf-8')
	return idx


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('src', help='Source file to chunk')
	parser.add_argument('--outdir', default='chunks_fine_overlap', help='Output directory')
	parser.add_argument('--size', type=int, default=1000, help='Chunk size in characters')
	parser.add_argument('--overlap', type=int, default=200, help='Overlap in characters between chunks')
	args = parser.parse_args()

	src = Path(args.src)
	text = src.read_text(encoding='utf-8')
	index = chunk_with_overlap(text, size=args.size, overlap=args.overlap)
	metadata = write_chunks(index, args.outdir)
	print(f'Wrote {len(metadata)} chunks to {args.outdir}')


if __name__ == '__main__':
	main()

