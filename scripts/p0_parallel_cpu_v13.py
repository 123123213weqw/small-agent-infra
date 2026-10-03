"""Bounded, ordered CPU work; no quality decisions, ledger writes or API calls."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import heapq
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
_tokenizer = None


def initialize(tokenizer=None):
    global _tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if tokenizer:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(str(tokenizer))


@contextmanager
def executor(workers, tokenizer=None):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('CPU workers must be 1..16')
    if workers == 1:
        initialize(tokenizer)
        yield None
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(tokenizer,)) as pool:
            yield pool


def ordered_map(pool, fn, iterable, inflight):
    """Unlike Executor.map on Python <3.14, never exhaust the input eagerly."""
    if inflight < 1:
        raise ValueError('Invalid prefetch bound')
    if pool is None:
        for item in iterable:
            yield fn(item)
        return
    iterator = iter(iterable)
    pending = deque()
    for _ in range(inflight):
        try: pending.append(pool.submit(fn, next(iterator)))
        except StopIteration: break
    while pending:
        yield pending.popleft().result()  # completion order cannot change corpus order
        try: pending.append(pool.submit(fn, next(iterator)))
        except StopIteration: pass


def parquet_chunks(path, size=64, columns=None):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=columns, use_threads=False):
        yield batch.to_pylist()


def encode_chunk(rows):
    import numpy as np
    result = []
    for r in rows:
        ids = _tokenizer.encode(r['text'], add_special_tokens=False).ids
        if len(ids) != r['own_text_tokens'] or _tokenizer.decode(ids, skip_special_tokens=False) != r['text']:
            raise ValueError('Token count/roundtrip differs')
        if any(not 0 <= t < 32768 or t == 2 for t in ids):
            raise ValueError('Invalid token or EOS collision')
        result.append((r['normalized_sha256'], r['split'], r['domain'], np.asarray(ids, dtype='<u2').tobytes()))
    return result


def encoded_documents(parquet, tokenizer, workers):
    import numpy as np
    from small_agent.data.sharded_pack import TokenizedDocument
    with executor(workers, str(tokenizer)) as pool:
        for chunk in ordered_map(pool, encode_chunk, parquet_chunks(parquet), workers * 2):
            for h, split, domain, data in chunk:
                yield TokenizedDocument(h, split, domain, np.frombuffer(data, dtype='<u2').tolist())


def verify_chunk(items):
    import numpy as np
    for text, data in items:
        ids = np.frombuffer(data, dtype='<u2').tolist()
        if (_tokenizer.decode(ids, skip_special_tokens=False) != text or
                _tokenizer.encode(text, add_special_tokens=False).ids != ids):
            raise ValueError('Packed document does not match full source')
    return len(items)


class CodecChecks:
    """Full-source checks with at most 2*workers chunks of 64 documents buffered."""
    def __init__(self, tokenizer, workers):
        self.workers = workers
        self.context = executor(workers, str(tokenizer))
        self.pending = deque()
        self.chunk = []
        self.checked = 0

    def __enter__(self):
        self.pool = self.context.__enter__()
        return self

    def submit(self):
        if not self.chunk: return
        chunk, self.chunk = self.chunk, []
        if self.pool is None: self.checked += verify_chunk(chunk)
        else:
            self.pending.append(self.pool.submit(verify_chunk, chunk))
            if len(self.pending) >= self.workers * 2:
                self.checked += self.pending.popleft().result()

    def add(self, text, packed_bytes):
        self.chunk.append((text, packed_bytes))
        if len(self.chunk) == 64: self.submit()

    def finish(self):
        self.submit()
        while self.pending: self.checked += self.pending.popleft().result()
        return self.checked

    def __exit__(self, *args):
        try:
            if args[0] is None: self.finish()
        finally: self.context.__exit__(*args)


def sketch_chunk(rows):
    from small_agent.curation.release_gates_v5 import sha, lexical
    out = []
    for r in rows:
        if sha(r['text']) != r['normalized_sha256']:
            raise ValueError('Inventory body hash differs')
        values = lexical(r['text'])
        out.append(dict(body_sha256=r['normalized_sha256'], split=r['split'],
                        shingle_count=len(values), sketch=heapq.nsmallest(64, values)))
    return out


def build_sketch(source, entry, root, workers):
    """Same v5 bottom64 cache schema/identity; hash both before and after reading."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from small_agent.curation.release_gates_v5 import write
    from small_agent.curation.supply_control import digest, file_sha
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    key = digest(entry); dest = root / (key + '.sketch.parquet'); marker = root / (key + '.done.json')
    if marker.exists():
        import json
        m = json.loads(marker.read_text())
        if m['input'] != entry or file_sha(dest) != m['sha256']:
            raise ValueError('Sketch cache changed')
        return m
    if file_sha(source) != entry['sha256']: raise ValueError('Inventory source changed')
    schema = pa.schema([('body_sha256', pa.string()), ('split', pa.string()), ('row', pa.int64()),
                        ('shingle_count', pa.int32()), ('sketch', pa.list_(pa.uint64()))])
    tmp = dest.with_suffix('.v13.tmp'); n = 0
    try:
        with executor(workers) as pool, pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
            for chunk in ordered_map(pool, sketch_chunk,
                                     parquet_chunks(source, columns=['text', 'normalized_sha256', 'split']), workers * 2):
                for r in chunk: r['row'] = n; n += 1
                writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
        if file_sha(source) != entry['sha256']: raise ValueError('Inventory changed during indexing')
        os.replace(tmp, dest)
        m = dict(input=entry, path=str(dest), sha256=file_sha(dest), documents=n)
        write(marker, m)
        return m
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
