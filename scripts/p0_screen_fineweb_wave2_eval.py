#!/usr/bin/env python3
"""Narrow exact-phrase screen of deduplicated Wave2 against frozen evaluations.

Matches are quarantined for review, not silently deleted. This cannot detect
paraphrases, answer-only leakage, or benchmarks absent from the reference set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re


WORD = re.compile(r"\w+", re.UNICODE)
REFERENCE_FILES = {
    "arc_easy_validation": ("ai2_arc/ARC-Easy/validation-00000-of-00001.parquet",
                            "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74"),
    "arc_easy_test": ("ai2_arc/ARC-Easy/test-00000-of-00001.parquet",
                      "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"),
    "hellaswag_validation": ("hellaswag/data/validation-00000-of-00001.parquet",
                              "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def words(text: str) -> tuple[str, ...]:
    return tuple(WORD.findall(text.casefold()))


def frozen_phrases(reference: Path):
    import pyarrow.parquet as pq

    files = {}
    phrases = []
    skipped = Counter()
    for task, (relative, expected) in REFERENCE_FILES.items():
        path = reference / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen evaluation file hash mismatch: {path}")
        files[task] = {"relative_path": relative, "sha256": actual,
                       "bytes": path.stat().st_size}
        for row in pq.read_table(path).to_pylist():
            if task.startswith("arc_easy"):
                phrase = words(row["question"])
                if len(phrase) < 6:
                    skipped[task] += 1
                    continue
                phrases.append((task, str(row["id"]), phrase))
            else:
                for choice, ending in enumerate(row["endings"]):
                    phrase = words(row["ctx"] + " " + ending)
                    if len(phrase) < 12:
                        skipped[task] += 1
                        continue
                    phrases.append((task, f"{row['ind']}:{choice}", phrase))
    return phrases, files, dict(skipped)


class AnchorAutomaton:
    """Aho-Corasick over <=8-word anchors; verify full phrases at a hit."""

    def __init__(self, phrases: list[tuple[str, str, tuple[str, ...]]]):
        self.word_id = {}
        self.transitions = [{}]
        self.failure = [0]
        self.outputs = [[]]
        self.phrases = phrases
        for phrase_id, (_, _, phrase) in enumerate(phrases):
            anchor = phrase[:8]
            node = 0
            for word in anchor:
                token = self.word_id.setdefault(word, len(self.word_id) + 1)
                child = self.transitions[node].get(token)
                if child is None:
                    child = len(self.transitions)
                    self.transitions[node][token] = child
                    self.transitions.append({})
                    self.failure.append(0)
                    self.outputs.append([])
                node = child
            self.outputs[node].append((phrase_id, len(anchor)))
        queue = deque(self.transitions[0].values())
        while queue:
            node = queue.popleft()
            for token, child in self.transitions[node].items():
                queue.append(child)
                fallback = self.failure[node]
                while fallback and token not in self.transitions[fallback]:
                    fallback = self.failure[fallback]
                self.failure[child] = self.transitions[fallback].get(token, 0)
                self.outputs[child].extend(self.outputs[self.failure[child]])

    def find(self, document: tuple[str, ...]):
        node = 0
        seen = set()
        for end, word in enumerate(document):
            token = self.word_id.get(word, 0)
            while node and token not in self.transitions[node]:
                node = self.failure[node]
            node = self.transitions[node].get(token, 0)
            for phrase_id, anchor_len in self.outputs[node]:
                start = end - anchor_len + 1
                task, item_id, phrase = self.phrases[phrase_id]
                if (phrase_id not in seen and
                        document[start:start + len(phrase)] == phrase):
                    seen.add(phrase_id)
                    yield task, item_id, len(phrase)


def scan_one(root: str, reference: str, output: str, filename: str,
             excluded_sha256: str) -> dict:
    import pyarrow.parquet as pq

    root, reference, output = Path(root), Path(reference), Path(output)
    phrases, reference_files, skipped = frozen_phrases(reference)
    automaton = AnchorAutomaton(phrases)
    index = root / "audit/near_dedup_v1/sketch_wave2" / f"{filename}.sketch.parquet"
    index_done = json.loads(index.with_suffix(".done.json").read_text())
    if sha256(index) != index_done["output_sha256"]:
        raise ValueError(f"sketch index integrity failure: {index}")
    exclusions = root / "audit/near_dedup_v1/verified_v1/wave2_exclusion_index.parquet"
    if sha256(exclusions) != excluded_sha256:
        raise ValueError("near-duplicate exclusion index changed")
    exclude_rows = {
        row["raw_row"] for row in pq.read_table(exclusions, columns=["source_file", "raw_row"]).to_pylist()
        if Path(row["source_file"]).name == filename
    }
    pointers = {}
    for batch in pq.ParquetFile(index).iter_batches(batch_size=8192,
                                                    columns=["raw_row", "content_sha256",
                                                             "tokenizer_v1_tokens"]):
        for row in batch.to_pylist():
            if row["raw_row"] not in exclude_rows:
                pointers[row["raw_row"]] = row
    if len(pointers) + len(exclude_rows) != index_done["rows"]:
        raise ValueError(f"deduplicated pointer count mismatch: {filename}")
    output.mkdir(parents=True, exist_ok=True)
    hit_path = output / f"{filename}.hits.jsonl"
    marker = output / f"{filename}.done.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if (old["index_sha256"] != index_done["output_sha256"]
                or old["exclusion_sha256"] != excluded_sha256
                or old["hits_sha256"] != sha256(hit_path)):
            raise ValueError(f"completed screen integrity mismatch: {filename}")
        return old
    if hit_path.exists():
        raise FileExistsError(hit_path)
    temp = hit_path.with_suffix(".jsonl.tmp")
    temp.unlink(missing_ok=True)
    scanned = scanned_tokens = raw_rows = 0
    hit_documents = set()
    hits_by_task = Counter()
    try:
        with temp.open("w", encoding="utf-8") as dest:
            pf = pq.ParquetFile(root / "raw" / filename)
            for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
                for value in batch.column(0).to_pylist():
                    pointer = pointers.get(raw_rows)
                    if pointer is not None:
                        if not isinstance(value, str):
                            raise ValueError("missing raw text during evaluation screen")
                        text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
                        if hashlib.sha256(text.encode()).hexdigest() != pointer["content_sha256"]:
                            raise ValueError("raw text hash mismatch during evaluation screen")
                        scanned += 1
                        scanned_tokens += pointer["tokenizer_v1_tokens"]
                        for task, item_id, length in automaton.find(words(text)):
                            hit_documents.add(pointer["content_sha256"])
                            hits_by_task[task] += 1
                            dest.write(json.dumps({
                                "content_sha256": pointer["content_sha256"],
                                "source_file": filename, "raw_row": raw_rows,
                                "tokenizer_v1_tokens": pointer["tokenizer_v1_tokens"],
                                "task": task, "eval_item_id": item_id,
                                "matched_words": length,
                                "disposition": "quarantine_for_review_not_automatic_drop",
                            }, sort_keys=True) + "\n")
                    raw_rows += 1
        if scanned != len(pointers):
            raise ValueError("evaluation screen coverage mismatch")
        result = {"stage": "narrow_exact_phrase_screen_not_full_decontamination",
                  "file": filename, "raw_rows_scanned": raw_rows,
                  "documents_scanned": scanned, "tokenizer_v1_tokens_scanned": scanned_tokens,
                  "hit_documents": len(hit_documents), "hits_by_task": dict(hits_by_task),
                  "index_sha256": index_done["output_sha256"],
                  "exclusion_sha256": excluded_sha256,
                  "reference_files": reference_files,
                  "phrases_screened": len(phrases), "short_phrases_skipped": skipped,
                  "automaton_nodes": len(automaton.transitions),
                  "hits_sha256": sha256(temp)}
        temp.replace(hit_path)
        marker.write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    root, reference, output = (p.resolve() for p in (args.root, args.reference, args.output))
    near = json.loads((root / "audit/near_dedup_v1/verified_v1/SUMMARY.json").read_text())
    exclusion_sha256 = near["outputs_sha256"]["wave2_exclusion_index.parquet"]
    names = [Path(item["path"]).name for item in json.loads((root / "manifest.json").read_text())["items"]]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(scan_one, str(root), str(reference), str(output), name,
                             exclusion_sha256) for name in names]
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": len(results), "total": len(names),
                              "file": result["file"], "hit_documents": result["hit_documents"]}),
                  flush=True)
    results.sort(key=lambda x: x["file"])
    documents = sum(x["documents_scanned"] for x in results)
    tokens = sum(x["tokenizer_v1_tokens_scanned"] for x in results)
    expected_documents = near["wave2_candidates"] - near["wave2_excluded_documents"]
    gate = json.loads((root / "audit/quality_gate_v1/REPORT.json").read_text())
    expected_tokens = gate["post_domain_tokenizer_v1_tokens"] - near["wave2_excluded_tokens"]
    if documents != expected_documents or tokens != expected_tokens:
        raise ValueError("screen total differs from near-deduplicated population")
    hits = Counter()
    quarantine = {}
    for item in results:
        hits.update(item["hits_by_task"])
        hit_path = output / f"{item['file']}.hits.jsonl"
        if sha256(hit_path) != item["hits_sha256"]:
            raise ValueError(f"screen hit receipt mismatch: {hit_path}")
        for line in hit_path.open():
            hit = json.loads(line)
            key = (hit["source_file"], hit["raw_row"])
            pointer = {name: hit[name] for name in
                       ("content_sha256", "source_file", "raw_row", "tokenizer_v1_tokens")}
            if key in quarantine and quarantine[key] != pointer:
                raise ValueError(f"conflicting quarantine pointer: {key}")
            quarantine[key] = pointer
    if sum(hits.values()) != sum(1 for item in results
                                  for _ in (output / f"{item['file']}.hits.jsonl").open()):
        raise ValueError("hit count differs from shard receipts")
    import pyarrow as pa
    import pyarrow.parquet as pq

    quarantine_path = output / "quarantine_index.parquet"
    quarantine_tmp = output / "quarantine_index.parquet.tmp"
    schema = pa.schema([
        ("content_sha256", pa.string()), ("source_file", pa.string()),
        ("raw_row", pa.int64()), ("tokenizer_v1_tokens", pa.int64()),
    ])
    rows = [quarantine[key] for key in sorted(quarantine)]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), quarantine_tmp,
                   compression="zstd")
    quarantine_tmp.replace(quarantine_path)
    summary = {"stage": "narrow_exact_phrase_eval_screen_not_full_decontamination",
               "documents_scanned": documents, "tokenizer_v1_tokens_scanned": tokens,
               "hit_documents_quarantined": len(quarantine),
               "hit_document_tokens_quarantined": sum(x["tokenizer_v1_tokens"] for x in rows),
               "quarantine_index_sha256": sha256(quarantine_path),
               "hits_by_task": dict(hits),
               "reference_files": results[0]["reference_files"],
               "phrases_screened": results[0]["phrases_screened"],
               "short_phrases_skipped": results[0]["short_phrases_skipped"],
               "per_shard": results,
               "limitations": ["only frozen ARC-Easy validation/test and HellaSwag validation",
                               "exact contiguous word phrases only", "short ARC questions skipped",
                               "no paraphrase, answer-only, math/code benchmark screening"],
               "hits_are_quarantine_not_automatic_exclusion": True}
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "documents_scanned": documents,
                      "hit_documents_quarantined": len(quarantine),
                      "hits_by_task": dict(hits)}), flush=True)


if __name__ == "__main__":
    main()
