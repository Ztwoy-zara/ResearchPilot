"""Small, offline lexical retriever for fetched webpage evidence.

This is sparse retrieval, not embedding or vector search. Retrieved text is
untrusted evidence and must never be treated as an instruction.
"""

import math
import re
from collections import Counter


_ENGLISH_STOPWORDS = frozenset(
    {"a", "an", "and", "are", "do", "does", "for", "how", "in", "is", "of", "the", "to", "what"}
)


def _terms(text: str) -> list[str]:
    """Tokenize English words and overlapping Chinese character pairs."""

    terms: list[str] = []
    for item in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if "\u4e00" <= item[0] <= "\u9fff":
            terms.extend(item[i : i + 2] for i in range(len(item) - 1))
        elif item not in _ENGLISH_STOPWORDS:
            terms.append(item)
    return terms


def _expanded_query_terms(query: str) -> set[str]:
    """Add known English terms for Chinese LangGraph concepts."""
    terms = set(_terms(query))
    if "状态" in query:
        terms.update({"state", "snapshot", "schema"})
    if "节点" in query:
        terms.add("nodes")
    if "边" in query:
        terms.add("edges")
    if "会话" in query or "线程" in query:
        terms.update({"thread", "checkpointer", "checkpointers", "checkpoints"})
    if "暂停" in query or "中断" in query:
        terms.update({"pause", "interrupt", "interrupts"})
    if "恢复" in query:
        terms.update({"resume", "command"})
    return terms



#把网页按段落切成短块，长段落继续切，避免整页进入模型
def chunk_webpage(text: str, *, chunk_size: int = 500, overlap: int = 60) -> list[str]:
    """Split paragraphs, then split long paragraphs into bounded chunks."""

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue
        step = chunk_size - overlap
        for start in range(0, len(paragraph), step):
            chunk = paragraph[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(paragraph):
                break
    return chunks


#用BM25根据问题给片段排序，只选相关且不超过长度预算的证据
def retrieve_webpage_passages(
    text: str,
    query: str,
    *,
    max_chars: int = 1200,
    top_k: int = 2,
) -> tuple[list[tuple[int, str]], bool]:
    """Rank webpage chunks by BM25 and keep a bounded evidence context.

    Returns ``(numbered_chunks, matched)``. A no-match fallback is explicitly
    marked so downstream users do not mistake it for a relevant retrieval.
    """

    if max_chars <= 0 or top_k <= 0:
        raise ValueError("max_chars and top_k must be positive")

    chunks = chunk_webpage(text, chunk_size=min(500, max_chars))
    if not chunks:
        return [], False

    query_terms = _expanded_query_terms(query)
    frequencies = [Counter(_terms(chunk)) for chunk in chunks]
    lengths = [sum(counts.values()) for counts in frequencies]
    average_length = sum(lengths) / len(lengths) or 1.0
    document_frequency = Counter(term for counts in frequencies for term in counts)
    scores: list[float] = []
    for counts, length in zip(frequencies, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            idf = math.log(1 + (len(chunks) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        scores.append(score)

    matched = max(scores) > 0
    ranked = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))
    ranked = [index for index in ranked if scores[index] > 0] if matched else ranked[:1]

    selected: list[tuple[int, str]] = []
    remaining = max_chars
    for index in ranked:
        if len(selected) >= top_k or remaining <= 0:
            break
        passage = chunks[index][:remaining]
        if passage:
            selected.append((index + 1, passage))
            remaining -= len(passage)
    return selected, matched
