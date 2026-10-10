from __future__ import annotations

# ask AI for help
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import List, Tuple

from policy_loader import load_policy_documents

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = r"D:/Downloud/pyCharm/py_GEX3/models/qwen3-0.6b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MIN_SCORE = 0.35


def _split_documents(docs, chunk_size: int = 400, overlap: int = 60):
    chunks = []
    for d in docs:
        text = d.page_content
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append({
                "text": text[start:end],
                "policy_id": d.metadata.get("policy_id"),
                "topic": d.metadata.get("topic"),
                "source": d.metadata.get("source"),
            })
            if end == len(text):
                break
            start = end - overlap
    return chunks


_embedder = None
_index = None
_chunks: List[dict] = []


def _build_index():
    global _embedder, _index, _chunks
    if _index is not None:
        return

    documents = load_policy_documents()
    _chunks = _split_documents(documents)

    _embedder = SentenceTransformer(EMBED_MODEL)
    vectors = _embedder.encode(
        [c["text"] for c in _chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _index = np.asarray(vectors, dtype="float32")


def retrieve(query: str, top_k: int = TOP_K) -> List[Tuple[dict, float]]:
    _build_index()
    q = _embedder.encode([query], normalize_embeddings=True)[0]
    scores = _index @ q
    order = np.argsort(-scores)[:top_k]
    return [(_chunks[i], float(scores[i])) for i in order]


_llm = None
_tok = None


def _load_llm():
    global _llm, _tok
    if _llm is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    _llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        local_files_only=True,
    )
    if torch.cuda.is_available():
        _llm = _llm.to("cuda")
    _llm.eval()


SYSTEM = (
    "You are a university IT support assistant. "
    "Answer ONLY using the context provided below. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know. The university IT policies do not provide that information.\" "
    "Do NOT use any outside knowledge. Do NOT guess. "
    "Never invent a phone number or any other fact."
)


def _generate(prompt: str, max_new_tokens: int = 200) -> str:
    import torch
    _load_llm()
    inputs = _tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = _llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tok.eos_token_id,
        )
    text = _tok.decode(out[0], skip_special_tokens=True)
    # keep only the assistant part after the last "Assistant:"
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text.strip()


REFUSAL = "I don't know. The university IT policies do not provide that information."


def answer(query: str) -> str:
    hits = retrieve(query)
    best_chunk, best_score = hits[0]
    print(f"[retrieval] best score = {best_score:.3f} "
          f"({best_chunk['policy_id']} / {best_chunk['topic']})")

    if best_score < MIN_SCORE:
        return REFUSAL

    context = "\n\n".join(
        f"[{c['policy_id']} | {c['topic']}]\n{c['text']}"
        for c, s in hits if s >= MIN_SCORE
    )

    prompt = (
        f"{SYSTEM}\n\n"
        f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        f"User: {query}\n\nAssistant:"
    )

    out = _generate(prompt)

    if not out:
        return REFUSAL
    import re
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", out):
        return REFUSAL
    if "i don't know" in out.lower() or "not available" in out.lower():
        return REFUSAL
    return out


if __name__ == "__main__":
    q = "What exact phone number should I call for the University IT Service Desk?"
    print("Q:", q)
    print("A:", answer(q))