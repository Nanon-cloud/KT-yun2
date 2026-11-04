"""Local PDF knowledge retrieval and Gemini LLM explanation layer.

PDFs and the TF-IDF index remain on disk in ./data. Only the question, selected
quantitative metrics, and a small set of retrieved text excerpts are sent to
Gemini when the user asks for an AI explanation.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "index"
MAX_PDF_BYTES = 25 * 1024 * 1024
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 250


def ensure_data_dirs() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _clean_filename(name: str) -> str:
    basename = Path(name or "research.pdf").name
    normalized = unicodedata.normalize("NFKD", basename).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    if not cleaned:
        cleaned = "research.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:150]


def chunk_page_text(text: str, chunk_chars: int = CHUNK_CHARS,
                    overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring whitespace boundaries."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    if chunk_chars < 100 or overlap < 0 or overlap >= chunk_chars:
        raise ValueError("Chunk size must be >= 100 and overlap must be smaller than chunk size.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_chars // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def index_pdf_bytes(filename: str, payload: bytes) -> dict:
    """Extract page text and create a local JSON index for one PDF."""
    ensure_data_dirs()
    safe_name = _clean_filename(filename)
    if not payload:
        raise ValueError(f"{safe_name}: the uploaded file is empty.")
    if len(payload) > MAX_PDF_BYTES:
        raise ValueError(f"{safe_name}: file exceeds the 25 MB upload limit.")
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"{safe_name}: this does not look like a valid PDF file.")

    digest = hashlib.sha256(payload).hexdigest()
    index_path = INDEX_DIR / f"{digest}.json"
    stored_path = PAPERS_DIR / f"{digest[:12]}_{safe_name}"

    if index_path.exists() and stored_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        return {"status": "already_exists", "filename": existing["filename"],
                "pages": existing["page_count"], "chunks": len(existing["chunks"]),
                "sha256": digest}

    try:
        reader = PdfReader(BytesIO(payload), strict=False)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("PDF is password-protected and cannot be indexed.") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Could not read {safe_name}: {exc}") from exc

    chunks: list[dict] = []
    extracted_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            extracted_pages += 1
            for chunk_number, chunk in enumerate(chunk_page_text(text), start=1):
                chunks.append({
                    "text": chunk,
                    "source": safe_name,
                    "page": page_number,
                    "chunk": chunk_number,
                })

    if not chunks:
        raise ValueError(
            f"{safe_name}: no selectable text was found. This may be a scanned PDF; OCR is not included in this version."
        )

    stored_path.write_bytes(payload)
    record = {
        "filename": safe_name,
        "sha256": digest,
        "stored_filename": stored_path.name,
        "page_count": len(reader.pages),
        "pages_with_text": extracted_pages,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunks,
    }
    index_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return {"status": "indexed", "filename": safe_name,
            "pages": len(reader.pages), "chunks": len(chunks), "sha256": digest}


def list_library_documents() -> list[dict]:
    ensure_data_dirs()
    documents: list[dict] = []
    for index_path in sorted(INDEX_DIR.glob("*.json")):
        try:
            record = json.loads(index_path.read_text(encoding="utf-8"))
            stored_path = PAPERS_DIR / record.get("stored_filename", "")
            documents.append({
                "filename": record.get("filename", stored_path.name),
                "sha256": record.get("sha256", index_path.stem),
                "pages": int(record.get("page_count", 0)),
                "pages_with_text": int(record.get("pages_with_text", 0)),
                "chunks": len(record.get("chunks", [])),
                "size_bytes": stored_path.stat().st_size if stored_path.exists() else 0,
                "indexed_at": record.get("indexed_at", ""),
                "stored_filename": record.get("stored_filename", ""),
            })
        except (OSError, ValueError, TypeError):
            continue
    return documents


def delete_library_document(sha256: str) -> bool:
    """Delete a paper and its index using its content hash, never a path input."""
    if not re.fullmatch(r"[a-f0-9]{64}", sha256 or ""):
        raise ValueError("Invalid document identifier.")
    ensure_data_dirs()
    index_path = INDEX_DIR / f"{sha256}.json"
    if not index_path.exists():
        return False
    try:
        record = json.loads(index_path.read_text(encoding="utf-8"))
        stored_name = Path(record.get("stored_filename", "")).name
        if stored_name:
            stored_path = PAPERS_DIR / stored_name
            if stored_path.exists():
                stored_path.unlink()
        index_path.unlink()
        return True
    except (OSError, ValueError, TypeError):
        return False


def load_all_chunks() -> list[dict]:
    ensure_data_dirs()
    all_chunks: list[dict] = []
    for index_path in INDEX_DIR.glob("*.json"):
        try:
            record = json.loads(index_path.read_text(encoding="utf-8"))
            all_chunks.extend(record.get("chunks", []))
        except (OSError, ValueError, TypeError):
            continue
    return all_chunks


def retrieve_relevant_chunks(query: str, top_k: int = 5,
                              chunks: list[dict] | None = None) -> list[dict]:
    """Retrieve matching PDF chunks using local TF-IDF and cosine similarity."""
    query = (query or "").strip()
    corpus = chunks if chunks is not None else load_all_chunks()
    corpus = [c for c in corpus if isinstance(c.get("text"), str) and c["text"].strip()]
    if not query or not corpus:
        return []

    texts = [c["text"] for c in corpus]
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, matrix).ravel()
    except ValueError:
        return []

    ranked = similarities.argsort()[::-1]
    results: list[dict] = []
    for idx in ranked:
        score = float(similarities[idx])
        if score <= 0:
            continue
        result = dict(corpus[int(idx)])
        result["similarity"] = score
        results.append(result)
        if len(results) >= max(1, min(int(top_k), 10)):
            break
    return results


def make_quant_context(scores, source_note: str, start_date, end_date,
                       risk_free_rate: float, is_demo: bool) -> str:
    """Create a compact, transparent summary of calculated quantitative metrics."""
    columns = [
        "Quant score", "Relative assessment", "Period return", "6M momentum",
        "Sharpe ratio", "Annualised volatility", "Maximum drawdown", "Price trend",
    ]
    available = [c for c in columns if c in scores.columns]
    top = scores[available].head(12).copy()
    context = [
        f"Data source: {source_note}",
        f"Data period: {start_date} to {end_date}",
        f"Annual risk-free-rate assumption: {risk_free_rate:.2%}",
        f"Synthetic demonstration data: {'YES' if is_demo else 'NO'}",
        "The quant score is a peer-relative factor score, not a probability or forecast.",
        "Top of currently selected universe (up to 12 stocks):",
        top.to_string(),
    ]
    return "\n".join(context)


def generate_gemini_answer(api_key: str, model: str, question: str,
                           quant_context: str,
                           retrieved_chunks: list[dict],
                           conversation: list[dict] | None = None) -> str:
    """Call Gemini with quantitative metrics and locally retrieved paper excerpts."""
    if not api_key or not api_key.strip():
        raise ValueError("Enter your Gemini API key in the AI Assistant settings first.")
    if not question or not question.strip():
        raise ValueError("Enter a question first.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini SDK is missing. Run: python -m pip install -r requirements.txt") from exc

    if retrieved_chunks:
        paper_context = "\n\n".join(
            f"[SOURCE: {c.get('source', 'Unknown document')} | PAGE: {c.get('page', '?')} | RETRIEVAL SCORE: {c.get('similarity', 0):.3f}]\n{c['text']}"
            for c in retrieved_chunks
        )
    else:
        paper_context = "No relevant excerpts were retrieved from the uploaded research-paper library. Do not claim to have consulted any paper."

    recent_history = []
    for item in (conversation or [])[-6:]:
        role = "User" if item.get("role") == "user" else "Assistant"
        recent_history.append(f"{role}: {item.get('content', '')}")
    history_text = "\n".join(recent_history) if recent_history else "No prior conversation."

    system_instruction = (
        "You are a cautious quantitative-finance research assistant. You explain calculations and research, "
        "not provide personalized financial advice. Never claim that a score guarantees returns or that a stock "
        "is certain to rise. Do not invent market data, calculations, facts, or paper citations. Distinguish "
        "quantitative metrics computed by the application from claims in retrieved research excerpts and from "
        "your own general explanation. If the data is synthetic, clearly say so. If context is insufficient, "
        "state what is missing. Discuss limitations such as overfitting, transaction costs, survivorship bias, "
        "look-ahead bias, data quality, and the fact that historical performance does not guarantee future results. "
        "When using a retrieved passage, cite it inline as [Document filename, p. X]. Do not create a citation "
        "unless that source appears in the supplied excerpts."
    )

    prompt = f"""Answer the user's question using the evidence below.

CURRENT QUANTITATIVE APPLICATION RESULTS:
{quant_context}

LOCALLY RETRIEVED RESEARCH-PAPER EXCERPTS:
{paper_context}

RECENT CONVERSATION:
{history_text}

USER QUESTION:
{question}

Suggested response structure where appropriate:
1. Direct answer.
2. What the quantitative results indicate (refer to exact tickers and values supplied; do not recalculate or invent them).
3. Relevant research-paper evidence with page citations.
4. Key risks, uncertainty, and what additional analysis is needed.

If the selected universe uses demo data, do not make claims about real stocks. Do not issue an unconditional buy/sell instruction."""

    try:
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=(model or "gemini-3.8-flash").strip(),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.25,
                max_output_tokens=1600,
            ),
        )
    except Exception as exc:
        message = str(exc).replace(api_key.strip(), "[API KEY REDACTED]")
        raise RuntimeError(f"Gemini request failed: {message[:700]}") from exc

    answer = getattr(response, "text", None)
    if not answer:
        raise RuntimeError("Gemini returned an empty response. Check the model name, API key, or safety response.")
    return answer.strip()
