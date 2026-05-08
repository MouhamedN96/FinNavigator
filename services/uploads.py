"""File upload + RAG indexing.

Saves the file to data/uploads/ and indexes its text into the existing Chroma
knowledge base used by KnowledgeBaseSearchTool. Supports .txt, .md, .pdf, .docx
out of the box (pypdf and docx2txt are already in requirements.txt).
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("finnav.uploads")

UPLOAD_DIR = Path(os.getenv("FINNAV_UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        if suffix == ".docx":
            import docx2txt
            return docx2txt.process(str(path)) or ""
        # Default: read as utf-8 text (covers .txt, .md, .csv, .jsonl, …)
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.warning("Couldn't extract text from %s: %s", path, e)
        return ""


def _chunk(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Crude text splitter — same overlap as the SEC chunker. Avoids importing
    langchain just for this."""
    if not text:
        return []
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_size)
        chunk = text[i:end]
        out.append(chunk)
        if end == n:
            break
        i = max(0, end - overlap)
    return out


def save_and_index(
    filename: str,
    data: bytes,
    vectorstore: Any,
    embeddings: Any,
) -> Dict[str, Any]:
    """Save bytes to disk, extract text, chunk + index into the vectorstore.
    Returns metadata: {filename, path, size, chunks, indexed}."""
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()")
    target = UPLOAD_DIR / safe_name
    target.write_bytes(data)
    size = target.stat().st_size

    indexed = False
    chunks_n = 0
    err: Optional[str] = None

    if vectorstore is None:
        err = "vectorstore not initialised"
    else:
        text = _read_text(target)
        chunks = _chunk(text)
        chunks_n = len(chunks)
        if chunks:
            try:
                metadatas = [
                    {
                        "source": safe_name,
                        "chunk": idx,
                        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                    for idx in range(len(chunks))
                ]
                vectorstore.add_texts(texts=chunks, metadatas=metadatas)
                indexed = True
            except Exception as e:
                err = f"index_failed: {type(e).__name__}: {e}"
                log.exception("Indexing failed for %s", safe_name)

    return {
        "filename": safe_name,
        "path": str(target),
        "size": size,
        "chunks": chunks_n,
        "indexed": indexed,
        "error": err,
    }


def list_uploads() -> List[Dict[str, Any]]:
    """Files currently in the upload dir, with size + mtime."""
    if not UPLOAD_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(UPLOAD_DIR.iterdir(), key=lambda x: -x.stat().st_mtime):
        if p.is_file():
            stat = p.stat()
            out.append({
                "filename": p.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            })
    return out


def remove(filename: str) -> bool:
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()")
    target = UPLOAD_DIR / safe_name
    if target.exists() and target.is_file():
        target.unlink()
        return True
    return False
