import io
import json
from pathlib import Path
from typing import Union

def read_pdf(file_bytes: bytes, filename: str = "file.pdf") -> dict:
    """Extract text and metadata from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        full_text = []

        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i + 1, "text": text[:3000]})
                full_text.append(f"--- Page {i+1} ---\n{text}")

        metadata = doc.metadata or {}
        doc.close()

        combined = "\n\n".join(full_text)
        return {
            "filename": filename,
            "type": "pdf",
            "pages": len(pages),
            "metadata": {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
            },
            "content": combined[:12000],
            "page_previews": pages[:5],
            "total_chars": len(combined)
        }
    except ImportError:
        return {"error": "PyMuPDF not installed. Run: pip install PyMuPDF", "filename": filename}
    except Exception as e:
        return {"error": str(e), "filename": filename}


def read_csv(file_bytes: bytes, filename: str = "file.csv") -> dict:
    """Parse a CSV file and return summary + preview."""
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(file_bytes))

        # Basic stats
        stats = {}
        for col in df.select_dtypes(include="number").columns:
            stats[col] = {
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "mean": round(float(df[col].mean()), 4),
                "nulls": int(df[col].isna().sum())
            }

        return {
            "filename": filename,
            "type": "csv",
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(10).to_dict(orient="records"),
            "stats": stats,
            "nulls_total": int(df.isna().sum().sum())
        }
    except ImportError:
        return {"error": "pandas not installed. Run: pip install pandas", "filename": filename}
    except Exception as e:
        return {"error": str(e), "filename": filename}


def read_text(file_bytes: bytes, filename: str = "file.txt") -> dict:
    """Read plain text files."""
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return {
            "filename": filename,
            "type": "text",
            "lines": len(lines),
            "characters": len(text),
            "content": text[:10000],
            "preview": "\n".join(lines[:20])
        }
    except Exception as e:
        return {"error": str(e), "filename": filename}


def read_json(file_bytes: bytes, filename: str = "file.json") -> dict:
    """Parse JSON files."""
    try:
        data = json.loads(file_bytes.decode("utf-8"))
        preview = json.dumps(data, indent=2)[:5000]
        return {
            "filename": filename,
            "type": "json",
            "keys": list(data.keys()) if isinstance(data, dict) else None,
            "length": len(data) if isinstance(data, (list, dict)) else None,
            "content": preview
        }
    except Exception as e:
        return {"error": str(e), "filename": filename}


def dispatch_file(file_bytes: bytes, filename: str) -> dict:
    """Route file to correct reader based on extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return read_pdf(file_bytes, filename)
    elif ext == ".csv":
        return read_csv(file_bytes, filename)
    elif ext == ".json":
        return read_json(file_bytes, filename)
    elif ext in (".txt", ".md", ".log", ".py", ".js", ".html", ".css"):
        return read_text(file_bytes, filename)
    else:
        return {"error": f"Unsupported file type: {ext}", "filename": filename}
