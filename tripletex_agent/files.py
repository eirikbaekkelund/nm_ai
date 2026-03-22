import base64
import csv
import io
from typing import Dict, Any
from pdfminer.high_level import extract_text as pdf_extract_text
from PIL import Image
import pytesseract

def make_claude_file_block(content_base64: str, mime_type: str):
    if mime_type.startswith("image/"):
        block_type = "image"
    else:
        block_type = "document"

    return {
        "type": block_type,
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": content_base64
        }
    }

def parse_file(file_obj) -> Any:
    """
    Parse a single uploaded file (CSV, PDF, image).
    Returns structured or text data usable by LLM.
    """
    raw_bytes = base64.b64decode(file_obj.content_base64)
    mime_type = getattr(file_obj, "mime_type", "").lower()
    filename = getattr(file_obj, "filename", "unknown")

    if mime_type == "text/csv":
        # CSV → structured rows
        text = raw_bytes.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text), delimiter=";")
        rows = [row for row in reader]
        return {"type": "csv", "filename": filename, "data": rows}

    elif mime_type == "application/pdf":
        # PDF → extract text
        with io.BytesIO(raw_bytes) as pdf_file:
            text = pdf_extract_text(pdf_file)
        return {"type": "pdf", "filename": filename, "data": text}

    elif mime_type in ["image/png", "image/jpeg", "image/jpg"]:
        # Image → OCR text
        with io.BytesIO(raw_bytes) as img_file:
            img = Image.open(img_file)
            text = pytesseract.image_to_string(img)
        return {"type": "image", "filename": filename, "data": text}

    else:
        # Unknown → just return raw bytes in base64
        return {"type": "raw", "filename": filename, "data": file_obj["content_base64"]}
    

def parse_files(files: list) -> list[dict]:
    """
    Process multiple files and return structured data for LLM.
    """
    all_files = []
    for f in files:
        parsed = parse_file(f)
        all_files.append(parsed)
    return all_files