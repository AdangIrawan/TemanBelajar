
import os, re, sys
from typing import List, Optional, Set
import fitz  
import pytesseract
from PIL import Image
from io import BytesIO

PDF_PATH   = r"D:\Webchatbot\Dataset\Penjas\PJOK_BS_KLS_VI.pdf"
OUTPUT_TXT = r"D:\Webchatbot\Dataset\Penjas\Clean\Penjas Kelas VI.txt"
SKIP_PAGES = list(range(1, 22)) + list(range(200, 211)) + list(range(213, 226)) 
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 
OCR_LANG   = "ind+eng"
DPI        = 300  


if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|edu|gov|go|id|co)\S*)",
    flags=re.IGNORECASE,
)
BAB_LINE_RE = re.compile(
    r"^\s*(?:bab|BAB)\s*(?:[0-9]+|[IVXLCDM]+)\s*(?:[:\-–—]\s*.*)?\s*$"
)
BAB_PREFIX_RE = re.compile(
    r"^\s*(?:bab|BAB)\s*(?:[0-9]+|[IVXLCDM]+)\s*(?:[:\-–—]\s*)?",
    flags=re.IGNORECASE,
)

def clean_text(text: str) -> str:
    text = URL_RE.sub("", text or "")

    text = text.replace("\t", " ")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7EÀ-ÿ]", "", text)

    cleaned_lines: List[str] = []
    for raw_ln in text.splitlines():
        ln = re.sub(r"\s+", " ", raw_ln).strip()
        if not ln:
            continue

        if BAB_LINE_RE.match(ln):
            continue

        ln = BAB_PREFIX_RE.sub("", ln).strip()

        if not ln:
            continue

        cleaned_lines.append(ln)

    text_out = "\n".join(cleaned_lines).strip()
    return text_out

def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes))

def ocr_page(img: Image.Image, lang: str) -> str:
    return clean_text(pytesseract.image_to_string(img, lang=lang))

def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF tidak ditemukan: {PDF_PATH}")
        sys.exit(1)

    doc = fitz.open(PDF_PATH)
    total = doc.page_count
    skip: Set[int] = set(SKIP_PAGES or [])

    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)

    results: List[str] = []
    skipped = 0
    kept = 0

    print(f"[*] Total halaman: {total} | DPI render: {DPI}")
    for page_num in range(1, total + 1): 
        if page_num in skip:
            skipped += 1
            print(f"Halaman {page_num} dilewati.")
            continue

        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = pixmap_to_pil(pix)

        print(f"Halaman {page_num}: OCR …")
        try:
            txt = ocr_page(img, OCR_LANG)
        except Exception as e:
            print(f"[!] OCR gagal halaman {page_num}: {e}")
            txt = ""

        if txt.strip():
            results.append(txt.strip())
            kept += 1
        else:
            print(f"Halaman {page_num}: hasil kosong/pendek.")

    doc.close()

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for t in results:
            if not t.strip():
                continue
            f.write(t + "\n\n")

    print("\nRingkasan:")
    print(f"- Total halaman       : {total}")
    print(f"- Dilewati (skip)     : {skipped}")
    print(f"- Tersimpan (non-skip): {kept}")
    print(f"[*] Output: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
