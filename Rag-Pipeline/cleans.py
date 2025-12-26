import os
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


def extract_and_clean_pdf(path: str, skip_pages: list[int] = None) -> list[str]:
    skip_pages = skip_pages or []
    cleaned_pages = []

    for i, page_layout in enumerate(extract_pages(path), start=1):
        if i in skip_pages:
            print(f"Halaman {i} dilewati.")
            continue

        page_text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()

        cleaned_text = clean_text(page_text)
        cleaned_pages.append(cleaned_text)

    print(f"\nTotal halaman diambil: {len(cleaned_pages)} halaman (dari {i} total halaman).")
    return cleaned_pages


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r'[^\x20-\x7EÀ-ÿ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()


def save_cleaned_text(cleaned_pages: list[str], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for page in cleaned_pages:
            f.write(page + "\n\n")

    print(f"File teks berhasil disimpan ke:\n{output_path}")


if __name__ == "__main__":
    pdf_path = "D:\Webchatbot\Dataset\Penjas\PJOK_BS_KLS_V.pdf"
    output_txt = "D:\Webchatbot\Dataset\Penjas\Clean\Penjas Kelas V.txt"

    halaman_dihapus = []+ list(range(1,15)) + list(range(188,208))
    hasil = extract_and_clean_pdf(pdf_path, skip_pages=halaman_dihapus)

    if hasil:
        save_cleaned_text(hasil, output_txt)
    else:
        print("Tidak ada halaman yang diekstrak.")