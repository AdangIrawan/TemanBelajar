from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import glob
import json
import os


folder_path = "D:\Webchatbot\Dataset\Penjas\Clean"
file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

pages = []
for path in file_paths:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        pages.append(Document(page_content=text, metadata={"source": path}))

print(f" Total file terbaca: {len(file_paths)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

documents = text_splitter.split_documents(pages)
all_texts = [doc.page_content for doc in documents]


output_dir = "D:\Webchatbot\Dataset\Penjas\Chunk"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "penjas_chunks.json")

data_to_save = [
    {"id": i + 1, "text": chunk}
    for i, chunk in enumerate(all_texts)
]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=2)

print(f"Hasil chunk disimpan ke: {os.path.abspath(output_path)}")

for i, chunk in enumerate(all_texts[:3]):
    print(f"\n--- Chunk {i+1} ---\n{chunk}")
