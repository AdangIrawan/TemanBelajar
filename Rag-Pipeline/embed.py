import os, json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ===== PATH =====
JSON_PATH   = "D:\Webchatbot\Dataset\Penjas\Chunk\penjas_chunks.json"
OUTPUT_DIR  = "D:\Webchatbot\Dataset\Penjas\Embedd"
OUTPUT_NAME = "penjas_embeddings.npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ===== LOAD CHUNKS =====
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
texts = [item["text"] for item in data]
print(f"Total chunk: {len(texts)}")

# ===== MODEL =====
MODEL_NAME = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.no_grad()
def get_embeddings(texts, batch_size=32, max_length=512):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = [f"passage: {t}" for t in texts[i:i+batch_size]]  
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        outputs = model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embs.append(pooled.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"Processed: {i+len(batch)}/{len(texts)}")
    return torch.cat(embs, dim=0)

embeddings = get_embeddings(texts, batch_size=32)
print(f"Embeddings shape: {embeddings.shape}")

output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
np.save(output_path, embeddings.numpy())
print(f"Embeddings disimpan ke: {output_path}")
