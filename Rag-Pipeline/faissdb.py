import numpy as np
import faiss
import os

embeddings_path = "D:\Webchatbot\Dataset\Penjas\Embedd\penjas_embeddings.npy"

output_dir = "D:\Webchatbot\Rag-Pipeline\Vektor Database\Penjas"

embeddings_np = np.load(embeddings_path)
print(f"Embeddings shape: {embeddings_np.shape}")

dimension = embeddings_np.shape[1]  
index = faiss.IndexFlatL2(dimension) 
index.add(embeddings_np)
print(f"Total vectors di FAISS: {index.ntotal}")

faiss_index_path = os.path.join(output_dir, "PENJAS_index.index")
faiss.write_index(index, faiss_index_path)
print(f"FAISS index disimpan ke: {faiss_index_path}")

