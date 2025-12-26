# model.py
from llama_cpp import Llama

def load_model(path, n_ctx=2048, n_gpu_layers=-1, n_threads=14):
    return Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads)

def generate(llm, prompt, max_tokens=384, temperature=0.2, top_p=0.9, stop=None):
    stop = stop or []
    out = llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
    return out["choices"][0]["text"].strip()
