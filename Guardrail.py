import warnings
warnings.filterwarnings("ignore")
from transformers import logging

logging.set_verbosity_error()

from transformers import pipeline


classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

SAFE_LABELS = ["pertanyaan sejarah", "pertanyaan olahraga", "pertanyaan alam"]
UNSAFE_LABELS = ["kasar", "penghinaan", "berbahaya"]

PANCASILA_KEYWORDS = [
    "pancasila", "sila", "garuda pancasila", "lambang pancasila",
    "bhineka tunggal ika", "simbol sila", "dasar negara",
    "gotong royong", "tanggung jawab", "toleransi", "jujur", "adil",
    "sopan santun", "disiplin", "cinta tanah air", "musyawarah",
    "rukun", "patuh", "menghormati", "menolong", "persatuan",
    "kerjasama", "keadilan", "ketuhanan", "kemanusiaan", "kerakyatan",
    "warga negara", "hak", "kewajiban", "norma", "peraturan",
    "etika", "aturan", "nasionalisme", "patriotisme", "kejujuran"
]

def classify_text(text: str):
    labels = SAFE_LABELS + UNSAFE_LABELS
    result = classifier(text, candidate_labels=labels)
    scores = dict(zip(result["labels"], result["scores"]))

    text_lower = text.lower()

    if any(keyword in text_lower for keyword in PANCASILA_KEYWORDS):
        if "pertanyaan sejarah" in scores:
            scores["pertanyaan sejarah"] += 0.25 
            if scores["pertanyaan sejarah"] > 1.0:
                scores["pertanyaan sejarah"] = 1.0

 
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]

    print("\n=== Hasil Skor Setiap Label ===")
    for label, score in scores.items():
        print(f"{label}: {score:.4f}")

    return top_label, top_score, scores


def validate_input(text: str, threshold: float = 0.2) -> bool:
    if not text.strip():
        print("Input kosong, tidak lolos.")
        return False

    top_label, top_score, _ = classify_text(text)

    if top_label in SAFE_LABELS and top_score > threshold:
        print(" Lolos guardrail.")
        return True
    else:
        print(" Tidak lolos guardrail.")
        return False

