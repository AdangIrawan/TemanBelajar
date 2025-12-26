import os, json, re, time, logging
from functools import lru_cache
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from dotenv import load_dotenv
load_dotenv()

import secrets

def generate_token():
    return secrets.token_urlsafe(32)


# ========= ENV & LOGGING =========
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rag-app")

# ========= IMPORT EKSTERNAL =========
from Guardrail import validate_input
from Model import load_model, generate

# ========= KONFIGURASI RAG =========
MODEL_PATH   = r"models\DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"
CTX_WINDOW   = 4096
N_GPU_LAYERS = -1
N_THREADS    = 4

ENCODER_NAME   = "intfloat/multilingual-e5-large"
ENCODER_DEVICE = torch.device("cpu")

SUBJECTS: Dict[str, Dict[str, str]] = {
    "ipas": {
        "index":      r"D:\Webchatbot\Rag-Pipeline\Vektor Database\Ipas\IPA_index.index",
        "chunks":     r"D:\Webchatbot\Dataset\Ipas\Chunk\ipas_chunks.json",
        "embeddings": r"D:\Webchatbot\Dataset\Ipas\Embedd\ipas_embeddings.npy",
        "label":      "IPAS",
        "desc":       "Ilmu Pengetahuan Alam dan Sosial"
    },
    "penjas": {
        "index":      r"D:\Webchatbot\Rag-Pipeline\Vektor Database\Penjas\PENJAS_index.index",
        "chunks":     r"D:\Webchatbot\Dataset\Penjas\Chunk\penjas_chunks.json",
        "embeddings": r"D:\Webchatbot\Dataset\Penjas\Embedd\penjas_embeddings.npy",
        "label":      "PJOK",
        "desc":       "Pendidikan Jasmani, Olahraga, dan Kesehatan"
    },
    "pancasila": {
        "index":      r"D:\Webchatbot\Rag-Pipeline\Vektor Database\Pancasila\PANCASILA_index.index",
        "chunks":     r"D:\Webchatbot\Dataset\Pancasila\Chunk\pancasila_chunks.json",
        "embeddings": r"D:\Webchatbot\Dataset\Pancasila\Embedd\pancasila_embeddings.npy",
        "label":      "PANCASILA",
        "desc":       "Pendidikan Pancasila dan Kewarganegaraan"
    }
}

# Threshold dan fallback
TOP_K_FAISS = 24
TOP_K_FINAL = 10
MIN_COSINE  = 0.84
MIN_HYBRID  = 0.15

FALLBACK_TEXT        = "maaf pengetahuan tidak ada dalam database"
GUARDRAIL_BLOCK_TEXT = "maaf, pertanyaan ditolak oleh guardrail"
ENABLE_PROFILING     = False

# ========= APP =========
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-please-change")

# ========= GLOBAL MODEL =========
ENCODER_TOKENIZER = None
ENCODER_MODEL = None
LLM = None

@dataclass(frozen=True)
class SubjectAssets:
    index: faiss.Index
    texts: List[str]
    embs: np.ndarray

# ========= TEKS UTILITAS =========
STOPWORDS_ID = {
    "yang","dan","atau","pada","di","ke","dari","itu","ini","adalah","dengan",
    "untuk","serta","sebagai","oleh","dalam","akan","kamu","apa","karena",
    "agar","sehingga","terhadap","dapat","juga","para","diri",
}
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

def tok_id(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS_ID]
    
def lexical_overlap(query: str, sent: str) -> float:
    q = set(tok_id(query)); s = set(tok_id(sent))
    if not q or not s: return 0.0
    return len(q & s) / max(1, len(q | s))

QUESTION_LIKE_RE = re.compile(r"(^\s*(apa|mengapa|bagaimana|sebutkan|jelaskan)\b|[?]$)", re.IGNORECASE)
INSTRUCTION_RE   = re.compile(r"\b(jelaskan|sebutkan|uraikan|kerjakan|diskusikan|tugas|latihan|menurut\s+pendapatmu)\b", re.IGNORECASE)
META_PREFIX_PATTERNS = [
    r"berdasarkan\s+(?:kalimat|sumber|teks|konten|informasi)(?:\s+(?:di\s+atas|tersebut))?",
    r"menurut\s+(?:sumber|teks|konten)",
    r"merujuk\s+pada",
    r"mengacu\s+pada",
    r"bersumber\s+dari",
    r"dari\s+(?:kalimat|sumber|teks|konten)"
]
META_PREFIX_RE = re.compile(r"^\s*(?:" + r"|".join(META_PREFIX_PATTERNS) + r")\s*[:\-–—,]?\s*", re.IGNORECASE)

def clean_prefix(t: str) -> str:
    t = (t or "").strip()
    for _ in range(5):
        t2 = META_PREFIX_RE.sub("", t).lstrip()
        if t2 == t: break
        t = t2
    return t

def strip_meta_sentence(s: str) -> str:
    s = clean_prefix(s or "")
    if re.match(r"^\s*(berdasarkan|menurut|merujuk|mengacu|bersumber|dari)\b", s, re.IGNORECASE):
        s = re.sub(r"^\s*[^,.;!?]*[,.;!?]\s*", "", s) or s
        s = clean_prefix(s)
    return s.strip()

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
def split_sentences(text: str) -> List[str]:
    outs = []
    for p in SENT_SPLIT_RE.split(text or ""):
        s = clean_prefix((p or "").strip())
        if not s: continue
        if s[-1] not in ".!?": s += "."
        if QUESTION_LIKE_RE.search(s):  continue
        if INSTRUCTION_RE.search(s):    continue
        if len(s.strip()) < 10:         continue
        outs.append(s)
    return outs

# ========= MODEL WARMUP =========
def warmup_models():
    global ENCODER_TOKENIZER, ENCODER_MODEL, LLM
    if ENCODER_TOKENIZER is None or ENCODER_MODEL is None:
        log.info(f"[INIT] Load encoder: {ENCODER_NAME} (CPU)")
        ENCODER_TOKENIZER = AutoTokenizer.from_pretrained(ENCODER_NAME)
        ENCODER_MODEL = AutoModel.from_pretrained(ENCODER_NAME).to(ENCODER_DEVICE).eval()
    if LLM is None:
        log.info(f"[INIT] Load LLM: {MODEL_PATH}")
        LLM = load_model(MODEL_PATH, n_ctx=CTX_WINDOW, n_gpu_layers=N_GPU_LAYERS, n_threads=N_THREADS)

# ========= LOAD ASSETS PER-MAPEL =========
@lru_cache(maxsize=8)
def load_subject_assets(subject_key: str) -> SubjectAssets:
    if subject_key not in SUBJECTS:
        raise ValueError(f"Unknown subject: {subject_key}")
    cfg = SUBJECTS[subject_key]
    log.info(f"[ASSETS] Loading subject={subject_key} | index={cfg['index']}")
    if not os.path.exists(cfg["index"]): raise FileNotFoundError(cfg["index"])
    if not os.path.exists(cfg["chunks"]): raise FileNotFoundError(cfg["chunks"])
    if not os.path.exists(cfg["embeddings"]): raise FileNotFoundError(cfg["embeddings"])

    index = faiss.read_index(cfg["index"])
    with open(cfg["chunks"], "r", encoding="utf-8") as f:
        texts = [it["text"] for it in json.load(f)]
    embs = np.load(cfg["embeddings"])
    if index.ntotal != len(embs):
        raise RuntimeError(f"Mismatch ntotal({index.ntotal}) vs emb({len(embs)})")

    return SubjectAssets(index=index, texts=texts, embs=embs)

# ========= ENCODER & RETRIEVAL PER-MAPEL =========
@torch.inference_mode()
def encode_query_exact(text: str) -> np.ndarray:
    toks = ENCODER_TOKENIZER(text, padding=True, truncation=True, return_tensors="pt").to(ENCODER_DEVICE)
    out = ENCODER_MODEL(**toks)
    vec = out.last_hidden_state.mean(dim=1)
    return vec.cpu().numpy()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

def best_cosine_from_faiss(query: str, subject_key: str) -> float:
    assets = load_subject_assets(subject_key)
    q = encode_query_exact(query)
    _, I = assets.index.search(q, TOP_K_FAISS)
    qv = q.reshape(-1)
    best = -1.0
    for i in I[0]:
        if 0 <= i < len(assets.texts):
            best = max(best, cosine_sim(qv, assets.embs[i]))
    return best

def retrieve_rerank_cosine(query: str, subject_key: str):
    assets = load_subject_assets(subject_key)
    q = encode_query_exact(query)
    _, idx = assets.index.search(q, TOP_K_FAISS)

    qv = q.reshape(-1)
    pairs = []

    for i in idx[0]:
        if 0 <= i < len(assets.texts):
            cos = cosine_sim(qv, assets.embs[i])
            pairs.append((cos, assets.texts[i]))

    pairs.sort(key=lambda x: x[0], reverse=True)
    top = pairs[:TOP_K_FINAL]

    log.info(f"[RETRIEVE] subject={subject_key} | top={len(top)}")
    return top


def pick_best_sentences(query: str, chunks: List[str], top_k: int = 5):
    if not chunks:
        return []

    qv = encode_query_exact(query).reshape(-1)
    cands = []

    for ch in chunks:
        for s in split_sentences(ch):
            sv = encode_query_exact(s).reshape(-1)
            cos = cosine_sim(qv, sv)
            ovl = lexical_overlap(query, s)
            penalty = 0.25 if len(s) < 60 else 0.0
            score = 0.7 * cos + 0.3 * ovl - penalty

            if score >= MIN_HYBRID:
                cands.append((score, s, ch))  # ← SIMPAN CHUNK

    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[:top_k]



def build_prompt(user_query: str, sentences: List[str]) -> str:
    block = "\n".join(f"- {clean_prefix(s)}" for s in sentences)
    system = (
        "- Gunakan HANYA daftar kalimat fakta berikut sebagai sumber.\n"
        "- Jika tidak ada kalimat yang menjawab, balas: maap pengetahuan tidak ada dalam database\n"
        "- Jawab TEPAT 1 kalimat, ringkas, Bahasa Indonesia baku.\n"
        "- DILARANG menulis frasa meta seperti 'berdasarkan', 'menurut', 'merujuk', atau 'bersumber'."
    )
    return f"""{system}

KALIMAT SUMBER:
{block}

PERTANYAAN:
{user_query}

JAWAB (1 kalimat saja):
"""

@lru_cache(maxsize=512)
def validate_input_cached(q: str):
    try:
        result = validate_input(q)
        # fleksibel: bool atau tuple
        if isinstance(result, tuple):
            allowed, label = result
        else:
            allowed, label = result, "unknown"

        log.info(f"[GUARDRAIL] allowed={allowed} | label={label}")
        return allowed, label

    except Exception as e:
        log.exception(f"[GUARDRAIL] error: {e}")
        return False, "error"

    

def send_reset_email(email: str, token: str):
    reset_link = f"{os.environ['BASE_URL']}/auth/reset/{token}"

    message = Mail(
        from_email=os.environ["MAIL_FROM"],
        to_emails=email,
        subject="Reset Password Akun",
        html_content=f"""
        <p>Anda meminta reset password.</p>
        <p>Klik link berikut (berlaku 30 menit):</p>
        <a href="{reset_link}">Reset Password</a>
        """
    )

    sg = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
    sg.send(message)

# ========= AUTH (POSTGRES) =========
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, func, or_
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

POSTGRES_URL = os.environ.get("POSTGRES_URL")
if not POSTGRES_URL:
    raise RuntimeError("POSTGRES_URL tidak ditemukan di .env")

engine = create_engine(POSTGRES_URL, pool_pre_ping=True, future=True, echo=False)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True))
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id        = Column(Integer, primary_key=True)
    username  = Column(String(50), unique=True, nullable=False, index=True)
    email     = Column(String(120), unique=True, nullable=False, index=True)
    password  = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin  = Column(Boolean, default=False, nullable=False)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, nullable=False, index=True)
    subject_key = Column(String(50), nullable=False, index=True)
    role        = Column(String(10), nullable=False)
    message     = Column(Text, nullable=False)
    timestamp   = Column(Integer, server_default=func.extract("epoch", func.now()))

class PasswordReset(Base):
    __tablename__ = "password_resets"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, nullable=False, index=True)
    token      = Column(String(128), unique=True, nullable=False, index=True)
    expires_at = Column(Integer, nullable=False)

Base.metadata.create_all(bind=engine)

JKT_TZ = ZoneInfo("Asia/Jakarta")
@app.template_filter("fmt_ts")
def fmt_ts(epoch_int: int):
    try:
        dt = datetime.fromtimestamp(int(epoch_int), tz=JKT_TZ)
        return dt.strftime("%d %b %Y %H:%M")
    except Exception:
        return "-"
def db():
    return SessionLocal()

def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("auth_login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

def admin_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("auth_login"))
        if not session.get("is_admin"):
            flash("Hanya admin yang boleh mengakses halaman itu.", "error")
            return redirect(url_for("subjects"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


# ========= ROUTES =========
@app.route("/")
def root():
    return redirect(url_for("auth_login"))

@app.route("/auth/login", methods=["GET", "POST"])
def auth_login():
    if request.method == "POST":
        identity = (request.form.get("identity") or "").strip().lower()
        pw_input = (request.form.get("password") or "").strip()
        if not identity or not pw_input:
            flash("Mohon isi email/username dan password.", "error")
            return render_template("login.html"), 400
        s = db()
        try:
            user = (
                s.query(User)
                 .filter(or_(func.lower(User.username) == identity,
                             func.lower(User.email) == identity))
                 .first()
            )
            ok = bool(user and user.is_active and check_password_hash(user.password, pw_input))
        finally:
            s.close()
        if not ok:
            flash("Identitas atau password salah.", "error")
            return render_template("login.html"), 401
        session["logged_in"] = True
        session["user_id"]   = user.id
        session["username"]  = user.username
        session["is_admin"]  = bool(user.is_admin)
        return redirect(url_for("subjects"))
    return render_template("login.html")

@app.route("/auth/register", methods=["GET", "POST"])
def auth_register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        email    = (request.form.get("email") or "").strip().lower()
        pw       = (request.form.get("password") or "").strip()
        confirm  = (request.form.get("confirm") or "").strip()
        if not username or not email or not pw:
            flash("Semua field wajib diisi.", "error")
            return render_template("register.html"), 400
        if len(pw) < 6:
            flash("Password minimal 6 karakter.", "error")
            return render_template("register.html"), 400
        if pw != confirm:
            flash("Konfirmasi password tidak cocok.", "error")
            return render_template("register.html"), 400
        s = db()
        try:
            existed = (
                s.query(User)
                 .filter(or_(func.lower(User.username) == username,
                             func.lower(User.email) == email))
                 .first()
            )
            if existed:
                flash("Username/Email sudah terpakai.", "error")
                return render_template("register.html"), 409
            u = User(username=username, email=email, password=generate_password_hash(pw), is_active=True)
            s.add(u); s.commit()
        finally:
            s.close()
        flash("Registrasi berhasil. Silakan login.", "success")
        return redirect(url_for("auth_login"))
    return render_template("register.html")

@app.route("/auth/forgot", methods=["GET", "POST"])
def auth_forgot():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        if not username:
            flash("Username wajib diisi.", "error")
            return render_template("forgot.html")

        s = db()
        try:
            user = s.query(User).filter(func.lower(User.username) == username).first()
            if user:
                token = generate_token()
                expires = int(time.time()) + 1800  # 30 menit

                s.add(PasswordReset(
                    user_id=user.id,
                    token=token,
                    expires_at=expires
                ))
                s.commit()

                send_reset_email(user.email, token)
        finally:
            s.close()

        # RESPONSE SAMA (ANTI ENUMERATION)
        flash("Jika akun terdaftar, email reset telah dikirim.", "info")
        return redirect(url_for("auth_login"))

    return render_template("forgot.html")


@app.route("/auth/reset/<token>", methods=["GET", "POST"])
def auth_reset(token):
    s = db()
    try:
        pr = (
            s.query(PasswordReset)
            .filter(
                PasswordReset.token == token,
                PasswordReset.expires_at > int(time.time())
            )
            .first()
        )
        if not pr:
            flash("Token tidak valid atau sudah kedaluwarsa.", "error")
            return redirect(url_for("auth_forgot"))

        user = s.query(User).filter_by(id=pr.user_id).first()
        if not user:
            flash("User tidak ditemukan.", "error")
            return redirect(url_for("auth_forgot"))

        if request.method == "POST":
            pw = request.form.get("password", "")
            conf = request.form.get("confirm", "")

            if len(pw) < 6:
                flash("Password minimal 6 karakter.", "error")
                return render_template("reset.html")

            if pw != conf:
                flash("Konfirmasi password tidak cocok.", "error")
                return render_template("reset.html")

            user.password = generate_password_hash(pw)
            s.delete(pr)  # token sekali pakai
            s.commit()

            flash("Password berhasil direset. Silakan login.", "success")
            return redirect(url_for("auth_login"))

    finally:
        s.close()

    return render_template("reset.html")


@app.route("/auth/logout")
def auth_logout():
    session.clear()
    return redirect(url_for("auth_login"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/subjects")
@login_required
def subjects():
    return render_template("home.html", subjects=SUBJECTS)

@app.route("/chat/<subject_key>")
@login_required
def chat_subject(subject_key: str):
    if subject_key not in SUBJECTS:
        return redirect(url_for("subjects"))
    session["subject_selected"] = subject_key
    label = SUBJECTS[subject_key]["label"]

    s = db()
    try:
        uid = session.get("user_id")
        rows = (
            s.query(ChatHistory)
             .filter_by(user_id=uid, subject_key=subject_key)
             .order_by(ChatHistory.id.asc())
             .all()
        )
        history = [{"role": r.role, "message": r.message} for r in rows]
    finally:
        s.close()

    return render_template("chat.html", subject=subject_key, subject_label=label, history=history)

@app.route("/health")
def health():
    return jsonify({"ok": True, "encoder_loaded": ENCODER_MODEL is not None, "llm_loaded": LLM is not None})

def debug_print(title: str, data, indent: int = 0):
    pad = " " * indent
    log.info(f"{pad}=== {title} ===")
    if isinstance(data, list):
        for i, v in enumerate(data, 1):
            log.info(f"{pad}[{i}] {v}")
    elif isinstance(data, dict):
        for k, v in data.items():
            log.info(f"{pad}{k}: {v}")
    else:
        log.info(f"{pad}{data}")


@app.route("/ask/<subject_key>", methods=["POST"])
@login_required
def ask(subject_key: str):
    if subject_key not in SUBJECTS:
        return jsonify({"ok": False, "error": "invalid subject"}), 400

    t0 = time.perf_counter()
    data  = request.get_json(silent=True) or {}
    query = (data.get("message") or "").strip()

    if not query:
        return jsonify({"ok": False, "error": "empty query"}), 400
    
    allowed, gr_label = validate_input_cached(query)

    log.info(f"[QUESTION] {query}")
    log.info(f"[GUARDRAIL] RESULT → allowed={allowed}, label={gr_label}")

    if not allowed:
        return jsonify({"ok": True, "answer": GUARDRAIL_BLOCK_TEXT})

    try:
        _ = load_subject_assets(subject_key)
    except Exception as e:
        log.exception(f"[ASSETS] error: {e}")
        return jsonify({"ok": False, "error": f"subject assets error: {e}"}), 500

    best = best_cosine_from_faiss(query, subject_key)
    log.info(f"[RAG] Subject={subject_key.upper()} | Best cosine={best:.3f}")
    if best < MIN_COSINE:
        return jsonify({"ok": True, "answer": FALLBACK_TEXT})

    chunk_pairs = retrieve_rerank_cosine(query, subject_key)

    debug_print(
        "TOP CHUNKS (FAISS)",
        [f"cos={c:.4f} | {t[:120]}..." for c, t in chunk_pairs],
    )

    chunks = [t for _, t in chunk_pairs]

    if not chunks:
        return jsonify({"ok": True, "answer": FALLBACK_TEXT})
    
    scored = pick_best_sentences(query, chunks, top_k=5)

    if not scored:
        return jsonify({"ok": True, "answer": FALLBACK_TEXT})

    sentences = [sent for _, sent, _ in scored]

    context_chunks = []
    used = set()
    for score, sent, chunk in scored:
        if chunk not in used:
            context_chunks.append(chunk)
            used.add(chunk)

    context = "\n".join(context_chunks)

    debug_print(
        "TOP 5 SENTENCE SELECTORS",
        [f"{i+1}. score={score:.4f} | {sent}"
        for i, (score, sent, _) in enumerate(scored)]
    )

    prompt = build_prompt(query, split_sentences(context))


    try:
        answer = generate(
            LLM, prompt,
            max_tokens=224, temperature=0.2, top_p=1.0,
            stop=["\n\n", "\n###", "###", "\nUser:",
                  "Berdasarkan", "berdasarkan", "Menurut", "menurut",
                  "Merujuk", "merujuk", "Mengacu", "mengacu", "Bersumber", "bersumber"]
        ).strip()
    except Exception as e:
        log.exception(f"[LLM] generate error: {e}")
        return jsonify({"ok": True, "answer": FALLBACK_TEXT})

    m = re.search(r"(.+?[.!?])(\s|$)", answer)
    answer = (m.group(1) if m else answer).strip()  
    answer = strip_meta_sentence(answer)

    try:
        s = db()
        uid = session.get("user_id")
        s.add_all([
            ChatHistory(user_id=uid, subject_key=subject_key, role="user", message=query),
            ChatHistory(user_id=uid, subject_key=subject_key, role="bot", message=answer)
        ])
        s.commit()
    except Exception as e:
        log.exception(f"[DB] gagal simpan chat history: {e}")
    finally:
        s.close()


    if not answer or len(answer) < 2:
        answer = FALLBACK_TEXT

    if ENABLE_PROFILING:
        log.info({"latency_total": time.perf_counter() - t0, "subject": subject_key, "faiss_best": best})

    return jsonify({"ok": True, "answer": answer})

@app.route("/admin")
@admin_required
def admin_dashboard():
    s = db()
    try:
        total_users   = s.query(func.count(User.id)).scalar() or 0
        total_active  = s.query(func.count(User.id)).filter(User.is_active.is_(True)).scalar() or 0
        total_admins  = s.query(func.count(User.id)).filter(User.is_admin.is_(True)).scalar() or 0
        total_msgs    = s.query(func.count(ChatHistory.id)).scalar() or 0
    finally:
        s.close()
    return render_template("admin_dashboard.html",
                           total_users=total_users,
                           total_active=total_active,
                           total_admins=total_admins,
                           total_msgs=total_msgs)

@app.route("/admin/users")
@admin_required
def admin_users():
    q = (request.args.get("q") or "").strip().lower()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 5), 100)
    s = db()
    try:
        base = s.query(User)
        if q:
            base = base.filter(or_(
                func.lower(User.username).like(f"%{q}%"),
                func.lower(User.email).like(f"%{q}%")
            ))
        total = base.count()
        users = (base
                 .order_by(User.id.asc())
                 .offset((page - 1) * per_page)
                 .limit(per_page)
                 .all())
        user_ids = [u.id for u in users] or [-1]
        counts = dict(s.query(ChatHistory.user_id, func.count(ChatHistory.id))
                        .filter(ChatHistory.user_id.in_(user_ids))
                        .group_by(ChatHistory.user_id)
                        .all())
    finally:
        s.close()
    return render_template("admin_users.html",
                           users=users, counts=counts,
                           q=q, page=page, per_page=per_page, total=total)

@app.route("/admin/history")
@admin_required
def admin_history():
    q          = (request.args.get("q") or "").strip().lower()    # cari di message
    username   = (request.args.get("username") or "").strip().lower()
    subject    = (request.args.get("subject") or "").strip().lower()
    role       = (request.args.get("role") or "").strip().lower()
    page       = max(int(request.args.get("page", 1)), 1)
    per_page   = min(max(int(request.args.get("per_page", 30)), 5), 200)

    s = db()
    try:
       
        base = (s.query(ChatHistory, User)
                  .join(User, User.id == ChatHistory.user_id))

        if q:
            base = base.filter(func.lower(ChatHistory.message).like(f"%{q}%"))
        if username:
            base = base.filter(or_(
                func.lower(User.username) == username,
                func.lower(User.email) == username
            ))
        if subject:
            base = base.filter(func.lower(ChatHistory.subject_key) == subject)
        if role in ("user", "bot"):
            base = base.filter(ChatHistory.role == role)

        total = base.count()
        rows = (base
                .order_by(ChatHistory.id.desc())
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all())
    finally:
        s.close()

    items = [{
        "id": r.ChatHistory.id,
        "username": r.User.username,
        "email": r.User.email,
        "subject": r.ChatHistory.subject_key,
        "role": r.ChatHistory.role,
        "message": r.ChatHistory.message,
        "timestamp": r.ChatHistory.timestamp,
    } for r in rows]

    return render_template("admin_history.html",
                           items=items, subjects=SUBJECTS,
                           q=q, username=username, subject=subject, role=role,
                           page=page, per_page=per_page, total=total)

# =========================
#   ADMIN: Delete actions
# =========================

def _is_last_admin(s) -> bool:
    """Cek apakah hanya tersisa 1 admin aktif."""
    return (s.query(func.count(User.id)).filter(User.is_admin.is_(True)).scalar() or 0) <= 1

@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id: int):
    s = db()
    try:
        me_id = session.get("user_id")
        user = s.query(User).filter_by(id=user_id).first()
        if not user:
            flash("User tidak ditemukan.", "error")
            return redirect(request.referrer or url_for("admin_users"))

        if user.id == me_id:
            flash("Tidak bisa menghapus akun yang sedang login.", "error")
            return redirect(request.referrer or url_for("admin_users"))

        if user.is_admin and _is_last_admin(s):
            flash("Tidak bisa menghapus admin terakhir.", "error")
            return redirect(request.referrer or url_for("admin_users"))

        s.query(ChatHistory).filter(ChatHistory.user_id == user.id).delete(synchronize_session=False)
        s.delete(user)
        s.commit()
        flash(f"User #{user_id} beserta seluruh riwayatnya telah dihapus.", "success")
    except Exception as e:
        s.rollback()
        log.exception(f"[ADMIN] delete user error: {e}")
        flash("Gagal menghapus user.", "error")
    finally:
        s.close()
    return redirect(request.referrer or url_for("admin_users"))

@app.route("/admin/users/<int:user_id>/history/clear", methods=["POST"])
@admin_required
def admin_clear_user_history(user_id: int):
    s = db()
    try:
        exists = s.query(User.id).filter_by(id=user_id).first()
        if not exists:
            flash("User tidak ditemukan.", "error")
            return redirect(request.referrer or url_for("admin_history"))

        deleted = s.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete(synchronize_session=False)
        s.commit()
        flash(f"Riwayat chat user #{user_id} dihapus ({deleted} baris).", "success")
    except Exception as e:
        s.rollback()
        log.exception(f"[ADMIN] clear history error: {e}")
        flash("Gagal menghapus riwayat.", "error")
    finally:
        s.close()
    return redirect(request.referrer or url_for("admin_history"))

@app.route("/admin/history/<int:chat_id>/delete", methods=["POST"])
@admin_required
def admin_delete_chat(chat_id: int):
    s = db()
    try:
        row = s.query(ChatHistory).filter_by(id=chat_id).first()
        if not row:
            flash("Baris riwayat tidak ditemukan.", "error")
            return redirect(request.referrer or url_for("admin_history"))
        s.delete(row)
        s.commit()
        flash(f"Riwayat chat #{chat_id} dihapus.", "success")
    except Exception as e:
        s.rollback()
        log.exception(f"[ADMIN] delete chat error: {e}")
        flash("Gagal menghapus riwayat.", "error")
    finally:
        s.close()
    return redirect(request.referrer or url_for("admin_history"))


if __name__ == "__main__":
    warmup_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
