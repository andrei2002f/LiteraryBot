"""
Evaluate the RAG chatbot on q&a CSV:
question,expected_answer,difficulty  (simple|medium|complex)
"""

import csv, pathlib, statistics, collections, re, string, unicodedata
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ----------  CONFIGURABLE CONSTANTS  ----------
TEST_FILE      = pathlib.Path("C:/Users/farca/Documents/POLI/Licenta/Aplicatie/tests/qa.csv")

LLM_MODEL_TAG  = "gemma3:12b"          # change model here: deepseek-r1:32b, qwen2.5:72b, llama4:16x17b, llama3.3:latest, gemma3:12b
SIM_MODEL      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIM_THRESHOLDS = {"simple": 0.7, "medium": 0.6, "complex": 0.5}                   # sentence-level cosine ≥ threshold ⇒ correct
TOP_K_DOCS     = 3                      # docs sent to LLM
NO_DATA_MARK   = "nu există suficiente date"  # model's 'no info' reply
DEBUG          = True                   # ← flip to False for quiet mode
DEBUG_ROWS     = False                   # e.g. 5 to limit rows in quick tests
# ---------------------------------------------

# --- project imports ---
from src.search import hybrid_search
from src.generator import generate_answer
from src.config import INDEX_ALL
from src.search1 import build_filtered_context
from src.context_filter import build_filtered_context_highlights
# ----------------------

embedder = SentenceTransformer(SIM_MODEL)
cos      = lambda a, b: float(util.cos_sim(a, b))

# ---------- helpers --------------------------------------------------
def normalize(txt: str) -> str:
    txt = txt.lower()
    # Remove every char whose Unicode category starts with "P" (punctuation)
    txt = "".join(
        ch for ch in txt
        if not unicodedata.category(ch).startswith("P")
    )
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def contains_all_keywords(gold: str, pred: str) -> bool:
    g_tokens = {t for t in normalize(gold).split() if len(t) > 2}
    print(f"Gold tokens: {g_tokens}")  # Debugging
    p_tokens = set(normalize(pred).split())
    print(f"Prediction tokens: {p_tokens}")  # Debugging
    return g_tokens.issubset(p_tokens)

def max_sentence_similarity(gold: str, prediction: str) -> float:
    gold_vec = embedder.encode(gold, convert_to_tensor=True)
    sentences = re.split(r'(?<=[.!?]) +', prediction)
    sims = [
        cos(gold_vec, embedder.encode(sent, convert_to_tensor=True))
        for sent in sentences if sent.strip()
    ]
    return max(sims) if sims else 0.0
# ---------------------------------------------------------------------

def run_rag(question: str) -> str:
    hits = hybrid_search(question, index_name=INDEX_ALL, k=TOP_K_DOCS)
    if DEBUG:
        print(f"  • Retrieved top-{TOP_K_DOCS} docs:")
        for h in hits:
            print(f"    - {h['name'][:60]}…")

    # context = "\n\n".join(f"{h['name']}:\n{h['description']}" for h in hits)
    # context = build_filtered_context(hits, question, top_n_sentences=5)
    context = build_filtered_context_highlights(hits, question, top_n_sentences=5)
    answer = generate_answer(question, context, model=LLM_MODEL_TAG).strip()
    if DEBUG:
        print(f"  • Generated answer: {answer}")
    return answer

def evaluate():
    stats = collections.defaultdict(lambda: {"total":0,"correct":0,"no_data":0,"sims":[]})
    with TEST_FILE.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx,row in enumerate(tqdm(reader, desc="Evaluating")):
            if DEBUG_ROWS and idx >= DEBUG_ROWS: break

            q     = row["question"].strip()
            gold  = row["expected_answer"].strip()
            diff  = row["difficulty"].strip().lower()

            if DEBUG:
                print(f"\n=== [{idx+1}]  {diff.upper()}  ===============================")
                print("Q:", q)
                print("Gold:", gold)

            pred  = run_rag(q)
            rec   = stats[diff];  rec["total"] += 1

            # --- no-data shortcut
            if NO_DATA_MARK in pred.lower():
                rec["no_data"] += 1
                if DEBUG: print("  ⚠  Model returned NO-DATA string.")
                continue

            # --- containment shortcut
            if contains_all_keywords(gold, pred):
                rec["correct"] += 1
                rec["sims"].append(1.0)
                if DEBUG: print("  ✔  Containment satisfied → correct.")
                continue

            # --- windowed cosine
            max_sim = max_sentence_similarity(gold, pred)
            rec["sims"].append(max_sim)
            if DEBUG: print(f"  • Max sentence-cosine = {max_sim:.3f}")

            SIM_THRESHOLD = SIM_THRESHOLDS[diff]
            if max_sim >= SIM_THRESHOLD:
                rec["correct"] += 1
                if DEBUG: print(f"  ✔  Above threshold ({SIM_THRESHOLD}) → correct.")
            elif DEBUG:
                print("  ✘  Below threshold → incorrect.")

    # ------------- report -------------
    print("\n===========  RESULTS  ===========")
    overall_total = overall_correct = 0
    for diff in ("simple","medium","complex"):
        t = stats[diff]["total"];  c = stats[diff]["correct"];  nd = stats[diff]["no_data"]
        if not t: continue
        acc = c/t
        avg = statistics.mean(stats[diff]["sims"]) if stats[diff]["sims"] else 0
        print(f"{diff.capitalize():>7}: total={t:3d}  correct={c:3d}  no-data={nd:3d}  "
              f"accuracy={acc:.2%}  avg-sim={avg:.3f}")
        overall_total   += t
        overall_correct += c
    print("---------------------------------")
    if overall_total:
        print(f"Overall accuracy: {overall_correct/overall_total:.2%}")

if __name__ == "__main__":
    evaluate()
