import re
from sentence_transformers import util
from src.embedder import embed

# ---------------------------------------------------------------------------
#  Abbreviation‑aware sentence tokenizer (Romanian)
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    # administrativ / tehnic
    "j.", "jud.", "nr.", "vol.", "cap.", "fig.", "art.", "pag.", "sec.", "cca.",
    "str.", "bl.", "sc.", "ap.", "mun.", "com.", "loc.", "nrcrt.", "crt.",
    # formule de adresare & titluri
    "d.", "dl.", "dna.", "dvs.", "dv.", "dom.", "dr.", "prof.", "ing.", "conf.",
    "lect.", "acad.", "col.", "lt.", "cpt.", "mr.", "gen.",
    # altele frecvente
    "etc.", "cf.", "op.cit.", "ibid.", "n.b.", "p.s.",
    # prenume abreviate uzuale
    "Al.", "Gh.", "I.", "Ion.", "Gr.", "V.", "M.", "O.", "G.", "E.", "Șt.", "T.",
    # expresii
    "a.c.", "l.c.", "d.e.", "de ex.", "p.a.", "î.e.n.", "e.n.",
    # enumerate
    "ș.a.", "ș.a.m.d.",
}

_INITIAL_REGEX = re.compile(r"^[A-ZȘȚĂÂÎ]\.?")          # O.
_TWO_LETTER_REGEX = re.compile(r"^[A-ZȘȚĂÂÎ][a-zăâîșț]\.?")  # Al.


def _is_bad_break(token: str) -> bool:
    token = token.strip()
    if any(token.endswith(abbr) for abbr in _ABBREVIATIONS):
        return True
    if _INITIAL_REGEX.fullmatch(token):
        return True
    if _TWO_LETTER_REGEX.fullmatch(token):
        return True
    if re.fullmatch(r"(?:[A-Z]\\.){2,}[A-Z]?", token):  # C.F.R.
        return True
    return False


def sent_tokenize(text: str) -> list[str]:
    """Sentence tokenizer that avoids splitting after common Romanian abbreviations."""
    if not text:
        return []

    chunks = re.split(r"(?<=[.!?]) +", text.strip())
    sentences, i = [], 0
    while i < len(chunks):
        cur = chunks[i]
        while _is_bad_break(cur) and i + 1 < len(chunks):
            i += 1
            cur += " " + chunks[i]
        sentences.append(cur.strip())
        i += 1
    return sentences

# ---------------------------------------------------------------------------
#  Sentence relevance scoring
# ---------------------------------------------------------------------------

def extract_top_sentences_anchored(text: str, query: str, anchor: str, top_n: int = 5) -> list[str]:
    if not text.strip():
        return []

    sents = sent_tokenize(text)
    if not sents:
        return []

    full_q = f"{query} {anchor}"
    q_emb = embed([full_q])[0]
    s_embs = embed(sents)

    scores = [util.cos_sim(q_emb, e)[0][0].item() for e in s_embs]
    ranked = sorted(zip(sents, scores), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]

# ---------------------------------------------------------------------------
#  Context builder (prefers ES highlights)
# ---------------------------------------------------------------------------

def build_filtered_context_highlights(results: list[dict], query: str, top_n_sentences: int = 5) -> str:
    """Builds compressed LLM context. Order of preference per document:
    1. Elasticsearch highlight fragments (if any)
    2. Top-N semantically ranked sentences.
    """
    parts = []
    for res in results:
        name = res["name"]
        hl = res.get("highlight", [])  # list[str] if ES returned highlight
        if hl:
            snippet = " ".join(hl)
        else:
            snippet = " ".join(
                extract_top_sentences_anchored(res["description"], query, name, top_n_sentences)
            )
        parts.append(f"{name}: {snippet}")
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
#  Quick manual test
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     from src.search import hybrid_search
#     from src.config import INDEX_ALL

#     query = "În ce an a fost publicată prima lucrare editorială a lui Solomon Marcus în domeniul poeticii?"
#     hits = hybrid_search(query, INDEX_ALL, k=3)
#     ctx = build_filtered_context_highlights(hits, query)
#     print(">>> Context construit:\n")
#     print(ctx)
