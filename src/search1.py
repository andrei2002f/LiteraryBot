import re
from sentence_transformers import util
from src.embedder import embed

# --- Abbreviation handling -------------------------------------------------
# A curated list of *very common* abrevieri româneşti, gathered from surse normative
# precum DOOM 3 şi articolul „XXVIII. Abrevierile” de la dexonline.ro (Mioara Avram).
# Nu e exhaustiv, dar reduce majoritatea împărţirilor greşite.
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
    "a.c.", "l.c.", "d.e.", "de ex.", "p.a.", "î.e.n.", "e.n.",
    # enumerate
    "ș.a.", "ș.a.m.d.",
}

# Regex‑uri pentru iniţiale (O., V.) şi prescurtări de 2 litere cu punct (Al.)
_INITIAL_REGEX = re.compile(r"^[A-ZȘȚĂÂÎ]\.?")  # single uppercase + punct
_TWO_LETTER_REGEX = re.compile(r"^[A-ZȘȚĂÂÎ][a-zăâîșț]\.?")


def _is_bad_break(token: str) -> bool:
    """Returnează True dacă *token* se termină cu o abreviere care NU trebuie
    să declanşeze sfârşit de propoziţie."""
    token_strip = token.strip()
    # verificare listă fixă
    if any(token_strip.endswith(abbr) for abbr in _ABBREVIATIONS):
        return True
    # iniţiale de o literă (O.)
    if _INITIAL_REGEX.fullmatch(token_strip):
        return True
    # abreviere de două litere cu majusculă + minusculă (Al.)
    if _TWO_LETTER_REGEX.fullmatch(token_strip):
        return True
    # acronime tip C.F.R. sau U.S.A. (majuscule + puncte interioare)
    if re.fullmatch(r"(?:[A-Z]\.){2,}[A-Z]?", token_strip):
        return True
    return False


# ---------------------------------------------------------------------------
# Sentence tokenizer
# ---------------------------------------------------------------------------

def sent_tokenize(text: str) -> list[str]:
    """Tokenizer de propoziţii care evită despărţirea după abrevieri uzuale.

    Nu necesită resurse externe şi funcţionează bine pe texte literare/biografice.
    """
    if not text:
        return []

    # despărţire preliminară la . ! ? urmate de spaţiu
    chunks = re.split(r"(?<=[.!?]) +", text.strip())

    sentences, i = [], 0
    while i < len(chunks):
        current = chunks[i]
        # uneşte cât timp se termină cu o abreviere/iniţială
        while _is_bad_break(current) and i + 1 < len(chunks):
            i += 1
            current += " " + chunks[i]
        sentences.append(current.strip())
        i += 1
    return sentences


# ---------------------------------------------------------------------------
# Sentence extraction & context builder
# ---------------------------------------------------------------------------

def extract_top_sentences_anchored(text: str, query: str, anchor: str, top_n: int = 3) -> list[str]:
    """Returnează cele mai relevante *top_n* propoziţii din *text* pentru *query*.

    Scoring: cosineSimilarity(embedding(query + " " + anchor), embedding(sentence)).
    """
    if not text.strip():
        return []

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # Embed query+anchor o singură dată
    full_query = f"{query} {anchor}"
    query_emb = embed([full_query])[0]
    sent_embs = embed(sentences)

    scores = [util.cos_sim(query_emb, se)[0][0].item() for se in sent_embs]
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]


def build_filtered_context(results: list[dict], query: str, top_n_sentences: int = 3) -> str:
    """Generează context compact de tipul "Nume: s1 s2 s3" pentru LLM."""
    parts = []
    for res in results:
        name = res["name"]
        desc = res["description"]
        top_sents = extract_top_sentences_anchored(desc, query, name, top_n_sentences)
        parts.append(f"{name}: {' '.join(top_sents)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.search import hybrid_search
    from src.config import INDEX_ALL

    q = "Cum este caracterizată abordarea tematicii istorice și legendare în opera lui Gheorghe Andrei?"
    hits = hybrid_search(q, index_name=INDEX_ALL, k=3)
    ctx = build_filtered_context(hits, q, 3)
    print(">>> Context filtrat:\n")
    print(ctx)
