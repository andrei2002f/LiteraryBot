import ollama
from src.config import OLLAMA_HOST, OLLAMA_AUTH_TOKEN

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"Authorization": f"Bearer {OLLAMA_AUTH_TOKEN}"}
)

def generate_answer(query, context, model="llama3.3:latest"):
    """
    Generates an answer using a language model, given a question and the retrieved context.
    """
    prompt = (
    f"Mai jos este un context extras dintr-o bază de date literară. "
    f"Folosește DOAR informațiile din acest context pentru a răspunde corect și complet la întrebare. "
    f"Răspunsul trebuie să fie concis, informativ și lung de 1-3 propoziții dacă nu este precizat altfel.\n\n"
    f"Dacă nu găsești răspunsul in contextul dat, răspunde doar 'Nu există suficiente date.', nu oferi alte explicații.\n\n"
    f"{context}\n\n"
    f"Întrebare: {query}\n\n"
    f"Răspuns:"
)

    response = client.generate(model=model, prompt=prompt)
    return response["response"].strip()
