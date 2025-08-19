from sentence_transformers import SentenceTransformer
from src.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, show_progress_bar=False).tolist()
