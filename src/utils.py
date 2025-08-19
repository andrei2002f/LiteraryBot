import json
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

def parse_bulk_json(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(lines[i]) for i in range(1, len(lines), 2)]

def clean_text(doc, field):
    """
    Cleans a specified text field in a document by:
    - Flattening to string if list
    - Removing HTML tags
    - Stripping and normalizing whitespace
    """
    raw_text = doc.get(field, "")

    # If the field is a list, join it into a single string
    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)

    # Remove HTML tags
    text = BeautifulSoup(raw_text, "html.parser").get_text()

    # Normalize whitespace
    return text.strip()

def extract_keywords(text, language="romanian"):
    """
    Extracts keywords from a text using TF-IDF and stopword removal.
    The number of keywords is dynamically determined based on the text length.
    :param text: The input text string.
    :param language: The language for stopword removal (default: Romanian).
    :return: A list of extracted keywords.
    """
    stop_words = stopwords.words(language)
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Dynamically calculate the number of keywords
    num_keywords = max(10, len(text.split()) // 5)  # 1 keyword per 5 words, minimum 10

    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [word for word, score in keywords[:num_keywords]]

