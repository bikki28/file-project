# ============================================================
# recommender.py
# Movie Recommendation System - NLP Core Logic
# Uses TF-IDF + Cosine Similarity for content-based filtering
# ============================================================

import os
import ast
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Download NLTK data ─────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


# ─── Helper: safely parse stringified JSON/list columns ─────
def parse_list(obj):
    """Convert stringified list (from CSV) into a Python list."""
    try:
        return ast.literal_eval(obj)
    except (ValueError, SyntaxError):
        return []


# ─── Feature Extractors ─────────────────────────────────────
def extract_names(obj, limit=None):
    """Extract 'name' fields from a list of dicts."""
    items = parse_list(obj)
    names = [i['name'].replace(" ", "") for i in items if 'name' in i]
    return names[:limit] if limit else names


def extract_director(obj):
    """Extract director name from crew list."""
    crew = parse_list(obj)
    for member in crew:
        if member.get('job') == 'Director':
            return [member['name'].replace(" ", "")]
    return []


# ─── NLP Preprocessing ──────────────────────────────────────
def preprocess_text(text):
    """
    Lowercase, tokenize, and remove stopwords from a string.
    Returns a clean space-joined string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    return " ".join(filtered)


# ─── Load & Preprocess Dataset ──────────────────────────────
def load_and_preprocess(movies_path='dataset/movies.csv',
                        credits_path='dataset/credits.csv'):
    """
    Load TMDB movies + credits CSVs, merge them, extract features,
    build a 'tags' column, and return a clean DataFrame.
    """
    print("[INFO] Loading datasets...")
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Rename for merge compatibility
    if 'movie_id' in credits.columns:
        credits.rename(columns={'movie_id': 'id'}, inplace=True)
    if 'title' in credits.columns and 'title' in movies.columns:
        credits.drop(columns=['title'], inplace=True, errors='ignore')

    # Merge on id
    df = movies.merge(credits, on='id')

    # Keep only useful columns
    cols = ['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
    df = df[cols].copy()

    print("[INFO] Preprocessing features...")

    # Extract structured fields
    df['genres']   = df['genres'].apply(extract_names)
    df['keywords'] = df['keywords'].apply(extract_names)
    df['cast']     = df['cast'].apply(lambda x: extract_names(x, limit=5))
    df['crew']     = df['crew'].apply(extract_director)

    # Fill missing overviews
    df['overview'] = df['overview'].fillna('')

    # Build combined 'tags' feature
    def build_tags(row):
        parts = (
            row['overview'].split()
            + row['genres']
            + row['keywords']
            + row['cast']
            + row['crew']
        )
        return " ".join(parts)

    df['tags'] = df.apply(build_tags, axis=1)

    # NLP clean the tags
    df['tags'] = df['tags'].apply(preprocess_text)

    # Drop rows with empty tags
    df = df[df['tags'].str.strip() != ''].reset_index(drop=True)

    print(f"[INFO] Dataset ready: {len(df)} movies loaded.")
    return df[['id', 'title', 'tags']]


# ─── Build TF-IDF + Similarity Matrix ───────────────────────
def build_similarity(df):
    """
    Vectorize tags with TF-IDF and compute cosine similarity matrix.
    Returns (vectorizer, similarity_matrix).
    """
    print("[INFO] Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['tags'])

    print("[INFO] Computing cosine similarity matrix...")
    similarity = cosine_similarity(tfidf_matrix)

    return tfidf, similarity


# ─── Save / Load Model ──────────────────────────────────────
def save_model(df, similarity, path='model/similarity.pkl'):
    """Persist the movie DataFrame and similarity matrix to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'df': df, 'similarity': similarity}, f)
    print(f"[INFO] Model saved to {path}")


def load_model(path='model/similarity.pkl'):
    """Load persisted model from disk."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['df'], data['similarity']


# ─── Recommendation Function ────────────────────────────────
def recommend(movie_title, df, similarity, top_n=10):
    """
    Given a movie title, return the top_n most similar movies.

    Parameters
    ----------
    movie_title : str   - Title to search
    df          : DataFrame - must contain 'title' column
    similarity  : ndarray   - cosine similarity matrix
    top_n       : int       - number of recommendations

    Returns
    -------
    list of dicts [{'title': ..., 'score': ...}, ...]
    """
    # Case-insensitive partial match
    title_lower = movie_title.strip().lower()
    mask = df['title'].str.lower().str.contains(title_lower, na=False)
    matches = df[mask]

    if matches.empty:
        return []

    # Use the first match
    idx = matches.index[0]
    sim_scores = list(enumerate(similarity[idx]))

    # Sort by similarity (descending), skip the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

    results = []
    for i, score in sim_scores:
        results.append({
            'title': df.iloc[i]['title'],
            'score': round(float(score) * 100, 1)
        })

    return results


# ─── CLI Entry Point (for training) ─────────────────────────
if __name__ == '__main__':
    df = load_and_preprocess()
    _, similarity = build_similarity(df)
    save_model(df, similarity)
    print("\n[DONE] Model training complete!")

    # Quick sanity check
    recs = recommend("The Dark Knight", df, similarity)
    print("\nTest recommendations for 'The Dark Knight':")
    for r in recs:
        print(f"  {r['title']}  ({r['score']}%)")
