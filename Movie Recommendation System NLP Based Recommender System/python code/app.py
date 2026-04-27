# ============================================================
# app.py
# Movie Recommendation System — Flask Web Application
# ============================================================

import os
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify
from recommender import load_model, load_and_preprocess, build_similarity, save_model, recommend

app = Flask(__name__)

# ─── OMDb API ────────────────────────────────────────────────
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '393a6ed6')
OMDB_BASE    = 'https://www.omdbapi.com/'

# ─── Global model state ─────────────────────────────────────
df_movies  = None
similarity = None
raw_movies_df = None
MODEL_PATH = 'model/similarity.pkl'


def get_model():
    global df_movies, similarity
    if df_movies is None or similarity is None:
        if os.path.exists(MODEL_PATH):
            print("[INFO] Loading pre-trained model...")
            df_movies, similarity = load_model(MODEL_PATH)
        else:
            print("[INFO] No saved model found. Training now...")
            df_movies = load_and_preprocess()
            _, similarity = build_similarity(df_movies)
            save_model(df_movies, similarity, MODEL_PATH)
    return df_movies, similarity


# ─── OMDb Poster Helper ──────────────────────────────────────
def fetch_poster(title):
    if not OMDB_API_KEY:
        return None
    try:
        res = requests.get(OMDB_BASE, params={'apikey': OMDB_API_KEY, 't': title}, timeout=3)
        data = res.json()
        poster = data.get('Poster', '')
        if poster and poster != 'N/A':
            return poster.replace('http://', 'https://')
        res2 = requests.get(OMDB_BASE, params={'apikey': OMDB_API_KEY, 's': title, 'type': 'movie'}, timeout=3)
        data2 = res2.json()
        results = data2.get('Search', [])
        if results:
            poster2 = results[0].get('Poster', '')
            if poster2 and poster2 != 'N/A':
                return poster2.replace('http://', 'https://')
    except Exception:
        pass
    return None


# ─── OMDb Trending Helper ────────────────────────────────────
def fetch_trending():
    if not OMDB_API_KEY:
        return []
    popular_titles = [
        'The Dark Knight', 'Inception', 'Interstellar',
        'The Avengers', 'Avatar', 'The Godfather'
    ]
    trending = []
    for title in popular_titles:
        try:
            res = requests.get(OMDB_BASE, params={'apikey': OMDB_API_KEY, 't': title}, timeout=3)
            data = res.json()
            if data.get('Response') == 'True':
                poster = data.get('Poster', '')
                trending.append({
                    'title':  data.get('Title', title),
                    'poster': poster.replace('http://', 'https://') if poster and poster != 'N/A' else None,
                    'rating': data.get('imdbRating', 'N/A')
                })
        except Exception:
            continue
    return trending


# ─── Movie Details Route ─────────────────────────────────────
def get_omdb_details(title):
    """Fetch full OMDb details by title, with search fallback."""
    res = requests.get(OMDB_BASE, params={
        'apikey': OMDB_API_KEY, 't': title, 'plot': 'full'
    }, timeout=4)
    data = res.json()

    if data.get('Response') == 'False':
        # fallback: search mode
        res2 = requests.get(OMDB_BASE, params={
            'apikey': OMDB_API_KEY, 's': title, 'type': 'movie'
        }, timeout=4)
        search_results = res2.json().get('Search', [])
        if not search_results:
            return None
        imdb_id = search_results[0].get('imdbID', '')
        res3 = requests.get(OMDB_BASE, params={
            'apikey': OMDB_API_KEY, 'i': imdb_id, 'plot': 'full'
        }, timeout=4)
        data = res3.json()

    return data if data.get('Response') == 'True' else None


def get_local_plot(title):
    """Fallback plot from local dataset when OMDb details are unavailable."""
    global raw_movies_df
    if raw_movies_df is None:
        if not os.path.exists('dataset/movies.csv'):
            return None
        try:
            raw_movies_df = pd.read_csv('dataset/movies.csv', usecols=['title', 'overview'])
        except Exception:
            return None

    q = title.strip().lower()
    if not q:
        return None
    matches = raw_movies_df[raw_movies_df['title'].str.lower().str.contains(q, na=False)]
    if matches.empty:
        return None
    overview = matches.iloc[0].get('overview', '')
    return overview if isinstance(overview, str) and overview.strip() else None


# ─── Routes ─────────────────────────────────────────────────
@app.route('/')
def index():
    trending = fetch_trending()
    return render_template('index.html', trending=trending, has_api_key=bool(OMDB_API_KEY))


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data  = request.get_json()
    title = (data or {}).get('title', '').strip()
    if not title:
        return jsonify({'error': 'Please enter a movie title.'}), 400

    df, sim = get_model()
    results = recommend(title, df, sim, top_n=10)
    if not results:
        return jsonify({'error': f'Movie "{title}" not found in our database. Try another title!'}), 404

    for r in results:
        r['poster'] = fetch_poster(r['title'])

    return jsonify({'movie': title, 'recommendations': results})


@app.route('/movie_details', methods=['GET'])
def movie_details():
    """Return full movie info for the modal: plot, ratings, trailer URL."""
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({'error': 'No title provided.'}), 400

    try:
        data = get_omdb_details(title)
        if not data:
            plot = get_local_plot(title)
            if not plot:
                return jsonify({'error': 'Movie details not found.'}), 404
            query = f"{title} official trailer"
            trailer_url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
            return jsonify({
                'title': title,
                'year': 'N/A',
                'rated': 'N/A',
                'runtime': 'N/A',
                'genre': 'N/A',
                'director': 'N/A',
                'actors': 'N/A',
                'plot': plot,
                'awards': 'N/A',
                'imdb': 'N/A',
                'poster': None,
                'reviews': [],
                'trailer': trailer_url,
            })

        # Ratings → review lines
        reviews = [
            f"{r.get('Source','')}: {r.get('Value','')}"
            for r in data.get('Ratings', [])
        ]

        # YouTube trailer search URL
        query       = f"{data.get('Title', title)} {data.get('Year', '')} official trailer"
        trailer_url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"

        poster = data.get('Poster', '')

        return jsonify({
            'title':    data.get('Title', title),
            'year':     data.get('Year', 'N/A'),
            'rated':    data.get('Rated', 'N/A'),
            'runtime':  data.get('Runtime', 'N/A'),
            'genre':    data.get('Genre', 'N/A'),
            'director': data.get('Director', 'N/A'),
            'actors':   data.get('Actors', 'N/A'),
            'plot':     data.get('Plot', 'No description available.'),
            'awards':   data.get('Awards', 'N/A'),
            'imdb':     data.get('imdbRating', 'N/A'),
            'poster':   poster.replace('http://', 'https://') if poster and poster != 'N/A' else None,
            'reviews':  reviews,
            'trailer':  trailer_url,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search_suggestions', methods=['GET'])
def search_suggestions():
    query = request.args.get('q', '').strip().lower()
    if len(query) < 2:
        return jsonify([])
    df, _ = get_model()
    mask  = df['title'].str.lower().str.contains(query, na=False)
    return jsonify(df[mask]['title'].head(8).tolist())


# ─── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[INFO] Initializing recommendation engine...")
    get_model()
    print("[INFO] Flask app starting at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)