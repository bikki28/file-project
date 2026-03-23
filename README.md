# 🎬 CineMatch — Movie Recommendation System

A **content-based movie recommendation web app** built with Flask, scikit-learn, and NLTK.
Enter any movie title and get the **top 10 similar movies** instantly — powered by TF-IDF vectorization and cosine similarity.

---

## 📸 Tech Stack

| Layer      | Technology                         |
|------------|------------------------------------|
| Backend    | Python 3.10+, Flask                |
| ML / NLP   | scikit-learn, NLTK                 |
| Data       | pandas, NumPy                      |
| Frontend   | HTML5, CSS3, Bootstrap 5, Vanilla JS |
| Dataset    | TMDB 5000 Movie Dataset (Kaggle)   |

---

## 📁 Project Structure

```
Movie-Recommendation-System/
│
├── dataset/
│   ├── movies.csv          ← TMDB movies metadata
│   └── credits.csv         ← Cast & crew data
│
├── model/
│   └── similarity.pkl      ← Saved similarity matrix (auto-generated)
│
├── templates/
│   └── index.html          ← Main UI page
│
├── static/
│   ├── css/style.css       ← Custom dark-theme styling
│   └── js/script.js        ← Frontend logic + autocomplete
│
├── app.py                  ← Flask web application
├── recommender.py          ← ML core: preprocessing, TF-IDF, cosine similarity
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## ⚡ Quick Start

### Step 1 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
2. Download the two CSV files:
   - `tmdb_5000_movies.csv` → rename to `movies.csv`
   - `tmdb_5000_credits.csv` → rename to `credits.csv`
3. Place both files inside the `dataset/` folder

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Train the Model (first run only)

```bash
python recommender.py
```

This will:
- Load and preprocess the dataset
- Build TF-IDF vectors from movie tags
- Compute the cosine similarity matrix
- Save `model/similarity.pkl`

> ⏱ Takes ~30–60 seconds on first run. Subsequent app starts load instantly.

### Step 4 — Run the App

```bash
python app.py
```

### Step 5 — Open in Browser

```
http://127.0.0.1:5000
```

---

## 🔑 Optional: TMDB API Key (for Movie Posters + Trending)

1. Create a free account at https://www.themoviedb.org/
2. Go to Settings → API → Request an API Key
3. Set it as an environment variable:

```bash
# Windows (Command Prompt)
set TMDB_API_KEY=your_api_key_here

# macOS / Linux
export TMDB_API_KEY=your_api_key_here
```

4. Then run `python app.py` — posters and trending movies will appear!

---

## 🧠 How It Works

```
1. Dataset Loading
   movies.csv + credits.csv  →  merged DataFrame

2. Feature Extraction
   genres, keywords, overview, cast (top 5), director

3. Tag Building
   All features joined into a single "tags" string per movie

4. NLP Preprocessing
   Lowercase → Tokenize → Remove Stopwords

5. TF-IDF Vectorization
   10,000 features, unigrams + bigrams

6. Cosine Similarity Matrix
   (4800 × 4800) — measures angle between vectors

7. Recommendation
   Find movie index → sort by similarity score → return top 10
```

---

## 🎯 Example Queries

| Input Movie        | You'll Get Recommendations Like         |
|--------------------|-----------------------------------------|
| The Dark Knight    | Batman Begins, The Dark Knight Rises    |
| Inception          | The Matrix, Interstellar, Memento       |
| Avatar             | Guardians of the Galaxy, John Carter   |
| The Avengers       | Iron Man, Thor, Captain America         |

---

## 📊 Model Performance

- Dataset: TMDB 5000 movies
- Vectorizer: TF-IDF (max 10,000 features, 1–2 ngrams)
- Similarity: Cosine Similarity
- Approach: Content-Based Filtering

---

## 🙌 Credits

- Dataset: [TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle
- Poster API: [The Movie Database (TMDB)](https://www.themoviedb.org/)

---

*Built for college projects, hackathons, and ML portfolios.*
