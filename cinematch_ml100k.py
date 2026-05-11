"""
CineMatch - Movie Recommendation Model
Dataset : MovieLens ML-100K
Model   : TF-IDF (genre soup) + Cosine Similarity
          Weighted blend: 0.8 * text_sim + 0.2 * numeric_sim
          Genre diversity penalty (max 3 per primary genre)

Requirements:
    pip install pandas numpy scikit-learn

Usage:
    python cinematch_ml100k.py
    >>> recommend("Toy Story")
    >>> recommend("Star Wars")
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "ml-100k"          # path to the ml-100k folder
MIN_VOTES      = 10                 # minimum ratings a movie must have
TEXT_WEIGHT    = 0.8                # weight for TF-IDF cosine similarity
NUMERIC_WEIGHT = 0.2                # weight for numeric (rating + year) similarity
MAX_RECS       = 10                 # top-N recommendations to return
MAX_GENRE_REPS = 3                  # max movies per primary genre in results
TFIDF_FEATURES = 7000               # TF-IDF vocabulary size
TFIDF_NGRAMS   = (1, 2)            # unigrams + bigrams

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(data_dir: str) -> pd.DataFrame:
    """Load and merge u.item + u.data, return clean DataFrame."""
    items = pd.read_csv(
        f"{data_dir}/u.item", sep="|", encoding="latin-1",
        names=["item_id", "title", "release", "video_release", "imdb"] + GENRE_COLS,
        usecols=["item_id", "title"] + GENRE_COLS,
    )

    ratings_raw = pd.read_csv(
        f"{data_dir}/u.data", sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    avg_ratings = (ratings_raw.groupby("item_id")["rating"]
                               .mean().reset_index()
                               .rename(columns={"rating": "avg_rating"}))
    vote_counts = (ratings_raw.groupby("item_id")["rating"]
                               .count().reset_index()
                               .rename(columns={"rating": "votes"}))

    df = items.merge(avg_ratings, on="item_id").merge(vote_counts, on="item_id")
    df = df[df["votes"] >= MIN_VOTES].copy()
    df["avg_rating"] = df["avg_rating"].round(2)

    df["genres_str"]  = df.apply(_make_genre_str, axis=1)
    df["year"]        = df["title"].apply(_extract_year)
    df["clean_title"] = df["title"].apply(_clean_title)
    df = df.reset_index(drop=True)

    return df


def _make_genre_str(row) -> str:
    genres = [g for g in GENRE_COLS if row[g] == 1]
    return " ".join(genres) if genres else "Unknown"


def _extract_year(title: str) -> int:
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else 1995


def _clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


# ── Feature engineering ────────────────────────────────────────────────────────
def build_soup(row) -> str:
    """
    Genre string repeated twice (double weight) + lowercased title words.
    This is the same 'soup' used in the original CineMatch model.
    """
    genres_doubled = (row["genres_str"] + " ") * 2
    title_words    = re.sub(r"[^a-z0-9 ]", " ", row["clean_title"].lower())
    return f"{genres_doubled}{title_words}"


# ── Model training ─────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    """
    Fit TF-IDF on genre soup, compute cosine similarity matrices,
    blend text + numeric similarity.

    Returns
    -------
    sim : np.ndarray  shape (n_movies, n_movies)
    """
    df["soup"] = df.apply(build_soup, axis=1)

    # Text similarity
    tfidf  = TfidfVectorizer(
        ngram_range=TFIDF_NGRAMS,
        max_features=TFIDF_FEATURES,
        sublinear_tf=True,
        min_df=2,
    )
    T        = tfidf.fit_transform(df["soup"])
    text_sim = cosine_similarity(T)

    # Numeric similarity  (avg_rating + year, both scaled 0-1)
    scaler   = MinMaxScaler()
    N        = scaler.fit_transform(df[["avg_rating", "year"]])
    num_sim  = cosine_similarity(N)

    # Weighted blend
    sim = TEXT_WEIGHT * text_sim + NUMERIC_WEIGHT * num_sim

    print(f"Model trained on {len(df)} movies | "
          f"TF-IDF vocab: {T.shape[1]} features")
    return sim


# ── Recommendation ─────────────────────────────────────────────────────────────
def get_recommendations(
    title:    str,
    df:       pd.DataFrame,
    sim:      np.ndarray,
    n:        int = MAX_RECS,
) -> pd.DataFrame:
    """
    Return top-N recommendations for a given movie title.
    Applies genre diversity penalty: max MAX_GENRE_REPS per primary genre.

    Parameters
    ----------
    title : str   exact or partial movie title (case-insensitive search)
    df    : DataFrame returned by load_data()
    sim   : similarity matrix returned by train()
    n     : number of recommendations

    Returns
    -------
    pd.DataFrame with columns: title, genres_str, year, avg_rating, score
    """
    # Fuzzy title match
    mask = df["clean_title"].str.lower().str.contains(title.lower(), regex=False)
    if not mask.any():
        print(f'No movie found matching "{title}"')
        return pd.DataFrame()

    idx = df[mask].index[0]
    matched = df.loc[idx, "clean_title"]
    print(f'Recommendations for: "{matched}" ({int(df.loc[idx, "year"])})')

    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx]

    primary_count = {}
    results = []
    for s_idx, score in scores:
        if len(results) >= n:
            break
        row     = df.iloc[s_idx]
        primary = row["genres_str"].split()[0]
        if primary_count.get(primary, 0) >= MAX_GENRE_REPS:
            continue
        primary_count[primary] = primary_count.get(primary, 0) + 1
        results.append({
            "title":      row["clean_title"],
            "genres":     row["genres_str"],
            "year":       int(row["year"]),
            "avg_rating": row["avg_rating"],
            "score":      round(float(score), 4),
        })

    return pd.DataFrame(results)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading ML-100K dataset...")
    df  = load_data(DATA_DIR)

    print("Training model...")
    sim = train(df)

    # Demo queries
    for query in ["Toy Story", "Star Wars", "Fargo", "Pulp Fiction"]:
        print("\n" + "─" * 55)
        recs = get_recommendations(query, df, sim)
        if not recs.empty:
            print(recs.to_string(index=False))

    # Interactive mode
    print("\n" + "═" * 55)
    print("Interactive mode — type a movie name (or 'q' to quit)")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
        recs = get_recommendations(query, df, sim)
        if not recs.empty:
            print(recs.to_string(index=False))
