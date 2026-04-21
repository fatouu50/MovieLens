"""
recommender.py — Moteur de recommandation de films
Implémente :
  - User-Based Collaborative Filtering (cosine similarity)
  - Item-Based Collaborative Filtering (cosine similarity)
  - Content-Based Filtering (profil genres utilisateur)
  - Recommandation LIVE (mise à jour instantanée dès qu'une note change)
  - Gestion de la REDONDANCE (dé-duplication inter-méthodes)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from functools import lru_cache
import streamlit as st

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
ITEM_COLS = [
    "movie_id", "title", "release_date", "video_date", "imdb_url",
] + GENRE_COLS


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (mis en cache Streamlit)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement du dataset MovieLens…")
def load_data():
    ratings = pd.read_csv(
        "data/u.data", sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        "data/u.item", sep="|",
        names=ITEM_COLS, encoding="latin-1",
    )
    # Nettoyage : extraire l'année du titre
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").fillna("")
    movies["genres_list"] = movies[GENRE_COLS].apply(
        lambda row: [g for g in GENRE_COLS if row[g] == 1], axis=1
    )
    return ratings, movies


@st.cache_data(show_spinner="Construction de la matrice user-film…")
def build_user_movie_matrix(ratings_hash: str):
    ratings, _ = load_data()
    matrix = ratings.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    )
    return matrix


@st.cache_data(show_spinner="Calcul des similarités utilisateurs…")
def get_user_similarity(ratings_hash: str):
    matrix = build_user_movie_matrix(ratings_hash)
    filled = matrix.fillna(0)
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


@st.cache_data(show_spinner="Calcul des similarités films…")
def get_item_similarity(ratings_hash: str):
    matrix = build_user_movie_matrix(ratings_hash)
    filled = matrix.fillna(0).T
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


@st.cache_data(show_spinner="Calcul des similarités content-based…")
def get_content_similarity():
    _, movies = load_data()
    feat = movies.set_index("movie_id")[GENRE_COLS].astype(float)
    sim = cosine_similarity(feat)
    return pd.DataFrame(sim, index=feat.index, columns=feat.index), feat


def _ratings_hash(ratings: pd.DataFrame) -> str:
    """Hash stable des données de ratings pour invalider le cache si besoin."""
    return str(hash(pd.util.hash_pandas_object(ratings).sum()))


# ─────────────────────────────────────────────
# USER-BASED COLLABORATIVE FILTERING
# ─────────────────────────────────────────────

def recommend_user_based(
    user_ratings: dict,
    n: int = 5,
    n_neighbors: int = 20,
    exclude_ids: set = None,
) -> pd.DataFrame:
    """
    Recommandation User-Based.
    user_ratings : {movie_id(int): rating(float)}
    Retourne un DataFrame Top-N avec colonnes [movie_id, title, genres_list, score, method]
    """
    ratings, movies = load_data()
    rh = _ratings_hash(ratings)
    matrix = build_user_movie_matrix(rh)
    user_sim_df = get_user_similarity(rh)

    # Construire le vecteur du nouvel utilisateur
    new_vec = pd.Series(0.0, index=matrix.columns)
    for mid, r in user_ratings.items():
        if int(mid) in new_vec.index:
            new_vec[int(mid)] = r

    # Similarité avec tous les utilisateurs existants
    filled = matrix.fillna(0)
    sims = cosine_similarity(new_vec.values.reshape(1, -1), filled.values)[0]
    sim_series = pd.Series(sims, index=matrix.index).sort_values(ascending=False)
    neighbors = sim_series.head(n_neighbors)

    rated_ids = {int(k) for k in user_ratings.keys()}
    if exclude_ids:
        rated_ids |= exclude_ids

    scores = {}
    for movie_id in matrix.columns:
        if movie_id in rated_ids:
            continue
        num, den = 0.0, 0.0
        for uid, sim_score in neighbors.items():
            r = matrix.loc[uid, movie_id]
            if not np.isnan(r):
                num += sim_score * r
                den += abs(sim_score)
        if den > 0:
            scores[movie_id] = num / den

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return _build_result_df(top, movies, method="👥 User-Based")


# ─────────────────────────────────────────────
# ITEM-BASED COLLABORATIVE FILTERING
# ─────────────────────────────────────────────

def recommend_item_based(
    user_ratings: dict,
    n: int = 5,
    exclude_ids: set = None,
) -> pd.DataFrame:
    ratings, movies = load_data()
    rh = _ratings_hash(ratings)
    matrix = build_user_movie_matrix(rh)
    item_sim_df = get_item_similarity(rh)

    rated_ids = {int(k) for k in user_ratings.keys()}
    if exclude_ids:
        rated_ids |= exclude_ids

    scores = {}
    for movie_id in matrix.columns:
        if movie_id in rated_ids:
            continue
        if movie_id not in item_sim_df.index:
            continue
        num, den = 0.0, 0.0
        for mid, r in user_ratings.items():
            mid = int(mid)
            if mid in item_sim_df.columns:
                sim = item_sim_df.loc[movie_id, mid]
                num += sim * r
                den += abs(sim)
        if den > 0:
            scores[movie_id] = num / den

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return _build_result_df(top, movies, method="🎬 Item-Based")


# ─────────────────────────────────────────────
# CONTENT-BASED FILTERING
# ─────────────────────────────────────────────

def recommend_content_based(
    user_ratings: dict,
    n: int = 5,
    exclude_ids: set = None,
) -> pd.DataFrame:
    _, movies = load_data()
    content_sim_df, feat = get_content_similarity()

    # Profil utilisateur = moyenne pondérée des genres des films bien notés
    liked = {int(mid): r for mid, r in user_ratings.items() if float(r) >= 3.5}
    if not liked:
        liked = {int(mid): r for mid, r in user_ratings.items()}

    profile = np.zeros(len(GENRE_COLS))
    total_w = 0.0
    for mid, r in liked.items():
        if mid in feat.index:
            profile += feat.loc[mid].values * r
            total_w += r
    if total_w > 0:
        profile /= total_w

    rated_ids = {int(k) for k in user_ratings.keys()}
    if exclude_ids:
        rated_ids |= exclude_ids

    scores = {}
    for movie_id in feat.index:
        if movie_id in rated_ids:
            continue
        sim = cosine_similarity(
            profile.reshape(1, -1),
            feat.loc[movie_id].values.reshape(1, -1),
        )[0][0]
        scores[movie_id] = float(sim)

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return _build_result_df(top, movies, method="🏷️ Content-Based")


# ─────────────────────────────────────────────
# RECOMMANDATION LIVE
# ─────────────────────────────────────────────

def recommend_live(
    user_ratings: dict,
    method: str = "content",
    n: int = 5,
) -> pd.DataFrame:
    """
    Recommandation LIVE : appelée à chaque mise à jour d'une note.
    Utilise st.cache_data avec TTL court pour être quasi-instantanée.
    """
    if not user_ratings:
        return pd.DataFrame()

    if method == "user":
        return recommend_user_based(user_ratings, n=n)
    elif method == "item":
        return recommend_item_based(user_ratings, n=n)
    else:
        return recommend_content_based(user_ratings, n=n)


# ─────────────────────────────────────────────
# RECOMMANDATION SANS REDONDANCE (fusion des 3 méthodes)
# ─────────────────────────────────────────────

def recommend_no_redundancy(
    user_ratings: dict,
    n_per_method: int = 5,
    final_n: int = 10,
) -> pd.DataFrame:
    """
    Fusionne les résultats des 3 méthodes en éliminant les doublons.
    Stratégie : score moyen si un film apparaît dans plusieurs méthodes.
    Retourne les `final_n` meilleurs films uniques.
    """
    if not user_ratings:
        return pd.DataFrame()

    dfs = []
    # Chaque méthode calcule ses Top-N indépendamment
    for fn, method_name in [
        (recommend_user_based,   "User-Based"),
        (recommend_item_based,   "Item-Based"),
        (recommend_content_based,"Content-Based"),
    ]:
        try:
            df = fn(user_ratings, n=n_per_method)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Dé-duplication : grouper par movie_id, moyenner les scores
    agg = (
        combined.groupby("movie_id")
        .agg(
            title       = ("title",      "first"),
            genres_list = ("genres_list","first"),
            score       = ("score",      "mean"),
            methods     = ("method",     lambda x: " + ".join(sorted(set(x)))),
            appearances = ("method",     "count"),
        )
        .reset_index()
        .sort_values(["appearances", "score"], ascending=[False, False])
        .head(final_n)
    )
    return agg


# ─────────────────────────────────────────────
# ÉVALUATION RMSE
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Calcul du RMSE (évaluation)…")
def evaluate_rmse(sample_size: int = 300) -> dict:
    """
    Évalue les deux approches collaboratives avec RMSE sur un échantillon test.
    """
    from sklearn.model_selection import train_test_split

    ratings, _ = load_data()
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    sample = test.sample(min(sample_size, len(test)), random_state=42)

    # Matrice d'entraînement
    train_matrix = train.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)

    # USER-BASED RMSE
    user_sim = cosine_similarity(train_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=train_matrix.index, columns=train_matrix.index)

    preds_ub, actuals_ub = [], []
    for _, row in sample.iterrows():
        uid, mid, actual = int(row.user_id), int(row.movie_id), row.rating
        if uid not in user_sim_df.index or mid not in train_matrix.columns:
            continue
        neighbors = user_sim_df[uid].sort_values(ascending=False)[1:21]
        num = sum(s * train_matrix.loc[u, mid] for u, s in neighbors.items()
                  if train_matrix.loc[u, mid] > 0)
        den = sum(abs(s) for u, s in neighbors.items()
                  if train_matrix.loc[u, mid] > 0)
        if den > 0:
            preds_ub.append(num / den)
            actuals_ub.append(actual)

    rmse_ub = float(np.sqrt(mean_squared_error(actuals_ub, preds_ub))) if preds_ub else None

    # ITEM-BASED RMSE
    item_sim = cosine_similarity(train_matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=train_matrix.columns, columns=train_matrix.columns)

    preds_ib, actuals_ib = [], []
    for _, row in sample.iterrows():
        uid, mid, actual = int(row.user_id), int(row.movie_id), row.rating
        if uid not in train_matrix.index or mid not in item_sim_df.index:
            continue
        user_row = train_matrix.loc[uid]
        rated = user_row[user_row > 0]
        if rated.empty:
            continue
        sims = item_sim_df[mid][rated.index]
        den = sims.abs().sum()
        if den > 0:
            preds_ib.append((sims * rated).sum() / den)
            actuals_ib.append(actual)

    rmse_ib = float(np.sqrt(mean_squared_error(actuals_ib, preds_ib))) if preds_ib else None

    return {
        "user_based":  round(rmse_ub, 4) if rmse_ub else "N/A",
        "item_based":  round(rmse_ib, 4) if rmse_ib else "N/A",
        "best":        "User-Based" if (rmse_ub and rmse_ib and rmse_ub < rmse_ib) else "Item-Based",
        "n_evaluated": len(preds_ub),
    }


# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────

def _build_result_df(top: list, movies: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for movie_id, score in top:
        m = movies[movies["movie_id"] == movie_id]
        if m.empty:
            continue
        m = m.iloc[0]
        rows.append({
            "movie_id":   movie_id,
            "title":      m["title"],
            "genres_list":m["genres_list"],
            "year":       m["year"],
            "score":      round(float(score), 4),
            "method":     method,
        })
    return pd.DataFrame(rows)


def get_movie_stats(movie_id: int) -> dict:
    ratings, _ = load_data()
    subset = ratings[ratings["movie_id"] == movie_id]
    if subset.empty:
        return {}
    return {
        "avg":   round(subset["rating"].mean(), 2),
        "count": len(subset),
        "dist":  subset["rating"].value_counts().sort_index().to_dict(),
    }


def get_all_genres() -> list:
    return [g for g in GENRE_COLS if g != "unknown"]


def get_movies_by_genre(genre: str, limit: int = 30) -> pd.DataFrame:
    ratings, movies = load_data()
    avg = ratings.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
    avg.columns = ["movie_id", "avg_rating", "num_ratings"]
    merged = movies.merge(avg, on="movie_id")
    filtered = merged[merged[genre] == 1].sort_values("num_ratings", ascending=False)
    return filtered.head(limit)[["movie_id", "title", "genres_list", "year", "avg_rating", "num_ratings"]]
