"""
data_loader.py — Chargement & préparation des données MovieLens 100K
"""

import os
import pandas as pd
import numpy as np

# Chemins par défaut
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_ratings(filename: str = 'u.data') -> pd.DataFrame:
    """
    Charge le fichier des évaluations MovieLens.
    Colonnes : user_id, item_id, rating, timestamp
    """
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(
        path,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    df['rating'] = df['rating'].astype(float)
    return df


def load_items() -> pd.DataFrame:
    """
    Charge le fichier des films MovieLens (u.item).
    Colonnes : item_id, title, year, genres, genres_str, genres_list
    """
    path = os.path.join(DATA_DIR, 'u.item')

    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's",
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_names

    df = pd.read_csv(
        path,
        sep='|',
        names=cols,
        encoding='latin-1',
        on_bad_lines='skip'
    )

    # Extraire l'année depuis le titre (ex: "Toy Story (1995)")
    df['year'] = df['title'].str.extract(r'\((\d{4})\)$').astype(float)

    # Construire la liste de genres
    def get_genres(row):
        return [g for g in genre_names if row.get(g, 0) == 1]

    df['genres_list'] = df.apply(get_genres, axis=1)
    df['genres'] = df['genres_list']  # alias
    df['genres_str'] = df['genres_list'].apply(lambda gs: ', '.join(gs) if gs else 'Unknown')

    # Garder uniquement les colonnes utiles
    df = df[['item_id', 'title', 'year', 'genres', 'genres_list', 'genres_str']].copy()

    return df


def get_dataset_stats(ratings_df: pd.DataFrame, items_df: pd.DataFrame) -> dict:
    """Retourne les statistiques générales du dataset."""
    return {
        'n_users':   ratings_df['user_id'].nunique(),
        'n_items':   len(items_df),
        'n_ratings': len(ratings_df),
        'avg_rating': round(ratings_df['rating'].mean(), 2),
        'sparsity':  round(
            1 - len(ratings_df) / (ratings_df['user_id'].nunique() * len(items_df)), 4
        ),
    }
