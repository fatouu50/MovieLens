"""
Module de chargement et préparation des données MovieLens 100K
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]


def load_ratings(split='u.data'):
    """Charge les évaluations depuis un fichier de données."""
    path = os.path.join(DATA_DIR, split)
    sep = '\t' if split != 'u.data' else '\t'
    df = pd.read_csv(path, sep=sep, header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df


def load_users():
    """Charge les informations utilisateurs."""
    path = os.path.join(DATA_DIR, 'u.user')
    df = pd.read_csv(path, sep='|', header=None,
                     names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    return df


def load_items():
    """Charge les informations des films."""
    path = os.path.join(DATA_DIR, 'u.item')
    cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRE_COLS
    df = pd.read_csv(path, sep='|', header=None, names=cols,
                     encoding='latin-1', on_bad_lines='skip')
    df['genres'] = df[GENRE_COLS].apply(
        lambda row: [g for g, v in zip(GENRE_COLS, row) if v == 1], axis=1
    )
    df['genres_str'] = df['genres'].apply(lambda x: ', '.join(x) if x else 'Inconnu')
    df['year'] = df['release_date'].str.extract(r'(\d{4})').astype('Int64')
    return df


def load_genres():
    """Charge la liste des genres."""
    path = os.path.join(DATA_DIR, 'u.genre')
    df = pd.read_csv(path, sep='|', header=None, names=['genre', 'genre_id'])
    return df


def load_occupations():
    """Charge la liste des professions."""
    path = os.path.join(DATA_DIR, 'u.occupation')
    occupations = pd.read_csv(path, header=None, names=['occupation'])
    return occupations['occupation'].tolist()


def build_user_item_matrix(ratings_df):
    """Construit la matrice utilisateur-item."""
    matrix = ratings_df.pivot_table(
        index='user_id', columns='item_id', values='rating'
    )
    return matrix


def get_dataset_stats(ratings_df, users_df, items_df):
    """Retourne les statistiques générales du dataset."""
    return {
        'n_users': users_df['user_id'].nunique(),
        'n_items': items_df['item_id'].nunique(),
        'n_ratings': len(ratings_df),
        'avg_rating': round(ratings_df['rating'].mean(), 2),
        'sparsity': round(1 - len(ratings_df) / (users_df['user_id'].nunique() * items_df['item_id'].nunique()), 4),
        'ratings_per_user': round(ratings_df.groupby('user_id').size().mean(), 1),
        'ratings_per_item': round(ratings_df.groupby('item_id').size().mean(), 1),
    }
