"""
Modèles de recommandation pour MovieLens 100K
- Filtrage Collaboratif Utilisateur-Utilisateur (User-Based CF)
- Filtrage Collaboratif Item-Item (Item-Based CF)
- Recommandation basée sur le contenu (Content-Based)
- Recommandation populaire (Baseline)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


# ─────────────────────────────────────────────────────────────
# BASELINE : Films les plus populaires
# ─────────────────────────────────────────────────────────────

def recommend_popular(ratings_df, items_df, n=10, min_ratings=50):
    """Recommande les films les plus populaires (mieux notés avec assez de votes)."""
    stats = ratings_df.groupby('item_id').agg(
        avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count')
    ).reset_index()
    stats = stats[stats['n_ratings'] >= min_ratings]
    stats = stats.sort_values('avg_rating', ascending=False).head(n)
    result = stats.merge(items_df[['item_id', 'title', 'genres_str', 'year']], on='item_id')
    result['score'] = result['avg_rating'].round(2)
    return result[['item_id', 'title', 'genres_str', 'year', 'avg_rating', 'n_ratings', 'score']]


# ─────────────────────────────────────────────────────────────
# FILTRAGE COLLABORATIF UTILISATEUR-UTILISATEUR
# ─────────────────────────────────────────────────────────────

class UserBasedCF:
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_ids = None

    def fit(self, ratings_df):
        """Entraîne le modèle sur les données de ratings."""
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', columns='item_id', values='rating'
        )
        self.user_ids = self.user_item_matrix.index.tolist()
        matrix_filled = self.user_item_matrix.fillna(0).values
        self.similarity_matrix = cosine_similarity(matrix_filled)
        return self

    def recommend(self, user_id, items_df, n=10, exclude_seen=True):
        """Génère des recommandations pour un utilisateur donné."""
        if user_id not in self.user_ids:
            return pd.DataFrame()

        user_idx = self.user_ids.index(user_id)
        sim_scores = self.similarity_matrix[user_idx]

        # Top N voisins (excluant l'utilisateur lui-même)
        neighbor_indices = np.argsort(sim_scores)[::-1][1:self.n_neighbors + 1]

        user_ratings = self.user_item_matrix.iloc[user_idx]
        seen_items = set(user_ratings.dropna().index)

        # Calcul des scores pondérés
        scores = {}
        for nb_idx in neighbor_indices:
            nb_sim = sim_scores[nb_idx]
            nb_ratings = self.user_item_matrix.iloc[nb_idx]
            for item_id, rating in nb_ratings.dropna().items():
                if exclude_seen and item_id in seen_items:
                    continue
                if item_id not in scores:
                    scores[item_id] = {'weighted_sum': 0, 'sim_sum': 0, 'n_raters': 0}
                scores[item_id]['weighted_sum'] += nb_sim * rating
                scores[item_id]['sim_sum'] += abs(nb_sim)
                scores[item_id]['n_raters'] += 1

        if not scores:
            return pd.DataFrame()

        results = []
        for item_id, vals in scores.items():
            if vals['sim_sum'] > 0:
                predicted = vals['weighted_sum'] / vals['sim_sum']
                results.append({'item_id': item_id, 'score': round(predicted, 2), 'n_raters': vals['n_raters']})

        results_df = pd.DataFrame(results).sort_values('score', ascending=False).head(n)
        results_df = results_df.merge(items_df[['item_id', 'title', 'genres_str', 'year']], on='item_id', how='left')
        return results_df[['item_id', 'title', 'genres_str', 'year', 'score', 'n_raters']]

    def get_user_stats(self, user_id):
        """Retourne des statistiques sur l'historique de l'utilisateur."""
        if user_id not in self.user_ids:
            return {}
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx].dropna()
        return {
            'n_ratings': len(user_ratings),
            'avg_rating': round(user_ratings.mean(), 2),
            'min_rating': int(user_ratings.min()),
            'max_rating': int(user_ratings.max()),
        }


# ─────────────────────────────────────────────────────────────
# FILTRAGE COLLABORATIF ITEM-ITEM
# ─────────────────────────────────────────────────────────────

class ItemBasedCF:
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_ids = None

    def fit(self, ratings_df):
        """Entraîne le modèle sur les données de ratings."""
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', columns='item_id', values='rating'
        )
        self.item_ids = self.user_item_matrix.columns.tolist()
        matrix_filled = self.user_item_matrix.fillna(0).values.T
        self.item_similarity = cosine_similarity(matrix_filled)
        return self

    def get_similar_items(self, item_id, items_df, n=10):
        """Retourne les films les plus similaires à un film donné."""
        if item_id not in self.item_ids:
            return pd.DataFrame()

        item_idx = self.item_ids.index(item_id)
        sim_scores = self.item_similarity[item_idx]
        similar_indices = np.argsort(sim_scores)[::-1][1:n + 1]

        results = []
        for idx in similar_indices:
            results.append({
                'item_id': self.item_ids[idx],
                'score': round(sim_scores[idx], 4)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.merge(items_df[['item_id', 'title', 'genres_str', 'year']], on='item_id', how='left')
        return results_df[['item_id', 'title', 'genres_str', 'year', 'score']]

    def recommend(self, user_id, ratings_df, items_df, n=10):
        """Recommande des films basé sur les films déjà notés par l'utilisateur."""
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
        if user_ratings.empty:
            return pd.DataFrame()

        seen_items = set(user_ratings['item_id'].tolist())
        scores = {}

        for _, row in user_ratings.iterrows():
            item_id = row['item_id']
            rating = row['rating']
            if item_id not in self.item_ids:
                continue
            item_idx = self.item_ids.index(item_id)
            sim_scores = self.item_similarity[item_idx]

            for nb_idx, sim in enumerate(sim_scores):
                nb_item_id = self.item_ids[nb_idx]
                if nb_item_id in seen_items:
                    continue
                if nb_item_id not in scores:
                    scores[nb_item_id] = {'weighted_sum': 0, 'sim_sum': 0}
                scores[nb_item_id]['weighted_sum'] += sim * rating
                scores[nb_item_id]['sim_sum'] += abs(sim)

        if not scores:
            return pd.DataFrame()

        results = []
        for item_id, vals in scores.items():
            if vals['sim_sum'] > 0:
                predicted = vals['weighted_sum'] / vals['sim_sum']
                results.append({'item_id': item_id, 'score': round(predicted, 2)})

        results_df = pd.DataFrame(results).sort_values('score', ascending=False).head(n)
        results_df = results_df.merge(items_df[['item_id', 'title', 'genres_str', 'year']], on='item_id', how='left')
        return results_df[['item_id', 'title', 'genres_str', 'year', 'score']]


# ─────────────────────────────────────────────────────────────
# RECOMMANDATION BASÉE SUR LE CONTENU
# ─────────────────────────────────────────────────────────────

class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.item_ids = None
        self.similarity_matrix = None

    def fit(self, items_df):
        """Construit la matrice de similarité basée sur les genres."""
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(items_df['genres'])
        self.item_ids = items_df['item_id'].tolist()
        self.similarity_matrix = cosine_similarity(genre_matrix)
        self.items_df = items_df
        return self

    def get_similar_items(self, item_id, n=10):
        """Retourne les films similaires en termes de genres."""
        if item_id not in self.item_ids:
            return pd.DataFrame()

        item_idx = self.item_ids.index(item_id)
        sim_scores = self.similarity_matrix[item_idx]
        similar_indices = np.argsort(sim_scores)[::-1][1:n + 1]

        results = []
        for idx in similar_indices:
            results.append({
                'item_id': self.item_ids[idx],
                'score': round(sim_scores[idx], 4)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.merge(
            self.items_df[['item_id', 'title', 'genres_str', 'year']], on='item_id', how='left'
        )
        return results_df[['item_id', 'title', 'genres_str', 'year', 'score']]

    def recommend_by_genre(self, genres, n=10, ratings_df=None, min_ratings=10):
        """Recommande les films correspondant à des genres sélectionnés."""
        df = self.items_df.copy()
        df['match'] = df['genres'].apply(
            lambda g: len(set(g) & set(genres)) / max(len(set(genres)), 1)
        )
        df = df[df['match'] > 0]

        if ratings_df is not None and not ratings_df.empty:
            stats = ratings_df.groupby('item_id').agg(
                avg_rating=('rating', 'mean'),
                n_ratings=('rating', 'count')
            ).reset_index()
            df = df.merge(stats, on='item_id', how='left')
            df = df[df['n_ratings'] >= min_ratings]
            df['score'] = df['match'] * 0.5 + (df['avg_rating'] / 5) * 0.5
        else:
            df['score'] = df['match']

        df = df.sort_values('score', ascending=False).head(n)
        return df[['item_id', 'title', 'genres_str', 'year', 'score']].rename(
            columns={'score': 'score'}
        )


# ─────────────────────────────────────────────────────────────
# ÉVALUATION RMSE
# ─────────────────────────────────────────────────────────────

def compute_rmse(user_cf_model, test_df, sample_size=500):
    """Calcule le RMSE du modèle User-CF sur un échantillon du test set."""
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    errors = []
    for _, row in sample.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        true_rating = row['rating']

        if user_id not in user_cf_model.user_ids:
            continue
        if item_id not in user_cf_model.user_item_matrix.columns:
            continue

        user_idx = user_cf_model.user_ids.index(user_id)
        item_col = user_cf_model.user_item_matrix.columns.tolist().index(item_id)
        sim_scores = user_cf_model.similarity_matrix[user_idx]
        neighbor_indices = np.argsort(sim_scores)[::-1][1:user_cf_model.n_neighbors + 1]

        weighted_sum = 0
        sim_sum = 0
        for nb_idx in neighbor_indices:
            nb_rating = user_cf_model.user_item_matrix.iloc[nb_idx, item_col]
            if pd.isna(nb_rating):
                continue
            nb_sim = sim_scores[nb_idx]
            weighted_sum += nb_sim * nb_rating
            sim_sum += abs(nb_sim)

        if sim_sum > 0:
            predicted = weighted_sum / sim_sum
            errors.append((predicted - true_rating) ** 2)

    if not errors:
        return None
    return round(np.sqrt(np.mean(errors)), 4)
