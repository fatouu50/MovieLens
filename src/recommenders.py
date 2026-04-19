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

def jaccard_genres(genres_a, genres_b):
    """Calcule la similarité Jaccard entre deux listes de genres."""
    set_a = set(genres_a) if genres_a else set()
    set_b = set(genres_b) if genres_b else set()

    union = set_a | set_b
    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


# reduction de redondance avec la méthode MMR 

def mmr_rerank_items(candidates_df, items_df, n=10, lambda_param=0.75):
    """
    Réordonne les recommandations avec MMR pour réduire la redondance.

    lambda_param :
    - proche de 1 -> plus de pertinence
    - plus petit   -> plus de diversité
    """
    if candidates_df.empty:
        return candidates_df

    df = candidates_df.merge(
        items_df[['item_id', 'genres']],
        on='item_id',
        how='left'
    ).copy()

    selected = []
    remaining = df.to_dict('records')

    while remaining and len(selected) < n:
        best_item = None
        best_mmr_score = -float("inf")

        for candidate in remaining:
            relevance = float(candidate['score'])

            if not selected:
                redundancy_penalty = 0.0
            else:
                redundancy_penalty = max(
                    jaccard_genres(candidate['genres'], chosen['genres'])
                    for chosen in selected
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy_penalty

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_item = candidate

        selected.append(best_item)
        remaining.remove(best_item)

    result = pd.DataFrame(selected)
    return result.drop(columns=['genres'], errors='ignore')

class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.item_ids = None
        self.similarity_matrix = None
        self.items_df = None

    def fit(self, items_df):
        """Construit la matrice de similarité basée sur les genres."""
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(items_df['genres'])
        self.item_ids = items_df['item_id'].tolist()
        self.similarity_matrix = cosine_similarity(genre_matrix)
        self.items_df = items_df
        return self

    def get_similar_items(self, item_id, n=10, candidate_pool=30, lambda_param=0.75):
        """
        Retourne les films similaires avec réduction de redondance via MMR.
        """
        if item_id not in self.item_ids:
            return pd.DataFrame()

        item_idx = self.item_ids.index(item_id)
        sim_scores = self.similarity_matrix[item_idx]

        # On récupère plus de candidats que nécessaire
        similar_indices = np.argsort(sim_scores)[::-1][1:candidate_pool + 1]

        results = []
        for idx in similar_indices:
            results.append({
                'item_id': self.item_ids[idx],
                'score': float(sim_scores[idx])
            })

        results_df = pd.DataFrame(results)

        results_df = results_df.merge(
            self.items_df[['item_id', 'title', 'genres_str', 'year']],
            on='item_id',
            how='left'
        )

        diversified_df = mmr_rerank_items(
            results_df,
            self.items_df,
            n=n,
            lambda_param=lambda_param
        )

        diversified_df['score'] = diversified_df['score'].round(4)

        return diversified_df[['item_id', 'title', 'genres_str', 'year', 'score']]

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
        return df[['item_id', 'title', 'genres_str', 'year', 'score']]



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

