"""
recommenders.py — Modèles de recommandation MovieLens
  - ContentBasedRecommender : similarité cosinus sur les genres (MMR)
  - recommend_popular        : baseline par popularité
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Recommandation basée sur la similarité de genres (vecteurs binaires).
    Intègre MMR (Maximal Marginal Relevance) pour diversifier les résultats.
    """

    def __init__(self, mmr_lambda: float = 0.7):
        self.mmr_lambda = mmr_lambda
        self.items_df   = None
        self.item_ids   = None
        self.item_matrix = None
        self.sim_matrix  = None
        self._id_to_idx  = {}

    def fit(self, items_df: pd.DataFrame) -> 'ContentBasedRecommender':
        """
        Entraîne le modèle sur le DataFrame des films.
        Requiert les colonnes : item_id, genres (liste de str)
        """
        self.items_df = items_df.reset_index(drop=True).copy()

        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(
            self.items_df['genres'].apply(
                lambda g: g if isinstance(g, list) else []
            )
        )

        self.item_matrix = genre_matrix.astype(float)
        self.item_ids    = self.items_df['item_id'].tolist()
        self._id_to_idx  = {iid: i for i, iid in enumerate(self.item_ids)}
        self.sim_matrix  = cosine_similarity(self.item_matrix)

        return self

    def get_similar_items(self, item_id: int, n: int = 10) -> pd.DataFrame:
        """
        Retourne les n films les plus similaires à item_id,
        avec diversification MMR.
        """
        if item_id not in self._id_to_idx:
            return pd.DataFrame()

        idx = self._id_to_idx[item_id]
        scores = self.sim_matrix[idx].copy()
        scores[idx] = -1  # exclure le film lui-même

        # MMR
        selected_indices = []
        candidate_indices = list(range(len(self.item_ids)))
        candidate_indices.remove(idx)

        for _ in range(min(n, len(candidate_indices))):
            if not selected_indices:
                best = int(np.argmax([scores[i] for i in candidate_indices]))
                selected_indices.append(candidate_indices[best])
                candidate_indices.pop(best)
            else:
                mmr_scores = []
                for c in candidate_indices:
                    relevance = scores[c]
                    redundancy = max(
                        self.sim_matrix[c][s] for s in selected_indices
                    )
                    mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy
                    mmr_scores.append(mmr)
                best = int(np.argmax(mmr_scores))
                selected_indices.append(candidate_indices[best])
                candidate_indices.pop(best)

        result_ids = [self.item_ids[i] for i in selected_indices]
        result_scores = [scores[i] for i in selected_indices]

        result_df = self.items_df[self.items_df['item_id'].isin(result_ids)].copy()
        score_map = dict(zip(result_ids, result_scores))
        result_df['similarity'] = result_df['item_id'].map(score_map)
        result_df = result_df.sort_values('similarity', ascending=False).reset_index(drop=True)

        return result_df


def recommend_popular(ratings_df: pd.DataFrame, items_df: pd.DataFrame,
                      min_ratings: int = 20, n: int = 20) -> pd.DataFrame:
    """
    Recommandation baseline par popularité.
    Classe les films par note moyenne (avec seuil minimum de votes).
    """
    stats = ratings_df.groupby('item_id').agg(
        avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count')
    ).reset_index()

    popular = stats[stats['n_ratings'] >= min_ratings].copy()
    popular = popular.sort_values('avg_rating', ascending=False)
    popular = popular.merge(items_df[['item_id', 'title', 'genres', 'genres_str', 'year']],
                            on='item_id', how='left')

    return popular.head(n).reset_index(drop=True)
