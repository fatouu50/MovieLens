"""
🎬 Système de Recommandation de Films - MovieLens 100K
Interface Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_ratings, load_users, load_items, get_dataset_stats
from src.recommenders import (
    UserBasedCF, ItemBasedCF, ContentBasedRecommender, recommend_popular, compute_rmse
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎬 MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

GENRE_LIST = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (avec cache)
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    ratings = load_ratings('u.data')
    users = load_users()
    items = load_items()
    return ratings, users, items


@st.cache_resource
def train_models(ratings_df, items_df):
    user_cf = UserBasedCF(n_neighbors=30).fit(ratings_df)
    item_cf = ItemBasedCF(n_neighbors=20).fit(ratings_df)
    content_cb = ContentBasedRecommender().fit(items_df)
    return user_cf, item_cf, content_cb


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 MovieLens Recommender")
    st.caption("Dataset MovieLens 100K • 943 utilisateurs • 1682 films")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Accueil & Stats", "👤 Reco Utilisateur", "🎥 Films Similaires", "🎭 Par Genre", "📊 Évaluation"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Modèles disponibles :")
    st.caption("• User-Based CF")
    st.caption("• Item-Based CF")
    st.caption("• Content-Based (genres)")
    st.caption("• Popularité (baseline)")


# ─────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────

with st.spinner("Chargement des données..."):
    ratings_df, users_df, items_df = load_data()

with st.spinner("Entraînement des modèles..."):
    user_cf, item_cf, content_cb = train_models(ratings_df, items_df)


# ─────────────────────────────────────────────────────────────
# PAGE 1 : ACCUEIL & STATS
# ─────────────────────────────────────────────────────────────

if page == "🏠 Accueil & Stats":
    st.title("📊 Tableau de bord — Dataset MovieLens 100K")

    stats = get_dataset_stats(ratings_df, users_df, items_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👤 Utilisateurs", f"{stats['n_users']:,}")
    c2.metric("🎥 Films", f"{stats['n_items']:,}")
    c3.metric("⭐ Évaluations", f"{stats['n_ratings']:,}")
    c4.metric("⭐ Note moyenne", stats['avg_rating'])
    c5.metric("🕳️ Sparsité", f"{stats['sparsity']*100:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des notes")
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, y=rating_counts.values,
            labels={'x': 'Note', 'y': 'Nombre'},
            color=rating_counts.values,
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 genres les plus représentés")
        genre_counts = {}
        for _, row in items_df.iterrows():
            for g in row['genres']:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Nb Films']).sort_values('Nb Films', ascending=True).tail(10)
        fig2 = px.bar(genre_df, x='Nb Films', y='Genre', orientation='h',
                      color='Nb Films', color_continuous_scale='Teal')
        fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribution de l'âge des utilisateurs")
        fig3 = px.histogram(users_df, x='age', nbins=20, color_discrete_sequence=['#636EFA'])
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Répartition par genre (H/F)")
        gender_counts = users_df['gender'].value_counts()
        fig4 = px.pie(values=gender_counts.values, names=gender_counts.index,
                      color_discrete_sequence=['#636EFA', '#EF553B'])
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🏆 Films les plus populaires")
    popular = recommend_popular(ratings_df, items_df, n=10)
    st.dataframe(
        popular[['title', 'genres_str', 'year', 'avg_rating', 'n_ratings']].rename(columns={
            'title': 'Titre', 'genres_str': 'Genres', 'year': 'Année',
            'avg_rating': 'Note moy.', 'n_ratings': 'Nb votes'
        }),
        use_container_width=True, hide_index=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE 2 : RECOMMANDATIONS UTILISATEUR
# ─────────────────────────────────────────────────────────────

elif page == "👤 Reco Utilisateur":
    st.title("👤 Recommandations personnalisées")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_id = st.selectbox("Sélectionner un utilisateur (ID)", sorted(users_df['user_id'].tolist()))
    with col2:
        model_choice = st.selectbox("Modèle de recommandation", ["User-Based CF", "Item-Based CF"])
    with col3:
        n_recs = st.slider("Nombre de reco.", 5, 20, 10)

    # Profil utilisateur
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    user_stats = user_cf.get_user_stats(user_id)

    st.divider()
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Âge", user_info['age'])
    mc2.metric("Genre", "Homme" if user_info['gender'] == 'M' else "Femme")
    mc3.metric("Profession", user_info['occupation'].capitalize())
    mc4.metric("Films notés", user_stats.get('n_ratings', 0))

    # Films déjà notés
    with st.expander("📋 Historique des notes de cet utilisateur"):
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].merge(
            items_df[['item_id', 'title', 'genres_str']], on='item_id'
        ).sort_values('rating', ascending=False)
        st.dataframe(
            user_ratings[['title', 'genres_str', 'rating']].rename(
                columns={'title': 'Titre', 'genres_str': 'Genres', 'rating': 'Note'}
            ),
            use_container_width=True, hide_index=True
        )

    st.divider()
    st.subheader(f"🎯 Top {n_recs} recommandations — {model_choice}")

    with st.spinner("Calcul des recommandations..."):
        if model_choice == "User-Based CF":
            recs = user_cf.recommend(user_id, items_df, n=n_recs)
        else:
            recs = item_cf.recommend(user_id, ratings_df, items_df, n=n_recs)

    if recs.empty:
        st.warning("Aucune recommandation disponible pour cet utilisateur.")
    else:
        for i, (_, row) in enumerate(recs.iterrows()):
            with st.container():
                c1, c2, c3 = st.columns([4, 3, 1])
                c1.markdown(f"**{i+1}. {row['title']}**")
                c2.caption(f"🎭 {row['genres_str']}")
                c3.markdown(f"⭐ `{row['score']}`")


# ─────────────────────────────────────────────────────────────
# PAGE 3 : FILMS SIMILAIRES
# ─────────────────────────────────────────────────────────────

elif page == "🎥 Films Similaires":
    st.title("🎥 Films similaires")

    movie_titles = items_df[['item_id', 'title']].sort_values('title')
    title_to_id = dict(zip(movie_titles['title'], movie_titles['item_id']))

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_title = st.selectbox("Choisir un film", movie_titles['title'].tolist())
    with col2:
        sim_method = st.radio("Méthode", ["Item-Based CF", "Content-Based"])

    item_id = title_to_id[selected_title]
    item_info = items_df[items_df['item_id'] == item_id].iloc[0]

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    col_a.info(f"🎬 **Titre** : {item_info['title']}")
    col_b.info(f"🎭 **Genres** : {item_info['genres_str']}")
    col_c.info(f"📅 **Année** : {item_info['year']}")

    # Stats du film
    film_ratings = ratings_df[ratings_df['item_id'] == item_id]
    if not film_ratings.empty:
        st.caption(f"⭐ Note moyenne : **{film_ratings['rating'].mean():.2f}** sur {len(film_ratings)} votes")

    st.divider()
    st.subheader(f"Films similaires — {sim_method}")

    if sim_method == "Item-Based CF":
        similar = item_cf.get_similar_items(item_id, items_df, n=10)
    else:
        similar = content_cb.get_similar_items(item_id, n=10)

    if similar.empty:
        st.warning("Pas de films similaires trouvés.")
    else:
        fig = px.bar(
            similar.head(10), x='score', y='title',
            orientation='h', color='score',
            color_continuous_scale='Viridis',
            labels={'score': 'Score de similarité', 'title': ''},
        )
        fig.update_layout(coloraxis_showscale=False, height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            similar[['title', 'genres_str', 'year', 'score']].rename(columns={
                'title': 'Titre', 'genres_str': 'Genres', 'year': 'Année', 'score': 'Similarité'
            }),
            use_container_width=True, hide_index=True
        )


# ─────────────────────────────────────────────────────────────
# PAGE 4 : PAR GENRE
# ─────────────────────────────────────────────────────────────

elif page == "🎭 Par Genre":
    st.title("🎭 Recommandation par genre")

    selected_genres = st.multiselect(
        "Sélectionner un ou plusieurs genres",
        GENRE_LIST,
        default=["Action", "Adventure"]
    )

    n_recs = st.slider("Nombre de résultats", 5, 20, 10)

    if not selected_genres:
        st.warning("Veuillez sélectionner au moins un genre.")
    else:
        recs = content_cb.recommend_by_genre(selected_genres, n=n_recs, ratings_df=ratings_df)

        if recs.empty:
            st.warning("Aucun film trouvé pour ces genres.")
        else:
            st.subheader(f"🎯 Top {n_recs} films pour : {', '.join(selected_genres)}")
            for i, (_, row) in enumerate(recs.iterrows()):
                with st.container():
                    c1, c2, c3 = st.columns([4, 3, 1])
                    c1.markdown(f"**{i+1}. {row['title']}** ({row['year']})")
                    c2.caption(f"🎭 {row['genres_str']}")
                    c3.markdown(f"🎯 `{row['score']:.2f}`")

        # Heatmap genres disponibles
        st.divider()
        st.subheader("🔥 Distribution des genres dans le dataset")
        genre_counts = {}
        for _, row in items_df.iterrows():
            for g in row['genres']:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Films']).sort_values('Films', ascending=False)
        fig = px.bar(genre_df, x='Genre', y='Films', color='Films', color_continuous_scale='Sunset')
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 5 : ÉVALUATION
# ─────────────────────────────────────────────────────────────

elif page == "📊 Évaluation":
    st.title("📊 Évaluation des modèles")
    st.info("Calcul du RMSE (Root Mean Square Error) sur le split u1 (80K train / 20K test)")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧮 Calculer le RMSE (User-CF)", use_container_width=True):
            from src.data_loader import load_ratings
            with st.spinner("Entraînement sur u1.base et évaluation sur u1.test..."):
                train = load_ratings('u1.base')
                test = load_ratings('u1.test')
                eval_model = UserBasedCF(n_neighbors=30).fit(train)
                rmse = compute_rmse(eval_model, test, sample_size=500)
                if rmse:
                    st.success(f"✅ RMSE User-Based CF : **{rmse}**")
                else:
                    st.warning("RMSE non calculable.")

    with col2:
        st.metric("Baseline (note moy. globale)", "~1.03", delta=None, help="RMSE si on prédit toujours la note moyenne")

    st.divider()
    st.subheader("📈 Comparaison conceptuelle des modèles")

    model_comparison = pd.DataFrame({
        'Modèle': ['Popularité', 'User-Based CF', 'Item-Based CF', 'Content-Based'],
        'Type': ['Baseline', 'Collaboratif', 'Collaboratif', 'Contenu'],
        'Personnalisation': ['Faible', 'Haute', 'Haute', 'Moyenne'],
        'Cold Start User': ['OK', '❌', '⚠️', 'OK'],
        'Cold Start Item': ['OK', '❌', '❌', 'OK'],
        'Scalabilité': ['✅', '⚠️', '✅', '✅'],
        'Diversité': ['Faible', 'Haute', 'Moyenne', 'Faible'],
    })
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)

    st.subheader("📐 Définitions des métriques")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **RMSE** (Root Mean Square Error)  
        Mesure l'écart moyen entre la note prédite et la note réelle.  
        Plus la valeur est faible, meilleur est le modèle.  
        `RMSE = √( Σ(r̂ - r)² / n )`
        """)
    with c2:
        st.markdown("""
        **Sparsité** du dataset  
        Proportion de cases vides dans la matrice utilisateur-item.  
        `Sparsité = 1 - n_ratings / (n_users × n_items)`  
        Ici : **93.7%** de cases vides.
        """)
