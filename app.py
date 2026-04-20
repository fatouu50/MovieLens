"""
app.py — CineMatch : Système de Recommandation de Films
Streamlit App — MovieLens 100K
"""

import streamlit as st
import pandas as pd
import time
import hashlib

from auth import (
    register_user, login_user, verify_token,
    get_user_data, save_user_ratings, save_user_genres,
)
from recommender import (
    load_data, recommend_live, recommend_no_redundancy,
    evaluate_rmse, get_all_genres, get_movies_by_genre,
    recommend_user_based, recommend_item_based, recommend_content_based,
)

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Recommandation de Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)

# ─────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  /* HEADER */
  .main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(135deg, #f5c842, #e8a820);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .sub-title { color: #7070a0; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase; }
  /* CARDS */
  .rec-card {
    background: #13131a; border: 1px solid #2a2a3a;
    border-radius: 14px; padding: 18px 20px;
    margin-bottom: 12px; transition: border-color .2s;
  }
  .rec-card:hover { border-color: #f5c842; }
  .rec-rank { color: #f5c842; font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
  .rec-title { font-family: 'Playfair Display', serif; font-size: 1.05rem; font-weight: 700; margin: 4px 0 8px; }
  .genre-tag {
    display: inline-block; padding: 2px 10px;
    background: #1c1c26; border: 1px solid #2a2a3a;
    border-radius: 100px; font-size: 0.72rem; color: #7070a0;
    margin: 2px;
  }
  .method-badge {
    display: inline-block; padding: 3px 10px;
    background: rgba(245,200,66,0.12); border: 1px solid rgba(245,200,66,0.3);
    border-radius: 100px; font-size: 0.72rem; color: #f5c842;
  }
  .multi-badge {
    background: rgba(124,92,252,0.15); border-color: rgba(124,92,252,0.4); color: #a78bfa;
  }
  /* METRIC CARDS */
  .metric-box {
    background: #13131a; border: 1px solid #2a2a3a;
    border-radius: 12px; padding: 16px 20px; text-align: center;
  }
  .metric-val { font-size: 1.8rem; font-weight: 700; color: #f5c842; }
  .metric-lbl { font-size: 0.78rem; color: #7070a0; margin-top: 2px; }
  /* LIVE BADGE */
  .live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #5cfca0; border-radius: 50%;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .5; transform: scale(1.4); }
  }
  /* SCORE BAR */
  .score-bar-wrap { background: #2a2a3a; border-radius: 3px; height: 4px; margin-top: 8px; }
  .score-bar-fill { height: 4px; border-radius: 3px; background: linear-gradient(90deg,#f5c842,#e8a820); }
  /* SECTION DIVIDER */
  .section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700;
    border-bottom: 1px solid #2a2a3a;
    padding-bottom: 8px; margin: 24px 0 16px;
  }
  /* STAR RATING */
  .star-row { font-size: 1.3rem; cursor: pointer; }
  /* SIDEBAR MOBILE FIX */
  [data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
  }
  section[data-testid="stSidebar"] {
    display: block !important;
  }
  /* HIDE STREAMLIT BRANDING */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  header [data-testid="collapsedControl"] {visibility: visible !important;}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "jwt_token":     None,
        "user_email":    None,
        "user_name":     None,
        "user_ratings":  {},
        "genre_prefs":   [],
        "live_method":   "content",
        "auth_tab":      "login",
        "show_rmse":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def is_logged_in() -> bool:
    token = st.session_state.get("jwt_token")
    if not token:
        return False
    payload = verify_token(token)
    return payload is not None

def logout():
    for k in ["jwt_token","user_email","user_name","user_ratings","genre_prefs"]:
        st.session_state[k] = None if k == "jwt_token" else ([] if k in ["genre_prefs"] else ({} if k == "user_ratings" else ""))
    st.rerun()

def render_star_rating(movie_id: int, current: int, key_prefix: str = "") -> int:
    """Affiche 5 étoiles cliquables et retourne la note choisie."""
    stars = ""
    for i in range(1, 6):
        filled = "★" if i <= current else "☆"
        color  = "#f5c842" if i <= current else "#3a3a5a"
        stars += f'<span style="color:{color};font-size:1.3rem">{filled}</span>'
    st.markdown(stars, unsafe_allow_html=True)
    return st.select_slider(
        "", options=[1, 2, 3, 4, 5],
        value=current if current > 0 else 3,
        key=f"{key_prefix}_{movie_id}",
        label_visibility="collapsed",
    )

def render_rec_card(rank: int, row, show_method: bool = True):
    genres_html = "".join(
        f'<span class="genre-tag">{g}</span>'
        for g in (row.genres_list if isinstance(row.genres_list, list) else [])[:5]
    )
    method_html = ""
    if show_method and hasattr(row, "method"):
        extra = " multi-badge" if "+" in str(row.method) else ""
        method_html = f'<span class="method-badge{extra}">{row.method}</span>'
    appearances = getattr(row, "appearances", 1)
    consensus = f'<span style="color:#7070a0;font-size:0.72rem">{"🔥 Consensus " + str(appearances) + " méthodes" if appearances > 1 else ""}</span>'
    score_pct = min(int(float(row.score) * 100 / 5 * 100), 100) if float(row.score) <= 5 else min(int(float(row.score) * 20), 100)

    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-rank">#{rank} Recommandation</div>
      <div class="rec-title">{row.title}</div>
      <div>{genres_html}</div>
      <div style="margin-top:8px">{method_html} {consensus}</div>
      <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{score_pct}%"></div>
      </div>
      <div style="font-size:0.75rem;color:#7070a0;margin-top:4px">Score : {float(row.score):.4f}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE AUTH
# ─────────────────────────────────────────────
def page_auth():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="main-title">🎬 CineMatch</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Système de Recommandation de Films</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔑 Connexion", "✨ Inscription"])

        # ── CONNEXION ──
        with tab_login:
            st.markdown("#### Bienvenue !")
            email = st.text_input("Adresse e-mail", key="login_email", placeholder="vous@exemple.com")
            password = st.text_input("Mot de passe", type="password", key="login_pwd", placeholder="••••••••")

            if st.button("Se connecter →", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Veuillez remplir tous les champs.")
                else:
                    with st.spinner("Vérification…"):
                        ok, msg, token = login_user(email, password)
                    if ok:
                        st.success(msg)
                        st.session_state.jwt_token  = token
                        st.session_state.user_email = email.lower().strip()
                        data = get_user_data(email)
                        st.session_state.user_name    = data["name"]
                        st.session_state.user_ratings = {str(k): v for k, v in data["ratings"].items()}
                        st.session_state.genre_prefs  = data["genre_prefs"]
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("---")
            st.caption("🔒 Connexion sécurisée — JWT + bcrypt")

        # ── INSCRIPTION ──
        with tab_register:
            st.markdown("#### Créer un compte")
            name     = st.text_input("Nom complet", key="reg_name", placeholder="Votre nom")
            email_r  = st.text_input("Adresse e-mail", key="reg_email", placeholder="vous@exemple.com")
            pwd_r    = st.text_input("Mot de passe", type="password", key="reg_pwd",
                                     placeholder="Min. 8 car., 1 maj., 1 chiffre, 1 spécial")
            pwd_r2   = st.text_input("Confirmer le mot de passe", type="password", key="reg_pwd2")

            st.markdown("""
            <div style="font-size:0.78rem;color:#7070a0;margin-bottom:8px">
            🔐 Exigences : 8 caractères min · 1 majuscule · 1 chiffre · 1 caractère spécial
            </div>
            """, unsafe_allow_html=True)

            if st.button("Créer mon compte →", use_container_width=True, type="primary"):
                if pwd_r != pwd_r2:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    with st.spinner("Création du compte…"):
                        ok, msg = register_user(name, email_r, pwd_r)
                    if ok:
                        st.success(f"✅ {msg} Vous pouvez maintenant vous connecter.")
                    else:
                        st.error(msg)


# ─────────────────────────────────────────────
# PAGE PRINCIPALE (APP)
# ─────────────────────────────────────────────
def page_app():
    ratings_df, movies_df = load_data()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown(f'<div class="main-title" style="font-size:1.5rem">🎬 CineMatch</div>', unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.user_name}**")
        st.caption(f"✉️ {st.session_state.user_email}")
        st.markdown("---")

        # Navigation
        page = st.radio("Navigation", [
            "🏠 Accueil",
            "⭐ Noter des films",
            "✨ Mes recommandations",
            "🔄 Sans redondance (fusion)",
            "📊 Évaluation & Comparaison",
            "🔴 Live Recommandation",
        ], label_visibility="collapsed")

        st.markdown("---")

        # Genres favoris
        st.markdown("**🎭 Genres favoris**")
        all_genres = get_all_genres()
        for g in all_genres:
            val = g in st.session_state.genre_prefs
            if st.checkbox(g, value=val, key=f"genre_{g}"):
                if g not in st.session_state.genre_prefs:
                    st.session_state.genre_prefs.append(g)
                    save_user_genres(st.session_state.user_email, st.session_state.genre_prefs)
            else:
                if g in st.session_state.genre_prefs:
                    st.session_state.genre_prefs.remove(g)
                    save_user_genres(st.session_state.user_email, st.session_state.genre_prefs)

        st.markdown("---")
        if st.button("🚪 Déconnexion", use_container_width=True):
            logout()

    # ─────────────────────
    # HOME
    # ─────────────────────
    if page == "🏠 Accueil":
        st.markdown(f'<div class="main-title">Bonjour, {st.session_state.user_name.split()[0]} 👋</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Votre espace de recommandation personnalisé</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Métriques
        n_rated = len(st.session_state.user_ratings)
        avg_note = (sum(float(v) for v in st.session_state.user_ratings.values()) / n_rated) if n_rated else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{n_rated}</div><div class="metric-lbl">Films notés</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_note:.1f}⭐</div><div class="metric-lbl">Note moyenne</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{len(st.session_state.genre_prefs)}</div><div class="metric-lbl">Genres favoris</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-val">100K</div><div class="metric-lbl">Ratings dataset</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dataset info
        st.markdown('<div class="section-header">📂 À propos du dataset</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
**MovieLens 100K**
- 🎬 **{len(movies_df):,}** films
- 👥 **{ratings_df['user_id'].nunique():,}** utilisateurs
- ⭐ **{len(ratings_df):,}** évaluations
- Note moyenne : **{ratings_df['rating'].mean():.2f} / 5**
""")
        with col2:
            dist = ratings_df['rating'].value_counts().sort_index()
            st.bar_chart(dist, height=160)

        # Guide rapide
        st.markdown('<div class="section-header">🚀 Comment utiliser CineMatch</div>', unsafe_allow_html=True)
        steps = [
            ("1️⃣", "**Notez des films** — Allez dans « ⭐ Noter des films » et évaluez au moins 5 films"),
            ("2️⃣", "**Choisissez une méthode** — User-Based, Item-Based ou Content-Based"),
            ("3️⃣", "**Obtenez vos recommandations** — Top-5 personnalisées pour vous"),
            ("4️⃣", "**Mode Live** — Vos recommandations se mettent à jour en temps réel"),
            ("5️⃣", "**Sans redondance** — Fusion des 3 méthodes sans doublons"),
        ]
        for icon, text in steps:
            st.markdown(f"{icon} {text}")

    # ─────────────────────
    # NOTER DES FILMS
    # ─────────────────────
    elif page == "⭐ Noter des films":
        st.markdown('<div class="main-title" style="font-size:1.8rem">⭐ Noter des films</div>', unsafe_allow_html=True)
        st.caption(f"Films notés : **{len(st.session_state.user_ratings)}** — Notez au moins 5 films pour de meilleures recommandations")
        st.markdown("<br>", unsafe_allow_html=True)

        # Filtre par genre
        filter_genre = st.selectbox(
            "Filtrer par genre",
            ["Tous"] + get_all_genres(),
            key="filter_genre_selector"
        )

        # Charger les films à afficher
        if filter_genre == "Tous":
            avg = ratings_df.groupby("movie_id")["rating"].agg(["mean","count"]).reset_index()
            avg.columns = ["movie_id","avg_rating","num_ratings"]
            pool = movies_df.merge(avg, on="movie_id").sort_values("num_ratings", ascending=False).head(60)
        else:
            pool = get_movies_by_genre(filter_genre, limit=40)

        # Grille 3 colonnes
        cols = st.columns(3)
        for i, (_, row) in enumerate(pool.iterrows()):
            mid = int(row["movie_id"])
            current = int(st.session_state.user_ratings.get(str(mid), 0))
            with cols[i % 3]:
                with st.container(border=True):
                    genres_short = ", ".join(row["genres_list"][:3]) if row["genres_list"] else ""
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"{genres_short} · {row.get('avg_rating', 0):.1f}⭐")
                    new_rating = st.select_slider(
                        "Note",
                        options=["—", "1⭐", "2⭐", "3⭐", "4⭐", "5⭐"],
                        value=f"{current}⭐" if current > 0 else "—",
                        key=f"rate_{mid}",
                        label_visibility="collapsed",
                    )
                    if new_rating != "—":
                        r_val = int(new_rating[0])
                        if str(mid) not in st.session_state.user_ratings or st.session_state.user_ratings[str(mid)] != r_val:
                            st.session_state.user_ratings[str(mid)] = r_val
                            save_user_ratings(st.session_state.user_email, st.session_state.user_ratings)

        st.markdown(f"<br>✅ **{len(st.session_state.user_ratings)} film(s) noté(s)**", unsafe_allow_html=True)

    # ─────────────────────
    # RECOMMANDATIONS
    # ─────────────────────
    elif page == "✨ Mes recommandations":
        st.markdown('<div class="main-title" style="font-size:1.8rem">✨ Mes recommandations</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if len(st.session_state.user_ratings) < 3:
            st.warning("⚠️ Notez au moins 3 films pour obtenir des recommandations.")
            return

        # Choix de méthode
        method_col1, method_col2, method_col3 = st.columns(3)
        with method_col1:
            if st.button("👥 User-Based", use_container_width=True, type="primary"):
                st.session_state.rec_method = "user"
        with method_col2:
            if st.button("🎬 Item-Based", use_container_width=True):
                st.session_state.rec_method = "item"
        with method_col3:
            if st.button("🏷️ Content-Based", use_container_width=True):
                st.session_state.rec_method = "content"

        method = st.session_state.get("rec_method", "content")
        method_labels = {
            "user": "👥 User-Based Collaborative Filtering",
            "item": "🎬 Item-Based Collaborative Filtering",
            "content": "🏷️ Content-Based Filtering",
        }
        method_descs = {
            "user": "Recommande ce qu'ont aimé des utilisateurs similaires à vous (cosine similarity).",
            "item": "Recommande des films similaires (en termes de notes) à ceux que vous avez aimés.",
            "content": "Recommande des films avec les mêmes genres que vos films préférés.",
        }

        st.info(f"**{method_labels[method]}** — {method_descs[method]}")

        with st.spinner("Calcul des recommandations…"):
            results = recommend_live(st.session_state.user_ratings, method=method, n=5)

        if results.empty:
            st.warning("Pas assez de données pour générer des recommandations.")
            return

        st.markdown(f'<div class="section-header">🎬 Votre Top-5 — {method_labels[method]}</div>', unsafe_allow_html=True)

        col_cards, col_info = st.columns([2, 1])
        with col_cards:
            for i, row in enumerate(results.itertuples(), 1):
                render_rec_card(i, row)

        with col_info:
            st.markdown('<div class="section-header" style="font-size:1rem">📊 Vos films notés</div>', unsafe_allow_html=True)
            rated_display = []
            for mid, r in list(st.session_state.user_ratings.items())[:8]:
                title_row = movies_df[movies_df["movie_id"] == int(mid)]
                if not title_row.empty:
                    rated_display.append({"Film": title_row.iloc[0]["title"][:30], "Note": "⭐" * int(r)})
            if rated_display:
                st.dataframe(pd.DataFrame(rated_display), hide_index=True, use_container_width=True)

    # ─────────────────────
    # SANS REDONDANCE
    # ─────────────────────
    elif page == "🔄 Sans redondance (fusion)":
        st.markdown('<div class="main-title" style="font-size:1.8rem">🔄 Recommandations sans redondance</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#7070a0">
        Les 3 méthodes sont combinées. Les doublons sont éliminés et les films
        apparaissant dans plusieurs méthodes sont remontés (consensus).
        🔥 = film recommandé par plusieurs algorithmes simultanément.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if len(st.session_state.user_ratings) < 3:
            st.warning("⚠️ Notez au moins 3 films.")
            return

        n_final = st.slider("Nombre de recommandations finales", 5, 15, 10)

        with st.spinner("Fusion des 3 méthodes et dé-duplication…"):
            results = recommend_no_redundancy(
                st.session_state.user_ratings,
                n_per_method=8,
                final_n=n_final,
            )

        if results.empty:
            st.warning("Résultats insuffisants.")
            return

        # Stats de fusion
        multi = results[results["appearances"] > 1] if "appearances" in results.columns else pd.DataFrame()
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Films uniques trouvés", len(results))
        with c2: st.metric("Films consensus (≥2 méthodes)", len(multi))
        with c3: st.metric("Méthodes fusionnées", 3)

        st.markdown(f'<div class="section-header">🏆 Top-{n_final} Films (fusion sans redondance)</div>', unsafe_allow_html=True)

        for i, row in enumerate(results.itertuples(), 1):
            render_rec_card(i, row, show_method=True)

    # ─────────────────────
    # ÉVALUATION & COMPARAISON
    # ─────────────────────
    elif page == "📊 Évaluation & Comparaison":
        st.markdown('<div class="main-title" style="font-size:1.8rem">📊 Évaluation & Comparaison</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔬 Calculer le RMSE (peut prendre ~30s)", type="primary"):
            st.session_state.show_rmse = True

        if st.session_state.show_rmse:
            with st.spinner("Évaluation RMSE sur jeu de test…"):
                metrics = evaluate_rmse(sample_size=300)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("RMSE User-Based", metrics["user_based"],
                          delta="Meilleur" if metrics["best"] == "User-Based" else None)
            with c2:
                st.metric("RMSE Item-Based", metrics["item_based"],
                          delta="Meilleur" if metrics["best"] == "Item-Based" else None)
            with c3:
                st.metric("Échantillon évalué", metrics["n_evaluated"])

            st.success(f"🏆 Meilleur modèle : **{metrics['best']}** (RMSE le plus faible)")

        # Tableau comparatif
        st.markdown('<div class="section-header">🔍 Comparaison des 3 approches</div>', unsafe_allow_html=True)

        comparison = pd.DataFrame({
            "Approche": ["👥 User-Based CF", "🎬 Item-Based CF", "🏷️ Content-Based"],
            "Principe": [
                "Utilisateurs avec goûts similaires",
                "Films similaires en termes de notes",
                "Films similaires en termes de genres",
            ],
            "Données utilisées": ["Notes des utilisateurs", "Notes des films", "Genres des films"],
            "Avantage clé": [
                "Découverte, sérendipité",
                "Stable, précis, scalable",
                "Fonctionne sans historique",
            ],
            "Limite principale": [
                "Scalabilité limitée",
                "Cold start (nouveaux films)",
                "Sur-spécialisation",
            ],
            "Cold Start User": ["❌ Problème", "⚠️ Partiel", "✅ OK"],
            "Cold Start Item": ["⚠️ Partiel", "❌ Problème", "✅ OK"],
        })
        st.dataframe(comparison, hide_index=True, use_container_width=True)

        # Matrice user-film visualisation
        st.markdown('<div class="section-header">🗂️ Aperçu de la matrice User-Film</div>', unsafe_allow_html=True)
        st.caption("Extrait 10×10 de la matrice de ratings (NaN = non noté)")
        matrix_sample = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).iloc[:10, :10]
        st.dataframe(matrix_sample.style.background_gradient(cmap="YlOrBr", axis=None), use_container_width=True)

    # ─────────────────────
    # LIVE RECOMMANDATION
    # ─────────────────────
    elif page == "🔴 Live Recommandation":
        st.markdown('<div class="main-title" style="font-size:1.8rem">🔴 Live Recommandation</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#7070a0">
        <span class="live-dot"></span>&nbsp;
        Les recommandations se mettent à jour <strong>instantanément</strong>
        dès que vous modifiez une note ci-dessous.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        live_method = st.selectbox(
            "Méthode de recommandation live",
            ["content", "user", "item"],
            format_func=lambda x: {
                "content": "🏷️ Content-Based",
                "user":    "👥 User-Based",
                "item":    "🎬 Item-Based",
            }[x],
            key="live_method_selector",
        )

        # Sélection rapide de films à noter
        avg2 = ratings_df.groupby("movie_id")["rating"].agg(["mean","count"]).reset_index()
        avg2.columns = ["movie_id","avg_rating","num_ratings"]
        top_films = movies_df.merge(avg2, on="movie_id").sort_values("num_ratings", ascending=False).head(12)

        left_col, right_col = st.columns([1, 1.3])

        with left_col:
            st.markdown("**Notez des films ici (mise à jour instantanée) :**")
            live_ratings = dict(st.session_state.user_ratings)

            for _, row in top_films.iterrows():
                mid = int(row["movie_id"])
                current = int(live_ratings.get(str(mid), 0))
                with st.container(border=True):
                    st.caption(f"**{row['title']}**")
                    new_r = st.select_slider(
                        "", options=["—","1⭐","2⭐","3⭐","4⭐","5⭐"],
                        value=f"{current}⭐" if current > 0 else "—",
                        key=f"live_{mid}",
                        label_visibility="collapsed",
                    )
                    if new_r != "—":
                        live_ratings[str(mid)] = int(new_r[0])

            # Synchroniser avec la session
            if live_ratings != st.session_state.user_ratings:
                st.session_state.user_ratings = live_ratings
                save_user_ratings(st.session_state.user_email, live_ratings)

        with right_col:
            st.markdown("**🎬 Recommandations en temps réel :**")
            if len(live_ratings) >= 2:
                results = recommend_live(live_ratings, method=live_method, n=5)
                if not results.empty:
                    for i, row in enumerate(results.itertuples(), 1):
                        render_rec_card(i, row, show_method=False)
                else:
                    st.info("Continuez à noter des films…")
            else:
                st.info("Notez au moins 2 films pour voir les recommandations live.")


# ─────────────────────────────────────────────
# ROUTER PRINCIPAL
# ─────────────────────────────────────────────
if is_logged_in():
    page_app()
else:
    page_auth()
