"""
MovieLens — Plateforme de recommandation style Netflix
Interface complète : catalogue, filtres genres, détail film + recommandations
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_ratings, load_items, get_dataset_stats
from src.recommenders import ContentBasedRecommender, recommend_popular

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

TMDB_API_KEY = "56320d0d7909088298d3e32af5a16bb3"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_IMG_LARGE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="MovieLens",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GENRE_LIST = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]

GENRE_EMOJIS = {
    'Action': '💥', 'Adventure': '🗺️', 'Animation': '🎨',
    "Children's": '🧸', 'Comedy': '😂', 'Crime': '🔫',
    'Documentary': '📽️', 'Drama': '🎭', 'Fantasy': '🧙',
    'Film-Noir': '🕵️', 'Horror': '👻', 'Musical': '🎵',
    'Mystery': '🔍', 'Romance': '❤️', 'Sci-Fi': '🚀',
    'Thriller': '😱', 'War': '⚔️', 'Western': '🤠'
}

# ─────────────────────────────────────────────────────────────
# TMDB FETCH
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def fetch_poster(title: str, year=None, large=False) -> str:
    try:
        clean_title = title
        if '(' in title:
            clean_title = title[:title.rfind('(')].strip()
        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "language": "fr-FR",
        }
        if year and not pd.isna(year):
            params["year"] = int(year)
        resp = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
        data = resp.json()
        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            base = TMDB_IMG_LARGE if large else TMDB_IMG_BASE
            return base + results[0]["poster_path"]
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400)
def fetch_tmdb_details(title: str, year=None):
    """Fetch overview and details from TMDB."""
    try:
        clean_title = title
        if '(' in title:
            clean_title = title[:title.rfind('(')].strip()
        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "language": "fr-FR",
        }
        if year and not pd.isna(year):
            params["year"] = int(year)
        resp = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
        data = resp.json()
        results = data.get("results", [])
        if results:
            r = results[0]
            return {
                "overview": r.get("overview", ""),
                "poster": TMDB_IMG_LARGE + r["poster_path"] if r.get("poster_path") else None,
                "backdrop": "https://image.tmdb.org/t/p/w1280" + r["backdrop_path"] if r.get("backdrop_path") else None,
                "tmdb_rating": r.get("vote_average", None),
            }
    except Exception:
        pass
    return {"overview": "", "poster": None, "backdrop": None, "tmdb_rating": None}

# ─────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
""")

# ─────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* ── RESET & BASE ─────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background-color: #080810 !important;
    color: #e2e2ee !important;
    font-family: 'Outfit', 'Helvetica Neue', sans-serif !important;
}
[data-testid="stHeader"]     { background: transparent !important; height: 0 !important; }
[data-testid="stSidebar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.main .block-container       { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080810; }
::-webkit-scrollbar-thumb { background: #e50914; border-radius: 2px; }

/* ── NAVBAR ───────────────────────────────────────── */
.navbar {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: linear-gradient(180deg, #080810 70%, transparent 100%);
    padding: 0 3rem;
    height: 68px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.navbar-logo {
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700;
    letter-spacing: 4px;
    color: #e50914 !important;
    text-transform: uppercase;
    user-select: none;
    flex-shrink: 0;
}
.navbar-home-btn {
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6666aa;
    background: transparent;
    border: 1px solid #1e1e38;
    border-radius: 6px;
    padding: 0.45rem 1.1rem;
    cursor: pointer;
    outline: none;
    transition: all 0.15s ease;
    white-space: nowrap;
    flex-shrink: 0;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
    -webkit-tap-highlight-color: transparent;
}
.navbar-home-btn:visited { color: #6666aa; }
.navbar-home-btn:hover {
    color: #e8e8f0;
    border-color: #e50914;
    background: rgba(229,9,20,0.07);
}
.navbar-tagline {
    font-size: 0.68rem;
    color: #333355;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 400;
    margin-top: 0.15rem;
}
.navbar-stats {
    display: flex;
    gap: 2rem;
    align-items: center;
}
.navbar-stat {
    text-align: right;
}
.navbar-stat-val {
    font-family: 'Oswald', sans-serif;
    font-size: 1.1rem;
    color: #e8e8f0;
    font-weight: 500;
    letter-spacing: 1px;
}
.navbar-stat-lbl {
    font-size: 0.6rem;
    color: #2a2a40;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── GENRE PILLS ──────────────────────────────────── */
.genre-bar {
    background: #0d0d18;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding: 0.75rem 3rem;
    overflow-x: auto;
    white-space: nowrap;
    scrollbar-width: none;
    -ms-overflow-style: none;
    display: flex;
    gap: 0.5rem;
    align-items: center;
}
.genre-bar::-webkit-scrollbar { display: none; }
.genre-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.38rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid #1e1e35;
    background: #0f0f20;
    color: #6666aa;
    transition: all 0.18s ease;
    white-space: nowrap;
    text-decoration: none !important;
    user-select: none;
    outline: none;
    -webkit-appearance: none;
    appearance: none;
    -webkit-tap-highlight-color: transparent;
}
.genre-pill:visited { color: #6666aa; }
.genre-pill.active:visited { color: #fff; }
.genre-pill:hover {
    border-color: #e50914;
    color: #e8e8f0;
    background: rgba(229,9,20,0.08);
}
.genre-pill.active {
    background: #e50914;
    border-color: #e50914;
    color: #fff;
    font-weight: 600;
}
.genre-pill-all {
    font-family: 'Oswald', sans-serif;
    letter-spacing: 1.5px;
    font-size: 0.8rem;
}

/* ── CONTENT WRAPPER ──────────────────────────────── */
.content-wrap {
    padding: 1.5rem 3rem 3rem 3rem;
    max-width: 1600px;
    margin: 0 auto;
}

/* ── SECTION HEADER ───────────────────────────────── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}
.section-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 500;
    letter-spacing: 2px;
    color: #d0d0e8;
    text-transform: uppercase;
}
.section-count {
    font-size: 0.75rem;
    color: #333355;
    letter-spacing: 1px;
}

/* ── CATALOG GRID ─────────────────────────────────── */
.catalog-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
    gap: 0.85rem;
}

/* ── MOVIE CARD ───────────────────────────────────── */
.movie-card {
    position: relative;
    border-radius: 6px;
    overflow: hidden;
    aspect-ratio: 2/3;
    background: #111122;
    cursor: pointer;
    transition: transform 0.22s ease, box-shadow 0.22s ease, z-index 0s;
    border: 1px solid #181830;
    display: block;
    text-decoration: none;
}
.movie-card:hover {
    transform: scale(1.05) translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.8), 0 0 0 2px rgba(229,9,20,0.45);
    z-index: 5;
}
.movie-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: opacity 0.3s ease;
}
.movie-card-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, rgba(5,5,15,0.98) 0%, rgba(5,5,15,0.65) 55%, transparent 100%);
    padding: 1.4rem 0.6rem 0.65rem 0.7rem;
    transform: translateY(0);
}
.movie-card-title {
    font-size: 0.73rem;
    font-weight: 600;
    color: #f0f0ff;
    line-height: 1.3;
    margin-bottom: 0.2rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.movie-card-year {
    font-size: 0.65rem;
    color: #e50914;
    font-weight: 500;
}
.movie-card-badge {
    position: absolute;
    top: 0.5rem; right: 0.5rem;
    background: rgba(229,9,20,0.88);
    color: #fff;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    letter-spacing: 0.3px;
}
.movie-no-poster {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(145deg, #0f0f22, #161630);
    color: #222240;
    font-size: 2.5rem;
    gap: 0.5rem;
}
.movie-no-poster span {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #1e1e38;
    text-align: center;
    padding: 0 0.5rem;
}

/* ── DETAIL VIEW ──────────────────────────────────── */
.detail-backdrop {
    width: calc(100% + 6rem);
    margin: -1.5rem -3rem 0 -3rem;
    height: 400px;
    position: relative;
    overflow: hidden;
}
.detail-backdrop img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 20%;
    filter: brightness(0.35);
}
.detail-backdrop-fallback {
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, #0f0a20 0%, #1a0810 50%, #0a0f1a 100%);
}
.detail-backdrop-fade {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 200px;
    background: linear-gradient(to top, #080810 0%, transparent 100%);
}
.detail-backdrop-left {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 300px;
    background: linear-gradient(to right, #080810 0%, transparent 100%);
}

.detail-main {
    display: flex;
    gap: 2.5rem;
    margin-top: -180px;
    position: relative;
    z-index: 2;
    padding: 0 3rem 2rem 3rem;
    align-items: flex-end;
}
.detail-poster {
    width: 200px;
    min-width: 200px;
    aspect-ratio: 2/3;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.9), 0 0 0 2px rgba(229,9,20,0.3);
    flex-shrink: 0;
}
.detail-poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.detail-poster-fallback {
    width: 100%; height: 100%;
    background: linear-gradient(145deg, #111125, #1a1a38);
    display: flex; align-items: center; justify-content: center;
    font-size: 3rem;
}
.detail-info {
    flex: 1;
    padding-bottom: 0.5rem;
}
.detail-back {
    font-size: 0.72rem;
    color: #333355;
    text-transform: uppercase;
    letter-spacing: 2px;
    cursor: pointer;
    margin-bottom: 1rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    transition: color 0.15s ease;
}
.detail-back:hover { color: #e50914; }
.detail-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 600;
    color: #f5f5ff;
    letter-spacing: 1px;
    line-height: 1.1;
    margin-bottom: 0.6rem;
    text-shadow: 0 2px 20px rgba(0,0,0,0.8);
}
.detail-meta-row {
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}
.detail-year {
    font-family: 'Oswald', sans-serif;
    font-size: 1rem;
    color: #8888aa;
    letter-spacing: 1px;
}
.detail-genre-tag {
    padding: 0.2rem 0.65rem;
    background: rgba(229,9,20,0.15);
    border: 1px solid rgba(229,9,20,0.3);
    border-radius: 12px;
    font-size: 0.7rem;
    color: #ff8080;
    font-weight: 500;
}
.detail-overview {
    font-size: 0.9rem;
    color: #9090b8;
    line-height: 1.7;
    max-width: 680px;
    margin-bottom: 1.2rem;
    font-weight: 300;
}
.detail-overview-empty {
    font-size: 0.85rem;
    color: #444466;
    font-style: italic;
    margin-bottom: 1.2rem;
}

/* ── LOVE COUNTER ─────────────────────────────────── */
.love-counter {
    display: inline-flex;
    align-items: center;
    gap: 1.5rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e1e38;
    border-radius: 12px;
    padding: 0.85rem 1.5rem;
    margin-bottom: 1rem;
}
.love-item {
    text-align: center;
}
.love-value {
    font-family: 'Oswald', sans-serif;
    font-size: 1.5rem;
    color: #e50914;
    letter-spacing: 1px;
    font-weight: 600;
}
.love-label {
    font-size: 0.62rem;
    color: #333355;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.1rem;
}
.love-divider {
    width: 1px;
    height: 36px;
    background: #1e1e38;
}

/* ── RECO SECTION ─────────────────────────────────── */
.reco-section {
    border-top: 1px solid #111128;
    padding-top: 2rem;
    margin-top: 0.5rem;
}
.reco-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 500;
    letter-spacing: 2.5px;
    color: #888899;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.reco-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 1.1rem;
    background: #e50914;
    border-radius: 2px;
}

/* ── RECO GRID (horizontal scroll) ───────────────── */
.reco-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 0.75rem;
}

/* ── SEARCH BAR ───────────────────────────────────── */
.search-wrap {
    position: relative;
    max-width: 380px;
}

/* ── EMPTY STATE ──────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 5rem 0;
    color: #1e1e38;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 0.8rem; opacity: 0.4; }
.empty-state-text { font-size: 0.85rem; letter-spacing: 1.5px; text-transform: uppercase; }

/* ── FOOTER ───────────────────────────────────────── */
.footer {
    margin-top: 4rem;
    padding: 1.8rem 3rem;
    border-top: 1px solid #0f0f20;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.footer-logo {
    font-family: 'Oswald', sans-serif;
    font-size: 1.1rem;
    color: #e50914;
    letter-spacing: 3px;
    font-weight: 600;
}
.footer-info {
    font-size: 0.68rem;
    color: #1e1e35;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.footer-links {
    display: flex;
    gap: 1.5rem;
    font-size: 0.68rem;
    color: #222240;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── BACK LINK ────────────────────────────────────── */
.back-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    color: #e50914;
    border: 1px solid rgba(229,9,20,0.3);
    background: rgba(229,9,20,0.05);
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    text-decoration: none;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    transition: all 0.15s ease;
    margin-bottom: 1rem;
}
.back-link:hover {
    background: rgba(229,9,20,0.12);
    color: #ff4444;
}

/* ── STREAMLIT OVERRIDES ──────────────────────────── */
.stTextInput input {
    background: #0f0f20 !important;
    border: 1px solid #1e1e38 !important;
    border-radius: 6px !important;
    color: #e8e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus {
    border-color: #e50914 !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.15) !important;
}
.stTextInput input::placeholder { color: #2a2a45 !important; }
.stTextInput label { display: none !important; }

.stButton > button {
    background: transparent !important;
    color: #666688 !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 6px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #e50914 !important;
    color: #e8e8f0 !important;
    background: rgba(229,9,20,0.06) !important;
}
/* Back button style */
.back-btn > button {
    color: #e50914 !important;
    border-color: rgba(229,9,20,0.3) !important;
    background: rgba(229,9,20,0.05) !important;
}
.back-btn > button:hover {
    background: rgba(229,9,20,0.12) !important;
}

[data-testid="stSpinner"] > div { border-top-color: #e50914 !important; }
hr { border-color: #111128 !important; }
.stAlert { display: none !important; }

/* Hide streamlit elements that pollute the UI */
[data-testid="stMarkdownContainer"] > p:empty { display: none; }

/* ── SHIMMER LOADING ──────────────────────────────── */
@keyframes shimmer {
    0% { background-position: -600px 0; }
    100% { background-position: 600px 0; }
}
.shimmer-card {
    border-radius: 6px;
    aspect-ratio: 2/3;
    background: linear-gradient(90deg, #0f0f22 25%, #161630 50%, #0f0f22 75%);
    background-size: 600px 100%;
    animation: shimmer 1.5s infinite;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE — driven by query params
# ─────────────────────────────────────────────────────────────

# Read query params on every render
_qp = st.query_params

# Film detail navigation
if 'film' in _qp:
    try:
        st.session_state.selected_film = int(_qp['film'])
    except Exception:
        st.session_state.selected_film = None
else:
    st.session_state.selected_film = None

# Genre filter
if 'genre' in _qp:
    st.session_state.selected_genre = _qp['genre']
else:
    st.session_state.selected_genre = None

# Search query
if 'q' in _qp:
    st.session_state.search_query = _qp['q']
else:
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    ratings = load_ratings('u.data')
    items = load_items()
    return ratings, items

@st.cache_resource
def train_model(_items_df):
    return ContentBasedRecommender().fit(_items_df)

@st.cache_data
def get_rating_stats(_ratings_df):
    """Pre-compute per-item rating stats."""
    stats = _ratings_df.groupby('item_id').agg(
        avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count'),
        n_fans=('rating', lambda x: (x >= 4).sum())
    ).reset_index()
    return stats

with st.spinner("Chargement du catalogue MovieLens..."):
    try:
        ratings_df, items_df = load_data()
        cb_model = train_model(items_df)
        rating_stats = get_rating_stats(ratings_df)
        items_with_stats = items_df.merge(rating_stats, on='item_id', how='left')
        items_with_stats['n_ratings'] = items_with_stats['n_ratings'].fillna(0).astype(int)
        items_with_stats['n_fans'] = items_with_stats['n_fans'].fillna(0).astype(int)
        items_with_stats['avg_rating'] = items_with_stats['avg_rating'].fillna(0)
        data_loaded = True
    except Exception as e:
        st.error(f"❌ Impossible de charger les données : {e}")
        st.info("Lance d'abord : `python setup_data.py`")
        st.stop()

total_films = len(items_df)
total_ratings = len(ratings_df)
avg_note = round(ratings_df['rating'].mean(), 1)

# ─────────────────────────────────────────────────────────────
# HELPERS — CARDS HTML
# ─────────────────────────────────────────────────────────────

def movie_card_html(item_id, title, year, genres_str, avg_rating=0, n_ratings=0, badge=None):
    """Build a clickable movie card. Returns HTML string."""
    poster_url = fetch_poster(title, year)
    year_str = str(int(year)) if pd.notna(year) and year != 0 else "—"

    if poster_url:
        img_html = f'<img src="{poster_url}" alt="{title}" loading="lazy">'
    else:
        short = title[:18] + "…" if len(title) > 18 else title
        img_html = f'<div class="movie-no-poster">🎬<span>{short}</span></div>'

    badge_html = ""
    if badge:
        badge_html = f'<div class="movie-card-badge">{badge}</div>'
    elif avg_rating and avg_rating > 0:
        badge_html = f'<div class="movie-card-badge">★ {avg_rating:.1f}</div>'

    return f"""
    <a class="movie-card" href="?film={item_id}" target="_self">
        {img_html}
        {badge_html}
        <div class="movie-card-overlay">
            <div class="movie-card-title">{title}</div>
            <div class="movie-card-year">{year_str}</div>
        </div>
    </a>
    """


def render_catalog_grid(df, max_cols=None):
    """Render films in responsive grid using Streamlit columns."""
    COLS = 7
    rows_iter = list(df.iterrows())
    if not rows_iter:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🎬</div>
            <div class="empty-state-text">Aucun film trouvé</div>
        </div>""", unsafe_allow_html=True)
        return

    for row_start in range(0, len(rows_iter), COLS):
        chunk = rows_iter[row_start:row_start + COLS]
        cols = st.columns(COLS)
        for col_idx, (_, row) in enumerate(chunk):
            with cols[col_idx]:
                avg_r = row.get('avg_rating', 0)
                n_r = row.get('n_ratings', 0)
                html = movie_card_html(
                    item_id=row['item_id'],
                    title=row['title'],
                    year=row.get('year'),
                    genres_str=row.get('genres_str', ''),
                    avg_rating=avg_r if n_r >= 10 else 0,
                    n_ratings=n_r,
                )
                st.markdown(html, unsafe_allow_html=True)


def render_poster_row(df, cols_count=10):
    """Render a single horizontal row of posters (for recommendations)."""
    cols = st.columns(cols_count)
    row_iter = list(df.head(cols_count).iterrows())
    for col_idx, (_, row) in enumerate(row_iter):
        with cols[col_idx]:
            avg_r = row.get('avg_rating', 0)
            html = movie_card_html(
                item_id=row['item_id'],
                title=row['title'],
                year=row.get('year'),
                genres_str=row.get('genres_str', ''),
                avg_rating=avg_r if row.get('n_ratings', 0) >= 10 else 0,
            )
            st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="navbar">
    <div style="display:flex; align-items:center; gap:2rem;">
        <div>
            <div class="navbar-logo">MovieLens</div>
            <div class="navbar-tagline">Intelligent Recommendation Platform</div>
        </div>
        <a class="navbar-home-btn" href="?" target="_self">Accueil</a>
    </div>
    <div class="navbar-stats">
        <div class="navbar-stat">
            <div class="navbar-stat-val">{total_films:,}</div>
            <div class="navbar-stat-lbl">Films</div>
        </div>
        <div class="navbar-stat">
            <div class="navbar-stat-val">{total_ratings:,}</div>
            <div class="navbar-stat-lbl">Évaluations</div>
        </div>
        <div class="navbar-stat">
            <div class="navbar-stat-val">★ {avg_note}</div>
            <div class="navbar-stat-lbl">Note moy.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# VIEW: DETAIL D'UN FILM
# ─────────────────────────────────────────────────────────────

if st.session_state.selected_film is not None:
    film_id = st.session_state.selected_film
    film_row_matches = items_with_stats[items_with_stats['item_id'] == film_id]

    if film_row_matches.empty:
        st.session_state.selected_film = None
        st.rerun()

    film = film_row_matches.iloc[0]
    year_val = film.get('year')
    year_str = str(int(year_val)) if pd.notna(year_val) and year_val else "—"
    genres_list = film.get('genres', [])
    if not isinstance(genres_list, list):
        genres_list = []
    genres_str = film.get('genres_str', '—')
    n_ratings_val = int(film.get('n_ratings', 0))
    n_fans_val = int(film.get('n_fans', 0))
    avg_r_val = float(film.get('avg_rating', 0))

    # Fetch TMDB details
    tmdb = fetch_tmdb_details(film['title'], year_val)
    backdrop_url = tmdb.get('backdrop')
    poster_url = tmdb.get('poster') or fetch_poster(film['title'], year_val, large=True)
    overview = tmdb.get('overview', '')

    # ── BACKDROP ──
    if backdrop_url:
        backdrop_html = f'<img src="{backdrop_url}" alt="">'
    else:
        backdrop_html = '<div class="detail-backdrop-fallback"></div>'

    st.markdown(f"""
    <div class="detail-backdrop">
        {backdrop_html}
        <div class="detail-backdrop-fade"></div>
        <div class="detail-backdrop-left"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── POSTER + INFO ──
    if poster_url:
        poster_html = f'<img src="{poster_url}" alt="{film["title"]}">'
    else:
        poster_html = '<div class="detail-poster-fallback">🎬</div>'

    genres_tags = ''.join([f'<span class="detail-genre-tag">{g}</span>' for g in genres_list[:5]])
    if not genres_tags:
        genres_tags = f'<span class="detail-genre-tag">{genres_str}</span>'

    if overview:
        overview_html = f'<div class="detail-overview">{overview}</div>'
    else:
        overview_html = '<div class="detail-overview-empty">Pas de description pour ce film.</div>'

    safe_title = film['title'].replace("'", "&#39;").replace('"', '&quot;')

    st.markdown(f"""
    <div class="detail-main">
        <div class="detail-poster">{poster_html}</div>
        <div class="detail-info">
            <div class="detail-title">{safe_title}</div>
            <div class="detail-meta-row">
                <span class="detail-year">{year_str}</span>
                {genres_tags}
            </div>
            {overview_html}
            <div class="love-counter">
                <div class="love-item">
                    <div class="love-value">{n_ratings_val:,}</div>
                    <div class="love-label">Utilisateurs</div>
                </div>
                <div class="love-divider"></div>
                <div class="love-item">
                    <div class="love-value">{n_fans_val:,}</div>
                    <div class="love-label">Fans (≥4★)</div>
                </div>
                <div class="love-divider"></div>
                <div class="love-item">
                    <div class="love-value">★ {avg_r_val:.2f}</div>
                    <div class="love-label">Note moyenne</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── BACK BUTTON ──
    st.markdown("""
    <div style="padding: 0 3rem 0 3rem; margin-top: -0.5rem;">
        <a href="?" target="_self" class="back-link">← Retour</a>
    </div>
    """, unsafe_allow_html=True)

    # ── RECOMMENDATIONS ──
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="reco-section">', unsafe_allow_html=True)
    st.markdown('<div class="reco-title">Films similaires recommandés</div>', unsafe_allow_html=True)

    with st.spinner("Calcul des recommandations..."):
        similar = cb_model.get_similar_items(film_id, n=10)

    if not similar.empty:
        # Merge rating stats into recommendations
        similar = similar.merge(
            rating_stats[['item_id', 'avg_rating', 'n_ratings', 'n_fans']],
            on='item_id', how='left'
        )
        similar['avg_rating'] = similar['avg_rating'].fillna(0)
        similar['n_ratings'] = similar['n_ratings'].fillna(0).astype(int)
        similar['n_fans'] = similar['n_fans'].fillna(0).astype(int)
        render_poster_row(similar, cols_count=10)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text">Pas assez de données pour recommander</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # reco-section
    st.markdown('</div>', unsafe_allow_html=True)  # content-wrap

# ─────────────────────────────────────────────────────────────
# VIEW: CATALOGUE PRINCIPAL
# ─────────────────────────────────────────────────────────────

else:
    # ── GENRE PILLS (boutons vrais, sans URL visible) ──
    genre_html_parts = []
    all_active = "active" if st.session_state.selected_genre is None else ""
    genre_html_parts.append(f'<a class="genre-pill genre-pill-all {all_active}" href="?" target="_self">TOUS</a>')
    for g in GENRE_LIST:
        active = "active" if st.session_state.selected_genre == g else ""
        genre_html_parts.append(f'<a class="genre-pill {active}" href="?genre={g}" target="_self">{g}</a>')

    st.markdown(
        '<div class="genre-bar">' + ''.join(genre_html_parts) + '</div>',
        unsafe_allow_html=True
    )

    # ── CONTENT ──
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # Search bar
    col_search, col_info = st.columns([3, 6])
    with col_search:
        search_val = st.text_input(
            "Rechercher",
            value=st.session_state.search_query,
            placeholder="🔍  Rechercher un film...",
            key="search_input",
            label_visibility="collapsed"
        )
        if search_val != st.session_state.search_query:
            st.query_params['q'] = search_val
            st.rerun()

    # Filter catalog
    catalog = items_with_stats.copy()

    if st.session_state.search_query:
        q = st.session_state.search_query.lower()
        catalog = catalog[catalog['title'].str.lower().str.contains(q, na=False)]
        title_label = f"Résultats pour « {st.session_state.search_query} »"
    elif st.session_state.selected_genre:
        g = st.session_state.selected_genre
        catalog = catalog[catalog['genres'].apply(lambda gl: g in gl if isinstance(gl, list) else False)]
        title_label = f"{GENRE_EMOJIS.get(g, '')}  {g}"
    else:
        # Default: sort by popularity (most rated + highest rated)
        catalog = catalog[catalog['n_ratings'] >= 5].copy()
        catalog['pop_score'] = (
            catalog['avg_rating'] * 0.4 +
            (catalog['n_ratings'] / catalog['n_ratings'].max()) * 0.6
        )
        catalog = catalog.sort_values('pop_score', ascending=False)
        title_label = "Catalogue · Tous les films"

    # Sort by popularity within genre/search too
    if st.session_state.selected_genre or st.session_state.search_query:
        catalog['pop_score'] = (
            catalog['avg_rating'] * 0.4 +
            (catalog['n_ratings'] / max(catalog['n_ratings'].max(), 1)) * 0.6
        )
        catalog = catalog.sort_values('pop_score', ascending=False)

    n_found = len(catalog)

    # Section header
    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">{title_label}</div>
        <div class="section-count">{n_found} film{'s' if n_found > 1 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

    if catalog.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text">Aucun film trouvé pour cette sélection</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Chargement des affiches..."):
            render_catalog_grid(catalog)

    st.markdown('</div>', unsafe_allow_html=True)  # content-wrap


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <div>
        <div class="footer-logo">MovieLens</div>
    </div>
    <div class="footer-info">MovieLens 100K &nbsp;·&nbsp; Content-Based Filtering &nbsp;·&nbsp; TMDB Posters</div>
    <div class="footer-links">
        <span>943 utilisateurs</span>
        <span>1,682 films</span>
        <span>100,000 évaluations</span>
    </div>
</div>
""", unsafe_allow_html=True)
