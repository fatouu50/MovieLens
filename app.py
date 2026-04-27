"""
MovieLens — Plateforme de recommandation style Netflix
Interface complète : catalogue, filtres genres, detail film + recommandations,
authentification JWT/bcrypt, notation, live reco, fusion sans redondance, RMSE
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_ratings, load_items, get_dataset_stats
from src.recommenders import ContentBasedRecommender, recommend_popular

from auth import (
    register_user, login_user, login_finalize, verify_token,
    get_user_data, save_user_ratings, save_user_genres,
    send_otp, verify_otp, send_reset_otp, reset_password,
)
from recommender import (
    load_data as load_reco_data,
    recommend_live, recommend_no_redundancy,
    evaluate_rmse, get_all_genres, get_movies_by_genre,
    recommend_user_based, recommend_item_based, recommend_content_based,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

TMDB_API_KEY = "56320d0d7909088298d3e32af5a16bb3"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_IMG_LARGE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="MovieLens",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GENRE_LIST = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]

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
                "tmdb_id": r.get("id"),
            }
    except Exception:
        pass
    return {"overview": "", "poster": None, "backdrop": None, "tmdb_rating": None, "tmdb_id": None}

@st.cache_data(ttl=3600)
def tmdb_similar(tmdb_id: int) -> list:
    """Retourne les films similaires depuis TMDB."""
    try:
        r = requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}/similar",
            params={"api_key": TMDB_API_KEY, "language": "fr-FR"},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json().get("results", [])[:6]
    except Exception:
        pass
    return []

# ─────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600&display=swap');

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
.navbar-stat { text-align: right; }
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

/* ── AUTH PANEL ───────────────────────────────────── */
.auth-wrap {
    max-width: 420px;
    margin: 3rem auto;
    background: #0d0d1a;
    border: 1px solid #1e1e38;
    border-radius: 14px;
    padding: 2.5rem 2rem;
}
.auth-logo {
    font-family: 'Oswald', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: #e50914;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 0.2rem;
}
.auth-sub {
    font-size: 0.68rem;
    color: #333355;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1.8rem;
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
.love-item { text-align: center; }
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

/* ── RECO GRID ────────────────────────────────────── */
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

/* ── NOTATION CARD ────────────────────────────────── */
.rating-section {
    border-top: 1px solid #111128;
    padding-top: 1.5rem;
    margin-top: 1.5rem;
}
.rating-section-title {
    font-family: 'Oswald', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 2px;
    color: #888899;
    text-transform: uppercase;
    margin-bottom: 1rem;
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
.stButton > button[kind="primary"] {
    color: #e50914 !important;
    border-color: rgba(229,9,20,0.4) !important;
    background: rgba(229,9,20,0.08) !important;
}
.stButton > button[kind="primary"]:hover {
    background: rgba(229,9,20,0.15) !important;
}
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
/* Alertes visibles uniquement dans la page auth */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
}
/* Masquer alertes hors auth (ex: warning use_container_width) */
.main .block-container > div > div > div > [data-testid="stAlert"] { display: none !important; }
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

/* ── LIVE DOT ─────────────────────────────────────── */
.live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #5cfca0; border-radius: 50%; animation: pulse 1.5s infinite;
    vertical-align: middle; margin-right: 0.4rem;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .5; transform: scale(1.4); }
}

/* ── SCORE BAR ────────────────────────────────────── */
.score-bar-wrap { background: #1e1e38; border-radius: 3px; height: 3px; margin-top: 6px; }
.score-bar-fill { height: 3px; border-radius: 3px; background: linear-gradient(90deg,#e50914,#ff4444); }


/* Fix raw HTML text leak in Streamlit markdown */
[data-testid="stMarkdownContainer"] > div > p:not(:has(*)) { display: none !important; }

/* ── UNIFIED NAVBAR ───────────────────────────────── */
.unb {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: #080810;
    padding: 0 2.5rem;
    height: 62px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    gap: 1.5rem;
    margin-bottom: 0;
}
.unb-logo-wrap { display: flex; flex-direction: column; flex-shrink: 0; }
.unb-logo {
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700;
    letter-spacing: 4px;
    color: #e50914 !important;
    text-transform: uppercase;
    line-height: 1;
}
.unb-tag {
    font-size: 0.58rem;
    color: #252545;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 3px;
}
.unb-nav {
    display: flex;
    gap: 0.15rem;
    align-items: center;
    flex: 1;
    padding-left: 1.5rem;
}
.unb-pill {
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #44446a;
    background: transparent;
    border: none;
    border-radius: 20px;
    padding: 6px 14px;
    text-decoration: none !important;
    white-space: nowrap;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    -webkit-tap-highlight-color: transparent;
}
.unb-pill:visited { color: #44446a; }
.unb-pill:hover { color: #c8c8e0; background: rgba(255,255,255,0.05); }
.unb-pill.active {
    background: rgba(229,9,20,0.15);
    border: 1px solid rgba(229,9,20,0.35);
    color: #e50914 !important;
    font-weight: 600;
}
.unb-pill.active:visited { color: #e50914; }
.unb-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-shrink: 0;
}
.unb-user {
    font-size: 0.7rem;
    color: #33334a;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.unb-logout {
    background: transparent;
    border: 1px solid #1e1e38;
    color: #44446a;
    font-size: 0.68rem;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.15s;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
}
.unb-logout:visited { color: #44446a; }
.unb-logout:hover { color: #e50914; border-color: rgba(229,9,20,0.4); background: rgba(229,9,20,0.06); }
.unb-stat { text-align: right; }
.unb-sv {
    font-family: 'Oswald', sans-serif;
    font-size: 1rem;
    color: #e8e8f0;
    font-weight: 500;
    letter-spacing: 1px;
}
.unb-sl {
    font-size: 0.58rem;
    color: #2a2a40;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 1px;
}


</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

# Auth session state defaults — initialisés EN PREMIER
_auth_defaults = {
    "jwt_token":         None,
    "user_email":        None,
    "user_name":         None,
    "user_ratings":      {},
    "genre_prefs":       [],
    "show_rmse":         False,
    "rec_method":        "content",
    "active_page":       "catalogue",
    # OTP flow
    "otp_pending_email": None,
    "otp_dev_code":      None,
    # Reset password flow
    "reset_step":        None,   # None | "otp" | "new_pwd"
    "reset_email":       None,
    "reset_dev_code":    None,
}
for _k, _v in _auth_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Lecture des query params — APRÈS les defaults pour écraser correctement
_qp = st.query_params

# ── SE SOUVENIR DE MOI : relire le token depuis l'URL ──────────
# Si l'URL contient ?t=<jwt> et qu'on n'est pas encore connecté,
# on restaure automatiquement la session sans redemander le mot de passe.
if 't' in _qp and not st.session_state.jwt_token:
    _url_token = _qp['t']
    _payload = verify_token(_url_token)
    if _payload:
        _email = _payload.get("email", "")
        st.session_state.jwt_token  = _url_token
        st.session_state.user_email = _email
        _udata = get_user_data(_email)
        st.session_state.user_name    = _udata["name"]
        st.session_state.user_ratings = {str(k): v for k, v in _udata["ratings"].items()}
        st.session_state.genre_prefs  = _udata["genre_prefs"]
        st.session_state.active_page  = "catalogue"

if 'film' in _qp:
    try:
        st.session_state.selected_film = int(_qp['film'])
    except Exception:
        st.session_state.selected_film = None
else:
    st.session_state.selected_film = None

if 'genre' in _qp:
    st.session_state.selected_genre = _qp['genre']
else:
    st.session_state.selected_genre = None

if 'q' in _qp:
    st.session_state.search_query = _qp['q']
else:
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

if 'page' in _qp:
    valid_pages = {"catalogue", "noter", "reco", "fusion", "evaluation", "live"}
    if _qp['page'] in valid_pages:
        st.session_state.active_page = _qp['page']
    else:
        st.session_state.active_page = "catalogue"
elif 'film' not in _qp and 'genre' not in _qp and 'q' not in _qp and '__logout' not in _qp and 't' not in _qp:
    st.session_state.active_page = "catalogue"


# ─────────────────────────────────────────────────────────────
# HELPERS AUTH
# ─────────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    token = st.session_state.get("jwt_token")
    if not token:
        return False
    return verify_token(token) is not None

def do_logout():
    for k in ["jwt_token", "user_email", "user_name", "user_ratings", "genre_prefs"]:
        st.session_state[k] = None if k == "jwt_token" else ([] if k == "genre_prefs" else ({} if k == "user_ratings" else ""))
    st.session_state.active_page = "catalogue"
    st.rerun()


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
    except Exception as e:
        st.error(f"Impossible de charger les donnees : {e}")
        st.info("Lance d'abord : `python setup_data.py`")
        st.stop()

total_films = len(items_df)
total_ratings = len(ratings_df)
avg_note = round(ratings_df['rating'].mean(), 1)


# ─────────────────────────────────────────────────────────────
# HELPERS — CARDS HTML
# ─────────────────────────────────────────────────────────────

def movie_card_html(item_id, title, year, genres_str, avg_rating=0, n_ratings=0, badge=None):
    poster_url = fetch_poster(title, year)
    year_str = str(int(year)) if pd.notna(year) and year != 0 else "—"
    _t_val = st.query_params.get('t', '')
    _film_href = f"?film={item_id}&t={_t_val}" if _t_val else f"?film={item_id}"

    if poster_url:
        img_html = f'<img src="{poster_url}" alt="{title}" loading="lazy">'
    else:
        short = title[:18] + "..." if len(title) > 18 else title
        img_html = f'<div class="movie-no-poster"><span>{short}</span></div>'

    badge_html = ""
    if badge:
        badge_html = f'<div class="movie-card-badge">{badge}</div>'
    elif avg_rating and avg_rating > 0:
        badge_html = f'<div class="movie-card-badge">* {avg_rating:.1f}</div>'

    return f"""
    <a class="movie-card" href="{_film_href}" target="_self">
        {img_html}
        {badge_html}
        <div class="movie-card-overlay">
            <div class="movie-card-title">{title}</div>
            <div class="movie-card-year">{year_str}</div>
        </div>
    </a>
    """


def render_catalog_grid(df, max_cols=None):
    COLS = 7
    rows_iter = list(df.iterrows())
    if not rows_iter:
        st.markdown('<div class="empty-state"><div class="empty-state-text">Aucun film trouve</div></div>', unsafe_allow_html=True)
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


def render_rec_row(rank: int, row, item_id: int = None):
    """Carte de recommandation cliquable — sans bouton séparé."""
    genres_html = ""
    if hasattr(row, "genres_list") and isinstance(row.genres_list, list):
        genres_html = " · ".join(row.genres_list[:3])
    score_val = float(row.score)
    score_pct = min(int(score_val * 100 / 5 * 100), 100) if score_val <= 5 else min(int(score_val * 20), 100)
    appearances = getattr(row, "appearances", 1)
    consensus = (
        f'<span style="color:#5cfca0;margin-left:6px;font-size:0.6rem;">● {appearances} METHODES</span>'
        if appearances > 1 else ""
    )
    method_map = {"content": "CONTENT", "user": "USER-BASED", "item": "ITEM-BASED"}
    method_txt = method_map.get(getattr(row, "method", ""), getattr(row, "method", "").upper())
    badge_color = {"CONTENT": "#7c5cbf", "USER-BASED": "#2e86ab", "ITEM-BASED": "#e07b39"}.get(method_txt, "#555")

    href = f"?film={item_id}" if item_id else "#"
    _t_r = st.query_params.get('t', '')
    if _t_r and item_id:
        href = f"?film={item_id}&t={_t_r}"
    st.markdown(f"""<a href="{href}" target="_self" style="text-decoration:none;display:block;margin-bottom:8px;">
<div style="background:#0d0d1a;border:1px solid #1a1a30;border-radius:10px;padding:16px 20px;cursor:pointer;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;"><span style="font-size:0.6rem;font-weight:600;letter-spacing:2px;color:#fff;background:{badge_color};padding:2px 8px;border-radius:3px;">{method_txt}</span><span style="font-size:0.65rem;color:#333355;letter-spacing:1px;">#{rank}</span>{consensus}<span style="margin-left:auto;font-size:0.65rem;color:#22223a;">&#8594;</span></div>
<div style="font-family:Oswald,sans-serif;font-size:1.05rem;color:#f0f0ff;font-weight:500;margin-bottom:3px;">{row.title}</div>
<div style="font-size:0.72rem;color:#44445a;margin-bottom:10px;">{genres_html}</div>
<div style="background:#111128;border-radius:3px;height:3px;overflow:hidden;"><div style="height:3px;border-radius:3px;background:linear-gradient(90deg,#e50914,#ff4444);width:{score_pct}%;"></div></div>
<div style="font-size:0.62rem;color:#222240;margin-top:4px;">Score {score_val:.4f}</div>
</div></a>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# NAVBAR UNIFIÉE
# ─────────────────────────────────────────────────────────────

def render_navbar():
    """Navbar pure st.markdown — pas d'iframe, liens natifs Streamlit."""
    logged = is_logged_in()
    active = st.session_state.get("active_page", "catalogue")
    user_name = (st.session_state.user_name or st.session_state.user_email or "") if logged else ""

    pages = {
        "catalogue":  "Accueil",
        "noter":      "Noter",
        "reco":       "Recommandations",
        "fusion":     "Fusion",
        "evaluation": "Evaluation",
        "live":       "Live",
    }

    if logged:
        _t = st.query_params.get('t', '')
        _t_suffix = f"&t={_t}" if _t else ""
        nav_links_html = ""
        for key, label in pages.items():
            css_class = "unb-pill active" if key == active else "unb-pill"
            nav_links_html += f'<a class="{css_class}" href="?page={key}{_t_suffix}" target="_self">{label}</a>'
        right_html = (
            f'<span class="unb-user">{user_name}</span>'
            f'<a class="unb-logout" href="?__logout=1" target="_self">Deconnexion</a>'
        )
    else:
        nav_links_html = '<a class="unb-pill active" href="?" target="_self">Accueil</a>'
        right_html = (
            f'<div class="unb-stat"><div class="unb-sv">{total_films:,}</div><div class="unb-sl">Films</div></div>'
            f'<div class="unb-stat"><div class="unb-sv">{total_ratings:,}</div><div class="unb-sl">Evaluations</div></div>'
            f'<div class="unb-stat"><div class="unb-sv">&#9733; {avg_note}</div><div class="unb-sl">Note moy.</div></div>'
        )

    st.markdown(f"""
<div class="unb">
  <div class="unb-logo-wrap">
    <div class="unb-logo">MovieLens</div>
    <div class="unb-tag">Intelligent Recommendation Platform</div>
  </div>
  <nav class="unb-nav">{nav_links_html}</nav>
  <div class="unb-right">{right_html}</div>
</div>
""", unsafe_allow_html=True)

    if '__logout' in st.query_params:
        st.query_params.clear()
        do_logout()

render_navbar()


# ─────────────────────────────────────────────────────────────
# PAGE AUTH (si non connecte et acces a une page protegee)
# ─────────────────────────────────────────────────────────────

def _finalize_register(email: str):
    """Finalise la session après validation OTP inscription."""
    token = login_finalize(email, remember_me=False)
    st.session_state.jwt_token         = token
    st.session_state.user_email        = email
    st.session_state.otp_pending_email = None
    st.session_state.otp_dev_code      = None
    data = get_user_data(email)
    st.session_state.user_name    = data["name"]
    st.session_state.user_ratings = {str(k): v for k, v in data["ratings"].items()}
    st.session_state.genre_prefs  = data["genre_prefs"]
    st.session_state.active_page  = "catalogue"
    time.sleep(0.2)
    st.rerun()


def page_auth():
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="auth-logo">MovieLens</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Systeme de Recommandation</div>', unsafe_allow_html=True)

        # ── FLUX MOT DE PASSE OUBLIÉ : étape OTP ─────────────────
        if st.session_state.reset_step == "otp":
            reset_email = st.session_state.reset_email

            st.markdown(f"""
            <div style="
                background:#0d0d1a;
                border:1px solid #1e1e38;
                border-radius:14px;
                padding:2rem;
                text-align:center;
                margin-bottom:1rem;
            ">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🔑</div>
                <div style="font-family:'Oswald',sans-serif;font-size:1rem;color:#e2e2ee;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">
                    Réinitialisation du mot de passe
                </div>
                <div style="font-size:0.82rem;color:#44446a;line-height:1.5;">
                    Si un compte existe pour <strong style="color:#8888cc;">{reset_email}</strong>,<br>
                    un code à 6 chiffres vous a été envoyé.<br>
                    <span style="font-size:0.72rem;">Valide 5 minutes.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.reset_dev_code:
                st.info(f"🛠️ **Mode dev** — Code OTP : `{st.session_state.reset_dev_code}`\n\n*(Configurez SMTP dans secrets.toml pour l'envoi réel)*")

            otp_reset = st.text_input(
                "Code de vérification",
                key="reset_otp_field",
                placeholder="000000",
                max_chars=6,
            )
            new_pwd = st.text_input(
                "Nouveau mot de passe",
                type="password",
                key="reset_new_pwd",
                placeholder="Min. 8 car., 1 maj., 1 chiffre, 1 spécial",
            )
            new_pwd2 = st.text_input(
                "Confirmer le nouveau mot de passe",
                type="password",
                key="reset_new_pwd2",
            )

            if st.button("Réinitialiser mon mot de passe", use_container_width=True, type="primary"):
                if not otp_reset or len(otp_reset.strip()) != 6:
                    st.error("Entrez le code à 6 chiffres reçu par email.")
                elif not new_pwd or not new_pwd2:
                    st.error("Veuillez remplir les deux champs de mot de passe.")
                elif new_pwd != new_pwd2:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    with st.spinner("Réinitialisation..."):
                        ok, msg = reset_password(reset_email, otp_reset.strip(), new_pwd)
                    if ok:
                        st.success("✅ Mot de passe modifié avec succès ! Vous pouvez maintenant vous connecter.")
                        st.session_state.reset_step    = None
                        st.session_state.reset_email   = None
                        st.session_state.reset_dev_code = None
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Retour à la connexion", use_container_width=False):
                st.session_state.reset_step     = None
                st.session_state.reset_email    = None
                st.session_state.reset_dev_code = None
                st.rerun()

        # ── ÉTAPE OTP : uniquement après inscription ──────────────
        elif st.session_state.otp_pending_email:
            pending_email = st.session_state.otp_pending_email

            st.markdown(f"""
            <div style="
                background:#0d0d1a;
                border:1px solid #1e1e38;
                border-radius:14px;
                padding:2rem;
                text-align:center;
                margin-bottom:1rem;
            ">
                <div style="font-size:2rem;margin-bottom:0.5rem;">📧</div>
                <div style="font-family:'Oswald',sans-serif;font-size:1rem;color:#e2e2ee;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">
                    Confirmez votre email
                </div>
                <div style="font-size:0.82rem;color:#44446a;line-height:1.5;">
                    Un code de confirmation à 6 chiffres a été envoyé à<br>
                    <strong style="color:#8888cc;">{pending_email}</strong><br>
                    <span style="font-size:0.72rem;">Valide 5 minutes.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.otp_dev_code:
                st.info(f"🛠️ **Mode dev** — Code OTP : `{st.session_state.otp_dev_code}`\n\n*(Configurez SMTP dans secrets.toml pour l'envoi réel)*")

            otp_input = st.text_input(
                "Code de confirmation",
                key="otp_input_field",
                placeholder="000000",
                max_chars=6,
            )

            col_v, col_r = st.columns([2, 1])
            with col_v:
                if st.button("Valider mon compte", use_container_width=True, type="primary"):
                    if not otp_input or len(otp_input.strip()) != 6:
                        st.error("Entrez le code à 6 chiffres.")
                    else:
                        valid, vmsg = verify_otp(pending_email, otp_input.strip())
                        if valid:
                            _finalize_register(pending_email)
                        else:
                            st.error(vmsg)
            with col_r:
                if st.button("Renvoyer", use_container_width=True):
                    with st.spinner("Envoi..."):
                        ok, smsg = send_otp(pending_email)
                    if ok:
                        st.success("Nouveau code envoyé.")
                    else:
                        if "[DEV]" in smsg:
                            import re as _re
                            m = _re.search(r":\s*(\d{6})\s*\(", smsg)
                            if m:
                                st.session_state.otp_dev_code = m.group(1)
                        st.warning(smsg)
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Annuler", use_container_width=False):
                st.session_state.otp_pending_email = None
                st.session_state.otp_dev_code      = None
                st.rerun()

        else:
            # ── Connexion / Inscription ───────────────────────────
            tab_login, tab_register = st.tabs(["Connexion", "Inscription"])

            with tab_login:
                email    = st.text_input("Adresse e-mail", key="login_email", placeholder="vous@exemple.com")
                password = st.text_input("Mot de passe", type="password", key="login_pwd", placeholder="••••••••")

                col_chk, _ = st.columns([1, 1])
                with col_chk:
                    remember_me = st.checkbox("Se souvenir de moi (30 jours)", key="login_remember")

                if st.button("Se connecter", use_container_width=True, type="primary"):
                    if not email or not password:
                        st.error("Veuillez remplir tous les champs.")
                    else:
                        with st.spinner("Vérification..."):
                            ok, msg, token = login_user(email, password)
                        if ok and msg == "OK":
                            # Si remember_me coché → regénérer un token longue durée
                            if remember_me:
                                token = login_finalize(email.strip().lower(), remember_me=True)
                            st.session_state.jwt_token  = token
                            st.session_state.user_email = email.strip().lower()
                            data = get_user_data(email)
                            st.session_state.user_name    = data["name"]
                            st.session_state.user_ratings = {str(k): v for k, v in data["ratings"].items()}
                            st.session_state.genre_prefs  = data["genre_prefs"]
                            st.session_state.active_page  = "catalogue"
                            # Toujours stocker le token dans l'URL pour la navigation
                            st.query_params['t'] = token
                            time.sleep(0.2)
                            st.rerun()
                        else:
                            st.error(msg)

                # ── Lien Mot de passe oublié ──────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Mot de passe oublié ?", use_container_width=False, key="forgot_pwd_btn"):
                    st.session_state.reset_step  = "request"
                    st.rerun()

                st.caption("Connexion sécurisée — JWT + bcrypt")

            with tab_register:
                name    = st.text_input("Nom complet", key="reg_name", placeholder="Votre nom")
                email_r = st.text_input("Adresse e-mail", key="reg_email", placeholder="vous@exemple.com")
                pwd_r   = st.text_input("Mot de passe", type="password", key="reg_pwd",
                                        placeholder="Min. 8 car., 1 maj., 1 chiffre, 1 special")
                pwd_r2  = st.text_input("Confirmer le mot de passe", type="password", key="reg_pwd2")
                st.caption("8 caractères min · 1 majuscule · 1 chiffre · 1 caractère spécial")
                if st.button("Créer mon compte", use_container_width=True, type="primary"):
                    if pwd_r != pwd_r2:
                        st.error("Les mots de passe ne correspondent pas.")
                    else:
                        with st.spinner("Création du compte..."):
                            ok, msg, _ = register_user(name, email_r, pwd_r)
                        if ok and msg == "OTP_REQUIRED":
                            # Compte créé → envoyer OTP de confirmation
                            with st.spinner("Envoi du code de confirmation..."):
                                sent_ok, sent_msg = send_otp(email_r.strip().lower())
                            st.session_state.otp_pending_email = email_r.strip().lower()
                            if not sent_ok:
                                import re as _re
                                m = _re.search(r":\s*(\d{6})\s*\(", sent_msg)
                                if m:
                                    st.session_state.otp_dev_code = m.group(1)
                            st.rerun()
                        else:
                            st.error(msg)

        # ── FORMULAIRE DE DEMANDE DE RÉINITIALISATION ─────────────
        if st.session_state.reset_step == "request":
            st.markdown("""
            <div style="
                background:#0d0d1a;
                border:1px solid #1e1e38;
                border-radius:14px;
                padding:2rem;
                margin-top:1.5rem;
            ">
                <div style="font-family:'Oswald',sans-serif;font-size:1rem;color:#e2e2ee;letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem;">
                    🔑 Mot de passe oublié
                </div>
                <div style="font-size:0.82rem;color:#44446a;margin-bottom:0.5rem;">
                    Entrez votre adresse email. Si un compte existe, vous recevrez un code de réinitialisation.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            reset_email_input = st.text_input(
                "Adresse e-mail",
                key="forgot_email_input",
                placeholder="vous@exemple.com",
            )
            col_send, col_cancel = st.columns([2, 1])
            with col_send:
                if st.button("Envoyer le code", use_container_width=True, type="primary", key="send_reset_btn"):
                    if not reset_email_input or "@" not in reset_email_input:
                        st.error("Veuillez entrer une adresse email valide.")
                    else:
                        with st.spinner("Envoi en cours..."):
                            ok, smsg = send_reset_otp(reset_email_input.strip())
                        st.session_state.reset_email = reset_email_input.strip().lower()
                        st.session_state.reset_step  = "otp"
                        if not ok and "[DEV]" in smsg:
                            import re as _re
                            m = _re.search(r":\s*(\d{6})\s*", smsg)
                            if m:
                                st.session_state.reset_dev_code = m.group(1)
                        st.rerun()
            with col_cancel:
                if st.button("Annuler", use_container_width=True, key="cancel_reset_btn"):
                    st.session_state.reset_step = None
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE RECOMMANDATIONS (User-Based / Item-Based / Content)
# ─────────────────────────────────────────────────────────────

def page_recommandations():
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    if len(st.session_state.user_ratings) < 3:
        st.warning("Notez au moins 3 films pour obtenir des recommandations personnalisees.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("User-Based", use_container_width=True):
            st.session_state.rec_method = "user"
    with c2:
        if st.button("Item-Based", use_container_width=True):
            st.session_state.rec_method = "item"
    with c3:
        if st.button("Content-Based", use_container_width=True):
            st.session_state.rec_method = "content"

    method = st.session_state.rec_method
    method_labels = {"user": "User-Based CF", "item": "Item-Based CF", "content": "Content-Based"}
    method_descs = {
        "user": "Recommande ce qu'ont aime des utilisateurs similaires.",
        "item": "Recommande des films similaires a ceux que vous avez aimes.",
        "content": "Recommande des films avec les memes genres que vos preferes.",
    }
    st.markdown(f"""
    <div style="background:#0d0d1a;border:1px solid #1e1e38;border-radius:8px;padding:12px 16px;margin:12px 0 20px;">
        <div style="font-family:'Oswald',sans-serif;font-size:0.85rem;color:#e50914;letter-spacing:1px;">
            {method_labels[method]}
        </div>
        <div style="font-size:0.8rem;color:#6666aa;margin-top:4px;">{method_descs[method]}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Calcul des recommandations..."):
        results = recommend_live(st.session_state.user_ratings, method=method, n=5)

    if results.empty:
        st.warning("Pas assez de donnees pour generer des recommandations.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">Votre Top-5 — {method_labels[method]}</div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in enumerate(results.itertuples(), 1):
        col_p, col_c = st.columns([1, 6])
        with col_p:
            poster = fetch_poster(row.title)
            if poster:
                st.image(poster, width=70)
        with col_c:
            match = items_with_stats[items_with_stats['title'] == row.title]
            iid = int(match.iloc[0]['item_id']) if not match.empty else None
            render_rec_row(i, row, item_id=iid)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE SANS REDONDANCE (fusion 3 methodes)
# ─────────────────────────────────────────────────────────────

def page_sans_redondance():
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Recommandations sans redondance</div>
    </div>
    <div style="font-size:0.82rem;color:#44445a;margin-bottom:1.2rem;">
        Les 3 methodes sont combinees. Les doublons sont elimines et les films
        apparaissant dans plusieurs methodes sont remontés. Films en consensus = notes par plusieurs methodes.
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.user_ratings) < 3:
        st.warning("Notez au moins 3 films pour utiliser cette fonctionnalite.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    n_final = st.slider("Nombre de recommandations finales", 5, 15, 10)

    with st.spinner("Fusion des 3 methodes..."):
        results = recommend_no_redundancy(st.session_state.user_ratings, n_per_method=8, final_n=n_final)

    if results.empty:
        st.warning("Resultats insuffisants.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    multi = results[results["appearances"] > 1] if "appearances" in results.columns else pd.DataFrame()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Films uniques trouves", len(results))
    with c2: st.metric("Films consensus (2+ methodes)", len(multi))
    with c3: st.metric("Methodes fusionnees", 3)

    st.markdown(f"""
    <div class="section-header" style="margin-top:1.5rem;">
        <div class="section-title">Top-{n_final} Films</div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in enumerate(results.itertuples(), 1):
        col_p, col_c = st.columns([1, 6])
        with col_p:
            poster = fetch_poster(row.title)
            if poster:
                st.image(poster, width=70)
        with col_c:
            match = items_with_stats[items_with_stats['title'] == row.title]
            iid = int(match.iloc[0]['item_id']) if not match.empty else None
            render_rec_row(i, row, item_id=iid)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE EVALUATION & COMPARAISON
# ─────────────────────────────────────────────────────────────

def page_evaluation():
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Evaluation & Comparaison</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Calculer le RMSE (environ 30s)", type="primary"):
        st.session_state.show_rmse = True

    if st.session_state.show_rmse:
        with st.spinner("Evaluation RMSE..."):
            metrics = evaluate_rmse(sample_size=300)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("RMSE User-Based", metrics["user_based"],
                           delta="Meilleur" if metrics["best"] == "User-Based" else None)
        with c2: st.metric("RMSE Item-Based", metrics["item_based"],
                           delta="Meilleur" if metrics["best"] == "Item-Based" else None)
        with c3: st.metric("Echantillon evalue", metrics["n_evaluated"])
        st.success(f"Meilleur modele : {metrics['best']}")

    st.markdown("""
    <div class="reco-section">
        <div class="reco-title">Comparaison des 3 approches</div>
    </div>
    """, unsafe_allow_html=True)

    comparison = pd.DataFrame({
        "Approche":  ["User-Based CF", "Item-Based CF", "Content-Based"],
        "Principe":  ["Utilisateurs similaires", "Films similaires en notes", "Films similaires en genres"],
        "Avantage":  ["Decouverte", "Stable et precis", "Fonctionne sans historique"],
        "Limite":    ["Scalabilite", "Cold start items", "Sur-specialisation"],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="reco-section">
        <div class="reco-title">Apercu de la matrice User-Film</div>
    </div>
    """, unsafe_allow_html=True)
    matrix_sample = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating").iloc[:10, :10]
    st.dataframe(matrix_sample.style.background_gradient(cmap="YlOrBr", axis=None), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE NOTATION
# ─────────────────────────────────────────────────────────────

def _star_widget(mid: int, current: int) -> int | None:
    """Retourne la nouvelle note (1-5) si changée, sinon None."""
    stars_html = ""
    for s in range(1, 6):
        filled = "filled" if s <= current else ""
        stars_html += f'<span class="star {filled}" data-v="{s}">&#9733;</span>'
    clear_btn = f'<span class="clear-btn" onclick="clearRating()">&#10005;</span>' if current > 0 else ""
    widget = f"""<!DOCTYPE html><html><head><style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;overflow:hidden;display:flex;align-items:center;gap:4px;padding:2px 0;}}
.star{{font-size:1.4rem;cursor:pointer;color:#1e1e38;transition:color 0.1s,transform 0.1s;line-height:1;}}
.star.filled{{color:#e50914;}}
.star:hover,.star:hover~.star{{color:#ff4444 !important;}}
.stars:hover .star{{color:#ff4444;}}
.stars:hover .star:hover~.star{{color:#1e1e38;}}
.clear-btn{{font-size:0.65rem;color:#333355;cursor:pointer;padding:0 4px;transition:color 0.1s;margin-left:2px;}}
.clear-btn:hover{{color:#e50914;}}
</style></head><body>
<div class="stars" id="stars">{stars_html}</div>{clear_btn}
<script>
const stars=document.querySelectorAll(".star");
stars.forEach(s=>{{
  s.addEventListener("mouseover",()=>{{
    const v=+s.dataset.v;
    stars.forEach(x=>x.classList.toggle("filled",+x.dataset.v<=v));
  }});
  s.addEventListener("mouseout",()=>{{
    stars.forEach(x=>x.classList.toggle("filled",x.classList.contains("orig")));
  }});
  s.addEventListener("click",()=>{{
    const v=+s.dataset.v;
    stars.forEach(x=>x.classList.remove("orig"));
    stars.forEach(x=>{{if(+x.dataset.v<=v)x.classList.add("orig","filled");else x.classList.remove("filled");}});
    window.parent.postMessage({{type:"streamlit:setComponentValue",value:v}},"*");
  }});
}});
document.querySelectorAll(".star.filled").forEach(x=>x.classList.add("orig"));
function clearRating(){{
  stars.forEach(x=>{{x.classList.remove("filled","orig");}});
  window.parent.postMessage({{type:"streamlit:setComponentValue",value:0}},"*");
}}
</script></body></html>"""
    val = components.html(widget, height=34, scrolling=False)
    if val is not None and isinstance(val, (int, float)) and int(val) != current:
        return int(val)
    return None


def page_noter():
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Noter des films</div>
        <div class="section-count">Notez au moins 5 films pour de meilleures recommandations</div>
    </div>
    """, unsafe_allow_html=True)

    filter_genre = st.selectbox("Filtrer par genre", ["Tous"] + GENRE_LIST, key="noter_genre_filter")
    if filter_genre == "Tous":
        avg2 = ratings_df.groupby('item_id')['rating'].agg(['mean','count']).reset_index()
        avg2.columns = ['item_id','avg_rating','num_ratings']
        pool = items_df.merge(avg2, on='item_id').sort_values('num_ratings', ascending=False).head(30)
    else:
        pool = items_with_stats[
            items_with_stats['genres'].apply(
                lambda gl: filter_genre in gl if isinstance(gl, list) else False
            )
        ].sort_values('n_ratings', ascending=False).head(30)

    cols = st.columns(3)
    for i, (_, row) in enumerate(pool.iterrows()):
        mid = int(row['item_id'])
        current = int(st.session_state.user_ratings.get(str(mid), 0))
        genres_short = ", ".join(row['genres'][:2]) if isinstance(row.get('genres'), list) else ""
        year_s = str(int(row['year'])) if pd.notna(row.get('year')) and row.get('year') else ""
        with cols[i % 3]:
            poster = fetch_poster(row['title'], row.get('year'))
            no_poster_div = '<div style="width:100%;aspect-ratio:2/3;background:#0d0d1a;border-radius:6px 6px 0 0;display:flex;align-items:center;justify-content:center;"><span style="font-size:0.75rem;color:#333;">No poster</span></div>'
            poster_html = f'<img src="{poster}" style="width:100%;border-radius:6px 6px 0 0;display:block;" />' if poster else no_poster_div
            rated_indicator = f'<div style="position:absolute;top:8px;right:8px;background:#e50914;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;color:#fff;">{current}</div>' if current > 0 else ""
            _t_n = st.query_params.get('t', '')
            _noter_href = f"?film={mid}&t={_t_n}" if _t_n else f"?film={mid}"
            st.markdown(f"""<a href="{_noter_href}" target="_self" style="text-decoration:none;display:block;">
<div style="background:#0d0d1a;border:1px solid #1a1a30;border-radius:8px;overflow:hidden;margin-bottom:4px;position:relative;cursor:pointer;">
{poster_html}{rated_indicator}
<div style="padding:10px 12px 4px;">
<div style="font-family:Oswald,sans-serif;font-size:0.9rem;color:#e8e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{row['title'][:30]}</div>
<div style="font-size:0.68rem;color:#33334a;margin-top:2px;">{genres_short} {year_s}</div>
</div></div></a>""", unsafe_allow_html=True)
            new_r = _star_widget(mid, current)
            if new_r is not None:
                if new_r == 0:
                    st.session_state.user_ratings.pop(str(mid), None)
                else:
                    st.session_state.user_ratings[str(mid)] = new_r
                save_user_ratings(st.session_state.user_email, st.session_state.user_ratings)
                st.rerun()

    n_rated = len(st.session_state.user_ratings)
    st.markdown(f'<div style="text-align:center;font-size:0.72rem;color:#33334a;padding:1rem 0;">{n_rated} film(s) note(s)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE LIVE RECOMMANDATION
# ─────────────────────────────────────────────────────────────

def page_live():
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-title">
            <span class="live-dot"></span> Live Recommandation
        </div>
    </div>
    <div style="font-size:0.82rem;color:#44445a;margin-bottom:1.2rem;">
        Les recommandations se mettent a jour instantanement a chaque note ajoutee.
    </div>
    """, unsafe_allow_html=True)

    live_method = st.selectbox(
        "Methode",
        ["content", "user", "item"],
        format_func=lambda x: {"content": "Content-Based", "user": "User-Based", "item": "Item-Based"}[x],
        key="live_method_selector"
    )

    avg2 = ratings_df.groupby('item_id')['rating'].agg(['mean','count']).reset_index()
    avg2.columns = ['item_id','avg_rating','num_ratings']
    top_films = items_df.merge(avg2, on='item_id').sort_values('num_ratings', ascending=False).head(12)

    left_col, right_col = st.columns([1, 1.3])
    live_ratings = dict(st.session_state.user_ratings)

    with left_col:
        st.markdown('<div style="font-size:0.8rem;color:#6666aa;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.8rem;">Notez des films</div>', unsafe_allow_html=True)
        for _, row in top_films.iterrows():
            mid = int(row['item_id'])
            current = int(live_ratings.get(str(mid), 0))
            with st.container(border=True):
                st.caption(f"**{row['title']}**")
                new_r = st.select_slider(
                    "", options=["—","1","2","3","4","5"],
                    value=str(current) if current > 0 else "—",
                    key=f"live_{mid}", label_visibility="collapsed"
                )
                if new_r != "—":
                    live_ratings[str(mid)] = int(new_r)
        if live_ratings != st.session_state.user_ratings:
            st.session_state.user_ratings = live_ratings
            save_user_ratings(st.session_state.user_email, live_ratings)

    with right_col:
        st.markdown('<div style="font-size:0.8rem;color:#6666aa;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.8rem;">Recommandations en temps reel</div>', unsafe_allow_html=True)
        if len(live_ratings) >= 2:
            results = recommend_live(live_ratings, method=live_method, n=5)
            if not results.empty:
                for i, row in enumerate(results.itertuples(), 1):
                    col_p, col_c = st.columns([1, 5])
                    with col_p:
                        poster = fetch_poster(row.title)
                        if poster:
                            st.image(poster, width=60)
                    with col_c:
                        render_rec_row(i, row)
            else:
                st.info("Continuez a noter des films...")
        else:
            st.info("Notez au moins 2 films.")

    st.markdown('</div>', unsafe_allow_html=True)


# render_user_nav() supprimé — fusionné dans render_navbar()


# ─────────────────────────────────────────────────────────────
# DETAIL D'UN FILM — vue complète (avec notation intégrée)
# ─────────────────────────────────────────────────────────────

if st.session_state.selected_film is not None and not is_logged_in():
    st.query_params.clear()
    st.session_state.selected_film = None
    st.rerun()

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

    tmdb = fetch_tmdb_details(film['title'], year_val)
    backdrop_url = tmdb.get('backdrop')
    poster_url = tmdb.get('poster') or fetch_poster(film['title'], year_val, large=True)
    overview = tmdb.get('overview', '')
    tmdb_id = tmdb.get('tmdb_id')

    # Backdrop
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

    # Poster + info
    if poster_url:
        poster_html = f'<img src="{poster_url}" alt="{film["title"]}">'
    else:
        poster_html = '<div class="detail-poster-fallback">--</div>'

    genres_tags = ''.join([f'<span class="detail-genre-tag">{g}</span>' for g in genres_list[:5]])
    if not genres_tags:
        genres_tags = f'<span class="detail-genre-tag">{genres_str}</span>'

    if overview:
        overview_html = f'<div class="detail-overview">{overview}</div>'
    else:
        overview_html = '<div class="detail-overview-empty">Pas de description pour ce film.</div>'

    safe_title = film['title'].replace("'", "&#39;").replace('"', '&quot;')

    tmdb_rating_val = tmdb.get('tmdb_rating') or 0

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
                    <div class="love-label">Fans (4+)</div>
                </div>
                <div class="love-divider"></div>
                <div class="love-item">
                    <div class="love-value">* {avg_r_val:.2f}</div>
                    <div class="love-label">Note moyenne</div>
                </div>
                <div class="love-divider"></div>
                <div class="love-item">
                    <div class="love-value">{tmdb_rating_val:.1f}/10</div>
                    <div class="love-label">Note TMDB</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bouton retour
    st.markdown("""
    <div style="padding: 0 3rem 0 3rem; margin-top: -0.5rem;">
        <a href="javascript:history.back()" class="back-link">Retour</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # Notation personnelle (si connecte)
    if is_logged_in():
        st.markdown('<div class="rating-section"><div class="rating-section-title">Votre note</div></div>', unsafe_allow_html=True)
        current_r = int(st.session_state.user_ratings.get(str(film_id), 0))
        new_r = st.select_slider(
            "Notez ce film",
            options=["—", "1", "2", "3", "4", "5"],
            value=str(current_r) if current_r > 0 else "—",
            key=f"detail_rate_{film_id}",
            label_visibility="collapsed",
        )
        if new_r != "—":
            r_val = int(new_r)
            if st.session_state.user_ratings.get(str(film_id)) != r_val:
                st.session_state.user_ratings[str(film_id)] = r_val
                save_user_ratings(st.session_state.user_email, st.session_state.user_ratings)
                st.success("Note sauvegardee.")

    # Distribution des notes
    movie_ratings = ratings_df[ratings_df['item_id'] == film_id]['rating']
    if not movie_ratings.empty:
        st.markdown('<div class="reco-section"><div class="reco-title">Distribution des notes</div></div>', unsafe_allow_html=True)
        dist = movie_ratings.value_counts().sort_index()
        st.bar_chart(dist, height=160)

    # Recommandations basees sur ce film
    st.markdown('<div class="reco-section"><div class="reco-title">Films similaires recommandes</div></div>', unsafe_allow_html=True)

    with st.spinner("Calcul des recommandations..."):
        similar = cb_model.get_similar_items(film_id, n=10)

    if not similar.empty:
        similar = similar.merge(
            rating_stats[['item_id', 'avg_rating', 'n_ratings', 'n_fans']],
            on='item_id', how='left'
        )
        similar['avg_rating'] = similar['avg_rating'].fillna(0)
        similar['n_ratings'] = similar['n_ratings'].fillna(0).astype(int)
        similar['n_fans'] = similar['n_fans'].fillna(0).astype(int)
        render_poster_row(similar, cols_count=10)
    else:
        st.markdown('<div class="empty-state"><div class="empty-state-text">Pas assez de donnees pour recommander</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # content-wrap


# ─────────────────────────────────────────────────────────────
# VUE PRINCIPALE
# ─────────────────────────────────────────────────────────────

else:
    # Navigation utilisateur connecte
    if is_logged_in():
        page = st.session_state.active_page

        if page == "noter":
            page_noter()
        elif page == "reco":
            page_recommandations()
        elif page == "fusion":
            page_sans_redondance()
        elif page == "evaluation":
            page_evaluation()
        elif page == "live":
            page_live()
        elif page == "auth":
            page_auth()
        else:
            # Catalogue par defaut
            genre_html_parts = []
            all_active = "active" if st.session_state.selected_genre is None else ""
            _t_g = st.query_params.get('t', '')
            _tg = f"&t={_t_g}" if _t_g else ""
            genre_html_parts.append(f'<a class="genre-pill genre-pill-all {all_active}" href="?{_tg}" target="_self">TOUS</a>')
            for g in GENRE_LIST:
                active = "active" if st.session_state.selected_genre == g else ""
                genre_html_parts.append(f'<a class="genre-pill {active}" href="?genre={g}{_tg}" target="_self">{g}</a>')
            st.markdown('<div class="genre-bar">' + ''.join(genre_html_parts) + '</div>', unsafe_allow_html=True)

            st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
            col_search, _ = st.columns([3, 6])
            with col_search:
                search_val = st.text_input(
                    "Rechercher",
                    value=st.session_state.search_query,
                    placeholder="Rechercher un film...",
                    key="search_input",
                    label_visibility="collapsed"
                )
                if search_val != st.session_state.search_query:
                    st.query_params['q'] = search_val
                    st.rerun()

            catalog = items_with_stats.copy()
            if st.session_state.search_query:
                q = st.session_state.search_query.lower()
                catalog = catalog[catalog['title'].str.lower().str.contains(q, na=False)]
                title_label = f"Resultats pour « {st.session_state.search_query} »"
            elif st.session_state.selected_genre:
                g = st.session_state.selected_genre
                catalog = catalog[catalog['genres'].apply(lambda gl: g in gl if isinstance(gl, list) else False)]
                title_label = g
            else:
                catalog = catalog[catalog['n_ratings'] >= 5].copy()
                catalog['pop_score'] = (
                    catalog['avg_rating'] * 0.4 +
                    (catalog['n_ratings'] / catalog['n_ratings'].max()) * 0.6
                )
                catalog = catalog.sort_values('pop_score', ascending=False)
                title_label = "Catalogue · Tous les films"

            if st.session_state.selected_genre or st.session_state.search_query:
                catalog['pop_score'] = (
                    catalog['avg_rating'] * 0.4 +
                    (catalog['n_ratings'] / max(catalog['n_ratings'].max(), 1)) * 0.6
                )
                catalog = catalog.sort_values('pop_score', ascending=False)

            n_found = len(catalog)
            st.markdown(f"""
            <div class="section-header">
                <div class="section-title">{title_label}</div>
                <div class="section-count">{n_found} film{'s' if n_found > 1 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

            if catalog.empty:
                st.markdown('<div class="empty-state"><div class="empty-state-text">Aucun film trouve pour cette selection</div></div>', unsafe_allow_html=True)
            else:
                with st.spinner("Chargement des affiches..."):
                    render_catalog_grid(catalog)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Non connecte : uniquement la page d'authentification, rien d'autre
        page_auth()


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <div>
        <div class="footer-logo">MovieLens</div>
    </div>
    <div class="footer-info">MovieLens 100K · Content-Based Filtering · User-Based CF · Item-Based CF · TMDB Posters</div>
    <div class="footer-links">
        <span>943 utilisateurs</span>
        <span>1,682 films</span>
        <span>100,000 evaluations</span>
    </div>
</div>
""", unsafe_allow_html=True)