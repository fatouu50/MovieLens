"""
security.py — Sécurité applicative centralisée
  - Chargement sécurisé de la clé TMDB
  - Validation & sanitisation des entrées
  - Requêtes HTTP sécurisées (timeout, rate limiting, whitelist)
  - Vérification au démarrage
"""

import os
import re
import time
import requests
from collections import deque
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CLÉ TMDB
# ─────────────────────────────────────────────

def get_tmdb_key() -> str:
    """
    Cherche la clé TMDB dans l'ordre :
    1. Colab Secrets (google.colab.userdata)
    2. Variable d'environnement TMDB_API_KEY
    3. Fichier .env local
    """
    # 1. Colab
    try:
        from google.colab import userdata
        key = userdata.get('TMDB_API_KEY')
        if key:
            return key
    except Exception:
        pass

    # 2. Env var / .env
    key = os.environ.get('TMDB_API_KEY', '')
    if key:
        return key

    return ''


# ─────────────────────────────────────────────
# VALIDATION DES ENTRÉES
# ─────────────────────────────────────────────

def validate_film_id(film_id) -> bool:
    """Vérifie que l'identifiant de film est un entier positif valide."""
    try:
        return int(film_id) > 0
    except (TypeError, ValueError):
        return False


def validate_genre(genre: str, allowed_genres: list) -> bool:
    """Vérifie que le genre fait partie de la liste autorisée."""
    return isinstance(genre, str) and genre in allowed_genres


def sanitize_search_query(query: str, max_length: int = 100) -> str:
    """
    Nettoie une requête de recherche :
    - Supprime les caractères dangereux
    - Limite la longueur
    """
    if not isinstance(query, str):
        return ''
    query = query.strip()[:max_length]
    query = re.sub(r'[<>{}\[\]\\;`]', '', query)
    return query


# ─────────────────────────────────────────────
# REQUÊTES HTTP SÉCURISÉES
# ─────────────────────────────────────────────

_ALLOWED_DOMAINS = {'api.themoviedb.org'}
_rate_window: deque = deque()
_RATE_LIMIT = 40       # max requêtes
_RATE_WINDOW_SEC = 10  # par fenêtre (secondes)


def safe_tmdb_request(url: str, params: dict = None, retries: int = 2) -> dict | None:
    """
    Effectue une requête TMDB sécurisée :
    - Whitelist de domaine
    - Timeout 5s
    - Rate limiting (max 40 req / 10s)
    - Retry automatique
    """
    # Vérification domaine
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    if domain not in _ALLOWED_DOMAINS:
        return None

    # Rate limiting
    now = time.time()
    while _rate_window and now - _rate_window[0] > _RATE_WINDOW_SEC:
        _rate_window.popleft()
    if len(_rate_window) >= _RATE_LIMIT:
        return None
    _rate_window.append(now)

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            if attempt < retries:
                time.sleep(0.5)
    return None


# ─────────────────────────────────────────────
# VÉRIFICATION AU DÉMARRAGE
# ─────────────────────────────────────────────

def check_env(data_dir: str = 'data') -> list[str]:
    """
    Vérifie l'environnement au démarrage.
    Retourne une liste de messages d'avertissement (vide = tout OK).
    """
    warnings = []

    # Clé TMDB
    if not get_tmdb_key():
        warnings.append("TMDB_API_KEY manquante — les affiches ne se chargeront pas.")

    # Dossier data
    if not os.path.isdir(data_dir):
        warnings.append(f"Dossier '{data_dir}/' introuvable — lance setup_data.py.")
    else:
        required_files = ['u.data', 'u.item', 'u.user', 'u.genre']
        for f in required_files:
            if not os.path.isfile(os.path.join(data_dir, f)):
                warnings.append(f"Fichier manquant : {data_dir}/{f}")

    # .gitignore contient .env
    if os.path.isfile('.gitignore'):
        with open('.gitignore', 'r') as fh:
            content = fh.read()
        if '.env' not in content:
            warnings.append(".env n'est pas dans .gitignore — risque de fuite de clé API.")
    else:
        warnings.append(".gitignore introuvable.")

    return warnings
