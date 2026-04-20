"""
auth.py — Authentification sécurisée avec Supabase
Sécurité : bcrypt (hachage mots de passe) + JWT (sessions) + HTTPS-ready
"""

import bcrypt
import jwt
import os
import re
import time
import hashlib
import secrets
import requests
import json
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG SÉCURITÉ
# ─────────────────────────────────────────────
try:
    import streamlit as st
    SECRET_KEY    = st.secrets["security"]["JWT_SECRET"]
    SUPABASE_URL  = st.secrets["supabase"]["supabase_url"]
    SUPABASE_KEY  = st.secrets["supabase"]["anon_key"]
except Exception:
    SECRET_KEY    = os.environ.get("JWT_SECRET", secrets.token_hex(32))
    SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY  = os.environ.get("SUPABASE_ANON_KEY", "")

JWT_ALGO      = "HS256"
TOKEN_TTL     = 3600 * 8
BCRYPT_ROUNDS = 12
MAX_ATTEMPTS  = 5
LOCKOUT_SEC   = 300

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


# ─────────────────────────────────────────────
# SUPABASE HELPERS
# ─────────────────────────────────────────────

def _get_user(email: str) -> dict | None:
    url = f"{SUPABASE_URL}/rest/v1/cinematch_users?email=eq.{email}&limit=1"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code == 200 and r.json():
        return r.json()[0]
    return None

def _create_user(data: dict) -> bool:
    url = f"{SUPABASE_URL}/rest/v1/cinematch_users"
    r = requests.post(url, headers={**HEADERS, "Prefer": "return=minimal"},
                      json=data, timeout=10)
    return r.status_code in (200, 201)

def _update_user(email: str, data: dict) -> bool:
    url = f"{SUPABASE_URL}/rest/v1/cinematch_users?email=eq.{email}"
    r = requests.patch(url, headers={**HEADERS, "Prefer": "return=minimal"},
                       json=data, timeout=10)
    return r.status_code in (200, 204)


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def _validate_email(email: str) -> bool:
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email.strip()))

def _validate_password(pwd: str) -> tuple[bool, str]:
    if len(pwd) < 8:
        return False, "Minimum 8 caractères requis."
    if not re.search(r"[A-Z]", pwd):
        return False, "Au moins une majuscule requise."
    if not re.search(r"[0-9]", pwd):
        return False, "Au moins un chiffre requis."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pwd):
        return False, "Au moins un caractère spécial requis."
    return True, "OK"


# ─────────────────────────────────────────────
# ANTI BRUTE-FORCE
# ─────────────────────────────────────────────

def _check_lockout(user: dict) -> tuple[bool, int]:
    attempts  = user.get("failed_attempts", 0)
    last_fail = user.get("last_failed_at", 0) or 0
    if attempts >= MAX_ATTEMPTS:
        elapsed = time.time() - float(last_fail)
        if elapsed < LOCKOUT_SEC:
            return True, int(LOCKOUT_SEC - elapsed)
    return False, 0

def _record_failed(email: str, user: dict):
    attempts = user.get("failed_attempts", 0) + 1
    _update_user(email, {"failed_attempts": attempts, "last_failed_at": time.time()})

def _reset_attempts(email: str):
    _update_user(email, {"failed_attempts": 0, "last_failed_at": 0})


# ─────────────────────────────────────────────
# JWT TOKENS
# ─────────────────────────────────────────────

def _generate_token(email: str, name: str) -> str:
    payload = {
        "sub":   hashlib.sha256(email.encode()).hexdigest(),
        "name":  name,
        "email": email,
        "iat":   int(time.time()),
        "exp":   int(time.time()) + TOKEN_TTL,
        "jti":   secrets.token_hex(16),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGO)

def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGO])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ─────────────────────────────────────────────
# INSCRIPTION
# ─────────────────────────────────────────────

def register_user(name: str, email: str, password: str) -> tuple[bool, str]:
    name  = name.strip()
    email = email.strip().lower()

    if not name or len(name) < 2:
        return False, "Nom invalide (min. 2 caractères)."
    if not _validate_email(email):
        return False, "Adresse e-mail invalide."
    ok, msg = _validate_password(password)
    if not ok:
        return False, msg

    if _get_user(email):
        return False, "Un compte existe déjà avec cet email."

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS))
    data = {
        "name":            name,
        "email":           email,
        "password_hash":   hashed.decode("utf-8"),
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "failed_attempts": 0,
        "last_failed_at":  0,
        "ratings":         {},
        "genre_prefs":     [],
    }
    if _create_user(data):
        return True, "Compte créé avec succès."
    return False, "Erreur lors de la création du compte. Réessayez."


# ─────────────────────────────────────────────
# CONNEXION
# ─────────────────────────────────────────────

def login_user(email: str, password: str) -> tuple[bool, str, str | None]:
    email = email.strip().lower()
    user  = _get_user(email)

    if not user:
        return False, "Email ou mot de passe incorrect.", None

    locked, remaining = _check_lockout(user)
    if locked:
        mins = remaining // 60
        secs = remaining % 60
        return False, f"Compte temporairement bloqué. Réessayez dans {mins}m{secs:02d}s.", None

    stored_hash = user["password_hash"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        _record_failed(email, user)
        attempts_left = MAX_ATTEMPTS - (user.get("failed_attempts", 0) + 1)
        if attempts_left <= 0:
            return False, f"Trop de tentatives. Compte bloqué {LOCKOUT_SEC//60} min.", None
        return False, f"Mot de passe incorrect. {attempts_left} tentative(s) restante(s).", None

    _reset_attempts(email)
    _update_user(email, {"last_login": datetime.now(timezone.utc).isoformat()})
    token = _generate_token(email, user["name"])
    return True, "Connexion réussie.", token


# ─────────────────────────────────────────────
# DONNÉES UTILISATEUR
# ─────────────────────────────────────────────

def get_user_data(email: str) -> dict:
    user = _get_user(email.lower()) or {}
    ratings = user.get("ratings", {})
    if isinstance(ratings, str):
        try:
            ratings = json.loads(ratings)
        except Exception:
            ratings = {}
    genre_prefs = user.get("genre_prefs", [])
    if isinstance(genre_prefs, str):
        try:
            genre_prefs = json.loads(genre_prefs)
        except Exception:
            genre_prefs = []
    return {
        "name":        user.get("name", ""),
        "ratings":     ratings,
        "genre_prefs": genre_prefs,
        "created_at":  user.get("created_at", ""),
    }

def save_user_ratings(email: str, ratings: dict):
    _update_user(email.lower(), {"ratings": ratings})

def save_user_genres(email: str, genres: list):
    _update_user(email.lower(), {"genre_prefs": genres})
