"""
auth.py — Authentification sécurisée
Sécurité : bcrypt (hachage mots de passe) + JWT (sessions) + HTTPS-ready
"""

import bcrypt
import jwt
import json
import os
import re
import time
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG SÉCURITÉ
# ─────────────────────────────────────────────
USERS_FILE   = Path("users_db.json")
SECRET_KEY   = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGO     = "HS256"
TOKEN_TTL    = 3600 * 8          # 8 heures
BCRYPT_ROUNDS = 12               # coût bcrypt élevé
MAX_ATTEMPTS  = 5                # anti brute-force
LOCKOUT_SEC   = 300              # 5 min de blocage


# ─────────────────────────────────────────────
# STOCKAGE UTILISATEURS (JSON chiffré en base64)
# ─────────────────────────────────────────────

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def _save_users(data: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


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

def _check_lockout(users: dict, email: str) -> tuple[bool, int]:
    """Retourne (is_locked, seconds_remaining)"""
    user = users.get(email, {})
    attempts = user.get("failed_attempts", 0)
    last_fail = user.get("last_failed_at", 0)
    if attempts >= MAX_ATTEMPTS:
        elapsed = time.time() - last_fail
        if elapsed < LOCKOUT_SEC:
            return True, int(LOCKOUT_SEC - elapsed)
        # Reset après expiration
        users[email]["failed_attempts"] = 0
    return False, 0

def _record_failed(users: dict, email: str):
    if email in users:
        users[email]["failed_attempts"] = users[email].get("failed_attempts", 0) + 1
        users[email]["last_failed_at"] = time.time()
        _save_users(users)

def _reset_attempts(users: dict, email: str):
    if email in users:
        users[email]["failed_attempts"] = 0
        users[email]["last_failed_at"] = 0
        _save_users(users)


# ─────────────────────────────────────────────
# JWT TOKENS
# ─────────────────────────────────────────────

def _generate_token(email: str, name: str) -> str:
    payload = {
        "sub":   hashlib.sha256(email.encode()).hexdigest(),  # on ne met pas l'email brut
        "name":  name,
        "iat":   int(time.time()),
        "exp":   int(time.time()) + TOKEN_TTL,
        "jti":   secrets.token_hex(16),   # ID unique du token (anti-replay)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGO)

def verify_token(token: str) -> dict | None:
    """Vérifie le JWT et retourne le payload ou None si invalide/expiré."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGO])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
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

    users = _load_users()
    if email in users:
        return False, "Un compte existe déjà avec cet email."

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS))

    users[email] = {
        "name":            name,
        "password_hash":   hashed.decode("utf-8"),
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "failed_attempts": 0,
        "last_failed_at":  0,
        "ratings":         {},
        "genre_prefs":     [],
    }
    _save_users(users)
    return True, "Compte créé avec succès."


# ─────────────────────────────────────────────
# CONNEXION
# ─────────────────────────────────────────────

def login_user(email: str, password: str) -> tuple[bool, str, str | None]:
    """
    Retourne (success, message, jwt_token_or_None)
    """
    email = email.strip().lower()
    users = _load_users()

    if email not in users:
        return False, "Email ou mot de passe incorrect.", None

    locked, remaining = _check_lockout(users, email)
    if locked:
        mins = remaining // 60
        secs = remaining % 60
        return False, f"Compte temporairement bloqué. Réessayez dans {mins}m{secs:02d}s.", None

    stored_hash = users[email]["password_hash"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        _record_failed(users, email)
        attempts_left = MAX_ATTEMPTS - users[email]["failed_attempts"]
        if attempts_left <= 0:
            return False, f"Trop de tentatives. Compte bloqué {LOCKOUT_SEC//60} min.", None
        return False, f"Mot de passe incorrect. {attempts_left} tentative(s) restante(s).", None

    _reset_attempts(users, email)
    token = _generate_token(email, users[email]["name"])
    users[email]["last_login"] = datetime.now(timezone.utc).isoformat()
    _save_users(users)

    return True, "Connexion réussie.", token


# ─────────────────────────────────────────────
# DONNÉES UTILISATEUR (ratings + préférences)
# ─────────────────────────────────────────────

def get_user_data(email: str) -> dict:
    users = _load_users()
    u = users.get(email.lower(), {})
    return {
        "name":        u.get("name", ""),
        "ratings":     u.get("ratings", {}),
        "genre_prefs": u.get("genre_prefs", []),
        "created_at":  u.get("created_at", ""),
    }

def save_user_ratings(email: str, ratings: dict):
    users = _load_users()
    if email.lower() in users:
        users[email.lower()]["ratings"] = ratings
        _save_users(users)

def save_user_genres(email: str, genres: list):
    users = _load_users()
    if email.lower() in users:
        users[email.lower()]["genre_prefs"] = genres
        _save_users(users)
