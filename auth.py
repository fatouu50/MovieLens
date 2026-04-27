"""
auth.py — Authentification sécurisée avec Supabase
Sécurité : bcrypt (hachage mots de passe) + JWT (sessions) + HTTPS-ready
Nouvelles fonctionnalités : OTP email (vérification 2 étapes) + Se souvenir de moi (JWT 30 jours)
"""

import bcrypt
import jwt
import os
import re
import time
import hashlib
import secrets
import random
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG SÉCURITÉ
# ─────────────────────────────────────────────
def _load_secrets():
    """Charge les secrets depuis Streamlit ou les variables d'environnement."""
    jwt_secret   = None
    supabase_url = None
    supabase_key = None
    smtp_host    = None
    smtp_port    = None
    smtp_user    = None
    smtp_pass    = None
    smtp_from    = None

    # Tentative via st.secrets
    try:
        import streamlit as st
        jwt_secret   = st.secrets.get("security", {}).get("JWT_SECRET")
        supabase_url = st.secrets.get("supabase", {}).get("supabase_url")
        supabase_key = st.secrets.get("supabase", {}).get("anon_key")
        smtp_host    = st.secrets.get("smtp", {}).get("host")
        smtp_port    = st.secrets.get("smtp", {}).get("port", 587)
        smtp_user    = st.secrets.get("smtp", {}).get("username")
        smtp_pass    = st.secrets.get("smtp", {}).get("password")
        smtp_from    = st.secrets.get("smtp", {}).get("from_email")
    except Exception:
        pass

    # Fallback variables d'environnement
    if not jwt_secret:
        jwt_secret = os.environ.get("JWT_SECRET", secrets.token_hex(32))
    if not supabase_url:
        supabase_url = os.environ.get("SUPABASE_URL", "")
    if not supabase_key:
        supabase_key = os.environ.get("SUPABASE_ANON_KEY", "")
    if not smtp_host:
        smtp_host = os.environ.get("SMTP_HOST", "")
    if not smtp_port:
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
    if not smtp_user:
        smtp_user = os.environ.get("SMTP_USERNAME", "")
    if not smtp_pass:
        smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    if not smtp_from:
        smtp_from = os.environ.get("SMTP_FROM", smtp_user)

    # Validation : URL Supabase doit commencer par https://
    if supabase_url and not supabase_url.startswith("https://"):
        supabase_url = "https://" + supabase_url

    return jwt_secret, supabase_url, supabase_key, smtp_host, int(smtp_port or 587), smtp_user, smtp_pass, smtp_from

SECRET_KEY, SUPABASE_URL, SUPABASE_KEY, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM = _load_secrets()

# Vérification critique au démarrage
if not SUPABASE_URL:
    try:
        import streamlit as st
        st.error(
            "⚠️ **Configuration Supabase manquante.**\n\n"
            "Ajoutez dans `.streamlit/secrets.toml` :\n"
            "```toml\n[supabase]\nsupabase_url = \"https://xxxx.supabase.co\"\nanon_key = \"eyJ...\"\n\n[security]\nJWT_SECRET = \"votre_secret\"\n\n[smtp]\nhost = \"smtp.gmail.com\"\nport = 587\nusername = \"vous@gmail.com\"\npassword = \"mot_de_passe_app\"\nfrom_email = \"vous@gmail.com\"\n```"
        )
        st.stop()
    except Exception:
        raise RuntimeError(
            "SUPABASE_URL est vide. Définissez la variable d'environnement "
            "SUPABASE_URL ou configurez .streamlit/secrets.toml."
        )

JWT_ALGO         = "HS256"
TOKEN_TTL        = 3600 * 8          # 8 heures (session normale)
TOKEN_TTL_LONG   = 3600 * 24 * 30    # 30 jours (se souvenir de moi)
BCRYPT_ROUNDS    = 12
MAX_ATTEMPTS     = 5
LOCKOUT_SEC      = 300
OTP_TTL          = 300               # 5 minutes
OTP_LENGTH       = 6

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
    if r.status_code not in (200, 201):
        import streamlit as st
        st.error(f"Supabase error {r.status_code}: {r.text}")
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

def _generate_token(email: str, name: str, remember_me: bool = False) -> str:
    ttl = TOKEN_TTL_LONG if remember_me else TOKEN_TTL
    payload = {
        "sub":         hashlib.sha256(email.encode()).hexdigest(),
        "name":        name,
        "email":       email,
        "iat":         int(time.time()),
        "exp":         int(time.time()) + ttl,
        "jti":         secrets.token_hex(16),
        "remember_me": remember_me,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGO)

def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGO])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ─────────────────────────────────────────────
# OTP — GÉNÉRATION & ENVOI EMAIL
# ─────────────────────────────────────────────

def _generate_otp() -> str:
    """Génère un code OTP numérique à 6 chiffres."""
    return str(random.randint(100000, 999999))

def _send_otp_email(email: str, otp: str, name: str) -> tuple[bool, str]:
    """
    Envoie le code OTP par email via SMTP.
    Retourne (succès, message).
    Si SMTP non configuré, retourne (False, message) avec le code affiché
    en mode développement.
    """
    if not SMTP_HOST or not SMTP_USER:
        # Mode développement : pas de SMTP configuré
        return False, f"[DEV] Code OTP pour {email} : {otp} (configurez SMTP dans secrets.toml pour l'envoi réel)"

    subject = "MovieLens — Confirmez votre inscription"
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #080810; color: #e2e2ee; margin: 0; padding: 0; }}
    .container {{ max-width: 480px; margin: 40px auto; background: #0d0d1a; border: 1px solid #1e1e38; border-radius: 14px; padding: 40px 32px; }}
    .logo {{ font-size: 2rem; font-weight: 700; letter-spacing: 4px; color: #e50914; text-transform: uppercase; text-align: center; margin-bottom: 8px; }}
    .sub {{ font-size: 0.7rem; color: #333355; letter-spacing: 2px; text-transform: uppercase; text-align: center; margin-bottom: 32px; }}
    .greeting {{ font-size: 0.95rem; color: #a0a0c0; margin-bottom: 24px; }}
    .otp-box {{ background: #111128; border: 1px solid #e50914; border-radius: 10px; text-align: center; padding: 24px; margin: 24px 0; }}
    .otp-code {{ font-size: 2.8rem; font-weight: 700; letter-spacing: 12px; color: #ffffff; font-family: monospace; }}
    .otp-note {{ font-size: 0.75rem; color: #44446a; margin-top: 10px; }}
    .footer-note {{ font-size: 0.72rem; color: #222240; text-align: center; margin-top: 28px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="logo">MovieLens</div>
    <div class="sub">Système de Recommandation</div>
    <div class="greeting">Bienvenue {name} !</div>
    <p style="color:#8888aa;font-size:0.9rem;">Merci de vous inscrire sur MovieLens. Voici votre code de confirmation pour valider votre compte :</p>
    <div class="otp-box">
      <div class="otp-code">{otp}</div>
      <div class="otp-note">Valide pendant 5 minutes</div>
    </div>
    <p style="color:#6666aa;font-size:0.85rem;">Si vous n'avez pas créé de compte, ignorez cet email.</p>
    <div class="footer-note">MovieLens · Plateforme de recommandation de films<br>Ce code est à usage unique et confidentiel.</div>
  </div>
</body>
</html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_FROM or SMTP_USER
        msg["To"]      = email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM or SMTP_USER, email, msg.as_string())

        return True, "Code OTP envoyé avec succès."
    except smtplib.SMTPAuthenticationError:
        return False, "Erreur SMTP : identifiants incorrects. Vérifiez votre configuration."
    except smtplib.SMTPConnectError:
        return False, "Impossible de joindre le serveur SMTP. Vérifiez l'hôte et le port."
    except Exception as e:
        return False, f"Erreur lors de l'envoi de l'email : {str(e)}"

def send_otp(email: str) -> tuple[bool, str]:
    """
    Génère un OTP, le stocke dans Supabase et l'envoie par email.
    Retourne (succès, message).
    """
    user = _get_user(email.lower())
    if not user:
        return False, "Utilisateur introuvable."

    otp = _generate_otp()
    otp_expires = time.time() + OTP_TTL

    _update_user(email.lower(), {
        "otp_code":    otp,
        "otp_expires": otp_expires,
    })

    ok, msg = _send_otp_email(email, otp, user.get("name", ""))
    return ok, msg

def verify_otp(email: str, code: str) -> tuple[bool, str]:
    """
    Vérifie le code OTP saisi par l'utilisateur.
    Retourne (valide, message).
    """
    user = _get_user(email.lower())
    if not user:
        return False, "Utilisateur introuvable."

    stored_otp  = user.get("otp_code", "")
    otp_expires = user.get("otp_expires", 0) or 0

    if not stored_otp:
        return False, "Aucun code OTP en attente. Reconnectez-vous."

    if time.time() > float(otp_expires):
        _update_user(email.lower(), {"otp_code": None, "otp_expires": None})
        return False, "Code OTP expiré. Reconnectez-vous pour en recevoir un nouveau."

    if code.strip() != str(stored_otp):
        return False, "Code OTP incorrect."

    # OTP valide : on efface le code
    _update_user(email.lower(), {"otp_code": None, "otp_expires": None})
    return True, "Code vérifié."


# ─────────────────────────────────────────────
# INSCRIPTION
# ─────────────────────────────────────────────

def register_user(name: str, email: str, password: str) -> tuple[bool, str, str | None]:
    name  = name.strip()
    email = email.strip().lower()

    if not name or len(name) < 2:
        return False, "Nom invalide (min. 2 caractères).", None
    if not _validate_email(email):
        return False, "Adresse e-mail invalide.", None
    ok, msg = _validate_password(password)
    if not ok:
        return False, msg, None

    if _get_user(email):
        return False, "Un compte existe déjà avec cet email.", None

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
        "otp_code":        None,
        "otp_expires":     None,
    }
    if _create_user(data):
        # Compte créé → envoyer OTP pour valider l'email
        return True, "OTP_REQUIRED", None
    return False, "Erreur lors de la création du compte. Réessayez.", None


# ─────────────────────────────────────────────
# CONNEXION (étape 1 — vérification mot de passe)
# ─────────────────────────────────────────────

def login_user(email: str, password: str) -> tuple[bool, str, str | None]:
    """
    Étape 1 du login : vérifie email + mot de passe.
    Si OK, retourne (True, "OTP_REQUIRED", None) pour déclencher l'envoi OTP.
    """
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

    # Mot de passe correct → JWT direct (OTP uniquement à l'inscription)
    user_data = _get_user(email) or {}
    token = _generate_token(email, user_data.get("name", ""), remember_me=False)
    return True, "OK", token


def login_finalize(email: str, remember_me: bool = False) -> str:
    """
    Génère le JWT final avec remember_me — appelé après validation OTP inscription
    ou pour renouveler un token longue durée.
    """
    user = _get_user(email.lower()) or {}
    return _generate_token(email.lower(), user.get("name", ""), remember_me=remember_me)


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


# ─────────────────────────────────────────────
# MOT DE PASSE OUBLIÉ
# ─────────────────────────────────────────────

def send_reset_otp(email: str) -> tuple[bool, str]:
    """
    Génère un OTP de réinitialisation et l'envoie par email.
    Retourne (succès, message).
    """
    email = email.strip().lower()
    user = _get_user(email)
    if not user:
        # Message volontairement neutre pour éviter l'énumération d'emails
        return True, "Si un compte existe pour cet email, un code a été envoyé."

    otp = _generate_otp()
    otp_expires = time.time() + OTP_TTL

    _update_user(email, {
        "otp_code":    otp,
        "otp_expires": otp_expires,
    })

    # Email HTML pour la réinitialisation
    subject = "MovieLens — Réinitialisation de votre mot de passe"
    name = user.get("name", "")
    html_body = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #080810; color: #e2e2ee; margin: 0; padding: 0; }}
    .container {{ max-width: 480px; margin: 40px auto; background: #0d0d1a; border: 1px solid #1e1e38; border-radius: 14px; padding: 40px 32px; }}
    .logo {{ font-size: 2rem; font-weight: 700; letter-spacing: 4px; color: #e50914; text-transform: uppercase; text-align: center; margin-bottom: 8px; }}
    .sub {{ font-size: 0.7rem; color: #333355; letter-spacing: 2px; text-transform: uppercase; text-align: center; margin-bottom: 32px; }}
    .greeting {{ font-size: 0.95rem; color: #a0a0c0; margin-bottom: 24px; }}
    .otp-box {{ background: #111128; border: 1px solid #e50914; border-radius: 10px; text-align: center; padding: 24px; margin: 24px 0; }}
    .otp-code {{ font-size: 2.8rem; font-weight: 700; letter-spacing: 12px; color: #ffffff; font-family: monospace; }}
    .otp-note {{ font-size: 0.75rem; color: #44446a; margin-top: 10px; }}
    .footer-note {{ font-size: 0.72rem; color: #222240; text-align: center; margin-top: 28px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="logo">MovieLens</div>
    <div class="sub">Système de Recommandation</div>
    <div class="greeting">Bonjour {name} !</div>
    <p style="color:#8888aa;font-size:0.9rem;">Vous avez demandé la réinitialisation de votre mot de passe. Utilisez ce code pour en définir un nouveau :</p>
    <div class="otp-box">
      <div class="otp-code">{otp}</div>
      <div class="otp-note">Valide pendant 5 minutes</div>
    </div>
    <p style="color:#6666aa;font-size:0.85rem;">Si vous n'avez pas fait cette demande, ignorez cet email. Votre mot de passe reste inchangé.</p>
    <div class="footer-note">MovieLens · Plateforme de recommandation de films<br>Ce code est à usage unique et confidentiel.</div>
  </div>
</body>
</html>"""

    if not SMTP_HOST or not SMTP_USER:
        # Mode dev : retourne le code dans le message
        return False, f"[DEV] SMTP non configuré. Code OTP : {otp} (valable 5 min)"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_FROM or SMTP_USER
        msg["To"]      = email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM or SMTP_USER, email, msg.as_string())

        return True, "Si un compte existe pour cet email, un code a été envoyé."
    except Exception as e:
        return False, f"Erreur lors de l'envoi de l'email : {str(e)}"


def reset_password(email: str, otp_code: str, new_password: str) -> tuple[bool, str]:
    """
    Vérifie l'OTP et met à jour le mot de passe.
    Retourne (succès, message).
    """
    email = email.strip().lower()

    valid, vmsg = verify_otp(email, otp_code)
    if not valid:
        return False, vmsg

    ok, errmsg = _validate_password(new_password)
    if not ok:
        return False, errmsg

    hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS))
    updated = _update_user(email, {
        "password_hash":   hashed.decode("utf-8"),
        "failed_attempts": 0,
        "last_failed_at":  0,
    })

    if updated:
        return True, "Mot de passe réinitialisé avec succès."
    return False, "Erreur lors de la mise à jour. Réessayez."