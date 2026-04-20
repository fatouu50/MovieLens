# 🎬 CineMatch — Système de Recommandation de Films

Application Streamlit complète utilisant le dataset **MovieLens 100K**.

## Fonctionnalités

| Fonctionnalité | Détail |
|---|---|
| 🔐 Authentification | Inscription / Connexion sécurisées |
| 🔑 Sécurité | JWT (tokens) + bcrypt (hachage) + anti brute-force |
| 👥 User-Based CF | Similarité cosinus entre utilisateurs |
| 🎬 Item-Based CF | Similarité cosinus entre films |
| 🏷️ Content-Based | Profil genres de l'utilisateur |
| 🔴 Live | Recommandations en temps réel |
| 🔄 Sans redondance | Fusion des 3 méthodes, dé-duplication |
| 📊 Évaluation | RMSE sur jeu de test 80/20 |
| ⭐ Notation | Interface de notation intuitive |

## Structure du projet

```
cinematch/
├── app.py                  # Application Streamlit principale
├── auth.py                 # Authentification JWT + bcrypt
├── recommender.py          # Moteur de recommandation
├── requirements.txt        # Dépendances Python
├── .gitignore              # Exclut users_db.json et secrets
├── .streamlit/
│   ├── config.toml         # Thème et config Streamlit
│   └── secrets.toml        # ⚠️ NE PAS COMMITER (clé JWT)
└── data/
    ├── u.data              # 100K ratings
    ├── u.item              # 1682 films + genres
    ├── u.genre             # Liste des genres
    └── u.user              # Infos utilisateurs
```

## Installation locale

```bash
# 1. Cloner le projet
git clone https://github.com/TON_USERNAME/cinematch.git
cd cinematch

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé JWT
# Créer le fichier .streamlit/secrets.toml :
# [security]
# JWT_SECRET = "ta_cle_secrete_ici"

# 4. Lancer l'application
streamlit run app.py
```

## Déploiement sur Streamlit Cloud

### Étape 1 — Préparer le dépôt GitHub

```bash
git init
git add app.py auth.py recommender.py requirements.txt .streamlit/config.toml data/ .gitignore
# ⚠️ Ne PAS ajouter secrets.toml ni users_db.json
git commit -m "Initial commit — CineMatch"
git remote add origin https://github.com/TON_USERNAME/cinematch.git
git push -u origin main
```

### Étape 2 — Déployer sur Streamlit Cloud

1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Cliquer **"New app"**
3. Sélectionner ton dépôt GitHub → branche `main` → fichier `app.py`
4. Cliquer **"Advanced settings"** → **"Secrets"**
5. Coller :
   ```toml
   [security]
   JWT_SECRET = "ta_cle_generee_avec_secrets.token_hex(32)"
   ```
6. Cliquer **"Deploy"** ✅

### Étape 3 — Générer une clé JWT sécurisée

```python
import secrets
print(secrets.token_hex(32))
# Exemple : a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1
```

## Sécurité

- **bcrypt** (12 rounds) : hachage irréversible des mots de passe
- **JWT** (HS256, TTL 8h) : sessions sans état côté serveur
- **Anti brute-force** : blocage 5 min après 5 tentatives échouées
- **Validation** : email, mot de passe (8 car., maj., chiffre, spécial)
- **users_db.json** : exclu du dépôt git (`.gitignore`)
- **XSRF Protection** : activé dans `config.toml`

## Technologies

- Python 3.10+
- Streamlit 1.32+
- Pandas, NumPy, Scikit-learn
- bcrypt, PyJWT
