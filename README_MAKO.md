# Branche `mako` — Securisation centralisee

## Pourquoi cette branche ?

Avant cette branche, la cle API TMDB etait ecrite directement dans le code.
Nimporte qui lisant le code pouvait la voir et lutiliser.
Cette branche centralise toute la securite dans un seul fichier : `src/security.py`.

---

## Les deux niveaux de securite

La securite d une application web se passe a deux niveaux differents :

### Niveau 1 — Transport reseau (TLS/SSL)
C est le cadenas visible dans le navigateur.
Il chiffre les donnees qui circulent entre l utilisateur et le serveur.

**Dans ce projet, TLS/SSL est deja actif sans configuration manuelle :**
- Les requetes vers TMDB utilisent `https://` -> chiffrement TLS actif automatiquement
- En local (localhost), Streamlit n a pas besoin de TLS
- En production sur Streamlit Cloud, TLS est fourni automatiquement par la plateforme

Tu n as rien a faire pour TLS — il est deja la.

### Niveau 2 — Securite applicative (branche mako)
C est ce que fait `src/security.py` : proteger les donnees a l interieur du code.
TLS protege le tunnel, la securite applicative protege ce qui se passe dedans.

---

## Ce qu ajoute src/security.py

### 1. Chargement securise de la cle TMDB
**Avant (dangereux) :**
```python
TMDB_API_KEY = "abc123monvraicle"  # visible par tout le monde sur GitHub
```

**Apres (securise) :**
```python
from src.security import get_tmdb_api_key
TMDB_API_KEY = get_tmdb_api_key()  # charge depuis .env, jamais dans le code
```

La fonction cherche la cle dans cet ordre :
1. Colab Secrets (si on tourne sur Google Colab)
2. Variable d environnement `TMDB_API_KEY`
3. Fichier `.env` local (jamais commite sur GitHub)

---

### 2. Validation des entrees utilisateur
Toutes les donnees venant de l utilisateur sont verifiees avant utilisation.

- `validate_film_id(id)` — verifie que l ID est un entier entre 1 et 1682
- `validate_genre(genre)` — verifie que le genre est dans la liste autorisee
- `sanitize_search_query(query)` — supprime les balises HTML et caracteres dangereux
- `sanitize_html(text)` — protege contre l injection HTML dans les templates

Sans ca, un utilisateur pourrait entrer `<script>alert('hack')</script>` dans la recherche.

---

### 3. Requetes HTTP securisees vers TMDB
La fonction `safe_tmdb_request()` remplace les appels directs a `requests.get()`.

Elle ajoute automatiquement :
- **Whitelist de domaines** : seul `api.themoviedb.org` est autorise
- **Timeout obligatoire** : 5 secondes max, evite que l app se bloque
- **Rate limiting** : pause automatique entre les appels (max 40 req/10s)
- **Retry automatique** : 2 tentatives si erreur reseau temporaire
- **Gestion des erreurs** : 401 (cle invalide), 429 (trop de requetes), 500 (serveur en panne)

---

### 4. Verification au demarrage
`check_env()` verifie que tout est en ordre quand l app se lance :
- La cle TMDB est bien configuree
- Le dossier `data/` existe
- Les fichiers `u.data`, `u.item`, `u.user` sont presents
- Le fichier `.env` est bien dans `.gitignore`

---

### 5. Protection des secrets avec .env
Le fichier `.env.example` montre la structure attendue.
Il faut le copier en `.env` et y mettre sa vraie cle :

```bash
cp .env.example .env
# Puis editer .env :
TMDB_API_KEY=ta_vraie_cle_ici
```

Le fichier `.env` est dans `.gitignore` donc il ne sera JAMAIS envoye sur GitHub.

---

## Tableau recapitulatif

| Securite | Technique | Qui s en occupe |
|---|---|---|
| Chiffrement reseau | TLS/SSL (https://) | Automatique - Streamlit Cloud |
| Cle API cachee | Variables d environnement (.env) | src/security.py |
| Validation entrees | Sanitisation + whitelist | src/security.py |
| Controle requetes | Rate limiting + timeout + whitelist domaines | src/security.py |
| Verification demarrage | check_env() | src/security.py |

---

## Fichiers ajoutes dans cette branche

| Fichier | Role |
|---|---|
| `src/security.py` | Toutes les fonctions de securite applicative |
| `.env.example` | Template de configuration (sans vraie cle) |
| `.gitignore` | Liste des fichiers exclus de Git |
| `README_MAKO.md` | Ce fichier |

---

## Lancer le projet

```bash
cp .env.example .env
# Ajouter votre cle TMDB dans .env
streamlit run app.py
```
