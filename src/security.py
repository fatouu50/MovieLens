import os, re, time, logging, requests

logging.basicConfig(level=logging.WARNING, format="[MovieLens Security] %(levelname)s - %(message)s")
logger = logging.getLogger("movielens.security")

def get_tmdb_api_key():
    try:
        from google.colab import userdata
        key = userdata.get("TMDB_API_KEY")
        if key and key.strip(): return key.strip()
    except Exception: pass
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if key: return key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("TMDB_API_KEY", "").strip()
        if key: return key
    except ImportError: pass
    raise EnvironmentError("Cle TMDB introuvable. Ajoutez TMDB_API_KEY dans .env")

def check_env():
    errors, warnings = [], []
    try: get_tmdb_api_key()
    except EnvironmentError as e: errors.append(str(e))
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.isdir(data_dir): errors.append(f"Dossier data/ introuvable")
    for f in ['u.data', 'u.item', 'u.user']:
        if not os.path.isfile(os.path.join(data_dir, f)): errors.append(f"Fichier manquant : data/{f}")
    return {'ok': len(errors)==0, 'errors': errors, 'warnings': warnings}

class RateLimiter:
    def __init__(self, min_interval=0.1): self._min_interval=min_interval; self._last_call=0.0
    def wait(self):
        elapsed = time.time()-self._last_call
        if elapsed < self._min_interval: time.sleep(self._min_interval-elapsed)
        self._last_call = time.time()

_tmdb_rate_limiter = RateLimiter(min_interval=0.1)
ALLOWED_TMDB_DOMAINS = {"api.themoviedb.org"}

def _is_allowed_url(url):
    from urllib.parse import urlparse
    return urlparse(url).netloc in ALLOWED_TMDB_DOMAINS

def safe_tmdb_request(url, params, timeout=5, retries=2):
    if not _is_allowed_url(url): return None
    for attempt in range(retries+1):
        try:
            _tmdb_rate_limiter.wait()
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout: pass
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 429: time.sleep(2)
            elif status < 500: return None
        except: return None
        if attempt < retries: time.sleep(0.5*(attempt+1))
    return None

_VALID_GENRES = {'Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'}

def validate_film_id(film_id):
    try:
        fid = int(film_id)
        return fid if 1 <= fid <= 1682 else None
    except: return None

def validate_genre(genre):
    return genre if isinstance(genre, str) and genre in _VALID_GENRES else None

def sanitize_search_query(query):
    if not query or not isinstance(query, str): return ""
    query = query[:100]
    query = re.sub(r'<[^>]+>', '', query)
    query = re.sub(r'[<>"\'\\]', '', query)
    return query.strip()

def sanitize_html(text):
    if not isinstance(text, str): return ""
    return text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;").replace("'","&#39;")

def display_security_report(st_module):
    report = check_env()
    for err in report['errors']: st_module.error(err)
    for warn in report['warnings']: st_module.warning(warn)
    if not report['ok']: st_module.stop()
