\
import re, json, time, hashlib, urllib.parse, datetime, queue, threading
import requests
import tldextract
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

HEADERS = {
    "User-Agent": "Ripley-Logistics-Ingestor/1.0 (+contact: ops@example.com)"
}
TIMEOUT = 15

def is_allowed(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return False

def fetch(url: str) -> str | None:
    if not is_allowed(url):
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.text
        return None
    except Exception:
        return None

def find_links(html: str, base_url: str, same_domain_only: bool = True) -> list[str]:
    """Extract links from *html* relative to *base_url*.

    Parameters
    ----------
    html: str
        Raw HTML contents to scan.
    base_url: str
        URL used to resolve relative links.
    same_domain_only: bool
        When True (default) only returns links that share the
        same registrable domain as ``base_url``. When False, links from
        any domain are returned.

    Returns
    -------
    list[str]
        Unique, ordered list of discovered URLs.
    """
    out = []
    if not html:
        return out
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(base_url, a["href"])
        if same_domain_only:
            if same_domain(base_url, href):
                out.append(href.split("#")[0])
        else:
            out.append(href.split("#")[0])
    return list(dict.fromkeys(out))  # unique, preserve order

def same_domain(a: str, b: str) -> bool:
    ea, eb = tldextract.extract(a), tldextract.extract(b)
    return (ea.domain, ea.suffix) == (eb.domain, eb.suffix)


def search_duckduckgo(query: str, max_results: int = 10) -> list[str]:
    """Return a list of result URLs from DuckDuckGo for *query*.

    This uses the public HTML endpoint which does not require an API key.
    Results are unverified and may include non-product pages.
    """
    try:
        params = {"q": query, "t": "hj", "ia": "web"}
        r = requests.get("https://duckduckgo.com/html/", params=params, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        urls = []
        for a in soup.select("a.result__a", limit=max_results):
            href = a.get("href")
            if href:
                urls.append(href)
        return urls
    except Exception:
        return []

def parse_json_ld_product(soup: BeautifulSoup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string)
            candidates = data if isinstance(data, list) else [data]
            for obj in candidates:
                if isinstance(obj, dict) and obj.get("@type") in ["Product", "schema:Product"]:
                    return obj
        except Exception:
            continue
    return None

def extract_number(text):
    if not text: return None
    import re
    t = str(text).lower().replace(",", ".")
    m = re.search(r"([\d]+(?:\.\d+)?)", t)
    return float(m.group(1)) if m else None

def parse_dimensions_from_text(txt):
    if not txt: return (None, None, None)
    t = txt.lower().replace(",", ".")
    m = re.search(r"([\d\.]+)\s*[x×]\s*([\d\.]+)\s*[x×]\s*([\d\.]+)\s*cm", t)
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return (None, None, None)

def extract_weight_kg_from_text(txt):
    if not txt: return None
    t = txt.lower().replace(",", ".")
    m = re.search(r"([\d\.]+)\s*(kg|kilo)", t)
    if m:
        val = float(m.group(1))
        if re.search(r"(carga|capacidad|lavadora|tambor|freezer|refrigerador|litros tambor)", t):
            return None
        return val
    m = re.search(r"([\d\.]+)\s*(g|gram)", t)
    if m:
        return float(m.group(1))/1000.0
    m = re.search(r"([\d\.]+)\s*(ml|l|litro)", t)
    if m:
        val = float(m.group(1))
        return val/1000.0 if m.group(2)=="ml" else val
    return None

def volumetric_kg(L,W,H, divisor=5000):
    try:
        return round((float(L)*float(W)*float(H))/divisor, 3)
    except:
        return None

def classify_by_thresholds(kg, thresholds):
    if kg is None: return ""
    selected = ""
    for row in sorted(thresholds, key=lambda r: float(r["kg_min"])):
        if kg >= float(row["kg_min"]):
            selected = row["clase"]
    return selected

def row_hash(d: dict) -> str:
    key = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def parse_product_page(html: str, url: str, thresholds, divisor_vol=5000):
    soup = BeautifulSoup(html, "lxml")
    jd = parse_json_ld_product(soup)
    name = brand = None
    peso_kg = None
    L = W = H = None

    if jd:
        name = jd.get("name")
        brand = jd.get("brand", {}).get("name") if isinstance(jd.get("brand"), dict) else jd.get("brand")
        if isinstance(jd.get("weight"), dict):
            peso_kg = extract_number(jd["weight"].get("value"))
            unit = (jd["weight"].get("unitCode") or "").lower()
            if unit in ["grm", "g"] and peso_kg is not None:
                peso_kg = peso_kg/1000.0
        elif isinstance(jd.get("weight"), str):
            peso_kg = extract_weight_kg_from_text(jd.get("weight"))
        props = jd.get("additionalProperty")
        if isinstance(props, list):
            for p in props:
                n = (p.get("name") or "").lower()
                v = p.get("value")
                if "peso" in n:
                    val = extract_weight_kg_from_text(str(v))
                    if val: peso_kg = val
                if ("alto" in n) or ("altura" in n) or ("height" in n):
                    H = extract_number(v)
                if ("ancho" in n) or ("width" in n):
                    W = extract_number(v)
                if ("largo" in n) or ("profundidad" in n) or ("depth" in n):
                    L = extract_number(v)

    if not name:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            name = og_title["content"]
    if not name and soup.title and soup.title.string:
        name = soup.title.string

    if not peso_kg:
        peso_kg = extract_weight_kg_from_text(name)

    if not (L and W and H):
        body_text = soup.get_text(separator=" ", strip=True)[:2000]
        l2, w2, h2 = parse_dimensions_from_text(body_text)
        L = L or l2; W = W or w2; H = H or h2

    vol_kg = volumetric_kg(L, W, H, divisor_vol)
    fact_kg = None if (peso_kg is None and vol_kg is None) else (vol_kg if peso_kg is None else (peso_kg if vol_kg is None else max(peso_kg, vol_kg)))
    clase = classify_by_thresholds(fact_kg, thresholds)

    now = datetime.datetime.utcnow().isoformat()
    row = {
        "product_name": name or "",
        "brand": brand or "",
        "peso_kg": peso_kg,
        "largo_cm": L, "ancho_cm": W, "alto_cm": H,
        "peso_vol_kg": vol_kg, "peso_fact_kg": fact_kg,
        "clase_logistica": clase,
        "source_url": url, "fetched_at": now
    }
    row["hash_row"] = row_hash(row)
    return row

def crawl_domain(start_url: str, thresholds, divisor_vol=5000, max_pages=25, delay=1.0):
    visited = set()
    q = queue.Queue()
    q.put(start_url)
    results = []

    while not q.empty() and len(visited) < max_pages:
        url = q.get()
        if url in visited: 
            continue
        visited.add(url)
        html = fetch(url)
        if not html:
            continue
        # Si la página tiene Product JSON-LD, parsea
        soup = BeautifulSoup(html, "lxml")
        if parse_json_ld_product(soup):
            try:
                row = parse_product_page(html, url, thresholds, divisor_vol)
                results.append(row)
            except Exception:
                pass
        # Expande enlaces del mismo dominio
        for link in find_links(html, url):
            if link not in visited and same_domain(start_url, link):
                q.put(link)
        time.sleep(delay)
    return results


def crawl_web(query: str, thresholds, divisor_vol=5000, max_pages=25, delay=1.0):
    """Crawl the web broadly starting from search results for *query*.

    The crawler uses DuckDuckGo to obtain seed URLs and then follows links
    without restricting to the same domain. The number of fetched pages is
    limited by *max_pages* to avoid unbounded crawling.
    """
    seeds = search_duckduckgo(query, max_results=max_pages)
    visited = set()
    q = queue.Queue()
    for url in seeds:
        q.put(url)
    results = []

    while not q.empty() and len(visited) < max_pages:
        url = q.get()
        if url in visited:
            continue
        visited.add(url)
        html = fetch(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        if parse_json_ld_product(soup):
            try:
                row = parse_product_page(html, url, thresholds, divisor_vol)
                results.append(row)
            except Exception:
                pass
        for link in find_links(html, url, same_domain_only=False):
            if link not in visited:
                q.put(link)
        time.sleep(delay)
    return results
