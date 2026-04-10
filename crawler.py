#!/usr/bin/env python3
"""
Discovers Indonesian video URLs by scraping news websites.
Uses article tags/categories to assign labels automatically.

Supported news sites:
    turnbackhoax.id — dedicated fact-check site
    kompas.com      — major news outlet
    detik.com       — major news outlet
    liputan6.com    — news outlet
    cnnindonesia.com — news outlet

Usage:
    python crawler.py --target 700
    python crawler.py --target 700 --sites turnbackhoax kompas
    python crawler.py --target 700 --labels disinformasi neutral

Outputs:
    data.csv — metadata per URL

Next step:
    python scraper.py data.csv --output ./dataset

Label mapping (auto-assigned from article tags):
    ujaran kebencian  — hate speech, SARA, intolerance articles
    fitnah            — defamation, false accusation articles
    disinformasi      — hoax, fact-check, misinfo articles
    neutral           — general news, lifestyle, sports, etc.
"""

import csv
import hashlib
import logging
import random
import re
import sys
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

import httpx


def setup_logging(log_file: str | None = None, level: int = logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root.handlers.clear()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return root


log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_FILE = "data.csv"
DEFAULT_TARGET = 100
DEFAULT_DELAY = 2.0
DEFAULT_CONCURRENCY = 8
MAX_PAGES_PER_TAG = 1

LABELS = ["ujaran kebencian", "fitnah", "disinformasi", "neutral"]

PAGINATION_QUERY = "query"  # ?page=N
PAGINATION_PATH = "path"  # /N
PAGINATION_BOTH = "both"  # try both styles

HEADERS = {
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


NEUTRAL_TOPICS = [
    "musik indonesia",
    "resep rumahan",
    "tips belajar",
    "olahraga pagi",
    "review gadget",
    "travel jakarta",
    "kuliner bandung",
    "film terbaru",
    "komedi lucu",
    "gaming mobile",
    "parenting anak",
    "kesehatan mental",
    "dekorasi rumah",
    "fashion hijab",
    "otomotif motor",
    "hewan lucu",
    "wisata jogja",
    "fotografi pemula",
    "produktif kerja",
    "berkebun rumah",
]

NEUTRAL_PLATFORM_SEARCHES = [
    ("youtube", "https://www.youtube.com/results?search_query={query}"),
    ("tiktok", "https://www.tiktok.com/search?q={query}"),
    ("instagram", "https://www.instagram.com/explore/search/keyword/?q={query}"),
    ("twitter", "https://x.com/search?q={query}&src=typed_query&f=live"),
    ("facebook", "https://www.facebook.com/search/videos?q={query}"),
]

TAG_TO_LABEL = {
    "disinformasi": [
        "hoax",
        "hoaks",
        "misinformasi",
        "disinformasi",
        "fakta",
        "cek fakta",
        "cekfakta",
        "klarifikasi",
        "fact check",
        "kabar bohong",
        "berita palsu",
        "informasi palsu",
        "klikbait",
        "salah kapih",
        "misleading",
    ],
    "fitnah": [
        "fitnah",
        "pencemaran nama baik",
        "defamasi",
        "tuduhan palsu",
        "adu domba",
        "provokasi",
        "black campaign",
        "kampanye hitam",
        "serangan pribadi",
        "character assassination",
    ],
    "ujaran kebencian": [
        "ujaran kebencian",
        "hate speech",
        "SARA",
        "rasisme",
        "intoleransi",
        "diskriminasi",
        "xenofobia",
        "bigotri",
        "ekstremisme",
        "radikalisasi",
        "perpecahan",
        "permusuhan",
        "kebencian",
    ],
}

CSV_FIELDS = [
    "id",
    "source_article",
    "url",
    "platform",
    "keyword",
    "discovered_at",
    "weak_label",
]

PLATFORM_MAP = {
    "instagram.com": "instagram",
    "tiktok.com": "tiktok",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "facebook.com": "facebook",
    "fb.watch": "facebook",
    "fb.com": "facebook",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
}


# ── Site Configuration ────────────────────────────────────────────────────────
# Each site defines: base_url, pagination type, article regex, tags, and label mapping
def _kompas_extract_tags(html: str) -> list[str]:
    return [
        m.group(1).replace("-", " ").replace("%20", " ")
        for m in re.finditer(r'href="https://[^"]*kompas\.com/tag/([^"]+)"', html)
    ]


def _detik_extract_tags(html: str) -> list[str]:
    return [
        m.group(1).replace("-", " ")
        for m in re.finditer(r'href="https://[^"]*detik\.com/tag/([^"]+)"', html)
    ]


def _liputan6_extract_tags(html: str) -> list[str]:
    return [
        m.group(1).replace("-", " ")
        for m in re.finditer(r'href="https://www\.liputan6\.com/tag/([^"]+)"', html)
    ]


def _cnnindonesia_extract_tags(html: str) -> list[str]:
    return [
        m.group(1).replace("-", " ")
        for m in re.finditer(r'href="https://www\.cnnindonesia\.com/tag/([^"]+)"', html)
    ]


def _cnnindonesia_skip(url: str) -> bool:
    return any(x in url for x in ["/tag/", "/search/", "/infografis/", "/foto/"])


def _turnbackhoax_skip(url: str) -> bool:
    return any(p in url for p in ["/tag/", "/category/", "/author/", "/page/"])


SITE_CONFIG = {
    "turnbackhoax": {
        "base_domain": "https://turnbackhoax.id",
        "pagination": PAGINATION_QUERY,  # Uses ?page=N style
        "article_regex": r'href="(https://turnbackhoax\.id/[^"#?]+)"',
        "extract_tags_fn": None,
        "skip_fn": _turnbackhoax_skip,
        "default_label": "disinformasi",
        "tag_pages": [
            ("/", "disinformasi"),
            ("/articles?category=all", "disinformasi"),
        ],
    },
    "kompas": {
        "base_domain": "https://www.kompas.com",
        "pagination": PAGINATION_BOTH,  # Kompas supports both ?page=N and /N styles
        "article_regex": r'href="(https://[^"]*kompas\.com/read/[^"]+)"',
        "extract_tags_fn": _kompas_extract_tags,
        "skip_fn": None,
        "default_label": "disinformasi",
        "tag_pages": [
            ("/cekfakta/hoaks-atau-fakta", "disinformasi"),
            ("/tag/hoaks", "disinformasi"),
            ("/tag/fitnah", "fitnah"),
            ("/tag/ujaran-kebencian", "ujaran kebencian"),
            ("/tag/sara", "ujaran kebencian"),
            ("/tag/olahraga", "neutral"),
            ("/tag/teknologi", "neutral"),
            ("/tag/kesehatan", "neutral"),
            ("/tag/hiburan", "neutral"),
            ("/tag/gaya-hidup", "neutral"),
        ],
    },
    "detik": {
        "base_domain": "https://www.detik.com",
        "pagination": PAGINATION_PATH,  # Uses /N style
        "article_regex": r'href="(https://[^"]*detik\.com/[^"]*(?:/d-|/berita)[^"]+)"',
        "extract_tags_fn": _detik_extract_tags,
        "skip_fn": None,
        "default_label": "disinformasi",
        "tag_pages": [
            ("/tag/hoax", "disinformasi"),
            ("/tag/hoaks", "disinformasi"),
            ("/tag/fakta", "disinformasi"),
            ("/tag/fitnah", "fitnah"),
            ("/tag/ujaran-kebencian", "ujaran kebencian"),
            ("/tag/olahraga", "neutral"),
            ("/tag/teknologi", "neutral"),
            ("/tag/health", "neutral"),
            ("/tag/entertainment", "neutral"),
            ("/tag/gaya-hidup", "neutral"),
        ],
    },
    "liputan6": {
        "base_domain": "https://www.liputan6.com",
        "pagination": PAGINATION_QUERY,  # Change to PAGINATION_PATH for /N style
        "article_regex": r'href="(https://www\.liputan6\.com/[a-z]+/read/[^"]+)"',
        "extract_tags_fn": _liputan6_extract_tags,
        "skip_fn": None,
        "default_label": "disinformasi",
        "tag_pages": [
            ("/tag/hoaks", "disinformasi"),
            ("/tag/fakta", "disinformasi"),
            ("/tag/fitnah", "fitnah"),
            ("/tag/ujaran-kebencian", "ujaran kebencian"),
            ("/tag/olahraga", "neutral"),
            ("/tag/teknologi", "neutral"),
            ("/tag/kesehatan", "neutral"),
            ("/tag/hiburan", "neutral"),
            ("/tag/bisnis", "neutral"),
        ],
    },
    "cnnindonesia": {
        "base_domain": "https://www.cnnindonesia.com",
        "pagination": PAGINATION_QUERY,  # Change to PAGINATION_PATH for /N style
        "article_regex": r'href="(https://www\.cnnindonesia\.com/[^"]+)"',
        "extract_tags_fn": _cnnindonesia_extract_tags,
        "skip_fn": _cnnindonesia_skip,
        "default_label": "disinformasi",
        "tag_pages": [
            ("/tag/hoaks", "disinformasi"),
            ("/tag/fakta", "disinformasi"),
            ("/tag/fitnah", "fitnah"),
            ("/tag/intoleransi", "ujaran kebencian"),
            ("/tag/olahraga", "neutral"),
            ("/tag/teknologi", "neutral"),
            ("/tag/gaya-hidup", "neutral"),
            ("/tag/hiburan", "neutral"),
        ],
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────


def make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def detect_content_platform(url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for domain, platform in PLATFORM_MAP.items():
        if domain in host:
            return platform
    return "unknown"


def normalize_tags(tags: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for tag in tags:
        normalized = re.sub(r"\s+", " ", (tag or "").strip()).strip(" -_,:/")
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(normalized)
    return cleaned


def build_keyword_from_tags(tags: list[str]) -> str:
    normalized_tags = normalize_tags(tags)
    if not normalized_tags:
        return ""
    return f"tag:{','.join(normalized_tags)}"[:120]


def build_seed_keyword(tag_path: str, fallback_label: str) -> str:
    match = re.search(r"/tag/([^/?#]+)", tag_path)
    if match:
        return build_keyword_from_tags([match.group(1).replace("-", " ")])
    if "category=" in tag_path:
        category = tag_path.split("category=", 1)[1].split("&", 1)[0]
        return f"category:{category}"[:120]
    if tag_path in {"/", ""}:
        return f"seed:{fallback_label}"[:120]
    slug = tag_path.strip("/").replace("/", ":").replace("-", " ")
    return f"seed:{slug or fallback_label}"[:120]


def build_search_keyword(topic: str, platform: str) -> str:
    return f"search:{platform}:{topic}"[:120]


def normalize_video_url(url: str) -> str:
    """Normalize supported social video URLs into stable canonical forms."""
    url = url.strip()

    yt = re.search(
        r"(?:youtube\.com/(?:embed/|watch\?v=)|m\.youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})",
        url,
    )
    if yt:
        return f"https://www.youtube.com/watch?v={yt.group(1)}"

    tiktok = re.search(
        r"(?:https?://)?(?:www\.)?tiktok\.com/[^\"'>\s]+/video/(\d+)",
        url,
    )
    if tiktok:
        full = tiktok.group(0)
        if not full.startswith("http"):
            full = "https://www." + full.lstrip("/")
        return full.split("?")[0]

    tiktok_short = re.search(
        r"(?:https?://)?(?:vm|vt)\.tiktok\.com/[^\"'>\s/]+/?",
        url,
    )
    if tiktok_short:
        full = tiktok_short.group(0)
        if not full.startswith("http"):
            full = "https://" + full.lstrip("/")
        return full.split("?")[0]

    fb_watch = re.search(r"(?:https?://)?(?:www\.)?fb\.watch/[^\"'>\s/]+/?", url)
    if fb_watch:
        full = fb_watch.group(0)
        if not full.startswith("http"):
            full = "https://" + full.lstrip("/")
        return full.split("?")[0]

    fb_video = re.search(
        r"(?:facebook\.com/[^\"'>\s]*(?:/videos/|watch\?v=)|facebook\.com/reel/)(\d+)",
        url,
    )
    if fb_video:
        return f"https://www.facebook.com/watch?v={fb_video.group(1)}"

    instagram = re.search(
        r"instagram\.com/(?P<kind>reel|p)/(?P<id>[A-Za-z0-9_-]+)",
        url,
    )
    if instagram:
        kind = instagram.group("kind")
        return f"https://www.instagram.com/{kind}/{instagram.group('id')}/"

    tweet = re.search(r"(?:twitter|x)\.com/\w+/status/(\d+)", url)
    if tweet:
        return f"https://x.com/i/web/status/{tweet.group(1)}"

    return url.split("?")[0]


def _extract_anchor_hrefs(html: str, base_url: str) -> list[str]:
    hrefs = []
    for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\']', html):
        href = m.group(1)
        if not href.startswith("http"):
            href = urljoin(base_url, href)
        hrefs.append(href)
    return hrefs


def _extract_video_urls_generic(html: str, base_url: str) -> set[str]:
    urls = set()

    # YouTube embeds
    for m in re.finditer(
        r"(?:youtube\.com/embed/|youtube\.com/watch\?v=|m\.youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})",
        html,
    ):
        urls.add(f"https://www.youtube.com/watch?v={m.group(1)}")

    # TikTok embeds
    for m in re.finditer(r'tiktok\.com/[^"\'>\s]+/video/(\d+)', html):
        urls.add(normalize_video_url(m.group(0)))
    for m in re.finditer(r'(?:vm|vt)\.tiktok\.com/[^"\'>\s/]+/?', html):
        urls.add(normalize_video_url(m.group(0)))

    # Facebook video embeds
    for m in re.finditer(r'facebook\.com/[^"\'>\s]*(?:/videos/|watch\?v=)(\d+)', html):
        urls.add(f"https://www.facebook.com/watch?v={m.group(1)}")
    for m in re.finditer(r"facebook\.com/reel/(\d+)", html):
        urls.add(f"https://www.facebook.com/watch?v={m.group(1)}")
    for m in re.finditer(r'fb\.watch/[^"\'>\s/]+/?', html):
        urls.add(normalize_video_url(m.group(0)))

    # Instagram embeds
    for m in re.finditer(r"instagram\.com/(?:reel|p)/[A-Za-z0-9_-]+", html):
        urls.add(normalize_video_url(m.group(0)))

    # Twitter/X embeds
    for m in re.finditer(r"(?:twitter|x)\.com/\w+/status/(\d+)", html):
        urls.add(f"https://x.com/i/web/status/{m.group(1)}")

    for href in _extract_anchor_hrefs(html, base_url):
        if re.search(
            r"(?:youtube\.com/(?:watch\?v=|embed/)|m\.youtube\.com/watch\?v=|youtu\.be/|"
            r"tiktok\.com/.+/video/|(?:vm|vt)\.tiktok\.com/|"
            r"(?:facebook\.com|fb\.com|fb\.watch).*(?:/videos/|watch\?v=|/reel/)|"
            r"instagram\.com/(?:reel|p)/|"
            r"(?:twitter|x)\.com/\w+/status/)",
            href,
        ):
            urls.add(normalize_video_url(href))

    return urls


def _extract_turnbackhoax_source_urls(html: str, base_url: str) -> set[str]:
    """Prefer claimed/source-post links over evidence/reference links."""
    blocks = []

    for pattern in (
        r"Narasi(.*?)(?:Penjelasan|Kesimpulan|Hasil Periksa fakta)",
        r"Salah Sumber:(.*?)(?:Referensi|Artikel terbaru|$)",
    ):
        match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            blocks.append(match.group(1))

    preferred_urls = set()
    for block in blocks:
        for href in _extract_anchor_hrefs(block, base_url):
            if re.search(
                r"(?:instagram\.com/(?:reel|p)/|"
                r"tiktok\.com/.+/video/|(?:vm|vt)\.tiktok\.com/|"
                r"(?:facebook\.com|fb\.com|fb\.watch).*(?:/videos/|watch\?v=|/reel/)|"
                r"(?:twitter|x)\.com/\w+/status/|"
                r"youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)",
                href,
            ):
                preferred_urls.add(normalize_video_url(href))

    return preferred_urls


def extract_video_urls_from_html(html: str, base_url: str) -> list[str]:
    """Extract all video URLs from HTML content (embeds and hyperlinks)."""
    if "turnbackhoax.id/articles/" in base_url:
        preferred_urls = _extract_turnbackhoax_source_urls(html, base_url)
        if preferred_urls:
            return list(preferred_urls)

    return list(_extract_video_urls_generic(html, base_url))


def classify_by_tags(tags: list[str]) -> str:
    tags_lower = [t.lower() for t in normalize_tags(tags)]
    for label, patterns in TAG_TO_LABEL.items():
        for tag in tags_lower:
            for pattern in patterns:
                if pattern in tag:
                    return label
    return "neutral"


def build_page_url(base_tag_url: str, pagination: str, page_num: int) -> str:
    if page_num == 1:
        return base_tag_url
    if pagination == PAGINATION_QUERY:
        separator = "&" if "?" in base_tag_url else "?"
        return f"{base_tag_url}{separator}page={page_num}"
    base = base_tag_url.rstrip("/")
    return f"{base}/{page_num}"


# ── HTTP Fetching ─────────────────────────────────────────────────────────────


async def fetch(client: httpx.AsyncClient, url: str, retries: int = 4) -> str | None:
    for attempt in range(retries):
        headers = {**HEADERS, "User-Agent": random.choice(USER_AGENTS)}
        try:
            resp = await client.get(
                url, headers=headers, follow_redirects=True, timeout=15
            )

            if resp.status_code == 429:
                wait = (2**attempt) * random.uniform(5, 10)  # exponential backoff
                log.warning(f"Rate limited on {url}. Waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 403:
                log.warning(f"403 Forbidden: {url}. Skipping.")
                return None

            if resp.status_code < 400:
                await asyncio.sleep(random.uniform(0.5, 2.0))
                return resp.text

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            wait = (2**attempt) * random.uniform(2, 5)
            log.debug(
                f"Attempt {attempt + 1} failed [{url}]: {e}. Retrying in {wait:.1f}s"
            )
            await asyncio.sleep(wait)

        except Exception as e:
            log.debug(f"Fetch failed [{url}]: {e}")
            return None

    log.warning(f"All {retries} attempts failed for {url}")
    return None


async def fetch_articles(
    client: httpx.AsyncClient,
    article_jobs: list[tuple[str, str, str]],
    extract_tags_fn=None,
    concurrency: int = 8,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def process_one(article_url: str, fallback_label: str, fallback_keyword: str):
        async with sem:
            html = await fetch(client, article_url)
            if not html:
                return
            tags = normalize_tags(extract_tags_fn(html) if extract_tags_fn else [])
            label = classify_by_tags(tags) if tags else fallback_label
            for vu in extract_video_urls_from_html(html, article_url):
                results.append(
                    {
                        "source_article": article_url,
                        "url": vu,
                        "tags": tags,
                        "label": label,
                        "fallback_keyword": fallback_keyword,
                    }
                )

    await asyncio.gather(
        *[
            process_one(article_url, fallback_label, fallback_keyword)
            for article_url, fallback_label, fallback_keyword in article_jobs
        ]
    )
    return results


async def fetch_search_results(
    client: httpx.AsyncClient,
    search_jobs: list[tuple[str, str, str]],
    concurrency: int = 8,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def process_one(search_url: str, fallback_label: str, fallback_keyword: str):
        async with sem:
            html = await fetch(client, search_url)
            if not html:
                return
            for vu in extract_video_urls_from_html(html, search_url):
                results.append(
                    {
                        "source_article": search_url,
                        "url": vu,
                        "tags": [],
                        "label": fallback_label,
                        "fallback_keyword": fallback_keyword,
                    }
                )

    await asyncio.gather(
        *[
            process_one(search_url, fallback_label, fallback_keyword)
            for search_url, fallback_label, fallback_keyword in search_jobs
        ]
    )
    return results


# ── Pagination Helpers ────────────────────────────────────────────────────────


async def collect_tag_articles_paginated(
    client: httpx.AsyncClient,
    base_tag_url: str,
    article_regex: str,
    default_label: str,
    seed_keyword: str,
    pagination: str,
    skip_fn=None,
    max_articles: int | None = None,
    max_pages: int = MAX_PAGES_PER_TAG,
) -> list[tuple[str, str, str]]:
    """Collect article URLs from a tag page using the configured pagination style."""
    article_jobs = []
    seen_urls = set()

    for page_num in range(1, max_pages + 1):
        if max_articles is not None and len(article_jobs) >= max_articles:
            break
        url = build_page_url(base_tag_url, pagination, page_num)
        html = await fetch(client, url)
        if not html:
            break

        found_on_page = 0
        for m in re.finditer(article_regex, html):
            a_url = m.group(1)
            if a_url not in seen_urls:
                if skip_fn is None or not skip_fn(a_url):
                    seen_urls.add(a_url)
                    article_jobs.append((a_url, default_label, seed_keyword))
                    found_on_page += 1
                    if max_articles is not None and len(article_jobs) >= max_articles:
                        break

        if found_on_page == 0:
            break
        await asyncio.sleep(random.uniform(0.2, 1.0))

    return article_jobs


# ── Unified Config-Driven Scraper ─────────────────────────────────────────────


async def scrape_site_from_config(
    client: httpx.AsyncClient,
    site_name: str,
    remaining_per_label: dict[str, int],
    concurrency: int = DEFAULT_CONCURRENCY,
    max_pages: int = MAX_PAGES_PER_TAG,
) -> list[dict]:
    """Scrape a site using its SITE_CONFIG entry. Pagination type is read from config."""
    config = SITE_CONFIG.get(site_name)
    if not config:
        log.warning(f"No config for site: {site_name}")
        return []
    if config.get("disabled_reason"):
        log.warning(f"[{site_name}] {config['disabled_reason']}")
        return []

    base_domain = config["base_domain"]
    pagination = config["pagination"]
    article_regex = config["article_regex"]
    extract_tags_fn = config.get("extract_tags_fn")
    skip_fn = config.get("skip_fn")
    tag_pages = config["tag_pages"]

    needed_labels = [
        label for label, remaining in remaining_per_label.items() if remaining > 0
    ]
    if not needed_labels:
        return []

    article_jobs = []
    for tag_path, tag_label in tag_pages:
        if tag_label not in needed_labels:
            continue
        remaining = remaining_per_label.get(tag_label, 0)
        max_articles = max(remaining * 3, 10)
        base_url = f"{base_domain}{tag_path}"
        seed_keyword = build_seed_keyword(tag_path, tag_label)

        if pagination in (PAGINATION_QUERY, PAGINATION_BOTH):
            log.info(
                f"  [{site_name}] collecting from {base_url} (pagination: ?page=N)"
            )
            jobs = await collect_tag_articles_paginated(
                client,
                base_url,
                article_regex,
                tag_label,
                seed_keyword,
                PAGINATION_QUERY,
                skip_fn=skip_fn,
                max_articles=max_articles,
                max_pages=max_pages,
            )
            article_jobs.extend(jobs)

        if pagination in (PAGINATION_PATH, PAGINATION_BOTH):
            log.info(f"  [{site_name}] collecting from {base_url} (pagination: /N)")
            jobs = await collect_tag_articles_paginated(
                client,
                base_url,
                article_regex,
                tag_label,
                seed_keyword,
                PAGINATION_PATH,
                skip_fn=skip_fn,
                max_articles=max_articles,
                max_pages=max_pages,
            )
            article_jobs.extend(jobs)

    seen = set()
    unique = []
    for url, label, keyword in article_jobs:
        if url not in seen:
            seen.add(url)
            unique.append((url, label, keyword))

    log.info(
        f"  [{site_name}] fetching {len(unique)} articles (concurrency={concurrency})"
    )
    results = await fetch_articles(
        client,
        unique,
        extract_tags_fn,
        concurrency,
    )
    return results


async def scrape_neutral_platform_searches(
    client: httpx.AsyncClient,
    remaining_per_label: dict[str, int],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    remaining = remaining_per_label.get("neutral", 0)
    if remaining <= 0:
        return []

    topics = list(NEUTRAL_TOPICS)
    random.shuffle(topics)
    search_jobs = []
    seen_search_urls = set()

    topics_needed = max(remaining * 2, 12)
    for topic in topics[:topics_needed]:
        encoded = quote_plus(topic)
        for platform, template in NEUTRAL_PLATFORM_SEARCHES:
            search_url = template.format(query=encoded)
            if search_url in seen_search_urls:
                continue
            seen_search_urls.add(search_url)
            search_jobs.append(
                (search_url, "neutral", build_search_keyword(topic, platform))
            )

    if not search_jobs:
        return []

    log.info(
        f"  [neutral-platforms] fetching {len(search_jobs)} direct platform search pages"
    )
    return await fetch_search_results(
        client,
        search_jobs,
        concurrency=min(concurrency, 6),
    )


# ── Site Scrapers (now config-driven) ─────────────────────────────────────────


async def scrape_turnbackhoax(
    client: httpx.AsyncClient,
    remaining_per_label: dict[str, int],
    concurrency: int = DEFAULT_CONCURRENCY,
    max_pages: int = MAX_PAGES_PER_TAG,
) -> list[dict]:
    return await scrape_site_from_config(
        client, "turnbackhoax", remaining_per_label, concurrency, max_pages
    )


async def scrape_kompas(
    client: httpx.AsyncClient,
    remaining_per_label: dict[str, int],
    concurrency: int = DEFAULT_CONCURRENCY,
    max_pages: int = MAX_PAGES_PER_TAG,
) -> list[dict]:
    return await scrape_site_from_config(
        client, "kompas", remaining_per_label, concurrency, max_pages
    )


async def scrape_detik(
    client: httpx.AsyncClient,
    remaining_per_label: dict[str, int],
    concurrency: int = DEFAULT_CONCURRENCY,
    max_pages: int = MAX_PAGES_PER_TAG,
) -> list[dict]:
    return await scrape_site_from_config(
        client, "detik", remaining_per_label, concurrency, max_pages
    )


async def scrape_liputan6(
    client: httpx.AsyncClient,
    remaining_per_label: dict[str, int],
    concurrency: int = DEFAULT_CONCURRENCY,
    max_pages: int = MAX_PAGES_PER_TAG,
) -> list[dict]:
    return await scrape_site_from_config(
        client, "liputan6", remaining_per_label, concurrency, max_pages
    )


async def scrape_cnnindonesia(
    client: httpx.AsyncClient,
    remaining_per_label: dict[str, int],
    concurrency: int = DEFAULT_CONCURRENCY,
    max_pages: int = MAX_PAGES_PER_TAG,
) -> list[dict]:
    return await scrape_site_from_config(
        client, "cnnindonesia", remaining_per_label, concurrency, max_pages
    )


# ── Persistence ───────────────────────────────────────────────────────────────


def load_existing(csv_path: Path) -> tuple[set[str], dict[str, int]]:
    if not csv_path.exists():
        return set(), {label: 0 for label in LABELS}
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen_urls = {r["url"] for r in rows if r.get("url")}
    per_label = {label: 0 for label in LABELS}
    for r in rows:
        lbl = r.get("weak_label") or r.get("label", "")
        if lbl in per_label:
            per_label[lbl] += 1
    return seen_urls, per_label


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def remaining_label_targets(
    per_label: dict[str, int], labels: list[str], target: int
) -> dict[str, int]:
    return {label: max(target - per_label.get(label, 0), 0) for label in labels}


def save_results(
    results: list[dict],
    csv_path: Path,
    seen_urls: set,
    target: int,
    active_labels: list[str],
) -> dict:
    per_label = {label: 0 for label in LABELS}
    _, existing = load_existing(csv_path)
    per_label.update(existing)

    new_rows = []
    for r in results:
        url = r["url"]
        if url in seen_urls:
            continue
        label = r.get("label", "neutral")
        if label not in active_labels:
            continue
        if per_label.get(label, 0) >= target:
            continue
        seen_urls.add(url)
        new_rows.append(
            {
                "id": make_id(url),
                "source_article": r.get("source_article", ""),
                "url": url,
                "platform": detect_content_platform(url),
                "keyword": build_keyword_from_tags(r.get("tags", []))
                or r.get("fallback_keyword", ""),
                "discovered_at": datetime.now().isoformat(timespec="seconds"),
                "weak_label": label,
            }
        )
        per_label[label] += 1

    append_rows(csv_path, new_rows)
    return per_label


# ── Main ──────────────────────────────────────────────────────────────────────

SITE_SCRAPERS = {
    "turnbackhoax": scrape_turnbackhoax,
    "kompas": scrape_kompas,
    "detik": scrape_detik,
    "liputan6": scrape_liputan6,
    "cnnindonesia": scrape_cnnindonesia,
}


async def run(args):
    csv_path = Path(args.output)

    seen_urls, per_label = load_existing(csv_path)
    log.info(f"Loaded existing: {sum(per_label.values())} total URLs")
    log.info(f"Target  : {args.target} per label")
    log.info(f"Sites   : {args.sites}")
    log.info(f"Labels  : {args.labels}")
    log.info(f"Concurrency: {args.concurrency}\n")

    async with httpx.AsyncClient(headers=HEADERS) as client:
        for site_name in args.sites:
            if all(per_label[label] >= args.target for label in args.labels):
                log.info("All requested labels have reached target. Stopping early.")
                break

            remaining_per_label = remaining_label_targets(
                per_label, args.labels, args.target
            )

            scraper_fn = SITE_SCRAPERS.get(site_name)
            if not scraper_fn:
                log.warning(f"Unknown site: {site_name}")
                continue

            log.info(f"{'=' * 55}")
            log.info(f"Scraping: {site_name}")
            log.info(f"{'=' * 55}")

            site_results = await scraper_fn(
                client, remaining_per_label, args.concurrency, args.max_pages
            )
            if remaining_per_label.get("neutral", 0) > 0:
                neutral_results = await scrape_neutral_platform_searches(
                    client,
                    remaining_per_label,
                    args.concurrency,
                )
                if neutral_results:
                    log.info(
                        f"  Added {len(neutral_results)} candidate URLs from direct platform searches"
                    )
                    site_results.extend(neutral_results)
            site_results = [r for r in site_results if r.get("label") in args.labels]
            log.info(f"  Found {len(site_results)} video URLs from {site_name}")

            per_label = save_results(
                site_results,
                csv_path,
                seen_urls,
                args.target,
                args.labels,
            )
            for lbl in args.labels:
                log.info(f"  {lbl}: {per_label[lbl]}/{args.target}")
            await asyncio.sleep(args.delay)

    _, final_counts = load_existing(csv_path)
    total = sum(final_counts.values())
    log.info(f"\n{'=' * 55}")
    log.info("Done. Summary:")
    for lbl in args.labels:
        count = final_counts[lbl]
        status = "OK" if count >= args.target else f"SHORT by {args.target - count}"
        log.info(f"  {lbl:<22} {count:>4}/{args.target}  {status}")
    log.info(f"  {'TOTAL':<22} {total:>4}")
    log.info(f"\nNext step:")
    log.info(f"  python scraper.py data.csv --output ./dataset")


def main():
    p = argparse.ArgumentParser(
        description="Discover video URLs from Indonesian news sites."
    )
    p.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"URLs to collect per label (default: {DEFAULT_TARGET})",
    )
    p.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file (default: {DEFAULT_OUTPUT_FILE})",
    )
    p.add_argument(
        "--sites",
        nargs="+",
        default=["turnbackhoax", "kompas", "detik", "liputan6", "cnnindonesia"],
        choices=list(SITE_SCRAPERS.keys()),
        help="News sites to scrape",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=LABELS,
        choices=LABELS,
        help="Labels to collect (default: all 4)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between site scrapes (default: {DEFAULT_DELAY})",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel article fetches per site (default: {DEFAULT_CONCURRENCY})",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES_PER_TAG,
        help=f"Max pagination pages per tag (default: {MAX_PAGES_PER_TAG})",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional file to write logs to (in addition to stdout)",
    )
    args = p.parse_args()
    setup_logging(args.log_file)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
