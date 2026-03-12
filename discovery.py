#!/usr/bin/env python3
"""
Discovers neutral Indonesian video URLs from TikTok, Instagram, Facebook, YouTube, Twitter/X.
Outputs a plain .txt file of URLs + a .csv with metadata.
Feed the .txt directly into scrape_audio.py.

Usage:
    python discovery.py --target 700 --output ./discovered
    python discovery.py --target 700 --platforms tiktok youtube --output ./discovered
    python discovery.py --target 700 --cookies-from-browser chrome --output ./discovered

Then:
    python scrape_audio.py discovered\\urls.txt --output ./dataset

Dependencies:
    pip install yt-dlp twikit
"""

import argparse
import asyncio
import csv
import logging
import random
import re
import time
import yt_dlp
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "./discovered"
DEFAULT_TARGET = 700
DEFAULT_PER_KEYWORD = 20
DEFAULT_DELAY = 2.0
DEFAULT_PLATFORMS = ["tiktok", "instagram", "facebook", "youtube"]

KEYWORDS = [
    # Food & cooking
    "resep masakan", "resep kue", "masak dirumah", "makanan enak",
    "kuliner indonesia", "street food indonesia", "minuman kekinian",
    "mukbang indonesia", "food review indonesia",
    # Daily life
    "vlog harian", "morning routine indonesia", "skincare routine",
    "outfit hari ini", "tips hemat", "dekorasi rumah", "DIY indonesia",
    # Travel & nature
    "wisata indonesia", "tempat wisata", "liburan",
    "pemandangan indah indonesia", "hiking indonesia",
    "pantai indonesia", "gunung indonesia",
    # Animals
    "kucing lucu", "anjing lucu", "hewan peliharaan indonesia",
    # Sports & fitness
    "olahraga pagi", "gym indonesia", "lari pagi",
    "highlights sepakbola indonesia", "futsal indonesia",
    # Entertainment
    "film indonesia terbaru", "review film", "unboxing indonesia",
    "konser indonesia", "stand up comedy indonesia",
    # Tech & education
    "tutorial indonesia", "belajar", "tips teknologi", "review gadget indonesia",
]

BLOCKLIST = [
    "pilkada", "pilpres", "capres", "cawapres", "partai", "korupsi",
    "bupati", "gubernur", "presiden", "dpr", "kpu", "pemilu", "kampanye",
    "kafir", "munafik", "pribumi", "yahudi", "khilafah", "syariat",
    "fitnah", "hoaks", "hoax", "bohong", "palsu", "tipu", "sesat",
    "hina", "bajingan", "bangsat", "babi", "laknat", "bunuh",
    "rasis", "diskriminasi", "pembunuhan", "pemerkosaan", "narkoba",
    "teroris", "bom", "israel", "palestina", "gaza",
]

CSV_FIELDS = ["id", "url", "platform", "keyword", "title", "author",
              "duration", "caption", "discovered_at", "label"]

BLOCKLIST_RE = re.compile(
    "|".join(r"\b" + re.escape(w) + r"\b" for w in BLOCKLIST),
    re.IGNORECASE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_neutral(text):
    if not text:
        return True
    return not BLOCKLIST_RE.search(text)


def build_query(platform, keyword, max_results):
    if platform == "tiktok":
        return f"tiktoksearch:{keyword}"
    elif platform == "instagram":
        hashtag = keyword.replace(" ", "")
        return f"https://www.instagram.com/explore/tags/{quote(hashtag)}/"
    elif platform == "facebook":
        return f"https://www.facebook.com/search/videos/?q={quote(keyword)}"
    elif platform == "youtube":
        return f"ytsearch{max_results}:{keyword} indonesia"
    return None


def load_existing(csv_path):
    if not csv_path.exists():
        return set(), set()
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ({r["id"] for r in rows if r.get("id")},
            {r["url"] for r in rows if r.get("url")})


def append_csv(rows, csv_path):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_urls(urls, url_path):
    existing = set()
    if url_path.exists():
        existing = set(url_path.read_text(encoding="utf-8").splitlines())
    new_urls = [u for u in urls if u not in existing]
    with open(url_path, "a", encoding="utf-8") as f:
        for u in new_urls:
            f.write(u + "\n")
    return len(new_urls)


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_platform(platform, keyword, max_results, cookies_browser, cookies_file):
    query = build_query(platform, keyword, max_results)
    if not query:
        return []

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max_results,
        "ignoreerrors": True,
        "skip_download": True,
    }
    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    results = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
        if not info:
            return results

        entries = info.get("entries") or [info]
        for entry in entries:
            if not entry:
                continue

            url = entry.get("webpage_url") or entry.get("url") or ""
            if not url or not url.startswith("http"):
                continue

            if entry.get("_type", "") in ("playlist", "channel"):
                continue

            caption = (
                entry.get("description")
                or entry.get("title")
                or entry.get("fulltitle")
                or ""
            ).strip()

            if not is_neutral(caption):
                log.debug(f"  blocked: {caption[:60]}")
                continue

            results.append({
                "id": f"{platform}_{entry.get('id', url)}",
                "url": url,
                "platform": platform,
                "keyword": keyword,
                "title": entry.get("title") or "",
                "author": (
                    entry.get("uploader")
                    or entry.get("channel")
                    or entry.get("creator")
                    or entry.get("uploader_id") or ""
                ),
                "duration": entry.get("duration") or "",
                "caption": caption[:300],
                "discovered_at": datetime.now().isoformat(timespec="seconds"),
                "label": "neutral",
            })
    except Exception as e:
        log.warning(f"  yt-dlp error [{platform}] '{keyword}': {e}")

    return results


async def discover_twitter(keyword, max_results, cookies_path):
    try:
        from twikit import Client
    except ImportError:
        log.warning("twikit not installed — skipping Twitter/X. Install: pip install twikit")
        return []

    if not cookies_path or not Path(cookies_path).exists():
        log.warning("No Twitter cookies — skipping Twitter/X. Generate with: python twitter_login.py")
        return []

    results = []
    try:
        client = Client(language="id-ID")
        client.load_cookies(cookies_path)
        tweets = await client.search_tweet(
            f"{keyword} lang:id -is:retweet has:videos",
            product="Latest",
            count=max_results,
        )
        for tweet in tweets:
            text = (tweet.full_text or tweet.text or "").strip()
            if not is_neutral(text):
                continue
            has_video = any(
                m.get("type") in ("video", "animated_gif")
                for m in (getattr(tweet, "media", None) or [])
            )
            if not has_video:
                continue
            results.append({
                "id": f"twitter_{tweet.id}",
                "url": f"https://x.com/i/web/status/{tweet.id}",
                "platform": "twitter",
                "keyword": keyword,
                "title": text[:80],
                "author": getattr(tweet.user, "screen_name", ""),
                "duration": "",
                "caption": text[:300],
                "discovered_at": datetime.now().isoformat(timespec="seconds"),
                "label": "neutral",
            })
    except Exception as e:
        log.warning(f"  Twitter error '{keyword}': {e}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "discovered.csv"
    url_path = output_dir / "urls.txt"

    seen_ids, seen_urls = load_existing(csv_path)
    collected = len(seen_urls)

    keywords = KEYWORDS.copy()
    random.shuffle(keywords)

    log.info(f"Target    : {args.target} URLs")
    log.info(f"Platforms : {args.platforms}")
    log.info(f"CSV       : {csv_path}")
    log.info(f"URLs      : {url_path}\n")

    for keyword in keywords:
        if collected >= args.target:
            break

        log.info(f"[{collected}/{args.target}] '{keyword}'")
        batch = []

        for platform in [p for p in args.platforms if p != "twitter"]:
            rows = discover_platform(
                platform=platform,
                keyword=keyword,
                max_results=args.per_keyword,
                cookies_browser=args.cookies_from_browser,
                cookies_file=args.cookies,
            )
            batch.extend(rows)
            time.sleep(args.delay)

        if "twitter" in args.platforms:
            rows = await discover_twitter(keyword, args.per_keyword, args.twitter_cookies)
            batch.extend(rows)
            time.sleep(args.delay)

        new_rows, new_urls = [], []
        for row in batch:
            if collected >= args.target:
                break
            if row["id"] in seen_ids or row["url"] in seen_urls:
                continue
            seen_ids.add(row["id"])
            seen_urls.add(row["url"])
            new_rows.append(row)
            new_urls.append(row["url"])
            collected += 1
            log.info(f"  ✓ [{collected}/{args.target}] {row['platform']:10s} | {row['title'][:60]}")

        if new_rows:
            append_csv(new_rows, csv_path)
            added = write_urls(new_urls, url_path)
            log.info(f"  → +{added} URLs written")

    log.info(f"\n{'='*50}")
    log.info(f"Done. {collected} URLs saved.")
    log.info(f"  python scrape_audio.py {url_path} --output ./dataset")

    if collected < args.target:
        log.warning(f"Only reached {collected}/{args.target}. Try --cookies-from-browser chrome.")


def main():
    p = argparse.ArgumentParser(description="Discover neutral Indonesian video URLs.")
    p.add_argument("--target", type=int, default=DEFAULT_TARGET, help=f"URLs to collect (default: {DEFAULT_TARGET})")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--platforms", nargs="+", default=DEFAULT_PLATFORMS,
                   choices=["tiktok", "instagram", "facebook", "youtube", "twitter"],
                   help=f"Platforms to search (default: {' '.join(DEFAULT_PLATFORMS)})")
    p.add_argument("--per-keyword", type=int, default=DEFAULT_PER_KEYWORD, help=f"Results per keyword per platform (default: {DEFAULT_PER_KEYWORD})")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Seconds between requests (default: {DEFAULT_DELAY})")
    p.add_argument("--cookies-from-browser", metavar="BROWSER", help="Browser to load cookies from (chrome, firefox, …)")
    p.add_argument("--cookies", metavar="FILE", help="Netscape cookies file")
    p.add_argument("--twitter-cookies", metavar="FILE", help="twikit cookies JSON (generate with twitter_login.py)")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()