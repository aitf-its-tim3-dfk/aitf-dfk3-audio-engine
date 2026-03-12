#!/usr/bin/env python3
"""
Audio scraper for Instagram, TikTok, Twitter/X, Facebook.
Given a list of URLs, downloads audio to a directory and writes a CSV report.

Usage:
    python scrape.py urls.txt --output ./dataset
    python scrape.py urls.txt --output ./dataset --format mp3 --cookies-from-browser chrome
    python scrape.py --urls "https://..." "https://..." --output ./dataset
"""

import argparse
import csv
import sys
import time
import yt_dlp
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_AUDIO_DIR = "audios"
DEFAULT_FORMAT = "wav"
DEFAULT_DELAY = 1.0

PLATFORM_MAP = {
    "instagram.com": "Instagram",
    "tiktok.com": "TikTok",
    "twitter.com": "Twitter/X",
    "x.com": "Twitter/X",
    "facebook.com": "Facebook",
    "fb.watch": "Facebook",
    "fb.com": "Facebook",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
}

CSV_FIELDS = ["url", "platform", "title", "uploader", "duration_sec", "filename", "status", "error", "scraped_at"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_platform(url):
    host = urlparse(url).netloc.lower().lstrip("www.")
    for domain, name in PLATFORM_MAP.items():
        if domain in host:
            return name
    return "Unknown"


def build_ydl_opts(audio_dir, audio_format, cookies_browser, cookies_file):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / "%(uploader)s - %(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": "192",
        }],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "ignoreerrors": False,
        "writeinfojson": False,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    if cookies_file:
        opts["cookiefile"] = cookies_file
    return opts


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── Core ──────────────────────────────────────────────────────────────────────

def scrape_url(url, ydl_opts):
    result = {
        "url": url,
        "platform": detect_platform(url),
        "title": "",
        "uploader": "",
        "duration_sec": "",
        "filename": "",
        "status": "",
        "error": "",
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
    }

    filenames = []

    def postprocessor_hook(d):
        if d.get("status") == "finished":
            filepath = d.get("info_dict", {}).get("filepath") or d.get("filename", "")
            if filepath:
                filenames.append(filepath)

    def download_hook(d):
        if d["status"] == "finished":
            filenames.append(d["filename"])

    opts = {**ydl_opts, "postprocessor_hooks": [postprocessor_hook], "progress_hooks": [download_hook]}

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                result["title"] = info.get("title", "")
                result["uploader"] = (
                    info.get("uploader")
                    or info.get("channel")
                    or info.get("creator")
                    or info.get("uploader_id")
                    or info.get("channel_id")
                    or ""
                )
                duration = info.get("duration") or info.get("duration_string")
                if isinstance(duration, float):
                    duration = int(duration)
                result["duration_sec"] = duration or ""
    except yt_dlp.utils.DownloadError as e:
        result["status"] = "failed"
        result["error"] = str(e).replace("\n", " ")
        return result
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"Unexpected error: {e}"
        return result

    if filenames:
        audio_ext = ydl_opts["postprocessors"][0]["preferredcodec"]
        p = Path(filenames[-1])
        final = p.with_suffix(f".{audio_ext}")
        result["filename"] = str(final) if final.exists() else str(p)
    else:
        result["filename"] = "(see output dir)"

    result["status"] = "ok"
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Scrape audio from social media URLs.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("url_file", nargs="?", help="Text file with one URL per line")
    src.add_argument("--urls", nargs="+", metavar="URL", help="One or more URLs directly")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help=f"Dataset directory (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help=f"Subdir name for audio files (default: {DEFAULT_AUDIO_DIR})")
    p.add_argument("--csv", default=None, help="CSV report path (default: <output>/results.csv)")
    p.add_argument("--format", "-f", default=DEFAULT_FORMAT, choices=["wav", "mp3", "m4a", "opus", "flac"], help=f"Audio format (default: {DEFAULT_FORMAT})")
    p.add_argument("--cookies-from-browser", metavar="BROWSER", help="Load cookies from browser (chrome, firefox, …)")
    p.add_argument("--cookies", metavar="FILE", help="Netscape-format cookies file")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Seconds between downloads (default: {DEFAULT_DELAY})")
    return p.parse_args()


def load_urls(args):
    if args.urls:
        return [u.strip() for u in args.urls if u.strip()]
    with open(args.url_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    args = parse_args()
    urls = load_urls(args)
    if not urls:
        print("No URLs found. Exiting.")
        sys.exit(1)

    output_dir = Path(args.output)
    audio_dir = output_dir / args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv else output_dir / "results.csv"
    ydl_opts = build_ydl_opts(audio_dir, args.format, args.cookies_from_browser, args.cookies)

    print(f"Scraping {len(urls)} URL(s)")
    print(f"Audio dir : {audio_dir}")
    print(f"Format    : {args.format}")
    print(f"CSV       : {csv_path}")
    print("-" * 50)

    results = []
    for i, url in enumerate(urls, 1):
        platform = detect_platform(url)
        print(f"[{i}/{len(urls)}] {platform:12s}  {url[:70]}")
        row = scrape_url(url, ydl_opts)
        results.append(row)

        if row["status"] == "ok":
            dur = f"  ({row['duration_sec']}s)" if row["duration_sec"] else ""
            print(f"           ✓  {row['title'][:60]}{dur}")
        else:
            print(f"           ✗  {row['error'][:80]}")

        write_csv(results, csv_path)

        if i < len(urls):
            time.sleep(args.delay)

    ok = sum(1 for r in results if r["status"] == "ok")
    print("-" * 50)
    print(f"Done. {ok}/{len(results)} succeeded.")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()