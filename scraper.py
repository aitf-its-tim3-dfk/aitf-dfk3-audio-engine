#!/usr/bin/env python3
"""
Download audio for social media URLs and write a CSV report.

Supports:
- plain text input with one URL per line
- CSV input, including current crawler.py output
- direct URLs via --urls
"""

import argparse
import csv
import hashlib
import multiprocessing
import random
import re
import signal
import sys
import time
import yt_dlp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from multiprocessing import Event as MPEvent
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_AUDIO_DIR = "raw"
DEFAULT_FORMAT = "wav"
DEFAULT_DELAY = 0.5
DEFAULT_WORKERS = 1

OUTPUT_FIELDS = [
    "url",
    "platform",
    "title",
    "uploader",
    "duration_sec",
    "filename",
    "status",
    "error",
    "scraped_at",
    "resolved_url",
    "weak_label",
    "source_article",
    "keyword",
]

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

PLATFORM_COOKIES = {
    "Instagram": "cookies/instagram_cookies.txt",
    "Twitter/X": "cookies/twitter_cookies.txt",
    "Facebook": "cookies/facebook_cookies.txt",
    "TikTok": "cookies/tiktok_cookies.txt",
    "YouTube": None,
}


stop_event = MPEvent()
active_executor: ProcessPoolExecutor | None = None


def detect_platform(url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for domain, name in PLATFORM_MAP.items():
        if domain in host:
            return name
    return "Unknown"


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "").strip()


def interruptible_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    if stop_event.wait(seconds):
        raise KeyboardInterrupt


def unique_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:10]


def build_filename(info: dict, url: str, audio_dir: Path, ext: str) -> Path:
    video_id = info.get("id") or unique_id(url)
    candidate = audio_dir / f"{video_id}.{ext}"
    if not candidate.exists():
        return candidate

    suffix = int(time.time() * 1000)
    while True:
        candidate = audio_dir / f"{video_id}-{suffix}.{ext}"
        if not candidate.exists():
            return candidate
        suffix += 1


def load_proxies_from_file(path: Path) -> list[str]:
    proxies = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            proxies.append(line)
    return proxies


def resolve_proxy_arg(proxy_arg: str | None) -> str | None:
    if not proxy_arg:
        return None
    proxy_path = Path(proxy_arg)
    if proxy_path.exists() and proxy_path.is_file():
        proxies = load_proxies_from_file(proxy_path)
        if not proxies:
            raise ValueError(f"No usable proxies found in file: {proxy_path}")
        return proxies[0]
    return proxy_arg


def build_ydl_opts(
    audio_dir: Path,
    audio_format: str,
    cookies_browser: str | None,
    cookies_file: str | None,
    proxy: str | None = None,
    platform: str | None = None,
) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "ignoreerrors": False,
        "writeinfojson": False,
        "nopart": True,
        "sleep_interval": 2,
        "max_sleep_interval": 6,
        "sleep_interval_requests": 1,
    }

    if platform and not cookies_file and not cookies_browser:
        platform_cookie = PLATFORM_COOKIES.get(platform)
        if platform_cookie and Path(platform_cookie).exists():
            opts["cookiefile"] = platform_cookie

    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    if proxy:
        opts["proxy"] = proxy
    return opts


def without_browser_cookies(ydl_opts: dict) -> dict:
    fallback_opts = dict(ydl_opts)
    fallback_opts.pop("cookiesfrombrowser", None)
    return fallback_opts


def is_valid_video_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    if not parsed.scheme or not parsed.netloc:
        return False
    if "login" in path:
        return False
    if host == "facebook.com" and "/videos/" not in path and "watch" in path:
        return bool(re.search(r"[?&]v=\d+", parsed.query))
    return True


def normalize_input_row(row: dict) -> dict:
    normalized = {
        k: (v or "").strip() if isinstance(v, str) else v for k, v in row.items()
    }
    normalized["url"] = normalized.get("url", "").strip()
    normalized["weak_label"] = normalized.get("weak_label") or normalized.get(
        "label", ""
    )
    normalized["source_article"] = normalized.get("source_article", "")
    normalized["keyword"] = normalized.get("keyword", "")
    return normalized


def load_input_items(args) -> list[dict]:
    if args.urls:
        return [
            {"url": url.strip(), "weak_label": "", "source_article": "", "keyword": ""}
            for url in args.urls
            if url.strip()
        ]

    input_path = Path(args.url_file)
    if input_path.suffix.lower() == ".csv":
        items = []
        with open(input_path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item = normalize_input_row(row)
                if (
                    item["url"]
                    and not item["url"].startswith("#")
                    and is_valid_video_url(item["url"])
                ):
                    items.append(item)
        return items

    with open(input_path, encoding="utf-8") as handle:
        return [
            {"url": line.strip(), "weak_label": "", "source_article": "", "keyword": ""}
            for line in handle
            if line.strip()
            and not line.startswith("#")
            and is_valid_video_url(line.strip())
        ]


def load_existing_rows(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8") as handle:
        return {row["url"]: row for row in csv.DictReader(handle) if row.get("url")}


def build_output_row(item: dict, result: dict) -> dict:
    row = {field: "" for field in OUTPUT_FIELDS}
    row.update(item)
    row.update(result)
    row["url"] = item["url"]
    row["platform"] = detect_platform(item["url"])
    row["weak_label"] = item.get("weak_label", "")
    row["source_article"] = item.get("source_article", "")
    row["keyword"] = item.get("keyword", "")
    row["resolved_url"] = item.get("resolved_url", item["url"])
    row["scraped_at"] = datetime.now().isoformat(timespec="seconds")
    return row


def write_results_csv(
    csv_path: Path, rows_by_url: dict[str, dict], input_order: list[dict]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        rows_by_url[item["url"]] for item in input_order if item["url"] in rows_by_url
    ]

    extra_fields = []
    for row in rows:
        for key in row:
            if key not in OUTPUT_FIELDS and key not in extra_fields:
                extra_fields.append(key)

    fieldnames = OUTPUT_FIELDS + sorted(extra_fields)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scrape_url(url: str, ydl_opts: dict, audio_dir: Path, retries: int = 3) -> dict:
    result = {"url": url, "status": "", "error": "", "filename": ""}
    current_ydl_opts = dict(ydl_opts)
    for attempt in range(retries):
        try:
            with yt_dlp.YoutubeDL(current_ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    result["status"] = "skipped"
                    return result

                ext = current_ydl_opts["postprocessors"][0]["preferredcodec"]
                final_path = build_filename(info, url, audio_dir, ext)
                downloaded = Path(ydl.prepare_filename(info))
                converted = downloaded.with_suffix(f".{ext}")
                source_path = converted if converted.exists() else downloaded
                if source_path.exists():
                    source_path.rename(final_path)

                result.update(
                    {
                        "status": "ok",
                        "filename": str(final_path),
                        "title": info.get("title", ""),
                        "uploader": info.get("uploader", ""),
                        "duration_sec": info.get("duration", ""),
                    }
                )
                return result
        except Exception as exc:
            error_text = strip_ansi(str(exc))
            lowered = error_text.lower()
            result["error"] = error_text

            if "dpapi" in lowered and "cookiesfrombrowser" in current_ydl_opts:
                print(
                    "  [cookies] Browser-cookie decryption failed on Windows; retrying without --cookies-from-browser"
                )
                current_ydl_opts = without_browser_cookies(current_ydl_opts)
                continue

            if any(
                token in lowered
                for token in [
                    "no video",
                    "not a video",
                    "no media",
                    "there is no video in this post",
                    "no video could be found in this tweet",
                    "media #1 is not a video",
                    "unsupported url: https://web.archive.org/",
                    "removed for violating",
                    "removed by the uploader",
                    "terms of service",
                    "empty media response",
                    "unable to load",
                    "cannot parse data",
                    "request range not satisfiable",
                    "range not satisfiable",
                ]
            ):
                result["status"] = "skipped"
                return result
            if any(
                token in lowered
                for token in [
                    "confirm you",
                    "sign in",
                    "login",
                    "private",
                    "registered users",
                    "authentication",
                    "permission to view this post",
                ]
            ):
                result["status"] = "needs-login"
                return result

            if stop_event.is_set():
                raise KeyboardInterrupt

            wait = (2**attempt) * random.uniform(3, 8)
            if any(
                token in lowered
                for token in [
                    "403",
                    "429",
                    "rate",
                    "blocked",
                    "forbidden",
                    "ip address is blocked",
                    "too many requests",
                ]
            ):
                wait = (2**attempt) * random.uniform(10, 20)
                print(
                    f"  [blocked] {url} - waiting {wait:.0f}s before retry {attempt + 1}/{retries}"
                )
            else:
                print(f"  [error] attempt {attempt + 1}/{retries}: {error_text[:80]}")
            interruptible_sleep(wait)

    result["status"] = "failed"
    return result


def scrape_url_worker(
    url: str, ydl_opts: dict, audio_dir_str: str, retries: int = 3
) -> dict:
    return scrape_url(url, ydl_opts, Path(audio_dir_str), retries=retries)


def terminate_executor_processes(executor: ProcessPoolExecutor | None) -> None:
    if executor is None:
        return

    processes = getattr(executor, "_processes", None) or {}
    for process in processes.values():
        if process is None or not process.is_alive():
            continue
        process.terminate()

    deadline = time.time() + 2
    for process in processes.values():
        if process is None:
            continue
        remaining = max(0, deadline - time.time())
        process.join(timeout=remaining)
        if process.is_alive():
            process.kill()


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape audio from social media URLs.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("url_file", nargs="?", help="File with URLs (txt or csv)")
    src.add_argument(
        "--urls", nargs="+", metavar="URL", help="One or more URLs directly"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dataset directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
        help=f"Subdir name for audio files (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--csv", default=None, help="CSV report path (default: <output>/metadata.csv)"
    )
    parser.add_argument(
        "--format",
        "-f",
        default=DEFAULT_FORMAT,
        choices=["wav", "mp3", "m4a", "opus", "flac"],
        help=f"Audio format (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Load cookies from browser (chrome, firefox, edge)",
    )
    parser.add_argument(
        "--cookies", metavar="FILE", help="Netscape-format cookies file"
    )
    parser.add_argument(
        "--proxy",
        metavar="URL",
        default=None,
        help="Proxy URL or a file containing proxy entries",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between downloads (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent workers (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args()


def main():
    def handle_sigint(sig, frame):
        stop_event.set()
        if active_executor is not None:
            terminate_executor_processes(active_executor)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    args = parse_args()
    if not args.url_file and not args.urls:
        args.url_file = "data.csv"

    try:
        args.proxy = resolve_proxy_arg(args.proxy)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    items = load_input_items(args)
    if not items:
        print("No URLs found. Exiting.")
        sys.exit(1)

    output_dir = Path(args.output)
    audio_dir = output_dir / args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else output_dir / "metadata.csv"

    existing_rows = load_existing_rows(csv_path)
    done_urls = {
        url
        for url, row in existing_rows.items()
        if row.get("status") in {"ok", "blocked", "skipped"}
    }

    print(f"Scraping {len(items)} URL(s)")
    print(f"Audio dir : {audio_dir}")
    print(f"Format    : {args.format}")
    print(f"Workers   : {args.workers}")
    if args.proxy:
        print(f"Proxy     : {args.proxy}")
    print(f"CSV       : {csv_path}")
    print("-" * 50)

    rows_by_url = {url: dict(row) for url, row in existing_rows.items()}
    scheduled_urls = set(done_urls)
    pending_items = []
    for item in items:
        url = item["url"]
        if url in scheduled_urls:
            if url not in rows_by_url:
                rows_by_url[url] = build_output_row(
                    item,
                    {"url": url, "status": "duplicate", "error": "", "filename": ""},
                )
            continue
        scheduled_urls.add(url)
        pending_items.append(item)

    global active_executor

    executor = None
    completed = 0
    interrupted = False
    try:
        mp_ctx = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx)
        active_executor = executor
        futures = {
            executor.submit(
                scrape_url_worker,
                item["url"],
                build_ydl_opts(
                    audio_dir,
                    args.format,
                    args.cookies_from_browser,
                    args.cookies,
                    args.proxy,
                    platform=detect_platform(item["url"]),
                ),
                str(audio_dir),
            ): item
            for item in pending_items
        }

        pending_futures = set(futures)
        while pending_futures:
            if stop_event.is_set():
                raise KeyboardInterrupt
            done_futures, pending_futures = wait(
                pending_futures,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )
            for future in done_futures:
                item = futures[future]
                result = future.result()
                rows_by_url[item["url"]] = build_output_row(item, result)
                completed += 1
                if args.delay:
                    interruptible_sleep(args.delay)
    except KeyboardInterrupt:
        interrupted = True
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
            terminate_executor_processes(executor)
        print("\nInterrupted. Stopping active downloads and saving partial results.")
    finally:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        active_executor = None
    for item in items:
        if item["url"] not in rows_by_url and item["url"] in existing_rows:
            rows_by_url[item["url"]] = dict(existing_rows[item["url"]])

    write_results_csv(csv_path, rows_by_url, items)

    print("-" * 50)
    print(f"Done. {completed} URL(s) processed.")
    print(f"CSV saved to: {csv_path}")
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
