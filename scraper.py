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
DEFAULT_MAX_DURATION = 600
DEFAULT_MIN_DURATION = 10
DEFAULT_WINDOW_SIZE = 30
DEFAULT_WINDOW_OVERLAP = 0
DEFAULT_MIN_CHUNK_DURATION = 5

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
    "strategy",
    "chunk_index",
    "chunk_start_sec",
    "chunk_end_sec",
    "parent_url",
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
    "YouTube": "cookies/youtube_cookies.txt",
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
        "js_runtimes": {"node": {}},
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
    normalized["strategy"] = normalized.get("strategy", "")
    return normalized


def load_input_items(args) -> list[dict]:
    if args.urls:
        return [
            {"url": url.strip(), "weak_label": "", "source_article": "", "keyword": ""}
            for url in args.urls
            if url.strip()
        ]

    input_path = Path(args.url_file)
    filter_labels = set(args.labels) if getattr(args, "labels", None) else None
    if input_path.suffix.lower() == ".csv":
        items = []
        with open(input_path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item = normalize_input_row(row)
                if not item["url"] or item["url"].startswith("#"):
                    continue
                if not is_valid_video_url(item["url"]):
                    continue
                if filter_labels and item.get("weak_label") not in filter_labels:
                    continue
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
    row["strategy"] = item.get("strategy", "")
    row["resolved_url"] = item.get("resolved_url", item["url"])
    row["scraped_at"] = datetime.now().isoformat(timespec="seconds")
    return row


def build_output_rows(item: dict, result: dict) -> list[dict]:
    base_row = build_output_row(item, result)
    chunks = result.get("chunks") or []
    if not chunks:
        return [base_row]

    rows = []
    for chunk in chunks:
        chunk_row = dict(base_row)
        chunk_row.update(
            {
                "filename": chunk.get("filename", base_row.get("filename", "")),
                "duration_sec": chunk.get(
                    "duration_sec", base_row.get("duration_sec", "")
                ),
                "chunk_index": chunk.get("chunk_index", ""),
                "chunk_start_sec": chunk.get("chunk_start_sec", ""),
                "chunk_end_sec": chunk.get("chunk_end_sec", ""),
                "parent_url": item["url"],
            }
        )
        rows.append(chunk_row)
    return rows


def write_results_csv(
    csv_path: Path, rows_by_url: dict, input_order: list[dict]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in input_order:
        if item["url"] not in rows_by_url:
            continue
        stored = rows_by_url[item["url"]]
        if isinstance(stored, list):
            rows.extend(stored)
        else:
            rows.append(stored)

    extra_fields = []
    for row in rows:
        for key in row:
            if key not in OUTPUT_FIELDS and key not in extra_fields:
                extra_fields.append(key)

    fieldnames = OUTPUT_FIELDS + sorted(extra_fields)
    exists = csv_path.exists()
    mode = "a" if exists else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def extract_video_id(url: str) -> str | None:
    # YouTube
    m = re.search(r"(?:watch\?v=|embed/|shorts/)([\w-]{11})", url)
    if m:
        return m.group(1)
    # TikTok
    m = re.search(r"/video/(\d+)", url)
    if m:
        return m.group(1)
    # Facebook
    m = re.search(r"watch\?v=(\d+)", url)
    if m:
        return m.group(1)
    # Instagram reel/p
    m = re.search(r"instagram\.com/(?:reel|p)/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    # Twitter/X
    m = re.search(r"/status/(\d+)", url)
    if m:
        return m.group(1)
    return None


def get_duration(filepath: Path) -> str:
    """Get audio duration using ffprobe."""
    import subprocess

    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return str(int(float(result.stdout.strip())))
    except Exception:
        pass
    return ""


def split_audio_into_chunks(
    input_path: Path,
    output_dir: Path,
    window_size: int,
    overlap: int,
    min_chunk_duration: int,
    sample_rate: int = 16000,
) -> list[dict]:
    """Split audio into fixed-size windows. Returns list of chunk info dicts."""
    import subprocess

    duration = get_duration(input_path)
    if not duration:
        return []
    duration = int(duration)

    if duration < min_chunk_duration:
        return []

    chunks = []
    step = window_size - overlap
    chunk_index = 0

    for start in range(0, duration, step):
        end = min(start + window_size, duration)
        chunk_duration = end - start

        if chunk_duration < min_chunk_duration:
            if chunks and start == 0:
                pass
            else:
                break

        output_name = f"{input_path.stem}_chunk{chunk_index:03d}{input_path.suffix}"
        output_path = output_dir / output_name

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-ss",
                str(start),
                "-t",
                str(chunk_duration),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=60)

            chunks.append(
                {
                    "filename": str(output_path),
                    "chunk_index": chunk_index,
                    "chunk_start_sec": start,
                    "chunk_end_sec": end,
                    "duration_sec": chunk_duration,
                }
            )
        except Exception:
            pass

        chunk_index += 1

        if end >= duration:
            break

    return chunks


def regenerate_metadata(data_csv: Path, audio_dir: Path, output_csv: Path) -> int:
    """Regenerate metadata by matching existing audio files to URLs."""
    url_to_info = {}
    with open(data_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url_to_info[row["url"]] = {
                "label": row.get("weak_label", ""),
                "source": row.get("source_article", ""),
                "keyword": row.get("keyword", ""),
                "discovered_at": row.get("discovered_at", ""),
                "id": row.get("id", ""),
            }

    existing_files = {}
    for f in audio_dir.iterdir():
        if f.suffix.lower() in (".wav", ".mp3", ".mp4", ".m4a", ".flac", ".opus"):
            existing_files[f.stem] = f

    matched = []
    for url in url_to_info:
        vid = extract_video_id(url)
        if not vid:
            continue
        for stem, path in existing_files.items():
            if stem == vid or stem.startswith(vid + "-"):
                info = url_to_info[url]
                duration = get_duration(path)
                matched.append(
                    {
                        "url": url,
                        "platform": detect_platform(url),
                        "title": "",
                        "uploader": "",
                        "duration_sec": duration,
                        "filename": str(path),
                        "status": "ok",
                        "error": "",
                        "scraped_at": datetime.now().isoformat(timespec="seconds"),
                        "resolved_url": url,
                        "weak_label": info["label"],
                        "source_article": info["source"],
                        "keyword": info["keyword"],
                        "discovered_at": info["discovered_at"],
                        "id": info["id"],
                    }
                )
                break

    if not matched:
        print("No matching audio files found")
        return 0

    # Append to CSV - include discovered_at, id
    fieldnames = OUTPUT_FIELDS + ["discovered_at", "id"]
    exists = output_csv.exists()
    mode = "a" if exists else "w"
    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(matched)

    print(f"Regenerated {len(matched)} metadata entries")
    return len(matched)


def get_video_duration(url: str, ydl_opts: dict) -> int | None:
    """Get video duration without downloading."""
    opts = dict(ydl_opts)
    opts["skip_download"] = True
    opts["quiet"] = True
    opts["no_warnings"] = True
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info:
                return info.get("duration")
    except Exception:
        pass
    return None


def check_duration_filter(
    url: str, ydl_opts: dict, max_duration: int | None, min_duration: int | None
) -> tuple[bool, str]:
    """Check if video passes duration filters. Returns (passes, reason)."""
    if max_duration is None and min_duration is None:
        return True, ""

    duration = get_video_duration(url, ydl_opts)
    if duration is None:
        return True, ""

    if max_duration is not None and duration > max_duration:
        return False, f"duration {duration}s exceeds max {max_duration}s"
    if min_duration is not None and duration < min_duration:
        return False, f"duration {duration}s below min {min_duration}s"

    return True, ""


def scrape_url(
    url: str,
    ydl_opts: dict,
    audio_dir: Path,
    retries: int = 3,
    max_duration: int | None = None,
    min_duration: int | None = None,
    window_size: int | None = None,
    window_overlap: int = 0,
    min_chunk_duration: int = 5,
    sample_rate: int = 16000,
) -> dict:
    result = {"url": url, "status": "", "error": "", "filename": ""}

    passes, reason = check_duration_filter(url, ydl_opts, max_duration, min_duration)
    if not passes:
        result["status"] = "skipped"
        result["error"] = reason
        return result

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

                if window_size:
                    chunks = split_audio_into_chunks(
                        final_path,
                        audio_dir,
                        window_size,
                        window_overlap,
                        min_chunk_duration,
                        sample_rate,
                    )
                    if chunks:
                        result["chunks"] = chunks
                        try:
                            final_path.unlink()
                        except OSError:
                            pass

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
                    "unsupported url",
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
    url: str,
    ydl_opts: dict,
    audio_dir_str: str,
    retries: int = 3,
    max_duration: int | None = None,
    min_duration: int | None = None,
    window_size: int | None = None,
    window_overlap: int = 0,
    min_chunk_duration: int = 5,
    sample_rate: int = 16000,
) -> dict:
    return scrape_url(
        url,
        ydl_opts,
        Path(audio_dir_str),
        retries,
        max_duration,
        min_duration,
        window_size,
        window_overlap,
        min_chunk_duration,
        sample_rate,
    )


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
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=[
            "ujaran kebencian",
            "fitnah",
            "disinformasi",
            "neutral",
            "fake",
            "real",
        ],
        help="Filter by label(s) from CSV",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Maximum number of items to process per label (default: no limit)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate metadata from existing audio files without re-downloading",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=DEFAULT_MAX_DURATION,
        help="Skip videos longer than N seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=DEFAULT_MIN_DURATION,
        help="Skip videos shorter than N seconds",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Window size in seconds for chunking audio (default: 30)",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=DEFAULT_WINDOW_OVERLAP,
        help="Overlap between windows in seconds (default: 0)",
    )
    parser.add_argument(
        "--min-chunk-duration",
        type=int,
        default=DEFAULT_MIN_CHUNK_DURATION,
        help="Minimum chunk duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for audio chunks (default: 16000)",
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

    if args.regenerate:
        data_csv = Path(args.url_file) if args.url_file else Path("data.csv")
        audio_dir = Path(args.output) / args.audio_dir
        csv_path = Path(args.csv) if args.csv else Path(args.output) / "metadata.csv"
        print(f"Regenerating metadata from {audio_dir}")
        regenerate_metadata(data_csv, audio_dir, csv_path)
        return

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
    permanently_done_urls = {
        url for url, row in existing_rows.items() if row.get("status") == "ok"
    }
    failed_urls = {
        url
        for url, row in existing_rows.items()
        if row.get("status") in {"failed", "blocked", "skipped", "needs-login"}
    }

    print(f"Scraping {len(items)} URL(s)")
    new_items = []
    retry_items = []
    label_counts = {label: 0 for label in (args.labels or [])}

    for item in items:
        url = item["url"]
        label = item.get("weak_label", "")

        if args.max_per_label and label in label_counts:
            if label_counts[label] >= args.max_per_label:
                continue
            label_counts[label] += 1

        if url in permanently_done_urls:
            continue
        if url in failed_urls:
            retry_items.append(item)
        else:
            new_items.append(item)
    pending_items = new_items + retry_items

    print(f"  New URLs: {len(new_items)}, Retries: {len(retry_items)}")
    print(f"Audio dir : {audio_dir}")
    print(f"Format    : {args.format}")
    print(f"Workers   : {args.workers}")
    if args.max_per_label:
        print(f"Max/label : {args.max_per_label}")
    if args.proxy:
        print(f"Proxy     : {args.proxy}")
    print(f"CSV       : {csv_path}")
    if args.window_size and args.window_size > 0:
        print(
            f"Chunking  : {args.window_size}s windows, {args.window_overlap}s overlap, {args.sample_rate}Hz"
        )
    print("-" * 50)

    rows_by_url = {url: dict(row) for url, row in existing_rows.items()}

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
                max_duration=args.max_duration,
                min_duration=args.min_duration,
                window_size=args.window_size,
                window_overlap=args.window_overlap,
                min_chunk_duration=args.min_chunk_duration,
                sample_rate=args.sample_rate,
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
                rows_by_url[item["url"]] = build_output_rows(item, result)
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
