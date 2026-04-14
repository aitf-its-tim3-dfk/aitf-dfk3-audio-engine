"""
scripter.py

Generates N spoken-style lines per label for TTS use.
- Calls the OpenRouter API in configurable batches to avoid ChunkedEncodingError.
- Strips HTML from RSS content so the model can actually read the context.
- Saves incrementally after every batch so a crash never loses progress.

Install:
    pip install requests feedparser

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python scripter.py
"""

import argparse
import html
import json
import os
import random
import re
import time
import requests
import feedparser
from collections import Counter
from datetime import datetime
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
DEFAULT_N = 1000
DEFAULT_BATCH_SIZE = 100
DEFAULT_OUTPUT_FILE = "scripts.json"
CHECKPOINT_FILE = "scripts_checkpoint.json"

DEFAULT_CONTEXT_POOL_SIZE = None
DEFAULT_FEEDS_PER_RUN = 8
DEFAULT_CONTEXT_PER_BATCH = 10

DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_DELAY = 60
DEFAULT_CALL_DELAY = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS scripts via OpenRouter API."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Items per label (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--context-pool-size",
        type=int,
        default=DEFAULT_CONTEXT_POOL_SIZE,
        help="Total headlines in pool (default: all)",
    )
    parser.add_argument(
        "--feeds-per-run",
        type=int,
        default=DEFAULT_FEEDS_PER_RUN,
        help=f"Feeds to sample (default: {DEFAULT_FEEDS_PER_RUN})",
    )
    parser.add_argument(
        "--context-per-batch",
        type=int,
        default=DEFAULT_CONTEXT_PER_BATCH,
        help=f"Headlines per batch (default: {DEFAULT_CONTEXT_PER_BATCH})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per batch (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=DEFAULT_RETRY_DELAY,
        help=f"Retry delay in seconds (default: {DEFAULT_RETRY_DELAY})",
    )
    parser.add_argument(
        "--call-delay",
        type=int,
        default=DEFAULT_CALL_DELAY,
        help=f"Delay between API calls (default: {DEFAULT_CALL_DELAY})",
    )
    return parser.parse_args()


# ── Labels ────────────────────────────────────────────────────────────────────

LABELS: dict[str, str] = {
    "disinformasi": (
        "Klaim fakta yang salah atau menyesatkan yang disampaikan sebagai kebenaran. "
        "Bukan opini, berisi pernyataan faktual palsu tentang kejadian, kebijakan, "
        "atau ilmu pengetahuan (UU ITE Pasal 28 ayat 1 dan 3)."
    ),
    "fitnah": (
        "Tuduhan spesifik terhadap individu nyata (nama jabatan atau nama orang) "
        "tentang tindak pidana atau pelanggaran moral TANPA bukti yang sah. "
        "Harus menyebutkan sosok tertentu, bukan kelompok (UU ITE Pasal 27A)."
    ),
    "ujaran_kebencian": (
        "Serangan atau penghinaan terhadap kelompok berdasarkan SARA. "
        "Berbeda dari fitnah karena menyerang KELOMPOK, bukan individu. "
        "Boleh berupa ajakan, stereotip kasar, atau dehumanisasi "
        "(UU ITE Pasal 28 ayat 2)."
    ),
    "neutral": (
        "Konten biasa sehari-hari. Tidak ada tuduhan, tidak ada kebohongan faktual, "
        "tidak ada serangan berbasis SARA. Boleh berupa keluhan, gosip ringan, "
        "kabar keluarga, ulasan produk, atau informasi publik yang benar."
    ),
}

# ── Persona and medium pools ──────────────────────────────────────────────────

PERSONAS = [
    "mahasiswa semester akhir yang suka bikin thread panjang di Twitter/X",
    "bapak-bapak 50-an yang nongkrong sore di warung kopi",
    "remaja SMA yang pakai bahasa gaul dan sering typo",
    "pegawai kantoran yang nulis di jam makan siang",
    "pedagang pasar yang baru belajar pakai smartphone",
    "aktivis LSM muda, gaya bahasa semi-formal",
    "netizen anonim yang rajin komentar panjang di YouTube",
    "guru SD, bahasa sopan tapi tidak kaku",
    "pensiunan, gaya bicara tegas dan singkat",
    "ibu muda milenial, nulis kayak caption Instagram",
    "pemilik warung yang suka forward berita tanpa filter",
    "konten kreator pemula yang lagi bikin script video",
    "warga desa yang baru punya akses internet",
    "ojek online yang ngobrol di pangkalan sambil pesan kopi",
    "karyawan pabrik yang kirim pesan di grup shift malam",
    "anak kuliahan rantau yang curhat sambil ngetik cepat di kos",
    "admin olshop yang balas chat sambil packing pesanan",
    "ibu-ibu pengajian yang sering kirim voice note panjang",
    "bocah warnet yang ngomong ceplas-ceplos dan suka lebay",
    "pengemudi travel antarkota yang cerita sambil nunggu penumpang penuh",
    "pegawai honorer yang gaya bicaranya hati-hati tapi emosinya gampang naik",
    "satpam komplek yang suka komentar soal kejadian sekitar",
    "barista coffee shop yang ngomong santai dan agak sok akrab",
    "pegawai salon yang suka gosipin kabar pelanggan",
    "host live shopping yang energik dan suka mengulang kalimat",
    "anak motor yang ngomong cepat, singkat, dan suka menyelipkan slang",
    "pengajar les privat yang bahasanya runtut dan jelas",
    "penjual makanan kaki lima yang suka promosi sambil bercanda",
    "penumpang KRL yang ngetik sambil berdiri dan emosinya campur aduk",
    "operator gudang yang ngomel di grup kerja setelah shift panjang",
    "vlogger keluarga sederhana yang narasinya spontan dan tidak rapi",
    "pegawai bank muda yang terbiasa formal tapi lagi kesel",
    "penyiar radio lokal yang dramatis saat bercerita",
    "warga kompleks yang aktif di grup WhatsApp RT",
    "freelancer desain yang suka ngetik panjang tanpa titik",
]

MEDIUMS = [
    "transkrip pesan suara WhatsApp (ada jeda, pengulangan, dan koreksi diri)",
    "komentar panjang di postingan Facebook",
    "caption video TikTok atau Instagram Reels",
    "pesan broadcast di grup WhatsApp keluarga atau RT/RW",
    "utas Twitter/X yang emosional",
    "rekaman obrolan santai (podcast informal, tidak diedit)",
    "komentar di kolom berita online",
    "status Facebook panjang yang ditulis malam hari",
    "pesan forward yang sudah muter di banyak grup",
    "cerita lisan kepada tetangga saat arisan",
    "ulasan produk di marketplace",
    "caption foto di grup alumni sekolah",
    "transkrip live TikTok yang spontan dan agak kacau",
    "voice note Telegram yang direkam sambil jalan",
    "komentar di siaran langsung YouTube",
    "pesan di grup kerja kantor yang semi-formal",
    "caption Threads yang reflektif tapi nyinyir",
    "storytime lisan untuk video YouTube Shorts",
    "curhat di DM Instagram ke teman dekat",
    "balasan komentar marketplace setelah barang diterima",
    "rekaman obrolan di motor atau mobil saat di perjalanan",
    "pesan suara di grup keluarga besar setelah dapat kabar viral",
    "cerita spontan di space audio komunitas lokal",
    "transkrip podcast duo teman nongkrong",
    "naskah pembuka video opini pendek untuk Reels",
    "status WhatsApp teks yang kemudian dibacakan",
    "komentar di forum parenting atau komunitas ibu-ibu",
    "catatan suara untuk tugas presentasi yang belum diedit",
    "pengumuman informal di grup warga atau grup sekolah",
    "monolog pendek untuk video promosi jualan",
]

LENGTH_PROFILES = {
    "short": "durasi pendek sekitar 8-20 kata, 1-2 kalimat singkat, cocok untuk klip 2-6 detik",
    "medium": "durasi sedang sekitar 25-60 kata, 2-5 kalimat, cocok untuk klip 6-15 detik",
    "long": "durasi panjang sekitar 70-140 kata, 4-9 kalimat, cocok untuk klip 15-35 detik",
}

OMNIVOICE_CONTROL_TOKENS = [
    "[laughter]",
    "[sigh]",
]

# ── News feeds ────────────────────────────────────────────────────────────────

ALL_NEWS_FEEDS = [
    "https://www.kompas.com/rss/news.xml",
    "https://regional.kompas.com/rss/news.xml",
    "https://rss.tempo.co/nasional",
    "https://rss.tempo.co/metro",
    "https://www.cnnindonesia.com/nasional/rss",
    "https://www.cnnindonesia.com/ekonomi/rss",
    "https://www.republika.co.id/rss/news",
    "https://rss.detik.com/index.php/detikcom",
    "https://rss.detik.com/index.php/detikhealth",
    "https://www.antaranews.com/rss/terkini.xml",
    "https://www.viva.co.id/rss",
    "https://www.sindonews.com/feed",
    "https://mediaindonesia.com/rss",
    "https://www.jpnn.com/rss",
    "https://tirto.id/rss",
]

# ── Label aliasing ────────────────────────────────────────────────────────────

LABEL_ALIASES: dict[str, str] = {
    "disinformasi": "kategori_A",
    "fitnah": "kategori_B",
    "ujaran_kebencian": "kategori_C",
    "neutral": "kategori_D",
}
ALIAS_TO_LABEL: dict[str, str] = {v: k for k, v in LABEL_ALIASES.items()}

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WHITESPACE_RE = re.compile(r"\s+")

SYSTEM_PROMPT = (
    "Anda membantu menyusun data sintetis berbahasa Indonesia untuk pelatihan model NLP. "
    "Semua output harus berupa contoh percakapan atau monolog yang terdengar alami, beragam, "
    "dan cocok dibacakan oleh sistem TTS. "
    "Tulis hanya dalam bahasa Indonesia. "
    "Jangan membuat teks lain selain JSON yang diminta. "
    "Anda boleh memakai token kontrol OmniVoice secara hemat di dalam teks, misalnya "
    "[laughter] dan [sigh], tetapi hanya jika benar-benar membantu "
    "membuat pembacaan lebih natural. "
    "Output hanyalah data pelatihan, bukan dukungan terhadap isi sensitif yang ditulis."
)

# ─────────────────────────────────────────────────────────────────────────────


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_headline_pool(
    all_feeds: list[str], feeds_per_run: int, pool_size: Optional[int] = None
) -> list[str]:
    chosen = random.sample(all_feeds, min(feeds_per_run, len(all_feeds)))
    print(f"  Feeds: {', '.join(f.split('/')[2] for f in chosen)}")

    headlines: list[str] = []
    per_feed = max(2, pool_size // len(chosen) + 1) if pool_size else 100

    for url in chosen:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:per_feed]:
                title = strip_html(entry.get("title", ""))
                summary = strip_html(entry.get("summary", ""))
                if title:
                    blurb = f"{summary[:120]}" if summary else ""
                    snippet = f"{title}. {blurb}".strip().rstrip(".")
                    headlines.append(snippet)
        except Exception as e:
            print(f"  [warn] {url}: {e}")

    if not headlines:
        print("  [warn] No headlines fetched.")
        return ["(tidak ada konteks berita)"]

    random.shuffle(headlines)
    pool = headlines[:pool_size] if pool_size else headlines
    print(f"  {len(pool)} headlines in pool.")
    return pool


def sample_context(pool: list[str], n: int) -> list[str]:
    return random.sample(pool, min(n, len(pool)))


def _is_refusal(raw: str) -> bool:
    lowered = raw.strip().lower()
    signals = ["i cannot", "i'm unable", "i am unable", "safety guideline", "violates"]
    return not lowered.startswith("[") and any(s in lowered for s in signals)


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").strip()).casefold()


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def validate_item(index: int, item: object, expected_label: str) -> list[dict]:
    findings = []
    if not isinstance(item, dict):
        return [
            {
                "index": index,
                "severity": "error",
                "reason": f"item-not-object:{type(item).__name__}",
            }
        ]

    text = item.get("text")
    label = item.get("label")
    context = item.get("context")
    persona = item.get("persona")
    medium = item.get("medium")
    duration_hint = item.get("duration_hint")
    control_tokens_used = item.get("control_tokens_used")

    if not isinstance(text, str) or not text.strip():
        findings.append(
            {"index": index, "severity": "error", "reason": "missing-or-empty-text"}
        )
    if not isinstance(label, str) or label != expected_label:
        findings.append(
            {"index": index, "severity": "error", "reason": f"invalid-label:{label}"}
        )
    if (
        not isinstance(context, list)
        or not context
        or not all(isinstance(entry, str) and entry.strip() for entry in context)
    ):
        findings.append(
            {"index": index, "severity": "error", "reason": "invalid-context"}
        )
    if not isinstance(persona, str) or not persona.strip():
        findings.append(
            {"index": index, "severity": "error", "reason": "missing-persona"}
        )
    if not isinstance(medium, str) or not medium.strip():
        findings.append(
            {"index": index, "severity": "error", "reason": "missing-medium"}
        )
    if duration_hint not in LENGTH_PROFILES:
        findings.append(
            {
                "index": index,
                "severity": "error",
                "reason": f"invalid-duration-hint:{duration_hint}",
            }
        )
    if not isinstance(control_tokens_used, list) or not all(
        isinstance(token, str) for token in control_tokens_used
    ):
        findings.append(
            {"index": index, "severity": "error", "reason": "invalid-control-tokens"}
        )

    if isinstance(text, str) and has_cjk(text):
        findings.append(
            {"index": index, "severity": "error", "reason": "text-contains-cjk"}
        )

    if isinstance(control_tokens_used, list):
        for token in control_tokens_used:
            if token not in OMNIVOICE_CONTROL_TOKENS:
                findings.append(
                    {
                        "index": index,
                        "severity": "error",
                        "reason": f"unknown-control-token:{token}",
                    }
                )
            elif isinstance(text, str) and token not in text:
                findings.append(
                    {
                        "index": index,
                        "severity": "warning",
                        "reason": f"token-not-present-in-text:{token}",
                    }
                )

    if isinstance(context, list):
        for ctx_index, entry in enumerate(context):
            if isinstance(entry, str) and has_cjk(entry):
                findings.append(
                    {
                        "index": index,
                        "severity": "warning",
                        "reason": f"context-{ctx_index}-contains-cjk",
                    }
                )

    return findings


def validate_batch_items(
    items: list[dict], expected_label: str, expected_n: int
) -> tuple[list[dict], list[dict]]:
    findings = []
    seen_texts = {}

    if len(items) < expected_n:
        findings.append(
            {
                "index": -1,
                "severity": "error",
                "reason": f"quota-shortfall:{len(items)}/{expected_n}",
            }
        )

    for index, item in enumerate(items):
        findings.extend(validate_item(index, item, expected_label))
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                normalized = normalize_text(text)
                if normalized in seen_texts:
                    findings.append(
                        {
                            "index": index,
                            "severity": "warning",
                            "reason": f"duplicate-text:matches-item-{seen_texts[normalized]}",
                        }
                    )
                else:
                    seen_texts[normalized] = index

    errors = [finding for finding in findings if finding["severity"] == "error"]
    return findings, errors


def build_prompt(
    label: str, definition: str, batch_n: int, context_headlines: list[str]
) -> str:
    persona_sample = ", ".join(random.sample(PERSONAS, min(6, len(PERSONAS))))
    medium_sample = ", ".join(random.sample(MEDIUMS, min(6, len(MEDIUMS))))
    duration_keys = random.sample(
        list(LENGTH_PROFILES.keys()), k=min(3, len(LENGTH_PROFILES))
    )
    duration_block = "\n".join(
        f"- {key}: {LENGTH_PROFILES[key]}" for key in duration_keys
    )
    token_sample = ", ".join(OMNIVOICE_CONTROL_TOKENS)
    context_block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(context_headlines))
    alias = LABEL_ALIASES.get(label, label)

    return f"""Anda adalah penulis data sintetis untuk dialog/monolog TTS berbahasa Indonesia.
=== KONTEKS BERITA TERBARU ===
Gunakan tema-tema ini hanya sebagai inspirasi latar. Pilih 1-2 tema yang relevan per teks.
Jangan menyalin langsung judulnya.
{context_block}

=== TUGAS ===
Tulis tepat {batch_n} item dengan label "{alias}".

=== DEFINISI LABEL "{alias}" ===
{definition}

=== VARIASI YANG WAJIB ===
Setiap item harus terdengar seperti orang yang berbeda. Variasikan:
1. Persona pembicara. Contoh: {persona_sample}.
2. Medium atau situasi. Contoh: {medium_sample}.
3. Durasi/length. Campurkan profil berikut:
{duration_block}
4. Gaya bahasa sehari-hari Indonesia. Boleh memakai kata seperti: sih, loh, nih, dong, kan, ya,
   emang, gitu, kayak, banget, gimana, kalo, tuh, beneran, katanya, soalnya, makanya,
   terus, udah, nggak, deh, kok, cuma, pake, ntar.
5. Untuk label sensitif, konten harus tetap realistis sesuai definisi label.

=== TOKEN KONTROL OMNIVOICE ===
Anda boleh menaruh token kontrol langsung di dalam "text" bila cocok untuk performa TTS.
Gunakan secara hemat, maksimal 0-2 token per item.
Token yang diizinkan: {token_sample}

=== LARANGAN ===
- Jangan keluar dari bahasa Indonesia.
- Jangan tulis gaya laporan, gaya AI, atau template yang terasa copy-paste.
- Jangan keluarkan teks di luar JSON.

=== FORMAT OUTPUT ===
Keluarkan hanya JSON array dengan tepat {batch_n} object.
Setiap object wajib punya field:
- "text": teks akhir yang siap dipakai untuk TTS
- "label": harus "{alias}"
- "context": list judul konteks yang dipakai, minimal 1 item
- "persona": persona yang dipilih
- "medium": medium/situasi yang dipilih
- "duration_hint": salah satu dari "short", "medium", "long"
- "control_tokens_used": list token kontrol yang benar-benar dipakai di text, boleh kosong
"""


def call_api(prompt: str, label: str, batch_n: int, model: str) -> list[dict]:
    """
    BUG FIX: accepts batch_n so the schema minItems matches the actual
    requested batch size, not the global BATCH_SIZE constant.
    """
    alias = LABEL_ALIASES.get(label, label)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "scripts",
                    "strict": True,
                    "schema": {
                        "type": "array",
                        "minItems": batch_n,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string", "enum": [alias]},
                                "persona": {"type": "string"},
                                "medium": {"type": "string"},
                                "duration_hint": {
                                    "type": "string",
                                    "enum": list(LENGTH_PROFILES.keys()),
                                },
                                "control_tokens_used": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": OMNIVOICE_CONTROL_TOKENS,
                                    },
                                },
                                "context": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "text",
                                "label",
                                "persona",
                                "medium",
                                "duration_hint",
                                "control_tokens_used",
                                "context",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        },
        timeout=300,
    )

    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    raw = data["choices"][0]["message"]["content"]

    if _is_refusal(raw):
        raise RuntimeError(f"Model refused: {raw[:120]}")

    parsed = json.loads(raw)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Expected JSON array, got {type(parsed).__name__}: {str(parsed)[:80]}"
        )

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Item {i} is {type(item).__name__}, expected dict: {str(item)[:80]}"
            )

    for item in parsed:
        item["label"] = ALIAS_TO_LABEL.get(item["label"], item["label"])

    findings, errors = validate_batch_items(parsed, label, batch_n)
    if errors:
        summary = ", ".join(finding["reason"] for finding in errors[:5])
        raise RuntimeError(f"Invalid batch from model: {summary}")
    warnings = [finding for finding in findings if finding["severity"] == "warning"]
    if warnings:
        summary = ", ".join(finding["reason"] for finding in warnings[:3])
        print(f"[warn] batch validation warnings: {summary}")

    return parsed


def save_checkpoint(all_items: list[dict], path: str) -> None:
    """Writes current items to a checkpoint file after every batch."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [warn] Checkpoint save failed: {e}")


def generate_label(
    label: str,
    definition: str,
    n: int,
    batch_size: int,
    headline_pool: list[str],
    all_items: list[dict],
    context_per_batch: int,
    max_retries: int,
    retry_delay: int,
    call_delay: int,
    model: str,
) -> tuple[list[dict], list[list[str]]]:
    results: list[dict] = []
    batch_contexts_used: list[list[str]] = []

    batches = []
    remaining = n
    while remaining > 0:
        batches.append(min(batch_size, remaining))
        remaining -= batch_size

    for b_idx, b_size in enumerate(batches, 1):
        batch_context = sample_context(headline_pool, context_per_batch)
        print(
            f"    batch {b_idx}/{len(batches)} ({b_size} items)...",
            end=" ",
            flush=True,
        )

        for attempt in range(1, max_retries + 1):
            try:
                prompt = build_prompt(label, definition, b_size, batch_context)
                items = call_api(prompt, label, b_size, model)
                results.extend(items)
                batch_contexts_used.append(batch_context)
                print(
                    f"ok ({len(items)} items, total so far: {len(all_items) + len(results)})"
                )
                save_checkpoint(all_items + results, CHECKPOINT_FILE)
                time.sleep(call_delay)
                break
            except Exception as e:
                print(f"fail [{attempt}/{max_retries}]: {e}")
                if attempt < max_retries:
                    print(f"    retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"    [error] Skipping batch {b_idx} for '{label}'.")

    return results, batch_contexts_used


def validate(results: list[dict], labels: dict, n: int) -> None:
    counts = Counter(item.get("label") for item in results)
    duration_counts = Counter(
        item.get("duration_hint")
        for item in results
        if isinstance(item, dict) and item.get("duration_hint")
    )
    token_counts = Counter()
    findings = []
    seen_texts = {}

    for index, item in enumerate(results):
        label = item.get("label") if isinstance(item, dict) else None
        findings.extend(validate_item(index, item, label))
        if isinstance(item, dict):
            text = item.get("text")
            for token in item.get("control_tokens_used", []):
                token_counts[token] += 1
            if isinstance(text, str) and text.strip():
                normalized = normalize_text(text)
                if normalized in seen_texts:
                    findings.append(
                        {
                            "index": index,
                            "severity": "warning",
                            "reason": f"duplicate-text:matches-item-{seen_texts[normalized]}",
                        }
                    )
                else:
                    seen_texts[normalized] = index

    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]

    print("\n[info] Final counts:")
    for label in labels:
        count = counts.get(label, 0)
        flag = "OK" if count >= n else f"LOW (expected {n})"
        print(f"  {label:20s}: {count:3d}  [{flag}]")
    print(f"  {'TOTAL':20s}: {len(results)}")
    print(f"  {'ERRORS':20s}: {len(errors)}")
    print(f"  {'WARNINGS':20s}: {len(warnings)}")
    if duration_counts:
        print("[info] Duration mix:")
        for key in ("short", "medium", "long"):
            print(f"  {key:20s}: {duration_counts.get(key, 0)}")
    if token_counts:
        print("[info] Control token usage:")
        for token, count in token_counts.most_common():
            print(f"  {token:20s}: {count}")
    if errors[:10]:
        print("[info] Sample errors:")
        for finding in errors[:10]:
            print(f"  item {finding['index']}: {finding['reason']}")
    if warnings[:10]:
        print("[info] Sample warnings:")
        for finding in warnings[:10]:
            print(f"  item {finding['index']}: {finding['reason']}")


def main() -> None:
    args = parse_args()

    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    print("=== TTS Script Generator ===\n")
    print(f"  Model           : {args.model}")
    print(f"  N per label     : {args.n}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Context pool    : {args.context_pool_size} headlines")
    print(f"  Context/batch   : {args.context_per_batch} headlines per call")
    print(f"  Labels          : {', '.join(LABELS.keys())}\n")

    start_time = time.time()

    print("Step 1: Building headline pool...")
    headline_pool = fetch_headline_pool(
        ALL_NEWS_FEEDS, args.feeds_per_run, args.context_pool_size
    )

    print(
        f"\nStep 2: Generating {args.n} x {len(LABELS)} = {args.n * len(LABELS)} items...\n"
    )
    all_items: list[dict] = []
    all_batch_contexts: dict[str, list[list[str]]] = {}

    for label, definition in LABELS.items():
        print(f"  [{label}]")
        items, batch_ctxs = generate_label(
            label,
            definition,
            args.n,
            args.batch_size,
            headline_pool,
            all_items,
            args.context_per_batch,
            args.max_retries,
            args.retry_delay,
            args.call_delay,
            args.model,
        )
        all_items.extend(items)
        all_batch_contexts[label] = batch_ctxs

    missing = [l for l in LABELS if not any(i["label"] == l for i in all_items)]
    if missing:
        print(f"\n[warn] Missing labels: {missing}. Re-running...")
        for label in missing:
            items, batch_ctxs = generate_label(
                label,
                LABELS[label],
                args.n,
                args.batch_size,
                headline_pool,
                all_items,
                args.context_per_batch,
                args.max_retries,
                args.retry_delay,
                args.call_delay,
                args.model,
            )
            all_items.extend(items)
            all_batch_contexts[label] = batch_ctxs

    validate(all_items, LABELS, args.n)

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "time_taken": f"{(time.time() - start_time) / 60:.1f} minutes",
            "model": args.model,
            "n_per_label": args.n,
            "batch_size": args.batch_size,
            "context_pool_size": args.context_pool_size,
            "context_per_batch": args.context_per_batch,
            "labels": list(LABELS.keys()),
            "headline_pool": headline_pool,
            "batch_contexts": all_batch_contexts,
        },
        "items": all_items,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\nSaved {len(all_items)} items to '{args.output}'.")


if __name__ == "__main__":
    main()
