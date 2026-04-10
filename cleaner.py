#!/usr/bin/env python3
"""
Filter a scraped audio dataset into a cleaner dataset directory.

The script reads a dataset directory produced by scraper.py, validates audio files,
applies a few practical quality heuristics, copies accepted files into a new
dataset directory, and writes kept/rejected CSV reports.

Examples:
    python cleaner.py ./dataset --output ./cleaned
    python cleaner.py ./dataset --output ./cleaned --min-duration 2 --max-duration 180
    python cleaner.py ./dataset --output ./cleaned --max-size-mb 100 --no-audio-check
"""

import argparse
import audioop
import csv
import hashlib
import shutil
import sys
import wave
from pathlib import Path


DEFAULT_OUTPUT_DIR = "dataset-clean"
DEFAULT_AUDIO_DIR = "raw"
DEFAULT_MIN_DURATION = 1.5
DEFAULT_MAX_DURATION = 300.0
DEFAULT_MIN_SIZE_MB = 0.05
DEFAULT_MAX_SIZE_MB = 250.0
DEFAULT_MIN_RMS = 200.0
DEFAULT_MAX_CLIP_RATIO = 0.02
DEFAULT_MIN_VAD_RATIO = 0.15
DEFAULT_MIN_VAD_SEGMENTS = 1
SUPPORTED_EXTENSIONS = {".wav"}
CSV_CANDIDATES = ("metadata.csv", "results.csv", "augmented_results.csv")

_SILERO_VAD_MODEL = None


def find_metadata_csv(dataset_dir: Path) -> Path:
    for name in CSV_CANDIDATES:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV metadata file found in {dataset_dir}")
    raise FileNotFoundError(
        f"Multiple CSV files found in {dataset_dir}; pass --csv explicitly. Found: {[p.name for p in csv_files]}"
    )


def resolve_audio_path(dataset_dir: Path, filename_value: str) -> Path:
    path = Path((filename_value or "").strip())
    if not path:
        return path
    if path.is_absolute():
        return path
    return (dataset_dir / path).resolve()


def safe_output_name(path: Path, seen_names: set[str]) -> str:
    candidate = path.name
    if candidate not in seen_names:
        seen_names.add(candidate)
        return candidate

    stem = path.stem
    suffix = path.suffix
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    candidate = f"{stem}-{digest}{suffix}"
    counter = 1
    while candidate in seen_names:
        candidate = f"{stem}-{digest}-{counter}{suffix}"
        counter += 1
    seen_names.add(candidate)
    return candidate


def get_silero_vad_model():
    global _SILERO_VAD_MODEL
    if _SILERO_VAD_MODEL is not None:
        return _SILERO_VAD_MODEL

    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD is required for cleaner.py. Install it with its runtime dependencies, "
            "for example: `uv pip install torch torchaudio silero-vad`."
        ) from exc

    _SILERO_VAD_MODEL = load_silero_vad()
    return _SILERO_VAD_MODEL


def compute_silero_vad(audio_path: Path, duration_sec: float) -> tuple[float, int]:
    try:
        from silero_vad import get_speech_timestamps, read_audio
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD is required for cleaner.py. Install it with its runtime dependencies, "
            "for example: `uv pip install torch torchaudio silero-vad`."
        ) from exc

    model = get_silero_vad_model()
    wav = read_audio(str(audio_path), sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        return_seconds=True,
    )

    speech_duration = 0.0
    for segment in speech_timestamps:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        speech_duration += max(0.0, end - start)

    vad_ratio = (speech_duration / duration_sec) if duration_sec > 0 else 0.0
    return vad_ratio, len(speech_timestamps)


def inspect_audio(audio_path: Path) -> dict:
    info = {
        "measured_duration_sec": "",
        "sample_rate": "",
        "channels": "",
        "peak": "",
        "rms": "",
        "clip_ratio": "",
        "vad_ratio": "",
        "vad_segments": "",
    }

    with wave.open(str(audio_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        info["measured_duration_sec"] = round(frames / float(sample_rate), 3) if sample_rate else 0.0
        info["sample_rate"] = sample_rate
        info["channels"] = channels

        preview_frames = min(frames, int(sample_rate * 20))
        raw = wav_file.readframes(preview_frames)
        if raw:
            rms = float(audioop.rms(raw, sample_width))
            peak = float(audioop.max(raw, sample_width))

            max_possible = float((1 << (8 * sample_width - 1)) - 1)
            clipped_samples = 0
            total_samples = 0
            threshold = max_possible * 0.999

            for offset in range(0, len(raw), sample_width):
                sample = int.from_bytes(raw[offset:offset + sample_width], byteorder="little", signed=True)
                total_samples += 1
                if abs(sample) >= threshold:
                    clipped_samples += 1

            clip_ratio = (clipped_samples / total_samples) if total_samples else 0.0

            info["peak"] = round(peak, 3)
            info["rms"] = round(rms, 3)
            info["clip_ratio"] = round(clip_ratio, 6)

    vad_ratio, vad_segments = compute_silero_vad(
        audio_path,
        float(info["measured_duration_sec"] or 0.0),
    )
    info["vad_ratio"] = round(vad_ratio, 6)
    info["vad_segments"] = vad_segments

    return info


def should_keep(row: dict, audio_path: Path, file_size_bytes: int, audio_info: dict, args) -> tuple[bool, str]:
    status = (row.get("status") or "").strip().lower()
    if status and status != "ok":
        return False, f"status={status}"
    if not audio_path.exists():
        return False, "missing-file"
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"unsupported-extension:{audio_path.suffix.lower()}"

    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb < args.min_size_mb:
        return False, f"too-small:{file_size_mb:.2f}MB"
    if file_size_mb > args.max_size_mb:
        return False, f"too-large:{file_size_mb:.2f}MB"

    duration = float(audio_info.get("measured_duration_sec") or 0.0)
    if duration < args.min_duration:
        return False, f"too-short:{duration:.2f}s"
    if duration > args.max_duration:
        return False, f"too-long:{duration:.2f}s"

    if args.audio_check:
        rms = float(audio_info.get("rms") or 0.0)
        clip_ratio = float(audio_info.get("clip_ratio") or 0.0)
        vad_ratio = float(audio_info.get("vad_ratio") or 0.0)
        vad_segments = int(audio_info.get("vad_segments") or 0)
        if rms < args.min_rms:
            return False, f"too-quiet:{rms:.5f}"
        if clip_ratio > args.max_clip_ratio:
            return False, f"too-clipped:{clip_ratio:.4f}"
        if args.vad_check:
            if vad_ratio < args.min_vad_ratio:
                return False, f"low-vad:{vad_ratio:.4f}"
            if vad_segments < args.min_vad_segments:
                return False, f"low-vad-segments:{vad_segments}"

    return True, "accepted"


def copy_clean_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter an audio dataset into a cleaner dataset directory.")
    parser.add_argument("input", help="Input dataset directory")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help=f"Output dataset directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--csv", default=None, help="Optional metadata CSV path. Defaults to metadata.csv/results.csv in the dataset directory")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help=f"Output audio subdirectory name (default: {DEFAULT_AUDIO_DIR})")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION, help=f"Minimum audio duration in seconds (default: {DEFAULT_MIN_DURATION})")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION, help=f"Maximum audio duration in seconds (default: {DEFAULT_MAX_DURATION})")
    parser.add_argument("--min-size-mb", type=float, default=DEFAULT_MIN_SIZE_MB, help=f"Minimum file size in MB (default: {DEFAULT_MIN_SIZE_MB})")
    parser.add_argument("--max-size-mb", type=float, default=DEFAULT_MAX_SIZE_MB, help=f"Maximum file size in MB (default: {DEFAULT_MAX_SIZE_MB})")
    parser.add_argument("--min-rms", type=float, default=DEFAULT_MIN_RMS, help=f"Minimum RMS loudness for preview audio (default: {DEFAULT_MIN_RMS})")
    parser.add_argument("--max-clip-ratio", type=float, default=DEFAULT_MAX_CLIP_RATIO, help=f"Maximum clipped-sample ratio for preview audio (default: {DEFAULT_MAX_CLIP_RATIO})")
    parser.add_argument("--min-vad-ratio", type=float, default=DEFAULT_MIN_VAD_RATIO, help=f"Minimum speech-active frame ratio for VAD filtering (default: {DEFAULT_MIN_VAD_RATIO})")
    parser.add_argument("--min-vad-segments", type=int, default=DEFAULT_MIN_VAD_SEGMENTS, help=f"Minimum number of speech segments required by VAD filtering (default: {DEFAULT_MIN_VAD_SEGMENTS})")
    parser.add_argument("--no-vad-check", dest="vad_check", action="store_false", help="Skip speech activity filtering")
    parser.add_argument("--no-audio-check", dest="audio_check", action="store_false", help="Skip RMS and clipping checks")
    parser.set_defaults(audio_check=True)
    parser.set_defaults(vad_check=True)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        print(f"Error: input dataset not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    output_audio_dir = output_dir / args.audio_dir
    metadata_csv = Path(args.csv).resolve() if args.csv else find_metadata_csv(input_dir)

    with metadata_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        print(f"Error: metadata CSV is empty: {metadata_csv}")
        sys.exit(1)

    print(f"Input dataset : {input_dir}")
    print(f"Metadata CSV  : {metadata_csv}")
    print(f"Output dataset: {output_dir}")
    print(f"Rows          : {len(rows)}")
    print("-" * 50)

    kept_rows = []
    rejected_rows = []
    seen_urls = set()
    seen_output_names = set()

    for index, row in enumerate(rows, 1):
        enriched = dict(row)
        audio_path = resolve_audio_path(input_dir, row.get("filename", ""))
        enriched["resolved_input_path"] = str(audio_path) if audio_path else ""

        url = (row.get("url") or "").strip()
        if url and url in seen_urls:
            enriched["filter_reason"] = "duplicate-url"
            rejected_rows.append(enriched)
            continue

        if audio_path.exists():
            file_size_bytes = audio_path.stat().st_size
        else:
            file_size_bytes = 0
        enriched["file_size_bytes"] = file_size_bytes

        try:
            audio_info = inspect_audio(audio_path) if audio_path.exists() else {
                "measured_duration_sec": "",
                "sample_rate": "",
                "channels": "",
                "peak": "",
                "rms": "",
                "clip_ratio": "",
            }
        except Exception as exc:
            enriched["filter_reason"] = f"audio-read-failed:{exc}"
            rejected_rows.append(enriched)
            continue

        enriched.update(audio_info)
        keep, reason = should_keep(row, audio_path, file_size_bytes, audio_info, args)
        if not keep:
            enriched["filter_reason"] = reason
            rejected_rows.append(enriched)
            continue

        output_name = safe_output_name(audio_path, seen_output_names)
        output_path = output_audio_dir / output_name
        copy_clean_audio(audio_path, output_path)

        if url:
            seen_urls.add(url)

        enriched["filename"] = str(output_path)
        enriched["filter_reason"] = "kept"
        kept_rows.append(enriched)

        if index % 25 == 0:
            print(f"Processed {index}/{len(rows)} rows...")

    base_fields = list(rows[0].keys())
    extra_fields = [
        "resolved_input_path",
        "file_size_bytes",
        "measured_duration_sec",
        "sample_rate",
        "channels",
        "peak",
        "rms",
        "clip_ratio",
        "vad_ratio",
        "vad_segments",
        "filter_reason",
    ]
    fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]

    write_csv(output_dir / "metadata.csv", kept_rows, fieldnames)
    write_csv(output_dir / "rejected.csv", rejected_rows, fieldnames)

    print("-" * 50)
    print(f"Kept     : {len(kept_rows)}")
    print(f"Rejected : {len(rejected_rows)}")
    print(f"Clean CSV: {output_dir / 'metadata.csv'}")
    print(f"Reject CSV: {output_dir / 'rejected.csv'}")
    print(f"Audio dir: {output_audio_dir}")


if __name__ == "__main__":
    main()
