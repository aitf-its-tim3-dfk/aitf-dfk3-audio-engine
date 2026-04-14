#!/usr/bin/env python3
"""
Filter a scraped audio dataset into a cleaner dataset directory.

The script reads a dataset directory produced by scraper.py, validates audio files,
applies a few practical quality heuristics, copies accepted files into a new
dataset directory, and writes kept/rejected CSV reports.

Audio processing (optional):
- Format standardization (sample rate, channels, codec)
- Silence trimming (leading/trailing silence)
- Loudness normalization (LUFS target)
- Length normalization (pad/truncate to fixed duration)
- Noise filtering (simple spectral gating)
- Label validation

Examples:
    python cleaner.py ./dataset --output ./cleaned
    python cleaner.py ./dataset --output ./cleaned --min-duration 2 --max-duration 180
    python cleaner.py ./dataset --output ./cleaned --max-size-mb 100 --no-audio-check
    python cleaner.py ./dataset --output ./cleaned --normalize-loudness --target-lufs=-16 --trim-silence
    python cleaner.py ./dataset --output ./cleaned --standardize-format --target-sample-rate 16000 --target-channels 1
    python cleaner.py ./dataset --output ./cleaned --length-normalize --target-length 30
"""

import argparse
import audioop
import csv
import hashlib
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_AUDIO_DIR = "cleaned"
DEFAULT_MIN_DURATION = 3
DEFAULT_MAX_DURATION = None
DEFAULT_MIN_SIZE_MB = None
DEFAULT_MAX_SIZE_MB = None
DEFAULT_MIN_RMS = 200.0
DEFAULT_MAX_CLIP_RATIO = 0.02
DEFAULT_MIN_VAD_RATIO = 0.15
DEFAULT_MIN_VAD_SEGMENTS = 1

DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_TARGET_CHANNELS = 1
DEFAULT_TARGET_LUFS = -16.0
DEFAULT_TARGET_LENGTH = None
DEFAULT_SILENCE_THRESHOLD_DB = -40
DEFAULT_SILENCE_MIN_DURATION = 0.3

SUPPORTED_EXTENSIONS = {".wav"}
CSV_CANDIDATES = ("metadata.csv",)

VALID_LABELS = {"ujaran kebencian", "fitnah", "disinformasi", "neutral", "fake", "real"}

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


def resolve_audio_path(dataset_dir: Path, filename_value: str) -> Path | None:
    raw_value = (filename_value or "").strip()
    if not raw_value:
        return None
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (dataset_dir / path).resolve()


def build_dedupe_key(row: dict, audio_path: Path) -> str:
    """Treat chunked rows as distinct while still filtering true duplicates."""
    if audio_path:
        resolved = str(audio_path)
        if resolved:
            return f"path:{resolved}"

    filename = (row.get("filename") or "").strip()
    if filename:
        return f"filename:{filename}"

    url = (row.get("url") or "").strip()
    chunk_index = (row.get("chunk_index") or "").strip()
    if url and chunk_index:
        return f"url:{url}#chunk:{chunk_index}"
    if url:
        return f"url:{url}"

    return ""


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


def get_vad_segments(audio_path: Path) -> list[dict]:
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
    return speech_timestamps


def extract_segment(
    input_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        str(start_sec),
        "-to",
        str(end_sec),
        "-c",
        "copy",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def run_ffmpeg(args_list: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return result


def standardize_audio(
    input_path: Path,
    target_sample_rate: int,
    target_channels: int,
    output_path: Path,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(target_sample_rate),
        "-ac",
        str(target_channels),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def trim_silence(
    input_path: Path,
    silence_threshold_db: float,
    min_silence_duration: float,
    output_path: Path,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"silenceremove=start_threshold={silence_threshold_db}dB:stop_threshold={silence_threshold_db}dB:detection=peak:start_duration={min_silence_duration}:stop_duration={min_silence_duration}",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def normalize_loudness(
    input_path: Path,
    target_lufs: float,
    output_path: Path,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def normalize_length(
    input_path: Path,
    target_length_sec: float,
    output_path: Path,
) -> None:
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
    current_duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0

    if current_duration < target_length_sec:
        pad_duration = target_length_sec - current_duration
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            f"apad=pad_duration={pad_duration}",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-t",
            str(target_length_sec),
            str(output_path),
        ]
    run_ffmpeg(cmd)


def denoise_audio(
    input_path: Path,
    output_path: Path,
    method: str = "afftdn",
    model_path: Path | None = None,
) -> None:
    if method == "rnnoise":
        if model_path and model_path.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-af",
                f"arnndl=model={str(model_path)}",
                str(output_path),
            ]
        else:
            print("    Warning: RNNoise model not found, falling back to afftdn")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-af",
                "afftdn=nf=-25",
                str(output_path),
            ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            "afftdn=nf=-25",
            str(output_path),
        ]
    run_ffmpeg(cmd)


def process_audio(
    input_path: Path,
    output_path: Path,
    args,
) -> tuple[bool, str]:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not any(
            [
                args.standardize_format,
                args.trim_silence,
                args.normalize_loudness,
                args.denoise,
                args.length_normalize,
            ]
        ):
            shutil.copy2(input_path, output_path)
            return True, ""

        temp_paths: list[Path] = []

        def make_temp_wav() -> Path:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            temp_paths.append(tmp_path)
            return tmp_path

        current = input_path

        def ensure_standardized() -> Path:
            nonlocal current
            if current != input_path:
                return current
            standardized = make_temp_wav()
            standardize_audio(
                current,
                args.target_sample_rate,
                args.target_channels,
                standardized,
            )
            current = standardized
            return current

        if args.standardize_format:
            standardized = make_temp_wav()
            standardize_audio(
                current,
                args.target_sample_rate,
                args.target_channels,
                standardized,
            )
            current = standardized

        if args.trim_silence:
            ensure_standardized()
            trimmed = make_temp_wav()
            trim_silence(
                current,
                args.silence_threshold_db,
                args.silence_min_duration,
                trimmed,
            )
            current = trimmed

        if args.normalize_loudness:
            ensure_standardized()
            normalized = make_temp_wav()
            normalize_loudness(current, args.target_lufs, normalized)
            current = normalized

        if args.denoise:
            ensure_standardized()
            denoised = make_temp_wav()
            model_path = Path(args.denoise_model) if args.denoise_model else None
            denoise_audio(
                current, denoised, method=args.denoise_method, model_path=model_path
            )
            current = denoised

        if args.length_normalize:
            ensure_standardized()
            length_normalized = make_temp_wav()
            normalize_length(current, args.target_length, length_normalized)
            current = length_normalized

        shutil.copy2(current, output_path)
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        for temp_path in locals().get("temp_paths", []):
            if temp_path.exists():
                temp_path.unlink()


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
        "vad_error": "",
    }

    with wave.open(str(audio_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        info["measured_duration_sec"] = (
            round(frames / float(sample_rate), 3) if sample_rate else 0.0
        )
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
                sample = int.from_bytes(
                    raw[offset : offset + sample_width], byteorder="little", signed=True
                )
                total_samples += 1
                if abs(sample) >= threshold:
                    clipped_samples += 1

            clip_ratio = (clipped_samples / total_samples) if total_samples else 0.0

            info["peak"] = round(peak, 3)
            info["rms"] = round(rms, 3)
            info["clip_ratio"] = round(clip_ratio, 6)

    try:
        vad_ratio, vad_segments = compute_silero_vad(
            audio_path,
            float(info["measured_duration_sec"] or 0.0),
        )
        info["vad_ratio"] = round(vad_ratio, 6)
        info["vad_segments"] = vad_segments
    except Exception:
        # Keep the cleaner usable even when optional VAD dependencies are missing.
        info["vad_ratio"] = ""
        info["vad_segments"] = ""
        info["vad_error"] = str(sys.exc_info()[1] or "")

    return info


def should_keep(
    row: dict, audio_path: Path, file_size_bytes: int, audio_info: dict, args
) -> tuple[bool, str]:
    status = (row.get("status") or "").strip().lower()
    if status and status != "ok":
        return False, f"status={status}"
    if not audio_path.exists():
        return False, "missing-file"
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"unsupported-extension:{audio_path.suffix.lower()}"

    if args.validate_labels:
        label = (row.get("weak_label") or row.get("label") or "").strip().lower()
        if label and label not in VALID_LABELS:
            return False, f"invalid-label:{label}"

    file_size_mb = file_size_bytes / (1024 * 1024)
    if args.min_size_mb is not None and file_size_mb < args.min_size_mb:
        return False, f"too-small:{file_size_mb:.2f}MB"
    if args.max_size_mb is not None and file_size_mb > args.max_size_mb:
        return False, f"too-large:{file_size_mb:.2f}MB"

    duration = float(audio_info.get("measured_duration_sec") or 0.0)
    if args.min_duration is not None and duration < args.min_duration:
        return False, f"too-short:{duration:.2f}s"
    if args.max_duration is not None and duration > args.max_duration:
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
        if args.vad_check and audio_info.get("vad_ratio") not in ("", None):
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
    parser = argparse.ArgumentParser(
        description="Filter an audio dataset into a cleaner dataset directory."
    )
    parser.add_argument("input", help="Input dataset directory")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dataset directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional metadata CSV path. Defaults to metadata.csv/results.csv in the dataset directory",
    )
    parser.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
        help=f"Output audio subdirectory name (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help=f"Minimum audio duration in seconds (default: {DEFAULT_MIN_DURATION})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help=f"Maximum audio duration in seconds (default: {DEFAULT_MAX_DURATION})",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=DEFAULT_MIN_SIZE_MB,
        help=f"Minimum file size in MB (default: {DEFAULT_MIN_SIZE_MB})",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=DEFAULT_MAX_SIZE_MB,
        help=f"Maximum file size in MB (default: {DEFAULT_MAX_SIZE_MB})",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=DEFAULT_MIN_RMS,
        help=f"Minimum RMS loudness for preview audio (default: {DEFAULT_MIN_RMS})",
    )
    parser.add_argument(
        "--max-clip-ratio",
        type=float,
        default=DEFAULT_MAX_CLIP_RATIO,
        help=f"Maximum clipped-sample ratio for preview audio (default: {DEFAULT_MAX_CLIP_RATIO})",
    )
    parser.add_argument(
        "--min-vad-ratio",
        type=float,
        default=DEFAULT_MIN_VAD_RATIO,
        help=f"Minimum speech-active frame ratio for VAD filtering (default: {DEFAULT_MIN_VAD_RATIO})",
    )
    parser.add_argument(
        "--min-vad-segments",
        type=int,
        default=DEFAULT_MIN_VAD_SEGMENTS,
        help=f"Minimum number of speech segments required by VAD filtering (default: {DEFAULT_MIN_VAD_SEGMENTS})",
    )
    parser.add_argument(
        "--no-vad-check",
        dest="vad_check",
        action="store_false",
        help="Skip speech activity filtering",
    )
    parser.add_argument(
        "--no-audio-check",
        dest="audio_check",
        action="store_false",
        help="Skip RMS and clipping checks",
    )
    parser.add_argument(
        "--standardize-format",
        action="store_true",
        help="Standardize audio format (sample rate, channels)",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help=f"Target sample rate for standardization (default: {DEFAULT_TARGET_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--target-channels",
        type=int,
        default=DEFAULT_TARGET_CHANNELS,
        help=f"Target number of channels for standardization (default: {DEFAULT_TARGET_CHANNELS})",
    )
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim leading/trailing silence from audio",
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD_DB,
        help=f"Silence threshold in dB for trimming (default: {DEFAULT_SILENCE_THRESHOLD_DB})",
    )
    parser.add_argument(
        "--silence-min-duration",
        type=float,
        default=DEFAULT_SILENCE_MIN_DURATION,
        help=f"Minimum silence duration in seconds to trim (default: {DEFAULT_SILENCE_MIN_DURATION})",
    )
    parser.add_argument(
        "--normalize-loudness",
        action="store_true",
        help="Normalize loudness to target LUFS",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=DEFAULT_TARGET_LUFS,
        help=f"Target LUFS for loudness normalization (default: {DEFAULT_TARGET_LUFS})",
    )
    parser.add_argument(
        "--length-normalize",
        action="store_true",
        help="Normalize audio length to fixed duration (pad or truncate)",
    )
    parser.add_argument(
        "--target-length",
        type=float,
        default=DEFAULT_TARGET_LENGTH,
        help="Target length in seconds for length normalization",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply noise reduction to audio",
    )
    parser.add_argument(
        "--denoise-method",
        type=str,
        default="afftdn",
        choices=["afftdn", "rnnoise"],
        help="Noise reduction method (default: afftdn)",
    )
    parser.add_argument(
        "--denoise-model",
        type=str,
        default=None,
        help="Path to RNNoise model file (.rnnn). Required if --denoise-method=rnnoise",
    )
    parser.add_argument(
        "--validate-labels",
        action="store_true",
        help="Validate that labels are from the allowed set",
    )
    parser.add_argument(
        "--split-vad-segments",
        action="store_true",
        help="Split audio into separate VAD segments and save with timestamps",
    )
    parser.set_defaults(audio_check=True)
    parser.set_defaults(vad_check=True)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        print(f"Error: input dataset not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    output_audio_dir = output_dir / args.audio_dir
    metadata_csv = (
        Path(args.csv).resolve() if args.csv else find_metadata_csv(input_dir)
    )

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
    if args.vad_check:
        try:
            get_silero_vad_model()
        except Exception as exc:
            args.vad_check = False
            print(f"VAD disabled  : {exc}")

    processing = []
    if args.standardize_format:
        processing.append(
            f"format_sr{args.target_sample_rate}_ch{args.target_channels}"
        )
    if args.trim_silence:
        processing.append("trim_silence")
    if args.normalize_loudness:
        processing.append(f"normalize_lufs{args.target_lufs}")
    if args.denoise:
        processing.append("denoise")
    if args.length_normalize:
        processing.append(f"length{args.target_length}s")
    if args.validate_labels:
        processing.append("validate_labels")

    if processing:
        print(f"Processing   : {', '.join(processing)}")
    else:
        print(f"Processing   : copy only")

    print("-" * 50)

    kept_rows = []
    rejected_rows = []
    seen_entries = set()
    seen_output_names = set()

    for index, row in enumerate(rows, 1):
        enriched = dict(row)
        audio_path = resolve_audio_path(input_dir, row.get("filename", ""))
        enriched["resolved_input_path"] = str(audio_path) if audio_path else ""

        dedupe_key = build_dedupe_key(row, audio_path)
        if dedupe_key and dedupe_key in seen_entries:
            enriched["filter_reason"] = "duplicate-entry"
            rejected_rows.append(enriched)
            continue

        if audio_path and audio_path.exists():
            file_size_bytes = audio_path.stat().st_size
        else:
            file_size_bytes = 0
        enriched["file_size_bytes"] = file_size_bytes

        try:
            audio_info = (
                inspect_audio(audio_path)
                if audio_path and audio_path.exists()
                else {
                    "measured_duration_sec": "",
                    "sample_rate": "",
                    "channels": "",
                    "peak": "",
                    "rms": "",
                    "clip_ratio": "",
                }
            )
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

        if args.split_vad_segments and audio_path and audio_path.exists():
            print(f"Processing VAD split for: {audio_path.name}")
            try:
                vad_segments = get_vad_segments(audio_path)
                print(f"  Found {len(vad_segments)} VAD segments")
                if vad_segments:
                    segments_kept = 0
                    for seg_idx, seg in enumerate(vad_segments):
                        seg_start = float(seg.get("start", 0.0))
                        seg_end = float(seg.get("end", 0.0))
                        seg_duration = seg_end - seg_start

                        print(
                            f"  Segment {seg_idx}: {seg_start:.2f}s - {seg_end:.2f}s ({seg_duration:.2f}s)"
                        )

                        if seg_duration < args.min_duration:
                            print(f"    Skipped (too short, min={args.min_duration}s)")
                            continue

                        seg_enriched = dict(enriched)
                        base_name = Path(output_name).stem
                        seg_name = f"{base_name}_seg{seg_idx:03d}.wav"
                        seg_output_path = output_audio_dir / seg_name

                        extract_segment(audio_path, seg_start, seg_end, seg_output_path)

                        seg_info = inspect_audio(seg_output_path)
                        seg_enriched["filename"] = str(seg_output_path)
                        seg_enriched["file_size_bytes"] = seg_output_path.stat().st_size
                        seg_enriched["sample_rate"] = seg_info.get("sample_rate", "")
                        seg_enriched["channels"] = seg_info.get("channels", "")
                        seg_enriched["peak"] = seg_info.get("peak", "")
                        seg_enriched["rms"] = seg_info.get("rms", "")
                        seg_enriched["clip_ratio"] = seg_info.get("clip_ratio", "")
                        seg_enriched["segment_index"] = seg_idx
                        seg_enriched["segment_start_sec"] = seg_start
                        seg_enriched["segment_end_sec"] = seg_end
                        seg_enriched["measured_duration_sec"] = round(seg_duration, 3)
                        seg_enriched["filter_reason"] = "kept"
                        kept_rows.append(seg_enriched)
                        segments_kept += 1
                        print(
                            f"    Saved: {seg_name} ({seg_info.get('rms', '?')} RMS, {seg_info.get('peak', '?')} peak)"
                        )

                    if segments_kept == 0:
                        print(
                            f"  No segments met min-duration, falling back to full audio"
                        )
                        output_path = output_audio_dir / output_name
                        processed, processing_error = process_audio(
                            audio_path, output_path, args
                        )
                        if not processed:
                            enriched["filter_reason"] = (
                                f"audio-processing-failed:{processing_error}"
                            )
                            rejected_rows.append(enriched)
                            continue

                        enriched["filename"] = str(output_path)
                        enriched["filter_reason"] = "kept"
                        kept_rows.append(enriched)
                else:
                    print(f"  No speech detected, copying full audio")
                    output_path = output_audio_dir / output_name
                    processed, processing_error = process_audio(
                        audio_path, output_path, args
                    )
                    if not processed:
                        enriched["filter_reason"] = (
                            f"audio-processing-failed:{processing_error}"
                        )
                        rejected_rows.append(enriched)
                        continue

                    if dedupe_key:
                        seen_entries.add(dedupe_key)

                    enriched["filename"] = str(output_path)
                    enriched["filter_reason"] = "kept"
                    kept_rows.append(enriched)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                enriched["filter_reason"] = f"vad-split-failed:{exc}"
                rejected_rows.append(enriched)
                continue
        else:
            output_name = safe_output_name(audio_path, seen_output_names)
            output_path = output_audio_dir / output_name

            processed, processing_error = process_audio(audio_path, output_path, args)
            if not processed:
                if processing_error:
                    enriched["filter_reason"] = (
                        f"audio-processing-failed:{processing_error}"
                    )
                else:
                    enriched["filter_reason"] = "audio-processing-failed"
                rejected_rows.append(enriched)
                continue

            if dedupe_key:
                seen_entries.add(dedupe_key)

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
        "vad_error",
        "filter_reason",
        "segment_index",
        "segment_start_sec",
        "segment_end_sec",
    ]
    fieldnames = base_fields + [
        field for field in extra_fields if field not in base_fields
    ]

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
