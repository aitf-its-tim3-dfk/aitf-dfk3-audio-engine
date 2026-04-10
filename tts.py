#!/usr/bin/env python3
"""
Batch OmniVoice TTS generation script with resumable local metadata output.

Expected input JSON format:
[
  {"text": "contoh kalimat", "label": "neutral"},
  ...
]

Example:
    python tts.py --input merged.json --output-dir synthetic_tts
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
import torchaudio
from omnivoice import OmniVoice
from tqdm.auto import tqdm


GENDERS = ["male", "female"]
AGES = ["child", "teenager", "young adult", "middle-aged", "elderly"]
PITCHES = [
    "very low pitch",
    "low pitch",
    "moderate pitch",
    "high pitch",
    "very high pitch",
]

MODEL_NAME = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
LANGUAGE_ID = "id"
AUDIO_FORMAT = "wav"
DEFAULT_BATCH_SIZE = 100
DEFAULT_INPUT = Path("scripts.json")
DEFAULT_OUTPUT_DIR = Path("tts")
MAX_GENERATION_ATTEMPTS = 3
MIN_DURATION_SECONDS = 0.8
MAX_DURATION_SECONDS = 45.0
MIN_RMS = 0.003
MAX_CLIP_RATIO = 0.03
ALLOWED_CONTROL_TOKENS = ["[pause]", "[laugh]", "[sigh]", "[cough]", "[breath]"]
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_TOKEN_RE = re.compile(r"\[(pause|laugh|sigh|cough|breath)\]")


def sample_voice_design_attr() -> tuple[str, str, str]:
    gender = random.choice(GENDERS)
    age = random.choice(AGES)

    if age == "child":
        pitch = random.choice(["high pitch", "very high pitch"])
    elif age == "elderly":
        pitch = random.choice(["low pitch", "moderate pitch"])
    else:
        pitch = random.choice(PITCHES)

    return gender, age, pitch


def sample_generation_params() -> dict:
    return {
        "speed": round(random.uniform(0.8, 1.2), 2),
        "num_step": 64,
    }


def load_model(device: str):
    dtype = torch.float16 if "cuda" in device else torch.float32
    return OmniVoice.from_pretrained(
        MODEL_NAME,
        device_map=device,
        dtype=dtype,
    )


def load_input_data(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Input JSON must be a list of items or an object with an 'items' list")


def load_processed_audio(meta_path: Path) -> set[str]:
    processed = set()
    if not meta_path.exists():
        return processed

    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            processed.add(json.loads(line)["audio"])
    return processed


def ensure_dirs(audio_dir: Path, meta_path: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_text_for_tts(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    if CJK_RE.search(cleaned):
        raise ValueError("text contains CJK characters")

    # Keep only the control tokens we explicitly support.
    tokens_found = set(re.findall(r"\[[^\]]+\]", cleaned))
    for token in tokens_found:
        if token not in ALLOWED_CONTROL_TOKENS:
            cleaned = cleaned.replace(token, " ")

    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        raise ValueError("text is empty after sanitization")
    return cleaned


def build_voice_design(item: dict) -> str:
    gender, age, pitch = sample_voice_design_attr()
    parts = [gender, age, pitch]

    duration_hint = item.get("duration_hint")
    if duration_hint == "short":
        parts.append("short utterance, concise")
    elif duration_hint == "medium":
        parts.append("natural conversational pacing")
    elif duration_hint == "long":
        parts.append("slightly sustained narration")

    return ", ".join(parts)


def ensure_audio_2d(audio_tensor: torch.Tensor) -> torch.Tensor:
    if audio_tensor.dim() == 1:
        return audio_tensor.unsqueeze(0)
    if audio_tensor.dim() == 2:
        return audio_tensor
    raise ValueError(f"unexpected audio tensor rank: {audio_tensor.dim()}")


def validate_generated_audio(audio_tensor: torch.Tensor, sample_rate: int) -> tuple[bool, str, dict]:
    audio_tensor = ensure_audio_2d(audio_tensor)
    if not torch.isfinite(audio_tensor).all():
        return False, "non-finite-values", {}

    duration = audio_tensor.shape[-1] / sample_rate
    if duration < MIN_DURATION_SECONDS:
        return False, f"too-short:{duration:.2f}s", {"duration": duration}
    if duration > MAX_DURATION_SECONDS:
        return False, f"too-long:{duration:.2f}s", {"duration": duration}

    mono = audio_tensor.mean(dim=0)
    peak = float(mono.abs().max().item()) if mono.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(mono.float() ** 2)).item()) if mono.numel() else 0.0
    clip_ratio = float((mono.abs() >= 0.999).float().mean().item()) if mono.numel() else 0.0

    if rms < MIN_RMS:
        return False, f"too-quiet:{rms:.6f}", {"duration": duration, "rms": rms, "peak": peak, "clip_ratio": clip_ratio}
    if clip_ratio > MAX_CLIP_RATIO:
        return False, f"too-clipped:{clip_ratio:.4f}", {"duration": duration, "rms": rms, "peak": peak, "clip_ratio": clip_ratio}

    return True, "ok", {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "clip_ratio": clip_ratio,
    }


def generate_valid_audio(model, text: str, item: dict):
    last_error = "generation-failed"
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        voice_design = build_voice_design(item)
        params = sample_generation_params()
        try:
            audio = model.generate(
                text=text,
                instruct=voice_design,
                **params,
            )
            audio_tensor = audio[0] if isinstance(audio, list) else audio
            audio_tensor = ensure_audio_2d(audio_tensor.detach().cpu())
            ok, reason, metrics = validate_generated_audio(audio_tensor, SAMPLE_RATE)
            if ok:
                return audio_tensor, voice_design, params, metrics
            last_error = f"attempt-{attempt}:{reason}"
        except Exception as exc:
            last_error = f"attempt-{attempt}:{exc}"

    raise RuntimeError(last_error)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch OmniVoice TTS generation with resumable local metadata output.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input JSON file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output dataset directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Progress flush interval (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"), help="Torch device / device_map for OmniVoice")
    parser.add_argument("--language-id", default=LANGUAGE_ID, help=f"Language id stored in metadata (default: {LANGUAGE_ID})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    output_dir = args.output_dir
    audio_dir = output_dir / "audio"
    meta_path = output_dir / "metadata.jsonl"

    ensure_dirs(audio_dir, meta_path)
    data = load_input_data(args.input)
    processed = load_processed_audio(meta_path)
    model = load_model(args.device)

    print(f"Using device: {args.device}")
    print(f"Loaded {len(data)} records from {args.input}")
    print(f"Output dir: {output_dir}")
    print(f"Batch size: {args.batch_size}")

    with meta_path.open("a", encoding="utf-8") as meta_file:
        for i, item in enumerate(tqdm(data)):
            try:
                text = sanitize_text_for_tts(item["text"])
            except Exception as exc:
                print("SKIP:", i, exc)
                continue
            label = item["label"]

            filename = f"{i:06d}.wav"
            rel_path = f"audio/{filename}"
            tmp_path = audio_dir / filename

            if rel_path in processed:
                continue

            try:
                audio_tensor, voice_design, params, metrics = generate_valid_audio(model, text, item)
                voice_parts = [part.strip() for part in voice_design.split(",")]
                gender = voice_parts[0] if len(voice_parts) > 0 else ""
                age = voice_parts[1] if len(voice_parts) > 1 else ""
                pitch = voice_parts[2] if len(voice_parts) > 2 else ""

                duration = metrics["duration"]
                torchaudio.save(str(tmp_path), audio_tensor, SAMPLE_RATE)

                row = {
                    "id": f"{i:06d}",
                    "text": text,
                    "label": label,
                    "audio": rel_path,
                    "format": AUDIO_FORMAT,
                    "duration": duration,
                    "sample_rate": SAMPLE_RATE,
                    "model": MODEL_NAME,
                    "instruct": voice_design,
                    "gender": gender,
                    "age": age,
                    "pitch": pitch,
                    "language_id": args.language_id,
                    "rms": metrics["rms"],
                    "peak": metrics["peak"],
                    "clip_ratio": metrics["clip_ratio"],
                    **params,
                }

                meta_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed.add(rel_path)

                if (i + 1) % args.batch_size == 0:
                    meta_file.flush()
                    os.fsync(meta_file.fileno())

            except Exception as exc:
                print("FAIL:", i, exc)
                continue

    print(f"Done. Audio saved in {audio_dir}")
    print(f"Metadata saved in {meta_path}")


if __name__ == "__main__":
    main()
