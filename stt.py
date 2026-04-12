#!/usr/bin/env python3
"""
Speech-to-text transcription CLI using Hugging Face models.

Usage:
    python stt.py audio.wav
    python stt.py audio.wav --output transcript.txt
    python stt.py ./dataset/raw --output ./transcripts
    python stt.py audio.wav --model cahya/whisper-small-id
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL = "cahya/whisper-medium-id"
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".opus",
    ".ogg",
    ".webm",
    ".mp4",
    ".mpeg",
    ".mpga",
}

CACHE_DIR = Path("models")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, device: str):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    return model, processor


transcriber = None
current_model = None
current_device = None


def get_transcriber(model_name: str):
    global transcriber, current_model, current_device
    device = get_device()

    if current_model != model_name or current_device != device:
        print(f"Loading model {model_name} on {device}...")
        model, processor = load_model(model_name, device)
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=processor.model.config.max_new_tokens,
            chunk_length_s=30,
            batch_size=16,
            device=device,
        )
        current_model = model_name
        current_device = device

    return transcriber


def iter_audio_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    return sorted(
        file_path
        for file_path in path.glob(pattern)
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_output_path(
    input_path: Path, source_root: Path, output: Path, multiple: bool
) -> Path:
    if not multiple:
        if output.suffix:
            return output
        return output / f"{input_path.stem}.txt"

    relative_root = source_root if source_root.is_dir() else input_path.parent
    relative = input_path.relative_to(relative_root)
    return output / relative.with_suffix(".txt")


def transcribe_file(transcriber, audio_path: Path):
    result = transcriber(
        str(audio_path),
        return_timestamps=False,
    )
    return result["text"].strip()


def run(args) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    audio_files = iter_audio_files(input_path, recursive=args.recursive)
    if not audio_files:
        print(f"Error: No supported audio files found in {input_path}")
        sys.exit(1)

    multiple = len(audio_files) > 1 or input_path.is_dir()
    output_path = (
        Path(args.output)
        if args.output
        else (Path("transcripts") if multiple else input_path.with_suffix(".txt"))
    )
    if multiple and output_path.suffix:
        print(
            "Error: Directory input requires --output to be a directory, not a single file"
        )
        sys.exit(1)

    transcriber = get_transcriber(args.model)
    failures = 0

    for index, audio_path in enumerate(audio_files, 1):
        destination = resolve_output_path(audio_path, input_path, output_path, multiple)
        print(f"[{index}/{len(audio_files)}] Transcribing {audio_path}")
        try:
            text = transcribe_file(transcriber, audio_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            print(f"  Saved: {destination}")
        except Exception as exc:
            failures += 1
            print(f"  Failed: {exc}")

    if failures:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Speech-to-text transcription")
    parser.add_argument("input", help="Audio file or directory of audio files")
    parser.add_argument("--output", "-o", help="Output text file or output directory")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Transcription model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for audio files",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
