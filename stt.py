#!/usr/bin/env python3
"""
Speech-to-text transcription CLI.

Usage:
    python stt.py audio.wav
    python stt.py audio.wav --output transcript.txt
    python stt.py ./dataset/raw --output ./transcripts --language id
    python stt.py audio.wav --json
"""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini-transcribe"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".opus", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga"}


def iter_audio_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    return sorted(
        file_path
        for file_path in path.glob(pattern)
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_output_path(input_path: Path, source_root: Path, output: Path, multiple: bool) -> Path:
    if not multiple:
        if output.suffix:
            return output
        return output / f"{input_path.stem}.txt"

    relative_root = source_root if source_root.is_dir() else input_path.parent
    relative = input_path.relative_to(relative_root)
    return output / relative.with_suffix(".txt")


def transcribe_file(client: OpenAI, audio_path: Path, model: str, language: str | None, prompt: str | None, response_format: str):
    with audio_path.open("rb") as audio_file:
        return client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language or None,
            prompt=prompt or None,
            response_format=response_format,
        )


def save_text_output(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_json_output(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    output_path = Path(args.output) if args.output else (Path("transcripts") if multiple else input_path.with_suffix(".txt"))
    if multiple and output_path.suffix:
        print("Error: Directory input requires --output to be a directory, not a single file")
        sys.exit(1)

    client = OpenAI()
    failures = 0

    for index, audio_path in enumerate(audio_files, 1):
        destination = resolve_output_path(audio_path, input_path, output_path, multiple)
        print(f"[{index}/{len(audio_files)}] Transcribing {audio_path}")
        try:
            result = transcribe_file(
                client=client,
                audio_path=audio_path,
                model=args.model,
                language=args.language,
                prompt=args.prompt,
                response_format="verbose_json" if args.json else "text",
            )
            if args.json:
                payload = result.model_dump()
                save_json_output(payload, destination.with_suffix(".json"))
                text = (getattr(result, "text", None) or payload.get("text", "")).strip()
                if text:
                    save_text_output(text, destination)
                    print(f"  Saved: {destination}")
                else:
                    print(f"  Saved JSON: {destination.with_suffix('.json')}")
            else:
                save_text_output(str(result).strip(), destination)
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Transcription model (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", default=None, help="Optional language hint, e.g. id or en")
    parser.add_argument("--prompt", default=None, help="Optional prompt to guide transcription")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for audio files")
    parser.add_argument("--json", action="store_true", help="Also save verbose JSON output alongside text")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
