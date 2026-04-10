#!/usr/bin/env python3
"""
Audio augmentation script for dataset expansion.
Given a dataset folder from scraper.py, augments the audio files and creates
a new dataset folder with augmented versions.

Usage:
    python augment.py ./dataset --output ./augmented
    python augment.py ./dataset --output ./augmented --augmentations pitch speed noise
    python augment.py ./dataset --output ./augmented --num-versions 3
    python augment.py ./dataset --output ./augmented --pitch-range -3 3 --speed-range 0.8 1.2
"""

import argparse
import numpy as np
import csv
import os
import random
import subprocess
import sys
import librosa
import soundfile as sf
from pathlib import Path


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "augmented"
DEFAULT_NUM_VERSIONS = 1  # Number of augmented versions per audio

# Available separate augmentation types
AUGMENTATIONS = [
    "pitch_up",
    "pitch_down",
    "speed_up",
    "speed_down",
    "noise_white",
    "noise_brown",
    "noise_pink",
    "reverb",
    "echo",
    "volume_up",
    "volume_down",
    "time_stretch_slow",
    "time_stretch_fast",
    "rvc",
]


# ── Augmentation Functions ─────────────────────────────────────────────────


def pitch_shift(audio, sr, n_steps):
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio, rate):
    """Stretch audio by rate (0.5 = slower, 2.0 = faster)."""
    return librosa.effects.time_stretch(audio, rate=rate)


def add_white_noise(audio, level):
    """Add white noise to audio."""
    noise = np.random.randn(len(audio)) * level
    return audio + noise


def add_brown_noise(audio, level):
    """Add brown (red) noise to audio."""
    noise = np.random.randn(len(audio))
    brown = np.cumsum(noise) / 100
    brown = brown / np.max(np.abs(brown)) * level * len(audio) * 0.01
    return audio + brown[: len(audio)]


def add_pink_noise(audio, level):
    """Add pink noise to audio."""
    noise = np.random.randn(len(audio))
    pink = np.zeros_like(noise)
    b0, b1, b2, b3, b4, b5, b6 = 0, 0, 0, 0, 0, 0, 0
    for i in range(len(noise)):
        white = noise[i]
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        b3 = 0.86650 * b3 + white * 0.3104856
        b4 = 0.55000 * b4 + white * 0.5329522
        b5 = -0.7616 * b5 - white * 0.0168980
        pink[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11
        b6 = white * 0.115926
    pink = pink / np.max(np.abs(pink)) * level
    return audio + pink


def apply_reverb(audio, sr, room_size=0.5):
    """Simple reverb effect using delay."""
    delay_samples = int(sr * 0.1 * room_size)
    reverb_audio = audio.copy()
    for i in range(delay_samples, len(audio)):
        reverb_audio[i] += audio[i - delay_samples] * 0.3 * room_size
    return reverb_audio


def apply_echo(audio, sr, delay_sec=0.3, decay=0.5):
    """Apply echo effect."""
    delay_samples = int(sr * delay_sec)
    echo_audio = audio.copy()
    for i in range(delay_samples, len(audio)):
        echo_audio[i] = audio[i] + echo_audio[i - delay_samples] * decay
    return echo_audio


def change_volume(audio, level):
    """Change volume by multiplier."""
    return audio * level


# ── Augmentation Map ─────────────────────────────────────────────────────────


def get_augmentation_func(aug_type):
    """Get augmentation function and default parameter range."""
    aug_map = {
        # Pitch shifts
        "pitch_up": (pitch_shift, {"n_steps": (1, 3)}),
        "pitch_down": (pitch_shift, {"n_steps": (-3, -1)}),
        # Speed changes
        "speed_up": (time_stretch, {"rate": (1.05, 1.2)}),
        "speed_down": (time_stretch, {"rate": (0.8, 0.95)}),
        # Noise types
        "noise_white": (add_white_noise, {"level": (0.003, 0.01)}),
        "noise_brown": (add_brown_noise, {"level": (0.03, 0.1)}),
        "noise_pink": (add_pink_noise, {"level": (0.03, 0.1)}),
        # Effects
        "reverb": (apply_reverb, {"room_size": (0.3, 0.7)}),
        "echo": (apply_echo, {"delay_sec": (0.2, 0.5), "decay": (0.3, 0.6)}),
        # Volume
        "volume_up": (change_volume, {"level": (1.1, 1.5)}),
        "volume_down": (change_volume, {"level": (0.5, 0.9)}),
        # Time stretch (different from speed)
        "time_stretch_slow": (time_stretch, {"rate": (0.7, 0.9)}),
        "time_stretch_fast": (time_stretch, {"rate": (1.1, 1.3)}),
    }
    return aug_map.get(aug_type, (None, {}))


def sample_augmentation_params(ranges):
    """Sample augmentation parameters from either scalar values or min/max tuples."""
    params = {}
    for key, value in ranges.items():
        if isinstance(value, tuple) and len(value) == 2:
            params[key] = random.uniform(value[0], value[1])
        else:
            params[key] = value
    return params


def run_rvc_augmentation(input_path, output_path, rvc_config):
    """Run an external RVC inference command using a user-provided template."""
    if not rvc_config or not rvc_config.get("command_template"):
        raise ValueError("RVC augmentation requires --rvc-command-template")
    if not rvc_config.get("model"):
        raise ValueError("RVC augmentation requires --rvc-model")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pitch_min, pitch_max = rvc_config.get("pitch_range", (-2, 2))
    pitch = random.randint(int(round(pitch_min)), int(round(pitch_max)))

    placeholders = {
        "input": str(Path(input_path).resolve()),
        "output": str(output_path.resolve()),
        "model": str(Path(rvc_config["model"]).resolve()),
        "index": str(Path(rvc_config["index"]).resolve())
        if rvc_config.get("index")
        else "",
        "pitch": str(pitch),
        "device": rvc_config.get("device", ""),
        "f0_method": rvc_config.get("f0_method", ""),
    }
    command = rvc_config["command_template"].format(**placeholders)
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"RVC command failed ({completed.returncode}): {stderr}")
    if not output_path.exists():
        raise RuntimeError("RVC command finished but did not create the output file")
    return True


def augment_audio(
    input_path, output_path, aug_type, param_ranges=None, runtime_config=None
):
    """Apply augmentation to audio file and save."""
    param_ranges = param_ranges or {}

    if aug_type == "rvc":
        try:
            return run_rvc_augmentation(input_path, output_path, runtime_config or {})
        except Exception as e:
            print(f"  Error augmenting {input_path} with RVC: {e}")
            return False

    func, default_ranges = get_augmentation_func(aug_type)
    if func is None:
        print(f"  Unknown augmentation: {aug_type}")
        return False

    try:
        ranges = {**default_ranges, **param_ranges}
        audio, sr = librosa.load(input_path, sr=None)
        params = sample_augmentation_params(ranges)
        audio = func(audio, sr, **params)
        audio = np.clip(audio, -1.0, 1.0)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sr)
        return True
    except Exception as e:
        print(f"  Error augmenting {input_path}: {e}")
        return False


# ── Core Functions ───────────────────────────────────────────────────────────


def find_audio_files(dataset_path):
    """Find all audio files in dataset folder."""
    audio_files = []
    audios_dir = dataset_path / "audios"

    if not audios_dir.exists():
        print(f"Error: audios directory not found in {dataset_path}")
        return audio_files

    for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.opus"]:
        audio_files.extend(audios_dir.glob(ext))

    return audio_files


def generate_augmented_filename(original_name, aug_type, version):
    """Generate filename for augmented audio."""
    stem = Path(original_name).stem
    ext = Path(original_name).suffix
    return f"{stem}_aug{version}_{aug_type}{ext}"


def process_dataset(args):
    """Main function to augment dataset."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV file found in {input_path}")
        sys.exit(1)

    csv_path = csv_files[0]
    print(f"Input dataset: {input_path}")
    print(f"Output dataset: {output_path}")
    print(f"CSV file: {csv_path}")

    output_audios = output_path / "audios"
    output_audios.mkdir(parents=True, exist_ok=True)

    original_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_rows = list(reader)

    print(f"Found {len(original_rows)} entries in CSV")

    successful_rows = [r for r in original_rows if r.get("status") == "ok"]
    print(f"Found {len(successful_rows)} successful entries")

    audio_files = {}
    for row in successful_rows:
        filename = row.get("filename", "")
        if filename:
            audio_path = Path(filename)
            if audio_path.exists():
                audio_files[audio_path.name] = audio_path

    print(f"Found {len(audio_files)} audio files")

    if not audio_files:
        print("Error: No audio files found")
        sys.exit(1)

    augmentations = args.augmentations if args.augmentations else AUGMENTATIONS
    num_versions = args.num_versions

    print(f"\nAugmentations: {augmentations}")
    print(f"Versions per audio: {num_versions}")
    print("-" * 50)

    augmented_count = 0
    failed_count = 0
    new_csv_rows = []

    for audio_name, audio_path in audio_files.items():
        available_augs = augmentations * (num_versions + 1)
        selected_augs = random.sample(
            available_augs, min(num_versions, len(available_augs))
        )

        for version, aug_type in enumerate(selected_augs, 1):
            aug_filename = generate_augmented_filename(audio_name, aug_type, version)
            output_audio_path = output_audios / aug_filename

            if input_path == output_path:
                if not audio_path.exists() or audio_path != output_audio_path:
                    import shutil
                    shutil.copy2(audio_path, output_audio_path)

            success = augment_audio(
                audio_path,
                output_audio_path,
                aug_type,
                param_ranges=args.param_ranges.get(aug_type),
                runtime_config=args.rvc_config,
            )

            if success:
                augmented_count += 1
                new_csv_rows.append(
                    {
                        "url": "",
                        "platform": "",
                        "title": f"{audio_name} (augmented: {aug_type})",
                        "uploader": "",
                        "duration_sec": "",
                        "filename": str(output_audio_path),
                        "status": "ok",
                        "error": "",
                        "scraped_at": "",
                    }
                )
            else:
                failed_count += 1

        if input_path == output_path:
            new_csv_rows.append(
                {
                    "url": "",
                    "platform": "",
                    "title": audio_name,
                    "uploader": "",
                    "duration_sec": "",
                    "filename": str(audio_path),
                    "status": "ok",
                    "error": "",
                    "scraped_at": "",
                }
            )

    output_csv_path = output_path / "augmented_results.csv"
    fieldnames = [
        "url",
        "platform",
        "title",
        "uploader",
        "duration_sec",
        "filename",
        "status",
        "error",
        "scraped_at",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_csv_rows)

    print("-" * 50)
    print(f"Augmentation complete!")
    print(f"  Original files: {len(audio_files)}")
    print(f"  Augmented files: {augmented_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total entries in CSV: {len(new_csv_rows)}")
    print(f"\nOutput CSV: {output_csv_path}")
    print(f"Output audios: {output_audios}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Augment audio dataset for training.")
    p.add_argument("input", help="Input dataset directory (from scraper.py)")
    p.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dataset directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--augmentations",
        nargs="+",
        choices=AUGMENTATIONS,
        default=None,
        help=f"Augmentation types to apply (default: all)",
    )
    p.add_argument(
        "--num-versions",
        type=int,
        default=DEFAULT_NUM_VERSIONS,
        help=f"Number of augmented versions per audio (default: {DEFAULT_NUM_VERSIONS})",
    )

    # Parameter ranges for each augmentation type
    p.add_argument(
        "--pitch-range",
        type=float,
        nargs=2,
        default=None,
        help="Pitch shift range in semitones (min max)",
    )
    p.add_argument(
        "--speed-range",
        type=float,
        nargs=2,
        default=None,
        help="Speed change range (min max)",
    )
    p.add_argument(
        "--noise-level",
        type=float,
        nargs=2,
        default=None,
        help="Noise level range (min max)",
    )
    p.add_argument(
        "--volume-range",
        type=float,
        nargs=2,
        default=None,
        help="Volume change range (min max)",
    )
    p.add_argument(
        "--reverb-range",
        type=float,
        nargs=2,
        default=None,
        help="Reverb room size range (min max)",
    )
    p.add_argument(
        "--rvc-command-template",
        default=None,
        help="Command template for external RVC inference. Use placeholders like {input}, {output}, {model}, {index}, {pitch}, {device}, {f0_method}",
    )
    p.add_argument(
        "--rvc-model", default=None, help="Path to the RVC model weights file"
    )
    p.add_argument(
        "--rvc-index", default=None, help="Optional path to the RVC index file"
    )
    p.add_argument(
        "--rvc-pitch-range",
        type=float,
        nargs=2,
        default=(-2, 2),
        help="RVC pitch shift range in semitones (default: -2 2)",
    )
    p.add_argument(
        "--rvc-device",
        default="",
        help="Optional device placeholder value for the RVC command template",
    )
    p.add_argument(
        "--rvc-f0-method",
        default="",
        help="Optional f0 method placeholder value for the RVC command template",
    )

    return p.parse_args()


def main():
    args = parse_args()

    param_ranges = {}
    if args.pitch_range:
        param_ranges["pitch_up"] = {
            "n_steps": (args.pitch_range[0], args.pitch_range[1])
        }
        param_ranges["pitch_down"] = {
            "n_steps": (-args.pitch_range[1], -args.pitch_range[0])
        }
    if args.speed_range:
        param_ranges["speed_up"] = {"rate": (args.speed_range[0], args.speed_range[1])}
        param_ranges["speed_down"] = {
            "rate": (2 - args.speed_range[1], 2 - args.speed_range[0])
        }
    if args.noise_level:
        param_ranges["noise_white"] = {
            "level": (args.noise_level[0], args.noise_level[1])
        }
        param_ranges["noise_brown"] = {
            "level": (args.noise_level[0] * 10, args.noise_level[1] * 10)
        }
        param_ranges["noise_pink"] = {
            "level": (args.noise_level[0] * 10, args.noise_level[1] * 10)
        }
    if args.volume_range:
        param_ranges["volume_up"] = {
            "level": (args.volume_range[0], args.volume_range[1])
        }
        param_ranges["volume_down"] = {
            "level": (2 - args.volume_range[1], 2 - args.volume_range[0])
        }
    if args.reverb_range:
        param_ranges["reverb"] = {
            "room_size": (args.reverb_range[0], args.reverb_range[1])
        }

    args.rvc_config = {
        "command_template": args.rvc_command_template,
        "model": args.rvc_model,
        "index": args.rvc_index,
        "pitch_range": tuple(args.rvc_pitch_range) if args.rvc_pitch_range else (-2, 2),
        "device": args.rvc_device,
        "f0_method": args.rvc_f0_method,
    }
    if args.augmentations and "rvc" in args.augmentations:
        if not args.rvc_command_template or not args.rvc_model:
            print(
                "Error: rvc augmentation requires --rvc-command-template and --rvc-model"
            )
            sys.exit(1)

    args.param_ranges = param_ranges

    process_dataset(args)


if __name__ == "__main__":
    main()
