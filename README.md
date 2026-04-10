# Audio Engine

Python tools for building an Indonesian social-video audio dataset: collect URLs, download audio, optionally augment it, transcribe speech, and synthesize TTS audio.

## Scripts

- `crawler.py`: collects candidate social-video URLs and weak labels into `data.csv`
- `scraper.py`: downloads audio with `yt-dlp` and writes a results CSV
- `augment.py`: creates augmented audio variants, including optional external RVC runs
- `stt.py`: speech-to-text for audio files or folders using the OpenAI transcription API
- `tts.py`: text-to-speech with OmniVoice
- `scripter.py`: generates spoken-style text prompts/scripts for TTS workflows

## Setup

### 1. Export cookies first

For Instagram, TikTok, X, or Facebook, log in in your browser and export Netscape-format cookie files.

Put them here:

```text
cookies/instagram_cookies.txt
cookies/tiktok_cookies.txt
cookies/twitter_cookies.txt
cookies/facebook_cookies.txt
```

Notes:
- `scraper.py` can auto-pick these files by platform
- cookie files are usually more reliable than `--cookies-from-browser`
- keep them private

### 2. Install Python dependencies

```bash
uv sync
```

Or:

```bash
pip install -e .
```

Python `3.11+` is required.

### 3. Optional: set your OpenAI API key

Needed for `stt.py`.

PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key-here"
```

## Recommended Workflow

### 1. Test one URL first

Before running a big batch, test one authenticated URL with low concurrency:

```bash
python scraper.py --urls "https://www.instagram.com/reel/..." --workers 1 --delay 3
```

### 2. Build `data.csv`

Use `crawler.py` to gather candidate URLs and weak labels.

```bash
python crawler.py --target 100
python crawler.py --target 200 --labels neutral disinformasi
python crawler.py --target 150 --sites kompas detik cnnindonesia
```

Notes:
- output defaults to `data.csv`
- labels are `ujaran kebencian`, `fitnah`, `disinformasi`, and `neutral`
- neutral collection is supplemented with direct platform search pages

### 3. Download audio

Use `scraper.py` on the generated CSV or on direct URLs.

```bash
python scraper.py data.csv --output ./dataset
python scraper.py data.csv --output ./dataset --format mp3
python scraper.py data.csv --output ./dataset --workers 1 --delay 3
```

If you want to force a single cookie file:

```bash
python scraper.py data.csv --cookies cookies/instagram_cookies.txt
```

Audio files are stored under `<output>/<audio-dir>`, which defaults to `dataset/raw`.

### 4. Augment audio

`augment.py` reads a scraped dataset and writes augmented variants plus an `augmented_results.csv`.

```bash
python augment.py ./dataset --output ./dataset_augmented
python augment.py ./dataset --output ./dataset_augmented --num-versions 3
python augment.py ./dataset --output ./dataset_augmented --augmentations pitch_up speed_down noise_white
```

RVC is available as an external augmentation backend:

```bash
python augment.py ./dataset --output ./dataset_augmented --augmentations rvc \
  --rvc-command-template "python infer_cli.py --input_path \"{input}\" --output_path \"{output}\" --model_path \"{model}\" --index_path \"{index}\" --pitch {pitch}" \
  --rvc-model "D:\path\to\voice.pth" \
  --rvc-index "D:\path\to\voice.index"
```

### 5. Transcribe audio

`stt.py` converts speech to text and can handle either one file or a directory.

```bash
python stt.py .\dataset\raw\sample.wav
python stt.py .\dataset\raw --output .\transcripts --recursive --language id
python stt.py .\dataset\raw --output .\transcripts --recursive --json
```

### 6. Generate TTS audio

`tts.py` is the OmniVoice text-to-speech entrypoint.

```bash
python tts.py --text "Halo dunia" --output halo.wav
python tts.py --file script.txt --output .\tts_output
```

## Outputs

### `crawler.py`

Main file: `data.csv`

Fields include:
- `id`
- `source_article`
- `url`
- `platform`
- `keyword`
- `discovered_at`
- `weak_label`

### `scraper.py`

CSV fields include:
- `url`
- `platform`
- `title`
- `uploader`
- `duration_sec`
- `filename`
- `status`
- `error`
- `scraped_at`
- `resolved_url`
- `weak_label`
- `source_article`
- `keyword`

## Notes

- `scraper.py` uses a process pool so interrupted `yt-dlp` jobs can be stopped more reliably than with threads
- `tts.py` requires OmniVoice and its local inference dependencies, which are not part of the base `pyproject.toml`

## License

MIT
