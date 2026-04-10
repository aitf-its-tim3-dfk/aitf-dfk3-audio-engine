# Audio Engine Agents

This file documents the main scripts and agents in the audio-engine project, which appears to be a pipeline for collecting, processing, and augmenting audio/video data from Indonesian news sources, likely for AI training or disinformation detection.

## Core Scripts

### crawler.py
- **Purpose**: Discovers Indonesian video URLs by scraping news websites (turnbackhoax.id, kompas.com, detik.com, liputan6.com, cnnindonesia.com).
- **Functionality**: Automatically assigns labels (ujaran kebencian, fitnah, disinformasi, neutral) based on article tags. Outputs metadata to data.csv.
- **Usage**: `python crawler.py --target 700 --sites turnbackhoax kompas`
- **Next Step**: Feeds into scraper.py for downloading content.

### scraper.py
- **Purpose**: Downloads and processes videos/audio from URLs listed in data.csv.
- **Functionality**: Creates a dataset in the ./dataset directory, likely extracting audio for further processing.
- **Usage**: `python scraper.py data.csv --output ./dataset`
- **Dependencies**: Requires yt-dlp for downloading.

### augment.py
- **Purpose**: Augments the collected audio data.
- **Functionality**: Likely applies transformations like noise addition, speed changes, or other augmentations to increase dataset size and diversity.
- **Usage**: Presumably `python augment.py [input] [output]`

### stt.py
- **Purpose**: Speech-to-Text conversion.
- **Functionality**: Transcribes audio files to text, possibly using models like Whisper or similar.
- **Usage**: `python stt.py [audio_file]`

### tts.py
- **Purpose**: Text-to-Speech synthesis.
- **Functionality**: Generates audio from text, useful for creating synthetic data or testing.
- **Usage**: `python tts.py [text] [output_audio]`

### cleaner.py
- **Purpose**: Cleans and preprocesses the data.
- **Functionality**: Removes noise, normalizes audio, filters invalid files, or cleans transcripts.
- **Usage**: `python cleaner.py [input] [output]`

### scripter.py
- **Purpose**: Scripting utilities or batch processing.
- **Functionality**: Likely automates workflows or runs sequences of the above scripts.
- **Usage**: `python scripter.py [options]`

## Pipeline Overview
1. **Discovery**: crawler.py collects URLs with labels.
2. **Collection**: scraper.py downloads videos/audio.
3. **Transcription**: stt.py converts audio to text.
4. **Cleaning**: cleaner.py preprocesses data.
5. **Augmentation**: augment.py expands the dataset.
6. **Synthesis**: tts.py generates additional audio if needed.
7. **Automation**: scripter.py orchestrates the pipeline.

## Dependencies
- httpx, asyncio for web scraping.
- yt-dlp for video downloading.
- librosa, soundfile for audio processing.
- transformers, torch for AI models (STT/TTS).
- pandas, numpy for data handling.

## Configuration
- Configurable via command-line arguments in each script.
- Logging to console and optional files.
- Concurrency and rate limiting for respectful scraping.

This project seems designed for building labeled audio datasets from social media and news videos, potentially for training models to detect hate speech, misinformation, or other content in Indonesian.