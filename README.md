# Audio Engine

A Python toolkit for discovering and scraping audio from Indonesian social media videos. Designed to collect neutral, non-political content for research and dataset building purposes.

## Features

- **Discovery**: Automatically finds neutral Indonesian video URLs from multiple platforms
- **Scraping**: Downloads audio from social media URLs in various formats
- **Platform Support**: TikTok, Instagram, Facebook, YouTube, Twitter/X
- **Filtering**: Built-in blocklist to filter out political and sensitive content

## Installation

```bash
# Clone the repository
cd aitf-dfk3-audio-engine

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Dependencies

- Python 3.11+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Video/audio downloading
- [twikit](https://github.com/disisdevel/twikit) - Twitter/X API client
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing

## Usage

### 1. Discover Videos (optional)

Discover neutral Indonesian video URLs from social media:

```bash
# Discover 700 URLs from all platforms (default)
python discovery.py --target 700 --output ./discovered

# Discover from specific platforms
python discovery.py --target 700 --platforms tiktok youtube --output ./discovered

# Use browser cookies for better access
python discovery.py --target 700 --cookies-from-browser chrome --output ./discovered

# For Twitter/X, generate cookies with twitter_login.py first
python discovery.py --target 700 --twitter-cookies twitter_cookies.json --output ./discovered
```

### 2. Scrape Audio

Download audio from discovered URLs:

```bash
# Using URLs from a file
python scraper.py discovered/urls.txt --output ./dataset

# Using URLs directly
python scraper.py --urls "https://instagram.com/reel/..." "https://tiktok.com/v/..." --output ./dataset

# Specify audio format
python scraper.py discovered/urls.txt --output ./dataset --format mp3

# Use browser cookies
python scraper.py discovered/urls.txt --output ./dataset --cookies-from-browser chrome
```

## Project Structure

```
audio-engine/
├── discovery.py      # URL discovery script
├── scraper.py        # Audio scraping script
└── README.md
```

## Configuration

### Discovery Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | 700 | Number of URLs to collect |
| `--platforms` | tiktok, instagram, facebook, youtube | Platforms to search |
| `--per-keyword` | 20 | Results per keyword per platform |
| `--delay` | 2.0 | Seconds between requests |

### Scraper Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | dataset | Output directory |
| `--format` | wav | Audio format (wav, mp3, m4a, opus, flac) |
| `--audio-dir` | audios | Subdirectory for audio files |
| `--delay` | 1.0 | Seconds between downloads |

## Blocklist

The discovery script includes a blocklist to filter out:
- Political content (pilkada, pilpres, capres, etc.)
- Sensitive topics (korupsi, teroris, etc.)
- Offensive language

## CSV Output

### Discovery Results

| Field | Description |
|-------|-------------|
| id | Unique identifier |
| url | Video URL |
| platform | Source platform |
| keyword | Search keyword |
| title | Video title |
| author | Content creator |
| duration | Video duration (seconds) |
| caption | Video description |
| discovered_at | Discovery timestamp |
| label | Content label (neutral) |

### Scraping Results

| Field | Description |
|-------|-------------|
| url | Source URL |
| platform | Source platform |
| title | Video title |
| uploader | Content creator |
| duration_sec | Audio duration |
| filename | Output file path |
| status | Download status (ok/failed) |
| error | Error message if failed |
| scraped_at | Scraping timestamp |

## License

MIT
