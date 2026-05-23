# 🔍 YouTube Visual Search

Search YouTube for videos matching a visual reference image. Define your own search queries, point it at a reference image, and it ranks results by how closely the video frames match — using OpenAI's CLIP model.

## What it does

1. **Searches YouTube** across a list of queries you define
2. **Downloads each video** at the lowest available quality to save bandwidth
3. **Scores every frame** using CLIP image-to-image cosine similarity against your reference image
4. **Saves the best matching frame** from each video as a thumbnail
5. **Exports a ranked CSV** of all videos sorted by their best match score

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, falls back to CPU)
- A reference image in the project root

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision transformers opencv-python pillow yt-dlp youtube-search-python pandas
```

## Usage

1. Place your reference image in the project root
2. Edit `script1.py` and set `REFERENCE_IMAGE_PATH` to your image filename
3. Edit the `QUERIES` list to match whatever you're searching for
4. Run it:

```bash
python script1.py
```

Results are saved to the configured `OUTPUT_FILE` and the best frame thumbnails go into the `FRAMES_DIR` folder.

## Customization

The main things to change are at the top of `script1.py`:

```python
REFERENCE_IMAGE_PATH = "my_image.png"   # Your reference image

QUERIES = [
    "your search term",
    "another variation",
    "yet another phrase",
]
```

The more query variations you add, the broader the search net. Use synonyms and phrasing variations to maximize coverage.

Other settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_RESULTS_PER_QUERY` | `15` | YouTube results per search query |
| `FRAME_SKIP` | `15` | Analyze every Nth frame |
| `MAX_SECONDS_TO_SCAN` | `120` | Only scan the first N seconds of each video |
| `OUTPUT_FILE` | `top_100_lemon_v2.csv` | Output CSV filename |
| `FRAMES_DIR` | `best_lemon_frames` | Folder for saved thumbnails |

## Output

| File | Description |
|------|-------------|
| `OUTPUT_FILE` (csv) | Ranked list of videos with score, timestamp, and URL |
| `FRAMES_DIR/` | Thumbnail of the best-matching frame from each video |

Scores are cosine similarity × 100 — higher means a closer visual match to your reference image.
