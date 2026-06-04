#!/bin/bash
# run.sh — LOCAL detection + ingest for Brigade Road store
# Usage: ./pipeline/run.sh [store_id] [clips_dir]
# For GPU detection, use colab_detection.ipynb instead

set -e

STORE_ID=${1:-"ST1008"}
CLIPS_DIR=${2:-"./data/clips"}
OUTPUT="./data/events.jsonl"
API_URL=${API_URL:-"http://localhost:8000"}
CLIP_START=${CLIP_START:-"2026-04-10T12:00:00Z"}

echo "=== Store Intelligence Pipeline (LOCAL CPU) ==="
echo "Store: $STORE_ID | Clips: $CLIPS_DIR"
echo "Note: For faster processing, use colab_detection.ipynb (T4 GPU)"
echo ""

# Downsample clips if ffmpeg available (3x speedup on CPU)
if command -v ffmpeg &> /dev/null && [ "${SKIP_DOWNSAMPLE}" != "1" ]; then
    echo "--- Downsampling clips to 640x360 @ 5fps ---"
    mkdir -p ./data/clips_small
    for mp4 in "$CLIPS_DIR"/*.mp4; do
        base=$(basename "$mp4" .mp4)
        out="./data/clips_small/${base}.mp4"
        if [ ! -f "$out" ]; then
            echo "  Downsampling $base..."
            ffmpeg -i "$mp4" -vf "scale=640:360,fps=5" -c:v libx264 -preset fast "$out" -y -loglevel error
        fi
    done
    CLIPS_DIR="./data/clips_small"
    echo "Done."
fi

python pipeline/detect.py \
  --store "$STORE_ID" \
  --clips-dir "$CLIPS_DIR" \
  --layout "./data/store_layout.json" \
  --pos "./data/pos_transactions.csv" \
  --output "$OUTPUT" \
  --clip-start "$CLIP_START" \
  --sample 3 \
  --debug

echo ""
echo "--- Ingesting into API ---"
python pipeline/ingest_events.py --events "$OUTPUT" --api "$API_URL"

echo ""
echo "--- Live metrics ---"
curl -s "$API_URL/stores/$STORE_ID/metrics" | python -m json.tool
