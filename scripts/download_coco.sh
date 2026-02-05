#!/bin/bash
# Download COCO val2017 dataset for pose estimation benchmarking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/coco"

echo "Downloading COCO val2017 dataset..."
echo "Target directory: $DATA_DIR"

# Create data directories
mkdir -p "$DATA_DIR/annotations"
mkdir -p "$DATA_DIR/val2017"

cd "$DATA_DIR"

# Download val2017 images (1GB zip, 5000 images)
if [ ! -f "val2017.zip" ]; then
    echo "Downloading val2017 images (1GB)..."
    curl -O http://images.cocodataset.org/zips/val2017.zip
else
    echo "val2017.zip already exists, skipping download"
fi

# Extract images
if [ ! -d "val2017" ] || [ -z "$(ls -A val2017)" ]; then
    echo "Extracting val2017 images..."
    unzip -q val2017.zip
else
    echo "val2017 images already extracted"
fi

# Download annotations (241MB zip)
if [ ! -f "annotations_trainval2017.zip" ]; then
    echo "Downloading annotations (241MB)..."
    curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
else
    echo "annotations_trainval2017.zip already exists, skipping download"
fi

# Extract annotations
if [ ! -f "annotations/person_keypoints_val2017.json" ]; then
    echo "Extracting annotations..."
    unzip -q annotations_trainval2017.zip
else
    echo "Annotations already extracted"
fi

echo ""
echo "✓ COCO val2017 dataset downloaded successfully!"
echo ""
echo "Dataset structure:"
echo "  $DATA_DIR/val2017/           (5000 images)"
echo "  $DATA_DIR/annotations/person_keypoints_val2017.json"
echo ""
echo "You can now run the benchmark with:"
echo "  python -m pose_bench.run_benchmark --config config.yaml"
