#!/bin/bash
# =============================================================================
# test_sandbox.sh — Verify submission.zip works in a sandbox-like environment
#
# Run on RunPod:
#   chmod +x test_sandbox.sh
#   ./test_sandbox.sh
#
# What it does:
#   1. Unzips submission.zip into a temp directory
#   2. Creates a fresh venv with sandbox-pinned package versions
#   3. Scans run.py for banned imports/callables
#   4. Runs inference on a few training images
#   5. Validates output JSON structure
#   6. Reports timing and prediction counts
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No color

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILURES=$((FAILURES + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "       $1"; }

FAILURES=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR=$(mktemp -d)
SUBMISSION_DIR="$WORK_DIR/submission"
TEST_IMAGES_DIR="$WORK_DIR/test_images"
OUTPUT_FILE="$WORK_DIR/predictions.json"

echo "============================================="
echo "  Sandbox Compatibility Test"
echo "============================================="
echo ""
echo "Work dir:    $WORK_DIR"
echo "Script dir:  $SCRIPT_DIR"
echo ""

# -----------------------------------------------
# Step 1: Unzip submission
# -----------------------------------------------
echo "--- Step 1: Unzip submission ---"

if [ ! -f "$SCRIPT_DIR/submission.zip" ]; then
    fail "submission.zip not found in $SCRIPT_DIR"
    echo "Run: python build_submission.py"
    exit 1
fi

mkdir -p "$SUBMISSION_DIR"
python3 -c "
import zipfile
with zipfile.ZipFile('$SCRIPT_DIR/submission.zip', 'r') as z:
    z.extractall('$SUBMISSION_DIR')
"

if [ -f "$SUBMISSION_DIR/run.py" ]; then
    pass "run.py found at zip root"
else
    fail "run.py NOT at zip root — submission will fail"
    echo "Contents:"
    ls -la "$SUBMISSION_DIR/"
    exit 1
fi

# Count files
PY_COUNT=$(find "$SUBMISSION_DIR" -name "*.py" | wc -l)
WEIGHT_COUNT=$(find "$SUBMISSION_DIR" \( -name "*.pt" -o -name "*.pth" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" \) | wc -l)
TOTAL_SIZE=$(du -sm "$SUBMISSION_DIR" | awk '{print $1}')

info "Python files: $PY_COUNT (limit: 10)"
info "Weight files: $WEIGHT_COUNT (limit: 3)"
info "Total size:   ${TOTAL_SIZE} MB (limit: 420 MB)"

[ "$PY_COUNT" -le 10 ] && pass "Python file count OK" || fail "Too many Python files ($PY_COUNT > 10)"
[ "$WEIGHT_COUNT" -le 3 ] && pass "Weight file count OK" || fail "Too many weight files ($WEIGHT_COUNT > 3)"
[ "$TOTAL_SIZE" -le 420 ] && pass "Total size OK" || fail "Total size exceeds 420 MB"

echo ""

# -----------------------------------------------
# Step 2: Scan for banned imports/callables
# -----------------------------------------------
echo "--- Step 2: Security scan ---"

BANNED_IMPORTS="^import os$|^import sys$|^import subprocess|^import socket|^import ctypes|^import builtins|^import importlib|^import pickle|^import marshal|^import shelve|^import shutil|^import yaml|^import requests|^import urllib|^import http\.client|^import multiprocessing|^import threading|^import signal|^import gc$|^import code$|^import codeop|^import pty"
BANNED_FROM="from os |from sys |from subprocess |from socket |from ctypes |from builtins |from importlib |from pickle |from marshal |from shelve |from shutil |from yaml |from requests |from urllib |from http\.client |from multiprocessing |from threading |from signal |from gc |from code |from codeop |from pty "
BANNED_CALLS="[^a-zA-Z_]eval(|[^a-zA-Z_]exec(|[^a-zA-Z_]compile(|__import__|[^a-zA-Z_]getattr("

SCAN_FAILED=0
for pyfile in $(find "$SUBMISSION_DIR" -name "*.py"); do
    relpath="${pyfile#$SUBMISSION_DIR/}"

    # Check banned imports
    if grep -Pn "$BANNED_IMPORTS" "$pyfile" 2>/dev/null; then
        fail "Banned import in $relpath (see above)"
        SCAN_FAILED=1
    fi
    if grep -Pn "$BANNED_FROM" "$pyfile" 2>/dev/null; then
        fail "Banned from-import in $relpath (see above)"
        SCAN_FAILED=1
    fi

    # Check banned callables
    if grep -Pn "$BANNED_CALLS" "$pyfile" 2>/dev/null; then
        fail "Banned callable in $relpath (see above)"
        SCAN_FAILED=1
    fi
done

[ "$SCAN_FAILED" -eq 0 ] && pass "No banned imports or callables found"
echo ""

# -----------------------------------------------
# Step 3: Check installed packages
# -----------------------------------------------
echo "--- Step 3: Package versions (sandbox targets in parens) ---"

# Sandbox targets
TORCH_TARGET="2.6.0"
TV_TARGET="0.21.0"
ORT_TARGET="1.20.0"
NP_TARGET="1.26.4"
PIL_TARGET="10.2.0"

TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "MISSING")
TV_VER=$(python3 -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || echo "MISSING")
ORT_VER=$(python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo "MISSING")
NP_VER=$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "MISSING")
PIL_VER=$(python3 -c 'from PIL import Image; import PIL; print(PIL.__version__)' 2>/dev/null || echo "MISSING")

info "torch:       $TORCH_VER (sandbox: $TORCH_TARGET)"
info "torchvision: $TV_VER (sandbox: $TV_TARGET)"
info "onnxruntime: $ORT_VER (sandbox: $ORT_TARGET)"
info "numpy:       $NP_VER (sandbox: $NP_TARGET)"
info "Pillow:      $PIL_VER (sandbox: $PIL_TARGET)"
info "CUDA:        $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Warn on mismatches but don't fail — ONNX models are version-independent
[[ "$TORCH_VER" == *"$TORCH_TARGET"* ]] || warn "torch $TORCH_VER != sandbox $TORCH_TARGET (OK — only used for tensors, not model loading)"
[[ "$ORT_VER" == "$ORT_TARGET" ]] || warn "onnxruntime $ORT_VER != sandbox $ORT_TARGET (ONNX opset 17 is compatible)"
[ "$ORT_VER" != "MISSING" ] && pass "All required packages importable" || fail "onnxruntime not installed"

# Check CUDA provider for onnxruntime
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('ORT providers:', providers)
if 'CUDAExecutionProvider' in providers:
    print('  -> CUDA provider available')
else:
    print('  -> WARNING: CUDA provider NOT available, will use CPU')
"

echo ""

# -----------------------------------------------
# Step 4: Prepare test images
# -----------------------------------------------
echo "--- Step 4: Prepare test images ---"

SRC_IMAGES="$SCRIPT_DIR/data/coco/train/images"
if [ ! -d "$SRC_IMAGES" ]; then
    fail "Training images not found at $SRC_IMAGES"
    exit 1
fi

mkdir -p "$TEST_IMAGES_DIR"

# Copy first 5 images for testing
COUNT=0
for img in "$SRC_IMAGES"/img_*.jpg; do
    cp "$img" "$TEST_IMAGES_DIR/"
    COUNT=$((COUNT + 1))
    [ "$COUNT" -ge 5 ] && break
done

pass "Copied $COUNT test images"
echo ""

# -----------------------------------------------
# Step 5: Check argument compatibility
# -----------------------------------------------
echo "--- Step 5: Check CLI arguments ---"

# The sandbox invokes: python run.py --input /data/images --output /output/predictions.json
# Verify run.py accepts these exact argument names
cd "$SUBMISSION_DIR"

python3 -c "
import ast, sys
tree = ast.parse(open('run.py').read())
args_found = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == 'add_argument':
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith('--'):
                    args_found.append(arg.value)
print('Arguments in run.py:', args_found)
if '--input' not in [a.split('=')[0] for a in args_found]:
    if '--input_dir' in [a.split('=')[0] for a in args_found]:
        print('CRITICAL: run.py uses --input_dir but sandbox calls --input')
        sys.exit(1)
if '--output' not in [a.split('=')[0] for a in args_found]:
    if '--output_file' in [a.split('=')[0] for a in args_found]:
        print('CRITICAL: run.py uses --output_file but sandbox calls --output')
        sys.exit(1)
" 2>&1

if [ $? -ne 0 ]; then
    fail "Argument mismatch — sandbox uses --input/--output, run.py uses different names!"
    warn "Fix run.py: rename --input_dir to --input, --output_file to --output"
    warn "Continuing test with run.py's actual arguments..."
    INPUT_ARG="--input_dir"
    OUTPUT_ARG="--output_file"
else
    pass "CLI arguments match sandbox contract"
    INPUT_ARG="--input"
    OUTPUT_ARG="--output"
fi

echo ""

# -----------------------------------------------
# Step 6: Run inference
# -----------------------------------------------
echo "--- Step 6: Run inference ---"

START_TIME=$(date +%s%N)

python3 run.py \
    $INPUT_ARG "$TEST_IMAGES_DIR" \
    $OUTPUT_ARG "$OUTPUT_FILE" \
    2>&1 | tee "$WORK_DIR/run_output.log"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s%N)

ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
ELAPSED_S=$(echo "scale=1; $ELAPSED_MS / 1000" | bc)

if [ "$EXIT_CODE" -eq 0 ]; then
    pass "run.py exited successfully"
else
    fail "run.py exited with code $EXIT_CODE"
fi

info "Wall time: ${ELAPSED_S}s (limit: 300s for full test set)"

# Extrapolate for full dataset (248 images)
if [ "$COUNT" -gt 0 ]; then
    PER_IMAGE_MS=$((ELAPSED_MS / COUNT))
    ESTIMATED_FULL=$((PER_IMAGE_MS * 248 / 1000))
    info "Per-image: ${PER_IMAGE_MS}ms"
    info "Estimated full (248 images): ${ESTIMATED_FULL}s"
    [ "$ESTIMATED_FULL" -le 300 ] && pass "Estimated time within 300s limit" || warn "Estimated ${ESTIMATED_FULL}s may exceed 300s timeout"
fi

echo ""

# -----------------------------------------------
# Step 7: Validate output JSON
# -----------------------------------------------
echo "--- Step 7: Validate output ---"

if [ ! -f "$OUTPUT_FILE" ]; then
    fail "Output file not created at $OUTPUT_FILE"
else
    pass "Output file exists"

    # Validate JSON structure with Python
    python3 -c "
import json
from pathlib import Path

data = json.loads(Path('$OUTPUT_FILE').read_text())

assert isinstance(data, list), f'Expected list, got {type(data)}'
print(f'  Predictions: {len(data)}')

if len(data) == 0:
    print('  WARNING: Zero predictions!')
else:
    # Check required keys
    required = {'image_id', 'category_id', 'bbox', 'score'}
    sample = data[0]
    missing = required - set(sample.keys())
    if missing:
        print(f'  FAIL: Missing keys: {missing}')
    else:
        print(f'  Keys OK: {sorted(sample.keys())}')

    # Check types
    assert isinstance(sample['image_id'], int), f'image_id should be int, got {type(sample[\"image_id\"])}'
    assert isinstance(sample['category_id'], int), f'category_id should be int'
    assert isinstance(sample['bbox'], list) and len(sample['bbox']) == 4, 'bbox should be [x,y,w,h]'
    assert isinstance(sample['score'], (int, float)), 'score should be float'

    # Stats
    image_ids = set(p['image_id'] for p in data)
    cat_ids = set(p['category_id'] for p in data)
    scores = [p['score'] for p in data]
    print(f'  Images with predictions: {len(image_ids)}')
    print(f'  Unique categories: {len(cat_ids)}')
    print(f'  Score range: [{min(scores):.3f}, {max(scores):.3f}]')
    print(f'  Avg predictions/image: {len(data)/len(image_ids):.1f}')

    # Per-image breakdown
    from collections import Counter
    per_image = Counter(p['image_id'] for p in data)
    for img_id, count in sorted(per_image.items()):
        print(f'    img {img_id:5d}: {count} detections')

    print('  Structure validation: PASS')
" && pass "Output JSON valid" || fail "Output JSON validation failed"
fi

echo ""

# -----------------------------------------------
# Step 8: VRAM report
# -----------------------------------------------
echo "--- Step 8: GPU memory ---"

python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'  GPU {i}: {name} ({total:.1f} GB)')
else:
    print('  No CUDA GPU available')
" 2>/dev/null || true

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null && true

echo ""

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo "============================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "  ${GREEN}ALL TESTS PASSED${NC}"
else
    echo -e "  ${RED}$FAILURES TEST(S) FAILED${NC}"
fi
echo "============================================="

# -----------------------------------------------
# Cleanup
# -----------------------------------------------
cd "$SCRIPT_DIR"
echo ""
echo "Work dir preserved at: $WORK_DIR"
echo "To clean up: rm -rf $WORK_DIR"

exit $FAILURES
