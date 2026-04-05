#!/bin/bash
# ============================================================================
# HackForge — Amazon Sustainability Challenge — Winning Demo Script
# ============================================================================
#
# BEFORE RUNNING:
#   1. pip install torchvision    ← CRITICAL: without this you get fallback
#                                   models with wrong param counts
#   2. pip install -e ".[all]"    ← installs all optional deps
#
# RECOMMENDED DEMO ORDER FOR JUDGES:
#   Step 1: Run the quick smoke test (2 min) to show it works
#   Step 2: Show the full run output (pre-captured) for real numbers
#   Step 3: Walk through the presentation slides
#
# ============================================================================

set -e
PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH

echo ""
echo "============================================================"
echo "  STEP 1: Quick smoke test (verifies everything runs)"
echo "============================================================"
echo ""

python -m tests.run_amazon_sustainability_demo_comprehensive \
    --quick --cnn-only \
    --seed 42

echo ""
echo "============================================================"
echo "  STEP 2: Full benchmark (pre-run for judging)"
echo "============================================================"
echo ""
echo "  For the full benchmark with 5 seeds and all scenarios:"
echo ""
echo "    python -m tests.run_amazon_sustainability_demo_comprehensive \\"
echo "        --seed 42 --seeds 5 --full \\"
echo "        --save-json results_full.json"
echo ""
echo "  Estimated time: ~15 min on Apple MPS, ~5 min on CUDA T4"
echo ""
echo "  For the classical ML story demos (strongest results):"
echo ""
echo "    python -u -m tests.run_story_ml_demo --scenario all --seeds 5"
echo "    python -u -m tests.run_story_dl_demo --seeds 3"
echo ""

echo ""
echo "============================================================"
echo "  STEP 3: Unit tests (98 tests, proves code quality)"
echo "============================================================"
echo ""
echo "  python -m pytest tests/test_smoke.py tests/test_dl_smoke.py -v"
echo ""
echo "============================================================"
echo "  DONE. See PRESENTATION_PROMPTS.md for slide-building guide."
echo "============================================================"
