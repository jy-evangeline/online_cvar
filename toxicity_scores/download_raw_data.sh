#!/bin/bash
# Fetch the raw llama3.2 RealToxicityPrompts generations from the Hugging Face
# Hub into ./data/llama3.2_real_toxic.
#
# You do NOT need this to reproduce the results: data_slim/ is committed to the
# repo and produces byte-identical output (see README, "Data"). Download the raw
# data only if you need the generated response texts or the perplexities.
#
# Requires the Hugging Face CLI and access to the (private) dataset:
#   pip install -U huggingface_hub && hf auth login
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="${REPO_ID:-Evangelinejy/online-cvar-llama3.2-real-toxic}"
DEST="${DEST:-${SCRIPT_DIR}/data}"

echo "Downloading ${REPO_ID} -> ${DEST}"
hf download "${REPO_ID}" --repo-type dataset --local-dir "${DEST}"

echo "Done. Run the sweep against the raw data with:"
echo "  DATA_ROOT=${DEST}/llama3.2_real_toxic bash ${SCRIPT_DIR}/run_ru.sh"
