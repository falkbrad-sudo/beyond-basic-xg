#!/usr/bin/env bash
# Fetches both public datasets used by this project into data/external/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$ROOT_DIR/data/external"

mkdir -p "$EXTERNAL_DIR"

METRICA_DIR="$EXTERNAL_DIR/metrica"
if [ ! -d "$METRICA_DIR" ]; then
  echo "Cloning Metrica Sports sample-data..."
  git clone --depth 1 https://github.com/metrica-sports/sample-data.git "$METRICA_DIR"
else
  echo "Metrica sample-data already present at $METRICA_DIR, skipping."
fi

STATSBOMB_DIR="$EXTERNAL_DIR/statsbomb"
if [ ! -d "$STATSBOMB_DIR" ]; then
  echo "Cloning StatsBomb open-data (this is a large repo, may take a while)..."
  git clone --depth 1 https://github.com/statsbomb/open-data.git "$STATSBOMB_DIR"
else
  echo "StatsBomb open-data already present at $STATSBOMB_DIR, skipping."
fi

echo "Done. Data available under $EXTERNAL_DIR/{metrica,statsbomb}."
echo
echo "Next:"
echo "  python -m src.pipeline              # fit both models, write data/processed/"
echo "  python -m scripts.render_figures    # write reports/figures/"
echo "  streamlit run app/streamlit_app.py  # interactive comparison"
