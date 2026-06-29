#!/usr/bin/env bash
# Ré-applique le patch OCR EasyOCR sur le processeur Docling de content-core (P3.1).
#
# POURQUOI : content-core appelle DocumentConverter() sans options -> moteur OCR
# RapidOCR, cassé. Le patch force EasyOCR (CPU). Normalement le patch survit aux
# redémarrages grâce à `uv run --no-sync` (start_local.sh). Mais si content-core
# est un jour RÉINSTALLÉ (uv sync sans --no-sync, mise à jour de paquet...), le
# fichier .venv repart à zéro. Ce script remet le patch en une commande.
#
# Usage :  bash patches/apply_ocr_patch.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=".venv/lib/python3.12/site-packages/content_core/processors/docling.py"
SRC="patches/docling.py.patched"

[ -f "$TARGET" ] || { echo "✗ Introuvable : $TARGET (content-core est-il installé ?)"; exit 1; }
[ -f "$SRC" ]    || { echo "✗ Sauvegarde du patch introuvable : $SRC"; exit 1; }

if grep -q "_build_converter" "$TARGET"; then
  echo "✓ Patch OCR déjà présent — rien à faire."
  exit 0
fi

cp "$TARGET" "$TARGET.orig.bak"          # garde l'original au cas où
cp "$SRC" "$TARGET"
echo "✓ Patch OCR EasyOCR ré-appliqué sur content-core."
echo "  (original sauvegardé en $TARGET.orig.bak)"
echo "  Redémarre le worker pour prise en compte : ./stop_local.sh && ./start_local.sh"
