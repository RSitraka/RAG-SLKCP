#!/usr/bin/env bash
# Arrête uniquement les services Open Notebook de ~/RAG (ne touche pas aux autres projets).
echo "🛑 Arrêt des services Open Notebook (RAG)..."
pkill -f "rocksdb:/home/rasix/RAG/open-notebook" 2>/dev/null || true
pkill -f "RAG/open-notebook.*run_api.py" 2>/dev/null || true
pkill -f "RAG/open-notebook.*surreal-commands-worker" 2>/dev/null || true
pkill -f "RAG/open-notebook/frontend" 2>/dev/null || true
echo "✅ Arrêté."
