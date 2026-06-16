#!/usr/bin/env bash
# Arrête tous les services Open Notebook lancés par start_local.sh
echo "🛑 Arrêt des services Open Notebook..."
pkill -f "next dev"                || true
pkill -f "next-server"            || true
pkill -f "npm run dev"            || true
pkill -f "surreal-commands-worker" || true
pkill -f "run_api.py"             || true
pkill -f "uvicorn api.main:app"   || true
pkill -f "surreal start"          || true
echo "✅ Services arrêtés."
