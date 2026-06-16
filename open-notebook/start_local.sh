#!/usr/bin/env bash
# Démarre Open Notebook en local (sans Docker) — à lancer dans WSL Ubuntu.
# Usage: bash start_local.sh
set -e

PROJECT_DIR="/home/rasix/RAG/open-notebook"
UV="/home/rasix/.local/bin/uv"
SURREAL="$HOME/.surrealdb/surreal"
NODE20_BIN="/home/rasix/.nvm/versions/node/v20.20.2/bin"
LOG_DIR="$PROJECT_DIR/.local-logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

echo "📊 SurrealDB..."
"$SURREAL" start --user root --pass root --bind 127.0.0.1:8000 \
  "rocksdb:$PROJECT_DIR/surreal_data/database.db" > "$LOG_DIR/surreal.log" 2>&1 &
sleep 3

echo "🔧 API (port 5055)..."
"$UV" run --env-file .env run_api.py > "$LOG_DIR/api.log" 2>&1 &
sleep 8

echo "⚙️  Worker..."
"$UV" run --env-file .env surreal-commands-worker --import-modules commands > "$LOG_DIR/worker.log" 2>&1 &
sleep 2

echo "🌐 Frontend (Next.js, port 3000/3001)..."
( export PATH="$NODE20_BIN:$PATH"; cd "$PROJECT_DIR/frontend" && npm run dev > "$LOG_DIR/frontend.log" 2>&1 & )
sleep 3

echo "✅ Démarré. Logs dans $LOG_DIR/"
echo "   Frontend : http://localhost:3000 (ou 3001 si 3000 est pris)"
echo "   API docs : http://localhost:5055/docs"
