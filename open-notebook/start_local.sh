#!/usr/bin/env bash
# Démarre Open Notebook en local (sans Docker) : SurrealDB + API + Worker + Frontend
# Chaque service est lancé détaché (setsid+nohup) pour survivre à la fermeture du shell.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs surreal_data

# Node 20 via nvm (le frontend Next.js 16 l'exige ; le node système est trop vieux).
# On force le PATH explicitement : `nvm use` ne prend pas dans un shell non-interactif.
NODE20_BIN="$(ls -d "$HOME"/.nvm/versions/node/v20*/bin 2>/dev/null | sort | tail -1)"
[ -n "$NODE20_BIN" ] && export PATH="$NODE20_BIN:$PATH"

# `uv` est dans ~/.local/bin (ajouté par ton shell interactif mais ABSENT du PATH
# d'un script non-interactif). On l'ajoute explicitement, sinon l'API/worker ne
# démarrent pas ("nohup: failed to run command 'uv'").
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

wait_for() { # url, label, timeout_s
  local url="$1" label="$2" max="${3:-60}" i=0
  while [ "$i" -lt "$max" ]; do
    curl -s --max-time 3 "$url" >/dev/null 2>&1 && { echo "  ✓ $label prêt"; return 0; }
    i=$((i+1)); sleep 1
  done
  echo "  ✗ $label n'a pas répondu après ${max}s (voir logs/)"; return 1
}

# 1) SurrealDB (port 8000)
if curl -s --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then
  echo "✓ SurrealDB déjà lancé"
else
  echo "→ Démarrage SurrealDB..."
  setsid nohup "$HOME/.surrealdb/surreal" start --user root --pass root \
    --bind 127.0.0.1:8000 rocksdb:surreal_data/database.db > logs/surreal.log 2>&1 &
  wait_for http://localhost:8000/health "SurrealDB" 30
fi

# 2) API FastAPI (port 5055) — lance les migrations automatiquement
if curl -s --max-time 3 http://localhost:5055/api/config >/dev/null 2>&1; then
  echo "✓ API déjà lancée"
else
  echo "→ Démarrage API..."
  setsid nohup uv run --env-file .env run_api.py > logs/api.log 2>&1 &
  wait_for http://localhost:5055/api/config "API" 90
fi

# 3) Worker (jobs async : extraction, embeddings, insights, podcasts)
if pgrep -f "surreal-commands-worker" >/dev/null 2>&1; then
  echo "✓ Worker déjà lancé"
else
  echo "→ Démarrage Worker..."
  setsid nohup uv run --env-file .env surreal-commands-worker --import-modules commands > logs/worker.log 2>&1 &
  sleep 2; echo "  ✓ Worker lancé"
fi

# 4) Frontend Next.js (port 3001 ; 3000 est pris par un autre projet)
if curl -s --max-time 3 http://localhost:3001 >/dev/null 2>&1; then
  echo "✓ Frontend déjà lancé"
else
  echo "→ Démarrage Frontend (port 3001, accessible sur le réseau)..."
  # -H 0.0.0.0 : écoute sur toutes les interfaces => joignable par les autres
  # PC du réseau local (grâce au mode WSL "mirrored"). API reste en localhost.
  ( cd frontend && setsid nohup env PORT=3001 npm run dev -- -H 0.0.0.0 > ../logs/frontend.log 2>&1 & )
  wait_for http://localhost:3001 "Frontend" 60
fi

echo ""
echo "================================================================"
echo " Open Notebook est lancé :"
echo "   • Interface     : http://localhost:3001"
echo "   • API (docs)    : http://localhost:5055/docs"
echo "   • Base SurrealDB: http://localhost:8000"
echo " Logs dans ./logs/   |   Arrêt : ./stop_local.sh"
echo "================================================================"
