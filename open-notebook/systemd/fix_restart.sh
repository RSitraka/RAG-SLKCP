#!/usr/bin/env bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

echo "== tue l orphelin sur :5055 =="
fuser -k 5055/tcp 2>/dev/null || true
sleep 2
if ss -ltn 2>/dev/null | grep -q ":5055"; then echo "  :5055 ENCORE pris"; else echo "  :5055 libre"; fi

echo "== relance API =="
systemctl --user restart onb-api
for i in $(seq 1 45); do
  if curl -s --max-time 2 http://localhost:5055/api/config >/dev/null 2>&1; then
    echo "  API UP apres ${i}s"; break
  fi
  sleep 1
done

echo "== relance worker =="
systemctl --user restart onb-worker
sleep 8

echo "== STATUTS =="
for s in onb-surreal onb-api onb-worker onb-frontend; do
  printf "  %-14s : %s\n" "$s" "$(systemctl --user is-active ${s}.service)"
done
