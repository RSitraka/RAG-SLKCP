#!/usr/bin/env bash
# Etat des services Open Notebook (systemd user).
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
echo "=== Services Open Notebook ==="
for s in onb-surreal onb-api onb-worker onb-frontend; do
  a="$(systemctl --user is-active ${s}.service 2>/dev/null)"
  e="$(systemctl --user is-enabled ${s}.service 2>/dev/null)"
  printf "  %-14s active=%-10s enabled=%s\n" "$s" "$a" "$e"
done
echo "=== Linger (persistance sans terminal) ==="
loginctl show-user "$USER" 2>/dev/null | grep -i linger
echo "=== Health ==="
curl -s --max-time 3 http://localhost:8000/health >/dev/null 2>&1 && echo "  SurrealDB OK" || echo "  SurrealDB DOWN"
curl -s --max-time 3 http://localhost:5055/api/config >/dev/null 2>&1 && echo "  API OK" || echo "  API DOWN"
curl -s --max-time 3 http://localhost:3001 >/dev/null 2>&1 && echo "  Frontend OK" || echo "  Frontend DOWN"
