#!/usr/bin/env bash
# Installe Open Notebook comme services systemd "user" (PERSISTANTS, sans sudo).
# Avantages vs start_local.sh : survit a la fermeture du terminal, redemarre tout
# seul en cas de crash, et (avec le linger) demarre tout seul avec WSL.
set -uo pipefail
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.config/systemd/user"

echo "== 1) Arret des anciens processus (liberation des ports) =="
[ -x "$HOME/RG/open-notebook/stop_local.sh" ] && bash "$HOME/RG/open-notebook/stop_local.sh" || true
pkill -f "surreal start"            2>/dev/null || true
pkill -f "run_api.py"              2>/dev/null || true
pkill -f "surreal-commands-worker" 2>/dev/null || true
pkill -f "next dev"                2>/dev/null || true
sleep 3

echo "== 2) Copie des units vers $DEST =="
mkdir -p "$DEST"
cp "$HERE"/onb-surreal.service "$HERE"/onb-api.service \
   "$HERE"/onb-worker.service "$HERE"/onb-frontend.service "$DEST"/
systemctl --user daemon-reload

echo "== 3) Activation + demarrage =="
systemctl --user enable --now onb-surreal.service onb-api.service onb-worker.service onb-frontend.service

echo "== 4) Statut =="
sleep 5
for s in onb-surreal onb-api onb-worker onb-frontend; do
  printf "  %-14s : %s\n" "$s" "$(systemctl --user is-active $s.service 2>/dev/null)"
done

echo ""
echo "Astuce : pour que tout demarre AUSSI sans terminal ouvert / apres reboot,"
echo "lance UNE fois (sudo, mot de passe demande) :"
echo "    sudo loginctl enable-linger $USER"
echo ""
echo "Commandes utiles :"
echo "  systemctl --user status onb-worker      # etat detaille"
echo "  journalctl --user -u onb-worker -f      # logs en direct"
echo "  systemctl --user restart onb-worker     # redemarrer un service"
echo "  systemctl --user stop onb-frontend      # arreter un service"
