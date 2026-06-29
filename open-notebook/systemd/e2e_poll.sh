#!/usr/bin/env bash
# Verifie que le WORKER systemd (deja lancé) traite une ingestion OCR de bout en bout.
# N'lance AUCUN worker : c'est systemd qui doit faire le travail.
set -uo pipefail
cd "$HOME/RG/open-notebook"
export PATH="$HOME/.local/bin:$PATH"
API="http://localhost:5055/api"
NB="notebook:9fml7nkge0slj3fxz4xe"
PDF="/tmp/e2e_systemd.pdf"

CUDA_VISIBLE_DEVICES= uv run --no-sync python - "$PDF" <<'PY'
import sys, fitz
from PIL import Image, ImageDraw, ImageFont
p=sys.argv[1]; img=Image.new("RGB",(1240,280),"white"); d=ImageDraw.Draw(img)
try: f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",34)
except Exception: f=ImageFont.load_default()
for i,l in enumerate(["RECU SYSTEMD - TEST PERSISTANCE","Code: PERSIST-2026-7777","Total: 4242 euros"]):
    d.text((40,30+i*80),l,fill="black",font=f)
png=p+".png"; img.save(png); doc=fitz.open(); pg=doc.new_page(width=1240,height=280)
pg.insert_image(fitz.Rect(0,0,1240,280),filename=png); doc.save(p); doc.close()
import os; os.remove(png); print("PDF cree")
PY

RESP=$(curl -s --max-time 30 -X POST "$API/sources" \
  -F "type=upload" -F "notebooks=[\"$NB\"]" -F "embed=true" \
  -F "async_processing=true" -F "title=E2E systemd" \
  -F "file=@$PDF;type=application/pdf")
SRC=$(echo "$RESP" | grep -oE '"id":"source:[a-z0-9]+"' | head -1 | cut -d'"' -f4)
echo "Source: $SRC"
[ -z "$SRC" ] && { echo "pas de source id: $RESP"; exit 1; }

for i in $(seq 1 48); do
  sleep 5
  FULL=$(curl -s --max-time 10 "$API/sources/$SRC" 2>/dev/null)
  if echo "$FULL" | grep -qi "PERSIST-2026\|RECU SYSTEMD\|4242"; then
    echo "OK [${i}x5s] : texte OCR present (worker systemd) !"
    echo "$FULL" | python3 -c "import sys,json;d=json.load(sys.stdin);t=d.get('full_text') or '';print('full_text(',len(t),'car):',repr(t[:200]));print('embedded_chunks:',d.get('embedded_chunks'))"
    exit 0
  fi
done
echo "TIMEOUT - dernier etat:"; echo "$FULL" | head -c 400
exit 1
