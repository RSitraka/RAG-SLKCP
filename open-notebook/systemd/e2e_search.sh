#!/usr/bin/env bash
# P3.4.f - E2E complet : upload -> OCR -> embed -> INDEX MTREE -> recherche vectorielle.
# Prouve que l'insertion d'un nouvel embedding met a jour l'index et que la recherche
# KNN (fn::vector_search) retrouve bien le document.
set -uo pipefail
cd "$HOME/RG/open-notebook"
export PATH="$HOME/.local/bin:$PATH"
API="http://localhost:5055/api"
NB="notebook:9fml7nkge0slj3fxz4xe"
PDF="/tmp/e2e_index.pdf"
MARK="ZEBRELUNE"   # mot rare et unique pour une recherche non ambigue

CUDA_VISIBLE_DEVICES= uv run --no-sync python - "$PDF" "$MARK" <<'PY'
import sys, fitz
from PIL import Image, ImageDraw, ImageFont
p, mark = sys.argv[1], sys.argv[2]
img=Image.new("RGB",(1240,280),"white"); d=ImageDraw.Draw(img)
try: f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",34)
except Exception: f=ImageFont.load_default()
for i,l in enumerate([f"PROJET {mark} - FICHE INDEXATION",
                      f"Le code projet est {mark} numero 9988.",
                      "Recherche vectorielle MTREE validee."]):
    d.text((40,30+i*80),l,fill="black",font=f)
png=p+".png"; img.save(png); doc=fitz.open(); pg=doc.new_page(width=1240,height=280)
pg.insert_image(fitz.Rect(0,0,1240,280),filename=png); doc.save(p); doc.close()
import os; os.remove(png); print("PDF cree avec marqueur", mark)
PY

echo "== upload =="
RESP=$(curl -s --max-time 30 -X POST "$API/sources" \
  -F "type=upload" -F "notebooks=[\"$NB\"]" -F "embed=true" \
  -F "async_processing=true" -F "title=E2E index $MARK" \
  -F "file=@$PDF;type=application/pdf")
SRC=$(echo "$RESP" | grep -oE '"id":"source:[a-z0-9]+"' | head -1 | cut -d'"' -f4)
echo "Source: $SRC"
[ -z "$SRC" ] && { echo "echec upload: $RESP"; exit 1; }

echo "== attente OCR + embeddings (index MTREE mis a jour a l insert) =="
for i in $(seq 1 48); do
  sleep 5
  FULL=$(curl -s --max-time 10 "$API/sources/$SRC" 2>/dev/null)
  CH=$(echo "$FULL" | grep -oE '"embedded_chunks":[0-9]+' | head -1 | cut -d: -f2)
  if [ -n "$CH" ] && [ "$CH" -gt 0 ] 2>/dev/null; then
    echo "  embeddings prets apres ${i}x5s (embedded_chunks=$CH)"; break
  fi
done

echo "== RECHERCHE VECTORIELLE (KNN/MTREE) via /api/search =="
SR=$(curl -s --max-time 20 -X POST "$API/search" -H "Content-Type: application/json" \
  -d "{\"query\":\"code projet $MARK indexation\",\"type\":\"vector\",\"limit\":5,\"search_sources\":true,\"search_notes\":false,\"minimum_score\":0.2}")
echo "$SR" | python3 -c "
import sys,json
d=json.load(sys.stdin)
res=d.get('results',[])
print('total resultats:', d.get('total_count'))
found=False
for r in res:
    pid=str(r.get('parent_id') or r.get('id'))
    sim=r.get('similarity')
    print(f\"  {pid}  sim={sim}\")
    if '$SRC'.split(':')[1] in pid: found=True
print('NOUVELLE SOURCE TROUVEE PAR LA RECHERCHE VECTORIELLE :', 'OUI ✅' if found else 'NON ❌')
"
