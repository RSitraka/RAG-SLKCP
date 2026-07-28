"""
Localisation des chunks dans leur document d'origine.

Objectif : pouvoir citer « Manuel RH 2025, p. 12, § 2.3 Congés payés » plutôt
que le seul identifiant technique `source:xxxx`.

Deux informations sont dérivées pour chaque chunk :

- **l'en-tête** (`heading`) : le chemin hiérarchique des titres markdown qui
  précèdent le chunk (« 2. Ressources humaines > 2.3 Congés payés »). La
  numérotation éventuelle vient telle quelle du document, on ne la réinvente
  pas.
- **la page** (`page`) : uniquement pour les PDF dont le fichier est encore sur
  disque. `content-core` concatène les pages sans aucun séparateur
  (`"".join(full_text)` dans processors/pdf.py), donc l'information est
  définitivement absente de `source.full_text` : il faut rouvrir le PDF.

Les deux sont **best-effort** et valent `None` en cas d'échec — jamais
d'exception : une citation imprécise vaut mieux qu'un embedding perdu.
"""

import bisect
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

# Longueur de l'extrait servant à retrouver un chunk dans le document.
# Assez long pour être discriminant, assez court pour survivre aux petites
# différences de nettoyage entre le texte extrait et le texte du PDF.
_PROBE_LEN = 80

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w]", re.UNICODE)

# Titres non-markdown : beaucoup de PDF (contrats, manuels, normes) numérotent
# leurs sections sans jamais utiliser de '#'. On ne retient qu'une ligne courte,
# sans ponctuation finale, introduite par un numéro ou un mot de structure.
_PLAIN_HEADING_RE = re.compile(
    r"^[ \t]*("
    r"(?:\d+(?:\.\d+)*[.)]?\s+\S.*)"  # 1., 2.3, 4) ...
    r"|(?:(?:article|section|chapitre|chapter|titre|annexe|appendix|part|partie)"
    r"\b.*)"
    r")[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_PLAIN_HEADING_MAX_LEN = 100


@dataclass
class ChunkLocation:
    """Position d'un chunk dans son document d'origine."""

    heading: Optional[str] = None
    page: Optional[int] = None


def _normalize(text: str) -> str:
    """Espaces réduits et casse ignorée, pour comparer deux extractions."""
    return _WS_RE.sub(" ", text).strip().lower()


def _alnum(text: str) -> str:
    """Ne garde que les caractères de mot, en minuscules.

    Indispensable pour retrouver un chunk dans le PDF d'origine :
    `clean_pdf_text()` de content-core réécrit la ponctuation — le tiret
    cadratin « — » du PDF devient « - » dans `source.full_text`. Une
    comparaison littérale échoue donc dès qu'un chunk contient ce genre de
    caractère. En ne conservant que `\\w` (unicode : accents et CJK inclus),
    les deux extractions redeviennent comparables.
    """
    return _NON_WORD_RE.sub("", text.lower())


# ---------------------------------------------------------------------------
# En-têtes markdown
# ---------------------------------------------------------------------------


def build_heading_index(text: str) -> List[Tuple[int, int, str]]:
    """Index des titres : (offset, niveau, texte du titre), trié par offset.

    Les titres markdown sont prioritaires. S'il n'y en a aucun — cas courant
    des PDF de contrats ou de manuels, qui numérotent leurs sections sans
    jamais écrire de '#' — on retombe sur la détection de titres numérotés
    (« Article 4 - Rémunération variable », « 2.3 Congés payés »).
    """
    markdown = [
        (m.start(), len(m.group(1)), m.group(2).strip())
        for m in _HEADING_RE.finditer(text)
    ]
    if markdown:
        return markdown

    plain: List[Tuple[int, int, str]] = []
    for m in _PLAIN_HEADING_RE.finditer(text):
        title = m.group(1).strip()
        if len(title) > _PLAIN_HEADING_MAX_LEN:
            continue
        # Une phrase se termine par une ponctuation, pas un titre.
        if title[-1] in ".,;:":
            continue
        # Le niveau suit la profondeur de numérotation : "2" -> 1, "2.3" -> 2.
        number = re.match(r"^(\d+(?:\.\d+)*)", title)
        level = number.group(1).count(".") + 1 if number else 1
        plain.append((m.start(), level, title))
    return plain


def heading_for_offset(
    index: List[Tuple[int, int, str]], offset: Optional[int]
) -> Optional[str]:
    """Chemin hiérarchique des titres actifs à cette position.

    On remonte l'index en ne gardant qu'un titre par niveau, et seulement les
    niveaux strictement plus hauts que le dernier retenu : c'est ce qui
    reconstruit « 2. RH > 2.3 Congés » sans y mêler « 2.2 Recrutement ».
    """
    if offset is None or not index:
        return None

    path: List[str] = []
    current_level = 99
    for pos, level, title in reversed(index):
        if pos > offset:
            continue
        if level < current_level:
            path.append(title)
            current_level = level
            if level == 1:
                break
    if not path:
        return None
    return " > ".join(reversed(path))


# ---------------------------------------------------------------------------
# Offsets des chunks dans le texte d'origine
# ---------------------------------------------------------------------------


def locate_offsets(text: str, chunks: List[str]) -> List[Optional[int]]:
    """Position de départ de chaque chunk dans `text`.

    Recherche séquentielle avec curseur : les chunks arrivent dans l'ordre du
    document, donc on ne revient jamais en arrière. Cela évite de retomber sur
    une occurrence antérieure d'un passage répété (pied de page, mention
    légale...). Si l'extrait est introuvable — le splitter markdown recolle
    parfois le contenu — on renvoie None pour ce chunk sans décaler le curseur.
    """
    offsets: List[Optional[int]] = []
    cursor = 0
    for chunk in chunks:
        probe = chunk.strip()[:_PROBE_LEN]
        pos = text.find(probe, cursor) if probe else -1
        if pos == -1 and probe:
            # Repli : le chunk a peut-être été normalisé par le splitter.
            pos = text.find(probe.split("\n", 1)[0], cursor)
        if pos == -1:
            offsets.append(None)
        else:
            offsets.append(pos)
            cursor = pos + 1
    return offsets


# ---------------------------------------------------------------------------
# Pages PDF
# ---------------------------------------------------------------------------


def _read_pdf_pages(file_path: Optional[str]) -> Optional[List[str]]:
    """Texte brut de chaque page du PDF, ou None si illisible."""
    if not file_path or not file_path.lower().endswith(".pdf"):
        return None

    try:
        import pymupdf  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - dépend de l'installation
        try:
            import fitz as pymupdf  # type: ignore[import-untyped,no-redef]
        except ImportError:
            logger.debug("PyMuPDF absent : pas de numéro de page dans les citations")
            return None

    try:
        with pymupdf.open(file_path) as doc:
            return [page.get_text() for page in doc]
    except Exception as e:
        logger.debug(f"Lecture PDF impossible ({file_path}): {e}")
        return None


def build_page_index(file_path: Optional[str]) -> Optional[Tuple[str, List[int]]]:
    """Texte normalisé de toutes les pages + offset de début de chaque page.

    Renvoie None si le fichier n'est pas un PDF lisible : l'appelant se
    contentera alors de l'en-tête, sans numéro de page.
    """
    pages = _read_pdf_pages(file_path)
    if not pages:
        return None

    parts: List[str] = []
    starts: List[int] = []
    total = 0
    for page_text in pages:
        starts.append(total)
        normalized = _alnum(page_text)
        parts.append(normalized)
        total += len(normalized)
    return "".join(parts), starts


def page_for_chunk(
    page_index: Optional[Tuple[str, List[int]]], chunk: str, cursor: int
) -> Tuple[Optional[int], int]:
    """Page (1-based) où commence ce chunk, et nouveau curseur de recherche.

    Un chunk peut chevaucher deux pages : on retient celle où il **commence**,
    qui est celle que le lecteur ouvrira.
    """
    if page_index is None:
        return None, cursor

    haystack, starts = page_index
    probe = _alnum(chunk)[:_PROBE_LEN]
    if not probe:
        return None, cursor

    pos = haystack.find(probe, cursor)
    if pos == -1:
        # Le chunk n'a pas été retrouvé à partir du curseur : on retente depuis
        # le début avant d'abandonner (l'ordre des chunks peut être bousculé
        # par un splitter structurel).
        pos = haystack.find(probe)
        if pos == -1:
            return None, cursor

    page_number = bisect.bisect_right(starts, pos)  # starts est trié, 1-based
    return page_number, pos + 1


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def annotate_pages(text: str, file_path: Optional[str] = None) -> str:
    """Insère des marqueurs `[p. N]` aux frontières de pages du PDF d'origine.

    Sert au contexte du chat, qui reçoit le document ENTIER et non des chunks :
    sans ces marqueurs le modèle n'a aucun moyen de citer une page, et s'il en
    invente une elle sera fausse. Avec eux, la page est lisible dans le texte
    lui-même.

    Renvoie le texte inchangé si ce n'est pas un PDF lisible, ou si les
    frontières ne sont pas retrouvables — jamais d'exception.
    """
    if not text or not file_path or not file_path.lower().endswith(".pdf"):
        return text

    try:
        pages = _read_pdf_pages(file_path)
        if not pages or len(pages) < 2:
            # Un document d'une seule page n'a pas besoin de marqueur interne.
            return f"[p. 1]\n{text}" if pages else text

        # Table de correspondance offset-alphanumérique -> offset d'origine,
        # nécessaire parce que la comparaison se fait sur le texte réduit à ses
        # caractères de mot (cf. _alnum) mais l'insertion doit viser le texte
        # réel.
        flat_chars: List[str] = []
        positions: List[int] = []
        for i, ch in enumerate(text):
            low = ch.lower()
            if not _NON_WORD_RE.match(low):
                flat_chars.append(low)
                positions.append(i)
        flat = "".join(flat_chars)

        # Offsets d'insertion, page par page, en avançant toujours (un curseur
        # évite de replacer un marqueur avant le précédent).
        insertions: List[Tuple[int, int]] = []
        cursor = 0
        for page_number, page_text in enumerate(pages, start=1):
            probe = _alnum(page_text)[:_PROBE_LEN]
            if not probe:
                continue
            found = flat.find(probe, cursor)
            if found == -1:
                continue
            insertions.append((positions[found], page_number))
            cursor = found + 1

        if not insertions:
            return text

        # Si la 1re page n'a pas été localisée (page de garde reformatée à
        # l'extraction, texte réordonné...) mais que la page 2 l'a été, alors le
        # texte qui précède appartient forcément à la page 1 : on le balise.
        # Volontairement limité à ce cas — au-delà d'une page manquante, on
        # préfère ne rien affirmer plutôt que de deviner un numéro.
        first_offset, first_page = insertions[0]
        if first_offset > 0 and first_page == 2:
            insertions.insert(0, (0, 1))

        out: List[str] = []
        previous = 0
        for offset, page_number in insertions:
            out.append(text[previous:offset])
            out.append(f"\n[p. {page_number}]\n")
            previous = offset
        out.append(text[previous:])
        logger.debug(f"{len(insertions)} marqueurs de page insérés ({file_path})")
        return "".join(out)
    except Exception as e:
        logger.debug(f"Annotation des pages impossible ({file_path}): {e}")
        return text


def locate_chunks(
    text: str, chunks: List[str], file_path: Optional[str] = None
) -> List[ChunkLocation]:
    """En-tête et page de chaque chunk. Best-effort, jamais bloquant."""
    if not chunks:
        return []

    try:
        heading_index = build_heading_index(text)
        offsets = locate_offsets(text, chunks)
        page_index = build_page_index(file_path)

        locations: List[ChunkLocation] = []
        cursor = 0
        for chunk, offset in zip(chunks, offsets):
            page, cursor = page_for_chunk(page_index, chunk, cursor)
            locations.append(
                ChunkLocation(
                    heading=heading_for_offset(heading_index, offset),
                    page=page,
                )
            )

        located = sum(1 for loc in locations if loc.page is not None)
        titled = sum(1 for loc in locations if loc.heading is not None)
        logger.debug(
            f"Localisation des chunks : {titled}/{len(chunks)} avec en-tête, "
            f"{located}/{len(chunks)} avec page"
        )
        return locations
    except Exception as e:
        # Une citation moins précise reste préférable à un embedding perdu.
        logger.warning(f"Localisation des chunks impossible : {e}")
        return [ChunkLocation() for _ in chunks]
