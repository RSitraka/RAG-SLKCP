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
from functools import lru_cache
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


# ---------------------------------------------------------------------------
# Repères temporels des vidéos
# ---------------------------------------------------------------------------
#
# Une vidéo n'a pas de page, mais elle a un minutage — et c'est exactement ce
# que l'utilisateur doit pouvoir retrouver. Le mécanisme est celui des pages :
# des marqueurs insérés dans le texte du contexte, que le modèle recopie et que
# `citations.py` vérifie.

# L'identifiant YouTube fait toujours 11 caractères, quelle que soit la forme
# de l'URL (watch?v=, youtu.be/, /embed/, /shorts/, /live/).
_YOUTUBE_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})"
)

# Les marqueurs suivent les phrases, pas une horloge. Deux énoncés voisins et
# presque identiques — « Customer credit analysis shows... » à 2:31, « Customer
# analysis shows... » à 2:38 — ne sont distinguables que si chacun porte son
# propre repère ; à intervalle fixe ils tombaient dans le même pas et la
# citation renvoyait à l'écran d'à côté.
#
# On marque donc les segments qui ouvrent une phrase, espacés d'au moins
# _MIN_GAP_S pour ne pas baliser chaque incise, et on force un repère au-delà
# de _MAX_GAP_S — un exposé sans ponctuation exploitable resterait sinon sans
# aucun point de repère.
_MIN_GAP_S = 5
_MAX_GAP_S = 30

# Début de phrase : ce qui suit une ponctuation forte, guillemet ou parenthèse
# fermante compris. `end()` de la correspondance donne la position du premier
# caractère du nouvel énoncé.
_SENTENCE_START_RE = re.compile(r"[.!?][\"'»)\]]*\s+")


def format_timecode(seconds: float) -> str:
    """« 14:20 », ou « 1:14:20 » au-delà de l'heure."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


@lru_cache(maxsize=64)
def _read_video_segments(url: Optional[str]) -> Optional[List[Tuple[float, str]]]:
    """(début en secondes, texte) de chaque segment de la transcription.

    La transcription stockée dans `source.full_text` est du texte plat : les
    horodatages n'existent que dans la source d'origine, il faut donc la
    redemander. None si ce n'est pas une vidéo exploitable.

    Mise en cache : le contexte est reconstruit à CHAQUE message du chat, et
    sans cache la même vidéo serait redemandée à YouTube à chaque fois — une
    latence inutile et un risque de blocage par le fournisseur. Le résultat ne
    change pas pendant la vie du processus ; un redémarrage suffit à le
    rafraîchir.
    """
    if not url:
        return None
    match = _YOUTUBE_ID_RE.search(url)
    if not match:
        return None

    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError:  # pragma: no cover - dépend de l'installation
        logger.debug("youtube-transcript-api absent : pas de repère temporel")
        return None

    try:
        fetched = YouTubeTranscriptApi().fetch(match.group(1))
        return [(float(seg.start), seg.text) for seg in fetched]
    except Exception as e:
        logger.debug(f"Transcription horodatée indisponible ({url}): {e}")
        return None


def _locate_segments(
    text: str, segments: List[Tuple[float, str]]
) -> List[Tuple[int, float, float]]:
    """(offset dans `text`, début, fin) de chaque segment retrouvé.

    Même table de correspondance que pour les pages : la recherche se fait sur
    le texte réduit à ses caractères de mot, la position renvoyée vise le texte
    réel. Les segments introuvables sont ignorés sans faire échouer le reste.
    """
    flat_chars: List[str] = []
    positions: List[int] = []
    for i, ch in enumerate(text):
        low = ch.lower()
        if not _NON_WORD_RE.match(low):
            flat_chars.append(low)
            positions.append(i)
    flat = "".join(flat_chars)

    located: List[Tuple[int, float, float]] = []
    cursor = 0
    for index, (start, segment_text) in enumerate(segments):
        probe = _alnum(segment_text)[:_PROBE_LEN]
        if not probe:
            continue
        found = flat.find(probe, cursor)
        if found == -1:
            continue
        # Fin = début du segment suivant. Pour le dernier, faute de mieux, on
        # lui prête la durée moyenne d'un segment de sous-titrage.
        end = segments[index + 1][0] if index + 1 < len(segments) else start + 4.0
        located.append((positions[found], start, end))
        cursor = found + 1
    return located


def _time_at_offset(
    located: List[Tuple[int, float, float]], offset: int
) -> Optional[float]:
    """Moment correspondant à une position du texte.

    Interpolation linéaire dans le segment qui la contient : une phrase qui
    commence aux trois quarts d'un segment est prononcée aux trois quarts de sa
    durée. Sans cela, tout ce qui tombe dans un même segment porterait le même
    minutage — précisément ce qui faisait confondre deux écrans voisins.
    """
    if not located:
        return None
    index = bisect.bisect_right([start for start, _, _ in located], offset) - 1
    if index < 0:
        return None

    seg_offset, begins, ends = located[index]
    following = located[index + 1][0] if index + 1 < len(located) else None
    span = (following - seg_offset) if following is not None else None
    if not span or span <= 0:
        return begins
    ratio = min(1.0, max(0.0, (offset - seg_offset) / span))
    return begins + ratio * max(0.0, ends - begins)


def _fill_gaps(
    insertions: List[Tuple[int, float]], located: List[Tuple[int, float, float]]
) -> List[Tuple[int, float]]:
    """Ajoute un repère de segment là où plus de _MAX_GAP_S s'écoule sans aucun."""
    if not located:
        return insertions

    filled = list(insertions)
    kept = sorted(moment for _, moment in insertions)
    for seg_offset, begins, _ in located:
        previous = max((moment for moment in kept if moment <= begins), default=None)
        if previous is None or begins - previous >= _MAX_GAP_S:
            filled.append((seg_offset, begins))
            kept.append(begins)
            kept.sort()
    filled.sort(key=lambda item: item[0])
    return filled


def annotate_timecodes(text: str, url: Optional[str] = None) -> str:
    """Insère des marqueurs `[t. MM:SS]` dans la transcription d'une vidéo.

    Pendant exact d'`annotate_pages` pour les PDF. Renvoie le texte inchangé si
    ce n'est pas une vidéo dont la transcription horodatée est accessible, ou
    si les segments ne sont pas retrouvables dans le texte — jamais d'exception.
    """
    if not text or not url:
        return text

    try:
        segments = _read_video_segments(url)
        if not segments:
            return text

        located = _locate_segments(text, segments)
        if not located:
            return text

        # Marquer les débuts de PHRASE, et non les débuts de segment : le
        # sous-titrage coupe toutes les 3 à 4 secondes, sans égard pour la
        # ponctuation. « ... further analysis. Customer » ferme un énoncé et
        # ouvre le suivant à l'intérieur d'un même segment ; caler le repère sur
        # le segment donnerait le même minutage aux deux, donc renverrait à
        # l'écran d'à côté.
        starts = [0] + [m.end() for m in _SENTENCE_START_RE.finditer(text)]

        insertions: List[Tuple[int, float]] = []
        last_kept = -_MAX_GAP_S
        for offset in starts:
            moment = _time_at_offset(located, offset)
            if moment is None:
                continue
            gap = moment - last_kept
            if gap < _MIN_GAP_S:
                continue
            insertions.append((offset, moment))
            last_kept = moment

        # Une phrase peut durer plus longtemps que _MAX_GAP_S sans ponctuation
        # exploitable : on complète alors avec les débuts de segment, pour ne
        # pas laisser de longues plages sans aucun repère.
        insertions = _fill_gaps(insertions, located)
        if not insertions:
            return text

        out: List[str] = []
        previous = 0
        for offset, start in insertions:
            out.append(text[previous:offset])
            out.append(f"\n[t. {format_timecode(start)}]\n")
            previous = offset
        out.append(text[previous:])
        logger.debug(f"{len(insertions)} repères temporels insérés ({url})")
        return "".join(out)
    except Exception as e:
        logger.debug(f"Annotation temporelle impossible ({url}): {e}")
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
