"""
Fiabilisation des citations produites par le LLM.

Un prompt ne garantit rien : les petits modèles locaux (llama3.2 3B et
consorts) recopient l'exemple du prompt, inventent un identifiant, ou citent
« p. 10 » dans un document qui n'a que 2 pages. On ne peut pas vérifier qu'un
passage dit bien ce que le modèle prétend, mais on peut vérifier — et corriger —
tout ce qui est factuel :

- l'identifiant existe-t-il dans le contexte fourni ? sinon la citation est
  supprimée : mieux vaut aucune référence qu'une référence morte ;
- le titre affiché est-il celui du document ? sinon il est remplacé par le vrai ;
- la page est-elle dans les bornes du document ? sinon elle est retirée.

Le texte de la réponse n'est jamais réécrit au-delà des citations elles-mêmes.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

# `[source:abc123]`, `[note:xyz]`, `[insight:...]` — éventuellement précédé d'un
# localisateur lisible entre parenthèses que l'on va reconstruire.
_CITATION_RE = re.compile(
    r"(?:\(\s*(?P<locator>[^()]{0,200}?)\s*\)[ \t,]*)?"
    r"\[(?P<id>[a-z_]+:[A-Za-z0-9_-]+)\]"
)
_PAGE_IN_LOCATOR_RE = re.compile(r"\bp\.?\s*(\d{1,4})\b", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"\[p\.\s*(\d{1,4})\]")
# Une vidéo n'a pas de page : son repère est le minutage, inséré par
# annotate_timecodes() sous la forme `[t. 14:20]`.
_TIME_MARKER_RE = re.compile(r"\[t\.\s*(\d{1,3}(?::\d{2}){1,2})\]")
_TIME_IN_LOCATOR_RE = re.compile(r"\b(\d{1,3}(?::\d{2}){1,2})\b")
# « , p. » suivi d'une fermeture au lieu d'un numéro : page amorcée puis
# abandonnée par le modèle.
_DANGLING_PAGE_RE = re.compile(r"[,;]?\s*\bp\.\s*(?=[^\s\d]|\s*$)", re.MULTILINE)

# Tentative de citation malformée : des crochets contenant un « type:valeur »
# noyé dans du texte libre. Observé en conditions réelles :
#   [source:insight_insight_type_DenseSummary content_RAPPORT... id_source:7pgq...]
# Ce n'est pas un identifiant, ça ne peut pas être validé, et laissé tel quel
# c'est illisible. On le supprime.
_MALFORMED_CITATION_RE = re.compile(r"\[[^\]\n]{0,300}?[a-z_]+:[^\]\n]{0,300}?\]")


def _to_seconds(stamp: str) -> int:
    """« 14:20 » -> 860, « 1:14:20 » -> 4460."""
    parts = [int(p) for p in stamp.split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


@dataclass
class CitationTarget:
    """Ce qu'on sait de source sûre à propos d'un document citable."""

    title: str
    max_page: Optional[int] = None
    # Vidéo : dernier repère temporel connu, en secondes. Borne le minutage que
    # le modèle a le droit de citer, comme max_page borne la page. None pour
    # tout ce qui n'est pas une vidéo horodatée.
    max_seconds: Optional[int] = None
    # Texte du document, marqueurs `[p. N]` / `[t. MM:SS]` compris : sert à
    # retrouver nous-mêmes d'où vient une réponse (cf. ground_references).
    text: str = ""

    @property
    def is_video(self) -> bool:
        return self.max_seconds is not None


def _collect(
    items: Any,
    registry: Dict[str, CitationTarget],
    fallback_title: Optional[str] = None,
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        title = str(item.get("title") or "").strip() or fallback_title or item_id

        # La page maximale se lit dans les marqueurs insérés par
        # annotate_pages() : elle borne ce que le modèle a le droit de citer.
        # Idem pour le minutage d'une vidéo (annotate_timecodes).
        max_page = None
        max_seconds = None
        text = item.get("full_text") or item.get("content") or ""
        if isinstance(text, str) and text:
            pages = [int(m.group(1)) for m in _PAGE_MARKER_RE.finditer(text)]
            if pages:
                max_page = max(pages)
            stamps = [
                _to_seconds(m.group(1)) for m in _TIME_MARKER_RE.finditer(text)
            ]
            if stamps:
                max_seconds = max(stamps)

        registry[item_id] = CitationTarget(
            title=title,
            max_page=max_page,
            max_seconds=max_seconds,
            text=text if isinstance(text, str) else "",
        )


def build_citation_registry(context: Any) -> Dict[str, CitationTarget]:
    """Documents réellement présents dans le contexte, par identifiant."""
    registry: Dict[str, CitationTarget] = {}
    # `POST /chat/context` renvoie {context, token_count, char_count} ; le
    # frontend n'en transmet que `context`, mais on accepte l'enveloppe pour
    # ne pas dépendre silencieusement de ce détail d'appelant.
    if isinstance(context, dict) and "sources" not in context:
        inner = context.get("context")
        if isinstance(inner, dict):
            context = inner
    if isinstance(context, dict):
        for key in ("sources", "notes"):
            _collect(context.get(key), registry)

        # Un insight n'a pas de titre : il est dérivé d'un document et hérite
        # du sien. Sans cet héritage la citation retomberait sur l'identifiant
        # technique — exactement ce qu'on cherche à éviter. Les sources sont
        # déjà enregistrées ici, on peut donc y retrouver le titre parent, que
        # les insights soient imbriqués dans leur source ou listés à part avec
        # un `source_id`.
        sources = context.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    _collect(
                        source.get("insights"),
                        registry,
                        fallback_title=str(source.get("title") or "").strip() or None,
                    )

        insights = context.get("insights")
        if isinstance(insights, list):
            for insight in insights:
                if not isinstance(insight, dict):
                    continue
                parent = registry.get(str(insight.get("source_id") or "").strip())
                _collect(
                    [insight],
                    registry,
                    fallback_title=parent.title if parent else None,
                )
    return registry


def _stamp_from_locator(
    locator: Optional[str], target: CitationTarget
) -> Optional[str]:
    """Minutage écrit par le modèle, s'il tombe dans la durée de la vidéo."""
    if not locator or target.max_seconds is None:
        return None
    match = _TIME_IN_LOCATOR_RE.search(locator)
    if not match:
        return None
    # Au-delà de la durée connue, le minutage est inventé : on le refuse comme
    # on refuse une page qui dépasse le nombre de pages.
    if _to_seconds(match.group(1)) > target.max_seconds:
        return None
    return match.group(1)


def _rebuild(
    target: CitationTarget, locator: Optional[str], offset: Optional[int]
) -> str:
    """Citation lisible : nom du document et repère, sans identifiant technique.

    Le modèle écrit `[source:7pgq2b8uy897pslkkrd7]` — c'est le seul moyen fiable
    de savoir DE QUEL document il parle. Mais cet identifiant ne dit rien à
    l'utilisateur : on le traduit en `(TechNova_Rapport_Annuel_2024.pdf, p. 2)`.

    Le repère dépend du support : une page pour un document paginé, un minutage
    pour une vidéo — `(IFS Cloud Finance, 14:20)`. Dans les deux cas il est
    d'abord relu dans ce qu'a écrit le modèle, borné par ce que le document
    contient réellement, et à défaut retrouvé nous-mêmes à partir de `offset`,
    la position du passage cité dans le document.
    """
    if target.is_video:
        stamp = _stamp_from_locator(locator, target)
        if stamp is None and offset is not None:
            stamp = _timecode_at(target.text, offset)
        return f"({target.title}, {stamp})" if stamp else f"({target.title})"

    page: Optional[int] = None
    if locator:
        match = _PAGE_IN_LOCATOR_RE.search(locator)
        if match:
            candidate = int(match.group(1))
            # Hors bornes = inventé. Sans marqueur connu, on ne peut rien
            # affirmer : on retire la page plutôt que de la laisser passer.
            if target.max_page and 1 <= candidate <= target.max_page:
                page = candidate

    if page is None and offset is not None:
        # Le modèle n'a pas donné de page exploitable : on la retrouve nous-
        # mêmes en localisant le contenu cité dans le document.
        page = _page_at(target.text, offset)

    if page:
        return f"({target.title}, p. {page})"
    return f"({target.title})"


def sanitize_citations(text: str, context: Any) -> str:
    """Corrige les citations d'une réponse à partir du contexte réellement fourni.

    Les identifiants inconnus sont supprimés avec leur localisateur ; les
    autres sont réécrits avec le titre exact du document et une page bornée.
    """
    if not text or not isinstance(text, str):
        return text

    registry = build_citation_registry(context)
    needles = set(_content_tokens(text))
    inferred: Dict[str, Optional[int]] = {}
    dropped = 0
    repaired = 0

    def infer_offset(citation_id: str, target: CitationTarget) -> Optional[int]:
        """Position du passage cité, retrouvée en localisant les mots de la
        réponse dans le document.

        C'est de cette position que se déduisent aussi bien la page que le
        minutage. None si la correspondance est trop faible pour affirmer quoi
        que ce soit — on préfère alors une citation sans repère.
        """
        if citation_id not in inferred:
            found = None
            if target.text and needles:
                score, offset = _best_window(target.text, needles)
                pinned = _unique_number_offset(target.text, needles)
                if pinned is not None:
                    offset = pinned
                if score >= _MIN_MATCHED_TOKENS and offset >= 0:
                    found = offset
            inferred[citation_id] = found
        return inferred[citation_id]

    def replace(match: re.Match) -> str:
        nonlocal dropped, repaired
        citation_id = match.group("id")
        target = registry.get(citation_id)
        # Identifiant absent du contexte, ou document sans titre lisible :
        # dans les deux cas il ne reste que l'identifiant technique à afficher,
        # ce qui ne renseigne personne. On supprime la citation.
        if target is None or target.title == citation_id:
            dropped += 1
            return ""
        rebuilt = _rebuild(
            target, match.group("locator"), infer_offset(citation_id, target)
        )
        if rebuilt != match.group(0):
            repaired += 1
        return rebuilt

    cleaned = _CITATION_RE.sub(replace, text)

    # Ce qui reste entre crochets et ressemble à une citation n'a pas pu être
    # validé (identifiant malformé) : on le retire plutôt que de l'afficher.
    cleaned, malformed = _MALFORMED_CITATION_RE.subn("", cleaned)
    dropped += malformed

    if dropped or repaired:
        logger.debug(
            f"Citations : {repaired} corrigée(s), {dropped} supprimée(s) "
            f"(identifiant absent du contexte ou document sans titre)"
        )

    # « p. » sans numéro : le modèle a amorcé une page puis n'a rien trouvé à
    # y mettre. Ce cas échappe au traitement ci-dessus quand il n'y a pas
    # d'identifiant à valider, d'où ce nettoyage indépendant.
    cleaned = _DANGLING_PAGE_RE.sub("", cleaned)

    # Le retrait d'une citation peut laisser des coquilles vides : parenthèses,
    # crochets, ou span de code si le modèle avait encadré sa citation de
    # backticks. La négation autour des deux backticks épargne les blocs ```.
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"(?<!`)``(?!`)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def list_known_ids(context: Any) -> List[str]:
    """Identifiants citables du contexte — pratique pour les tests et le debug."""
    return sorted(build_citation_registry(context))


# ---------------------------------------------------------------------------
# Référence calculée, sans passer par le modèle
# ---------------------------------------------------------------------------
#
# Supprimer une citation fausse ne suffit pas : l'utilisateur veut voir la
# source et la page. Plutôt que de faire confiance au modèle, on retrouve nous-
# mêmes d'où vient la réponse, en cherchant ses mots dans les documents du
# contexte. C'est vérifiable et reproductible.

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Mots trop fréquents pour situer quoi que ce soit. Volontairement court : les
# chiffres et les noms propres font l'essentiel du travail.
_STOPWORDS = {
    "alors", "aussi", "avec", "cela", "cette", "comme", "dans", "elle", "être",
    "leur", "mais", "meme", "même", "pour", "sans", "sont", "sous", "supérieur",
    "tout", "tous", "une", "des", "les", "que", "qui", "est", "par", "sur",
    "aux", "ces", "son", "ses", "plus", "moins", "the", "and", "for", "with",
    "that", "this", "from", "have", "has", "was", "were", "are", "its",
}
_MIN_TOKEN_LEN = 4
# En dessous, la correspondance relève du hasard : on préfère ne pas citer de
# page plutôt qu'en citer une fausse.
_MIN_MATCHED_TOKENS = 3

# Un mot partagé par plusieurs documents du contexte (« salariés », « maximum »,
# un petit chiffre) ne désigne aucune source en particulier : c'est du décor. Un
# mot qui n'apparaît QUE dans un document — « télétravail », « 480 » — le désigne
# sans ambiguïté. On n'attache une source que si elle partage avec la réponse au
# moins un tel mot distinctif ; sinon un rapport annuel se fait « citer » pour
# une réponse sur le télétravail qu'il ne mentionne pas, sur la seule foi de mots
# courants communs.
_MAX_DISTINCTIVE_DF = 1


@dataclass
class Reference:
    """Origine retrouvée d'une réponse."""

    id: str
    title: str
    page: Optional[int] = None
    # Vidéo : moment où commence le passage, « 14:20 ». Exclusif de `page`.
    timecode: Optional[str] = None
    score: int = 0

    @property
    def locator(self) -> Optional[str]:
        """Repère affichable, quel que soit le support."""
        if self.timecode:
            return self.timecode
        if self.page:
            return f"p. {self.page}"
        return None


def _fold(word: str) -> str:
    """Minuscule sans accents.

    Le modèle reformule : il écrit « developpement » là où le PDF a
    « développement », et l'inverse arrive tout autant. Comparer les formes
    accentuées ferait manquer la moitié des correspondances — donc la page.
    """
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _content_tokens(text: str) -> List[str]:
    """Mots porteurs de sens : longs, ou numériques. Accents neutralisés."""
    out = []
    for raw in _TOKEN_RE.findall(text):
        folded = _fold(raw)
        if folded.isdigit() or (
            len(folded) >= _MIN_TOKEN_LEN and folded not in _STOPWORDS
        ):
            out.append(folded)
    return out


def _page_at(text: str, offset: int) -> Optional[int]:
    """Numéro du dernier marqueur `[p. N]` situé avant cet offset."""
    page = None
    for match in _PAGE_MARKER_RE.finditer(text):
        if match.start() > offset:
            break
        page = int(match.group(1))
    return page


def _timecode_at(text: str, offset: int) -> Optional[str]:
    """Minutage du dernier marqueur `[t. MM:SS]` situé avant cet offset.

    Le passage cité commence après ce repère : c'est donc le moment à partir
    duquel l'utilisateur doit lancer la lecture pour l'entendre.
    """
    stamp = None
    for match in _TIME_MARKER_RE.finditer(text):
        if match.start() > offset:
            break
        stamp = match.group(1)
    return stamp


# Taille de la fenêtre de recherche, en caractères : l'ordre de grandeur d'un
# paragraphe. Une fenêtre large ferait gagner la page de garde, qui partage le
# titre du document avec la réponse, au détriment du paragraphe qui porte
# réellement le chiffre cité.
_WINDOW_CHARS = 600

# Au-delà de tant d'occurrences dans un même document, un mot est du décor
# (nom de la société, année de l'exercice) et non un repère.
_COMMON_TOKEN_MAX = 3


def _unique_number_offset(document: str, needles: set) -> Optional[int]:
    """Position d'un nombre de la réponse qui n'apparaît qu'une fois dans le document.

    C'est le repère le plus fort qui existe : « 318 » ne se trouve qu'à un
    endroit, là où le chiffre est énoncé. Un utilisateur qui lit « 318 millions »
    veut ouvrir cette page-là, pas celle qui partage le plus de vocabulaire avec
    la réponse. À plusieurs candidats, le nombre le plus long gagne — le plus
    spécifique.
    """
    occurrences: Dict[str, List[int]] = {}
    for match in _TOKEN_RE.finditer(document):
        word = _fold(match.group(0))
        if word.isdigit() and word in needles:
            occurrences.setdefault(word, []).append(match.start())

    unique = [(word, spots[0]) for word, spots in occurrences.items() if len(spots) == 1]
    if not unique:
        return None
    unique.sort(key=lambda item: len(item[0]), reverse=True)
    return unique[0][1]


def _best_window(document: str, needles: set) -> tuple:
    """Passage du document couvrant le plus de mots de la réponse.

    Renvoie (nombre de mots distincts retrouvés, offset du passage). À égalité,
    le passage contenant des chiffres l'emporte : un nombre est bien plus
    discriminant qu'un mot, et c'est presque toujours lui que l'utilisateur
    veut pouvoir retrouver.
    """
    hits = []
    for match in _TOKEN_RE.finditer(document):
        folded = _fold(match.group(0))
        if folded in needles:
            hits.append((match.start(), folded))
    if not hits:
        return 0, -1

    # Un mot présent partout dans le document ne situe rien : le nom de la
    # société et l'année figurent sur chaque page, et leur poids suffisait à
    # faire gagner la page de garde. On ne garde que les mots discriminants.
    counts: Dict[str, int] = {}
    for _, word in hits:
        counts[word] = counts.get(word, 0) + 1
    discriminant = [(o, w) for o, w in hits if counts[w] <= _COMMON_TOKEN_MAX]
    if len({w for _, w in discriminant}) >= _MIN_MATCHED_TOKENS:
        hits = discriminant

    best_score, best_offset, best_has_digit = 0, -1, False
    for i, (offset, _) in enumerate(hits):
        seen = set()
        matched: List[tuple] = []
        for j in range(i, len(hits)):
            if hits[j][0] - offset > _WINDOW_CHARS:
                break
            seen.add(hits[j][1])
            matched.append(hits[j])
        has_digit = any(word.isdigit() for word in seen)
        better = len(seen) > best_score or (
            len(seen) == best_score and has_digit and not best_has_digit
        )
        if better:
            best_score = len(seen)
            best_has_digit = has_digit
            best_offset = _representative_offset(matched)
    return best_score, best_offset


def _representative_offset(matched: List[tuple]) -> int:
    """Position qui représente le mieux un passage retrouvé.

    Surtout PAS son début : une fenêtre peut commencer sur le dernier mot d'une
    page — typiquement le nom de la société, présent partout — et couvrir le
    paragraphe de la page suivante qui porte l'information. Le début donnerait
    alors la mauvaise page.

    On prend donc la position médiane des mots retrouvés, et parmi les chiffres
    s'il y en a : un nombre ne se promène pas d'une page à l'autre comme le fait
    un nom propre, c'est le repère le plus sûr.
    """
    if not matched:
        return -1
    digits = [offset for offset, word in matched if word.isdigit()]
    positions = sorted(digits or [offset for offset, _ in matched])
    return positions[len(positions) // 2]


def ground_references(answer: str, context: Any) -> List[Reference]:
    """Documents du contexte d'où provient réellement la réponse, avec leur page.

    Trié du plus au moins probable. Liste vide si rien ne dépasse le seuil.
    """
    if not answer or not isinstance(answer, str):
        return []

    needles = set(_content_tokens(answer))
    if not needles:
        return []

    references: List[Reference] = []
    for citation_id, target in build_citation_registry(context).items():
        document = target.text or ""
        if not document:
            continue
        # Sans titre lisible (insight dérivé, entrée sans nom), on n'a rien à
        # montrer à l'utilisateur : afficher l'identifiant serait exactement le
        # défaut qu'on cherche à corriger.
        if target.title == citation_id:
            continue
        score, offset = _best_window(document, needles)
        if score < _MIN_MATCHED_TOKENS:
            continue
        # Un nombre unique dans le document prime sur le comptage de mots.
        pinned = _unique_number_offset(document, needles)
        if pinned is not None:
            offset = pinned
        located = offset >= 0
        references.append(
            Reference(
                id=citation_id,
                title=target.title,
                page=(
                    _page_at(document, offset)
                    if located and not target.is_video
                    else None
                ),
                timecode=(
                    _timecode_at(document, offset)
                    if located and target.is_video
                    else None
                ),
                score=score,
            )
        )

    references.sort(key=lambda r: r.score, reverse=True)
    return references


def format_reference(reference: Reference) -> str:
    """« (TechNova_Rapport_Annuel_2024.pdf, p. 2) », « (Ma vidéo, 14:20) »."""
    locator = reference.locator
    if locator:
        return f"({reference.title}, {locator})"
    return f"({reference.title})"


def attach_references(text: str, context: Any, limit: int = 2) -> str:
    """Ajoute la référence retrouvée si la réponse n'en porte aucune.

    Appelé APRÈS sanitize_citations : à ce stade le texte ne contient plus
    d'identifiant technique, les citations valides ont déjà pris leur forme
    lisible « (Titre, p. N) ». On ne complète donc que le silence — le cas où
    le modèle n'a rien cité, ou n'a cité que des documents inexistants.
    """
    if not text or not isinstance(text, str):
        return text

    registry = build_citation_registry(context)
    if any(f"({target.title}" in text for target in registry.values()):
        return text

    references = ground_references(text, context)
    if not references:
        return text

    # Un insight porte le titre du document dont il est tiré : sans ce filtre,
    # une source et son résumé produiraient deux fois la même référence. Les
    # références sont triées par score, la première d'un titre est donc la
    # meilleure — sauf si une suivante situe le passage, ce qui est plus utile.
    unique: Dict[str, Reference] = {}
    for reference in references:
        best = unique.get(reference.title)
        if best is None or (reference.locator and not best.locator):
            unique[reference.title] = reference
    references = list(unique.values())

    # On ne garde que les sources qui partagent avec la réponse un mot vraiment
    # discriminant (présent dans un seul document du contexte). Un document qui
    # ne recoupe la réponse que par des mots courants — communs à plusieurs
    # documents — n'en est pas la source : c'est ainsi qu'un rapport annuel se
    # faisait attacher à une réponse sur le télétravail. Trié par score, borné
    # par `limit`.
    kept = [r for r in references if _has_distinctive_overlap(r.id, text, registry)]
    kept = kept[:limit]
    if not kept:
        return text

    formatted = " ".join(format_reference(r) for r in kept)
    return f"{text.rstrip()}\n\n{formatted}"


def _has_distinctive_overlap(
    doc_id: str, answer: str, registry: Dict[str, CitationTarget]
) -> bool:
    """Le document partage-t-il avec la réponse un mot propre à lui seul ?

    Un mot présent dans plusieurs documents du contexte ne désigne aucune source
    (fréquence documentaire élevée) ; un mot présent dans un seul la désigne. Le
    document est retenu s'il partage au moins un mot de ce dernier type avec la
    réponse.
    """
    needles = set(_content_tokens(answer))
    if not needles:
        return False

    # Fréquence documentaire : dans combien de documents du contexte chaque mot
    # de la réponse apparaît-il. On EXCLUT les insights : un insight (résumé,
    # analyse) est dérivé de sa source et en reprend le vocabulaire. Le compter
    # ferait passer chaque mot de la source pour « partagé » (source + son propre
    # résumé = fréquence 2), et la source elle-même n'aurait plus aucun mot
    # distinctif — elle ne serait jamais attachée.
    doc_freq: Dict[str, int] = {}
    for cid, target in registry.items():
        if cid.startswith("insight:") or not target.text:
            continue
        for tok in set(_content_tokens(target.text)) & needles:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    target = registry.get(doc_id)
    if target is None or not target.text:
        return False
    own_tokens = set(_content_tokens(target.text)) & needles
    return any(doc_freq.get(tok, 0) <= _MAX_DISTINCTIVE_DF for tok in own_tokens)
