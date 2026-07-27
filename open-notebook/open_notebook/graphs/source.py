import operator
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from content_core import extract_content
from content_core.common import ProcessSourceState
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger
from typing_extensions import Annotated, TypedDict

from open_notebook.ai.models import Model, ModelManager
from open_notebook.domain.content_settings import ContentSettings
from open_notebook.domain.notebook import Asset, Source
from open_notebook.domain.transformation import Transformation
from open_notebook.graphs.transformation import graph as transform_graph

# --- Détection du type de source -------------------------------------------

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac", ".wma",
    ".aiff", ".aif", ".amr", ".mpga",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v",
    ".mpeg", ".mpg", ".3gp", ".ogv", ".m2ts", ".mts",
    # NB: .ts est volontairement absent — traité comme TypeScript (voir CODE_EXTENSIONS)
}
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".heic", ".heif", ".svg", ".avif", ".ico",
}
EBOOK_EXTENSIONS = {".epub", ".mobi", ".azw", ".azw3", ".fb2"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".ods", ".csv", ".tsv"}
PRESENTATION_EXTENSIONS = {".pptx", ".ppt", ".odp", ".key"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar", ".xz"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}
DATA_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml", ".toml", ".ini"}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".bash", ".zsh", ".sql", ".r", ".lua", ".pl", ".dart", ".vue", ".svelte",
}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".log"}
DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".pages", ".html", ".htm", ".mhtml",
}

# Extension -> type logique. L'ordre de construction n'a pas d'importance :
# une extension ne doit apparaître que dans un seul ensemble.
_EXTENSION_MAP: Dict[str, str] = {}
for _exts, _kind in (
    (AUDIO_EXTENSIONS, "audio"),
    (VIDEO_EXTENSIONS, "video"),
    (IMAGE_EXTENSIONS, "image"),
    (EBOOK_EXTENSIONS, "ebook"),
    (SPREADSHEET_EXTENSIONS, "spreadsheet"),
    (PRESENTATION_EXTENSIONS, "presentation"),
    (ARCHIVE_EXTENSIONS, "archive"),
    (SUBTITLE_EXTENSIONS, "subtitle"),
    (DATA_EXTENSIONS, "data"),
    (CODE_EXTENSIONS, "code"),
    (TEXT_EXTENSIONS, "plaintext"),
    (DOCUMENT_EXTENSIONS, "document"),
):
    for _ext in _exts:
        _EXTENSION_MAP[_ext] = _kind

VIDEO_HOSTS = {
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "dai.ly",
    "twitch.tv", "ted.com",
}
AUDIO_HOSTS = {
    "soundcloud.com", "podcasts.apple.com", "anchor.fm", "spotify.com",
    "open.spotify.com",
}

# Extensions qui, dans une URL, désignent un vrai fichier et non une page web.
_URL_FILE_KINDS = {
    "audio", "video", "image", "ebook", "spreadsheet",
    "presentation", "archive", "subtitle",
}
# À l'inverse, ces extensions restent des pages web même dans DOCUMENT_EXTENSIONS.
_WEB_PAGE_EXTENSIONS = {".html", ".htm", ".mhtml"}


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _url_suffix(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def _is_youtube(url: str) -> bool:
    return _host(url) in {"youtube.com", "m.youtube.com", "youtu.be"}


def detect_source_type(content_state: Dict[str, Any]) -> str:
    """Type logique de la source, utilisé pour choisir le moteur d'extraction.

    Valeurs possibles : youtube, video, audio, image, ebook, spreadsheet,
    presentation, archive, subtitle, data, code, plaintext, document,
    website, text.
    """
    url = (content_state.get("url") or "").strip()
    file_path = (content_state.get("file_path") or "").strip()

    if url:
        if _is_youtube(url):
            return "youtube"
        host = _host(url)
        if host in VIDEO_HOSTS:
            return "video"
        if host in AUDIO_HOSTS:
            return "audio"
        # URL pointant directement vers un fichier téléchargeable.
        # On ne fait confiance qu'aux extensions non ambiguës : ".php", ".js"
        # ou ".html" dans une URL désignent une page web, pas un fichier.
        suffix = _url_suffix(url)
        by_ext = _EXTENSION_MAP.get(suffix)
        if by_ext in _URL_FILE_KINDS:
            return by_ext
        if suffix in DOCUMENT_EXTENSIONS - _WEB_PAGE_EXTENSIONS:
            return "document"
        return "website"

    if file_path:
        return _EXTENSION_MAP.get(Path(file_path).suffix.lower(), "document")

    return "text"


# Types nécessitant un modèle speech-to-text (au moins en repli)
TRANSCRIBABLE = {"audio", "video", "youtube"}

# ATTENTION — ce qui peut réellement être passé à content-core.
# `extract_content(dict)` fait `ProcessSourceInput(**dict)`, et ce modèle
# pydantic n'accepte QUE ces champs :
#     content, file_path, url, document_engine, url_engine, output_format,
#     audio_provider, audio_model
# Pydantic est en `extra='ignore'` : toute autre clé posée sur `content_state`
# est silencieusement jetée, sans erreur. N'ajoute donc pas de clés ici en
# espérant configurer l'OCR, la langue ou un modèle vision — ça ne ferait rien.
#
# Les leviers réels :
#   - OCR / images / PDF scannés : `document_engine`. En "auto", content-core
#     route PDF, images (png/jpeg/tiff) et Office vers docling quand il est
#     installé, et c'est docling qui fait l'OCR. Aucune option `ocr_languages`
#     n'existe, aucun modèle vision n'est injectable.
#   - Langues des sous-titres YouTube : le CONFIG global de content-core,
#     voir `_apply_youtube_languages()`.
#   - Speech-to-text : `audio_provider` / `audio_model`, voir
#     `_apply_speech_to_text_config()`.

EXTRACTION_ERRORS = {
    "youtube": (
        "Impossible d'extraire le contenu de cette vidéo YouTube : aucun "
        "transcript ni sous-titre disponible. Configurez un modèle "
        "Speech-to-Text dans les Réglages pour transcrire l'audio."
    ),
    "video": (
        "Impossible d'extraire le contenu de cette vidéo. La piste audio est "
        "peut-être absente ou muette. Vérifiez qu'un modèle Speech-to-Text est "
        "configuré dans les Réglages."
    ),
    "audio": (
        "Impossible de transcrire ce fichier audio. Vérifiez qu'un modèle "
        "Speech-to-Text est configuré dans les Réglages et que le format est "
        "supporté."
    ),
    "image": (
        "Aucun texte n'a pu être extrait de cette image. Elle ne contient "
        "peut-être pas de texte lisible, ou aucun modèle Vision/OCR n'est "
        "configuré dans les Réglages."
    ),
    "ebook": (
        "Impossible d'extraire le texte de cet ebook. Le fichier est peut-être "
        "protégé par DRM ou corrompu."
    ),
    "spreadsheet": (
        "Aucune donnée lisible dans ce tableur. Les feuilles sont peut-être "
        "vides ou le fichier est protégé."
    ),
    "presentation": (
        "Aucun texte n'a pu être extrait de cette présentation. Les diapositives "
        "ne contiennent peut-être que des images."
    ),
    "archive": (
        "Aucun contenu exploitable dans cette archive. Elle est peut-être vide, "
        "chiffrée, ou ne contient que des fichiers non supportés."
    ),
    "subtitle": "Ce fichier de sous-titres est vide ou mal formé.",
    "data": (
        "Aucun contenu textuel dans ce fichier de données. Il est peut-être vide "
        "ou mal formé."
    ),
    "code": "Ce fichier source est vide.",
    "plaintext": "Ce fichier texte est vide.",
    "website": (
        "Impossible d'extraire du texte depuis cette page web. Elle est "
        "peut-être protégée, vide, ou rendue entièrement en JavaScript."
    ),
    "document": (
        "Impossible d'extraire du texte depuis ce document. Il est peut-être "
        "vide, scanné (image sans OCR), protégé par mot de passe, ou dans un "
        "format non supporté."
    ),
    "text": "Aucun contenu texte n'a pu être extrait de cette source.",
}


class SourceState(TypedDict):
    content_state: ProcessSourceState
    apply_transformations: List[Transformation]
    source_id: str
    notebook_ids: List[str]
    source: Source
    transformation: Annotated[list, operator.add]
    embed: bool


class TransformationState(TypedDict):
    source: Source
    transformation: Transformation


async def _apply_speech_to_text_config(content_state: Dict[str, Any]) -> None:
    """Injecte le modèle STT par défaut pour l'audio et la vidéo."""
    try:
        model_manager = ModelManager()
        defaults = await model_manager.get_defaults()
        if not defaults.default_speech_to_text_model:
            logger.warning(
                "Aucun modèle Speech-to-Text par défaut n'est configuré ; "
                "content-core utilisera son moteur par défaut."
            )
            return
        stt_model = await Model.get(defaults.default_speech_to_text_model)
        if stt_model:
            content_state["audio_provider"] = stt_model.provider
            content_state["audio_model"] = stt_model.name
            logger.debug(
                f"Using speech-to-text model: {stt_model.provider}/{stt_model.name}"
            )
    except Exception as e:
        logger.warning(f"Failed to retrieve speech-to-text model configuration: {e}")
        # Continue without custom audio model (content-core will use its default)


def _apply_youtube_languages(preferred: List[str]) -> None:
    """Applique l'ordre de préférence des sous-titres YouTube.

    Ne passe PAS par `content_state` : la clé y serait ignorée (voir la note
    au-dessus de EXTRACTION_ERRORS). `extract_youtube_transcript` lit
    `CONFIG["youtube_transcripts"]["preferred_languages"]` dans le CONFIG
    global de content-core, qui est un dict mutable importé par référence.

    Si aucune de ces langues n'est disponible, le repli pytubefix prend
    automatiquement la première piste existante : rien n'est perdu.
    """
    if not preferred:
        return
    try:
        from content_core.config import CONFIG

        CONFIG.setdefault("youtube_transcripts", {})["preferred_languages"] = list(
            preferred
        )
        logger.debug(f"YouTube preferred languages: {preferred}")
    except Exception as e:
        logger.warning(f"Failed to apply YouTube preferred languages: {e}")


async def content_process(state: SourceState) -> dict:
    content_settings = ContentSettings(
        default_content_processing_engine_doc="auto",
        default_content_processing_engine_url="auto",
        default_embedding_option="ask",
        auto_delete_files="yes",
        youtube_preferred_languages=[
            "fr",
            "mg",
            "en",
            "en-GB",
            "pt",
            "es",
            "de",
            "nl",
            "it",
            "ar",
            "hi",
            "ja",
            "zh",
        ],
    )
    content_state: Dict[str, Any] = state["content_state"]  # type: ignore[assignment]

    source_type = detect_source_type(content_state)
    logger.debug(f"Detected source type: {source_type}")

    content_state["url_engine"] = (
        content_settings.default_content_processing_engine_url or "auto"
    )
    # "auto" route PDF, images et documents Office vers docling quand il est
    # installé : c'est docling qui assure l'OCR des pages scannées.
    content_state["document_engine"] = (
        content_settings.default_content_processing_engine_doc or "auto"
    )
    content_state["output_format"] = "markdown"

    # Aucune langue n'est imposée à l'extraction : le texte est conservé dans
    # sa langue d'origine (fr, mg, en, ar, zh...). C'est la RÉPONSE qui suit la
    # langue de la question — voir les prompts dans prompts/.
    if source_type in {"youtube", "video"}:
        _apply_youtube_languages(content_settings.youtube_preferred_languages or [])

    # Modèle speech-to-text : audio, vidéo, et YouTube (en repli des sous-titres).
    # Le modèle détecte lui-même la langue parlée, on ne la force pas.
    if source_type in TRANSCRIBABLE:
        await _apply_speech_to_text_config(content_state)

    processed_state = await extract_content(content_state)

    if not processed_state.content or not processed_state.content.strip():
        raise ValueError(
            EXTRACTION_ERRORS.get(source_type, EXTRACTION_ERRORS["text"])
        )

    return {"content_state": processed_state}


async def save_source(state: SourceState) -> dict:
    content_state = state["content_state"]

    # Get existing source using the provided source_id
    source = await Source.get(state["source_id"])
    if not source:
        raise ValueError(f"Source with ID {state['source_id']} not found")

    # Update the source with processed content
    source.asset = Asset(url=content_state.url, file_path=content_state.file_path)
    source.full_text = content_state.content

    # Preserve user-set title; only overwrite placeholder or empty titles
    if content_state.title and (not source.title or source.title == "Processing..."):
        source.title = content_state.title

    await source.save()

    # NOTE: Notebook associations are created by the API immediately for UI responsiveness
    # No need to create them here to avoid duplicate edges

    if state["embed"]:
        if source.full_text and source.full_text.strip():
            logger.debug("Embedding content for vector search")
            await source.vectorize()
        else:
            logger.warning(
                f"Source {source.id} has no text content to embed, skipping vectorization"
            )

    return {"source": source}


def trigger_transformations(state: SourceState, config: RunnableConfig) -> List[Send]:
    if len(state["apply_transformations"]) == 0:
        return []

    to_apply = state["apply_transformations"]
    logger.debug(f"Applying transformations {to_apply}")

    return [
        Send(
            "transform_content",
            {
                "source": state["source"],
                "transformation": t,
            },
        )
        for t in to_apply
    ]


async def transform_content(state: TransformationState) -> Optional[dict]:
    source = state["source"]
    content = source.full_text
    if not content:
        return None
    transformation: Transformation = state["transformation"]

    logger.debug(f"Applying transformation {transformation.name}")
    result = await transform_graph.ainvoke(
        dict(input_text=content, transformation=transformation)  # type: ignore[arg-type]
    )
    await source.add_insight(transformation.title, result["output"])
    return {
        "transformation": [
            {
                "output": result["output"],
                "transformation_name": transformation.name,
            }
        ]
    }


# Create and compile the workflow
workflow = StateGraph(SourceState)

# Add nodes
workflow.add_node("content_process", content_process)
workflow.add_node("save_source", save_source)
workflow.add_node("transform_content", transform_content)
# Define the graph edges
workflow.add_edge(START, "content_process")
workflow.add_edge("content_process", "save_source")
workflow.add_conditional_edges(
    "save_source", trigger_transformations, ["transform_content"]
)
workflow.add_edge("transform_content", END)

# Compile the graph
source_graph = workflow.compile()
