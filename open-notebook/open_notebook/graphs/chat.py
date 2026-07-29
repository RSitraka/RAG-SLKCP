import asyncio
import concurrent.futures
import sqlite3
from typing import Annotated, Any, Optional

from ai_prompter import Prompter
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from open_notebook.ai.provision import provision_langchain_model
from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
from open_notebook.domain.notebook import Notebook
from open_notebook.exceptions import OpenNotebookError
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.citations import attach_references, sanitize_citations
from open_notebook.utils.error_classifier import classify_error
from open_notebook.utils.text_utils import extract_text_content


class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    notebook: Optional[Notebook]
    context: Optional[str]
    context_config: Optional[dict]
    model_override: Optional[str]


# Q/R factuel sur documents : on veut une réponse déterministe et fidèle au
# texte, pas de créativité. Sans température fixée, un petit modèle local varie
# d'une exécution à l'autre et pioche parfois la mauvaise ligne d'un tableau
# (ex. 2 800 000 au lieu de 1 100 000 pour un même salaire). 0 = déterministe.
_CHAT_TEMPERATURE = 0.0

# Plafond de génération. En CPU-only (~3 tok/s ici), 8192 tokens = jusqu'à ~45 min
# de génération si le modèle ne s'arrête pas tôt — au-delà du timeout de 10 min du
# front. Une réponse factuelle sourcée tient largement dans 1024 tokens ; ce plafond
# borne le pire cas sans tronquer les réponses utiles.
_CHAT_MAX_TOKENS = 1024


def _run_async(coro_factory):
    """Exécute une coroutine depuis ce nœud LangGraph synchrone.

    Les nœuds sont synchrones mais nos accès base de données sont asynchrones :
    on ouvre une boucle dédiée, dans un thread séparé si une boucle tourne déjà.
    `coro_factory` doit renvoyer une coroutine FRAÎCHE à chaque appel.
    """

    def in_new_loop():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(in_new_loop).result()
    except RuntimeError:
        return asyncio.run(coro_factory())


async def build_grounding_context(context: Any) -> Any:
    """Contexte enrichi du texte intégral des documents, pour le grounding seul.

    Le contexte envoyé au modèle peut n'être qu'un résumé (source ajoutée en
    mode « short ») : sans `full_text`, le grounding n'a aucun texte où retrouver
    la source, et la réponse s'affiche sans référence. On recharge donc le texte
    complet de chaque document du contexte — UNIQUEMENT pour retrouver la source,
    jamais renvoyé au modèle : aucun token supplémentaire dans le prompt.

    En cas d'échec de chargement, l'entrée d'origine est conservée telle quelle :
    le grounding ne doit jamais casser une réponse.
    """
    if not isinstance(context, dict):
        return context
    # Accepte l'enveloppe {context, token_count, char_count} de /chat/context.
    inner = context if "sources" in context else context.get("context")
    if not isinstance(inner, dict):
        return context

    from open_notebook.domain.notebook import Note, Source

    enriched_sources = []
    for src in inner.get("sources") or []:
        if not isinstance(src, dict):
            continue
        entry = dict(src)
        if not entry.get("full_text"):
            sid = str(entry.get("id") or "")
            try:
                source = await Source.get(sid)
                ctx = await source.get_context(context_size="long")
                if isinstance(ctx, dict) and ctx.get("full_text"):
                    entry["full_text"] = ctx["full_text"]
                    if not entry.get("insights"):
                        entry["insights"] = ctx.get("insights")
            except Exception as e:
                logger.debug(f"Grounding : full_text indisponible pour {sid} ({e})")
        enriched_sources.append(entry)

    enriched_notes = []
    for note in inner.get("notes") or []:
        if not isinstance(note, dict):
            continue
        entry = dict(note)
        if not entry.get("content"):
            nid = str(entry.get("id") or "")
            try:
                loaded = await Note.get(nid)
                if getattr(loaded, "content", None):
                    entry["content"] = loaded.content
            except Exception as e:
                logger.debug(f"Grounding : contenu indisponible pour {nid} ({e})")
        enriched_notes.append(entry)

    return {**inner, "sources": enriched_sources, "notes": enriched_notes}


def call_model_with_messages(state: ThreadState, config: RunnableConfig) -> dict:
    try:
        system_prompt = Prompter(prompt_template="chat/system").render(data=state)  # type: ignore[arg-type]
        payload = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )

        # Handle async model provisioning from sync context
        def run_in_new_loop():
            """Run the async function in a new event loop"""
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    provision_langchain_model(
                        str(payload),
                        model_id,
                        "chat",
                        max_tokens=_CHAT_MAX_TOKENS,
                        temperature=_CHAT_TEMPERATURE,
                    )
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an event loop, run in a thread with a new loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                model = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            model = asyncio.run(
                provision_langchain_model(
                    str(payload),
                    model_id,
                    "chat",
                    max_tokens=8192,
                    temperature=_CHAT_TEMPERATURE,
                )
            )

        ai_message = model.invoke(payload)

        # Clean thinking content from AI response (e.g., <think>...</think> tags)
        content = extract_text_content(ai_message.content)
        cleaned_content = clean_thinking_content(content)

        # Le prompt demande le bon format de citation, mais ne le garantit pas :
        # un petit modèle local invente un identifiant ou une page hors bornes.
        # On recale d'abord les citations sur le contexte réellement fourni,
        # puis — s'il n'en reste aucune de valide — on retrouve nous-mêmes
        # d'où vient la réponse. Supprimer une citation fausse sans la
        # remplacer priverait l'utilisateur de toute référence.
        # Le grounding a besoin du texte intégral des documents, même quand le
        # modèle n'a reçu qu'un résumé : on l'enrichit ici, sans rien renvoyer
        # au modèle (aucun token ajouté au prompt). Sinon les sources ajoutées
        # en mode « résumé » n'affichent aucune référence.
        context = _run_async(lambda: build_grounding_context(state.get("context")))
        cleaned_content = sanitize_citations(cleaned_content, context)
        cleaned_content = attach_references(cleaned_content, context)

        cleaned_message = ai_message.model_copy(update={"content": cleaned_content})

        return {"messages": cleaned_message}
    except OpenNotebookError:
        raise
    except Exception as e:
        error_class, user_message = classify_error(e)
        raise error_class(user_message) from e


conn = sqlite3.connect(
    LANGGRAPH_CHECKPOINT_FILE,
    check_same_thread=False,
)
memory = SqliteSaver(conn)

agent_state = StateGraph(ThreadState)
agent_state.add_node("agent", call_model_with_messages)
agent_state.add_edge(START, "agent")
agent_state.add_edge("agent", END)
graph = agent_state.compile(checkpointer=memory)
