"""Tests du recalage des citations produites par le LLM.

Les cas couverts sont ceux réellement observés avec llama3.2 3B en local :
identifiants recopiés depuis l'exemple du prompt, page hors bornes, titre
remplacé par un nom de section, citation absente.

Contrat visé : l'utilisateur voit `(NomDuDocument.pdf, p. 2)` — jamais un
identifiant technique, jamais une page qui n'existe pas.
"""

from open_notebook.utils.citations import (
    attach_references,
    build_citation_registry,
    ground_references,
    list_known_ids,
    sanitize_citations,
)

RAPPORT = (
    "[p. 1]\n"
    "Couverture et sommaire du document annuel de la societe\n"
    "[p. 2]\n"
    "4. Investissements et R&D. Les depenses de recherche et developpement "
    "atteignent 318 millions MGA en 2024, soit une hausse de 22 pourcent "
    "par rapport a 2023."
)

CONTEXT = {
    "sources": [
        {
            "id": "source:7pgq2b8uy897pslkkrd7",
            "title": "TechNova_Rapport_Annuel_2024.pdf",
            "full_text": RAPPORT,
            "insights": [
                {"id": "insight:abc", "title": "Dense Summary", "content": "resume"}
            ],
        }
    ],
    "notes": [{"id": "note:xyz", "title": "Ma note", "content": "contenu"}],
}


class TestRegistry:
    def test_collects_sources_notes_and_nested_insights(self):
        assert list_known_ids(CONTEXT) == [
            "insight:abc",
            "note:xyz",
            "source:7pgq2b8uy897pslkkrd7",
        ]

    def test_max_page_read_from_markers(self):
        registry = build_citation_registry(CONTEXT)
        assert registry["source:7pgq2b8uy897pslkkrd7"].max_page == 2

    def test_no_markers_means_no_page(self):
        registry = build_citation_registry(CONTEXT)
        assert registry["note:xyz"].max_page is None

    def test_empty_context_is_safe(self):
        assert list_known_ids(None) == []
        assert list_known_ids({}) == []

    def test_accepts_the_api_envelope(self):
        """`POST /chat/context` renvoie {context, token_count, char_count}."""
        envelope = {"context": CONTEXT, "token_count": 12, "char_count": 34}
        assert list_known_ids(envelope) == list_known_ids(CONTEXT)


class TestSanitize:
    def test_technical_id_never_reaches_the_user(self):
        """Le coeur de la demande : un id brut n'est pas une référence."""
        out = sanitize_citations("R&D. [source:7pgq2b8uy897pslkkrd7]", CONTEXT)
        assert "source:7pgq2b8uy897pslkkrd7" not in out
        assert "TechNova_Rapport_Annuel_2024.pdf" in out

    def test_invented_id_is_dropped(self):
        """Ids recopiés depuis l'exemple du prompt : ils n'existent pas."""
        text = "Le deep learning est un sous-ensemble du ML. [note:iuiodadalknda]"
        assert "iuiodadalknda" not in sanitize_citations(text, CONTEXT)

    def test_wrong_title_is_replaced_by_the_real_one(self):
        text = "R&D en hausse. (Investissements et R&D) [source:7pgq2b8uy897pslkkrd7]"
        out = sanitize_citations(text, CONTEXT)
        assert "TechNova_Rapport_Annuel_2024.pdf" in out
        assert "(Investissements et R&D)" not in out

    def test_out_of_range_page_falls_back_to_the_located_one(self):
        """Le document n'a que 2 pages : « p. 10 » est forcément inventé."""
        text = (
            "Les depenses de recherche et developpement atteignent 318 millions "
            "MGA. (Rapport, p. 10) [source:7pgq2b8uy897pslkkrd7]"
        )
        out = sanitize_citations(text, CONTEXT)
        assert "p. 10" not in out
        assert "(TechNova_Rapport_Annuel_2024.pdf, p. 2)" in out

    def test_valid_page_is_kept(self):
        text = "R&D. (Rapport, p. 2) [source:7pgq2b8uy897pslkkrd7]"
        out = sanitize_citations(text, CONTEXT)
        assert "(TechNova_Rapport_Annuel_2024.pdf, p. 2)" in out

    def test_page_omitted_when_document_has_no_markers(self):
        """Sans pagination connue, aucune page ne peut être confirmée."""
        out = sanitize_citations("Voir (Ma note, p. 3) [note:xyz]", CONTEXT)
        assert "p. 3" not in out
        assert "(Ma note)" in out

    def test_bare_id_gains_title_and_located_page(self):
        text = (
            "Les depenses de recherche et developpement atteignent 318 millions "
            "MGA en 2024. [source:7pgq2b8uy897pslkkrd7]"
        )
        out = sanitize_citations(text, CONTEXT)
        assert "(TechNova_Rapport_Annuel_2024.pdf, p. 2)" in out

    def test_prefix_is_not_reassigned(self):
        """`insight:abc` existe, `source:abc` non : ce dernier doit sauter."""
        out = sanitize_citations("A [insight:abc] B [source:abc]", CONTEXT)
        assert "(Dense Summary)" in out
        assert "source:abc" not in out

    def test_answer_text_is_preserved(self):
        text = "TechNova a dépensé 318 millions MGA. [source:7pgq2b8uy897pslkkrd7]"
        assert "318 millions MGA" in sanitize_citations(text, CONTEXT)

    def test_dangling_page_without_number_is_cleaned(self):
        """Le modèle amorce « p. » puis n'écrit aucun numéro."""
        text = "R&D en hausse.\n\n(TechNova SARL - Rapport Annuel 2024, p. )"
        out = sanitize_citations(text, CONTEXT)
        assert "p." not in out

    def test_empty_code_span_left_by_a_dropped_id_is_cleaned(self):
        """Le modèle encadre parfois sa citation de backticks."""
        out = sanitize_citations("Chiffre.\n\n`[source:inexistant]`", CONTEXT)
        assert "``" not in out
        assert out == "Chiffre."

    def test_code_fences_are_preserved(self):
        text = "Exemple :\n\n```python\nprint(1)\n```"
        assert "```python" in sanitize_citations(text, CONTEXT)

    def test_no_citation_no_change(self):
        text = "Une réponse sans aucune citation."
        assert sanitize_citations(text, CONTEXT) == text

    def test_handles_empty_and_non_string(self):
        assert sanitize_citations("", CONTEXT) == ""
        assert sanitize_citations(None, CONTEXT) is None


    def test_malformed_pseudo_id_is_removed(self):
        """Cas réel : le modèle fabrique un pseudo-identifiant illisible."""
        text = (
            "R&D en hausse. (Rapport Annuel 2024, p. 5, "
            "[source:insight_insight_type_DenseSummary content_RAPPORT "
            "ANNUEL 2024 id_source:7pgq2b8uy897pslkkrd7])"
        )
        out = sanitize_citations(text, CONTEXT)
        assert "DenseSummary" not in out
        assert "insight_insight_type" not in out
        assert "R&D en hausse." in out


# Un insight n'a pas de titre à lui. Deux formes rencontrées : imbriqué dans sa
# source (contexte du chat de notebook) ou listé à part avec un `source_id`
# (contexte construit par le chat de source).
INSIGHTS_SANS_TITRE = {
    "sources": [
        {
            "id": "source:7pgq2b8uy897pslkkrd7",
            "title": "TechNova_Rapport_Annuel_2024.pdf",
            "full_text": RAPPORT,
            "insights": [{"id": "insight:imbrique", "content": "resume"}],
        }
    ],
    "insights": [
        {
            "id": "insight:aplat",
            "source_id": "source:7pgq2b8uy897pslkkrd7",
            "content": "resume",
        }
    ],
}


class TestInsightTitles:
    """Un insight cité doit renvoyer au document dont il est tiré."""

    def test_nested_insight_inherits_the_source_title(self):
        out = sanitize_citations("Resume. [insight:imbrique]", INSIGHTS_SANS_TITRE)
        assert "(TechNova_Rapport_Annuel_2024.pdf)" in out
        assert "insight:" not in out

    def test_flat_insight_inherits_via_source_id(self):
        out = sanitize_citations("Resume. [insight:aplat]", INSIGHTS_SANS_TITRE)
        assert "(TechNova_Rapport_Annuel_2024.pdf)" in out
        assert "insight:" not in out

    def test_untitled_document_is_dropped_rather_than_shown_as_an_id(self):
        """Sans titre nulle part, il ne resterait que l'identifiant technique."""
        context = {"sources": [{"id": "source:anonyme", "full_text": RAPPORT}]}
        out = sanitize_citations("R&D. [source:anonyme]", context)
        assert "source:anonyme" not in out
        assert out == "R&D."

    def test_source_and_its_insight_are_not_cited_twice(self):
        answer = (
            "Les depenses de recherche et developpement atteignent 318 millions "
            "MGA en 2024."
        )
        out = attach_references(answer, INSIGHTS_SANS_TITRE)
        assert out.count("TechNova_Rapport_Annuel_2024.pdf") == 1


class TestGrounding:
    """Retrouver la source sans rien demander au modèle."""

    def test_locates_the_answer_in_the_right_document_and_page(self):
        answer = (
            "TechNova a depense 318 millions MGA en recherche et developpement "
            "en 2024, soit une hausse de 22 pourcent."
        )
        refs = ground_references(answer, CONTEXT)
        assert refs, "la réponse doit être rattachée à un document"
        assert refs[0].title == "TechNova_Rapport_Annuel_2024.pdf"
        assert refs[0].page == 2

    def test_unrelated_answer_is_not_attached(self):
        """Mieux vaut aucune référence qu'une référence au hasard."""
        refs = ground_references("La photosynthese produit du dioxygene.", CONTEXT)
        assert refs == []

    def test_reference_is_appended_when_the_model_cited_nothing(self):
        answer = (
            "TechNova a depense 318 millions MGA en recherche et developpement "
            "en 2024."
        )
        out = attach_references(answer, CONTEXT)
        assert out.endswith("(TechNova_Rapport_Annuel_2024.pdf, p. 2)")

    def test_existing_valid_citation_is_left_alone(self):
        answer = "Les depenses R&D. (TechNova_Rapport_Annuel_2024.pdf, p. 2)"
        assert attach_references(answer, CONTEXT) == answer

    def test_no_duplicate_reference_after_sanitize(self):
        """Chaîne complète : sanitize puis attach ne doit citer qu'une fois."""
        raw = (
            "Les depenses de recherche et developpement atteignent 318 millions "
            "MGA. [source:7pgq2b8uy897pslkkrd7]"
        )
        out = attach_references(sanitize_citations(raw, CONTEXT), CONTEXT)
        assert out.count("TechNova_Rapport_Annuel_2024.pdf") == 1


# Deux documents distincts : un manuel RH (télétravail) et un rapport annuel
# (finances). Ils partagent du vocabulaire courant (« salaries », « maximum »,
# petits chiffres) mais chacun a son contenu propre.
RH_MANUEL = (
    "[p. 1]\n"
    "Politique de teletravail. Les salaries reguliers peuvent teletravailler au "
    "maximum 3 jours par semaine. Les salaries avec plus de 6 mois d anciennete "
    "beneficient de la meme limite."
)
RAPPORT_FINANCE = (
    "[p. 1]\nCouverture\n[p. 2]\n"
    "4. Effectifs et investissements. La societe compte 120 salaries. Les "
    "depenses R&D atteignent 318 millions MGA en 2024. Objectif : maximum 3 sites."
)
DEUX_DOCS = {
    "sources": [
        {"id": "source:rh", "title": "Manuel_RH_2025.pdf", "full_text": RH_MANUEL},
        {"id": "source:fin", "title": "Rapport_2024.pdf", "full_text": RAPPORT_FINANCE},
    ]
}


class TestDistinctiveAttribution:
    """Ne pas attacher un document qui ne partage que des mots courants."""

    def test_spurious_second_source_is_not_attached(self):
        answer = (
            "Les salaries reguliers sont autorises au teletravail pour un maximum "
            "de 3 jours par semaine, selon leur anciennete."
        )
        out = attach_references(answer, DEUX_DOCS)
        assert "Manuel_RH_2025.pdf" in out
        assert "Rapport_2024.pdf" not in out

    def test_genuine_multi_source_answer_keeps_both(self):
        answer = (
            "Le teletravail est limite a 3 jours par semaine pour les salaries "
            "reguliers, et les depenses R&D ont atteint 318 millions MGA en 2024."
        )
        out = attach_references(answer, DEUX_DOCS)
        assert "Manuel_RH_2025.pdf" in out
        assert "Rapport_2024.pdf" in out

    def test_source_own_insight_does_not_hide_it(self):
        """Un insight reprend le vocabulaire de sa source ; il ne doit pas la
        rendre « non distinctive » et la faire disparaître des références."""
        docs = {
            "sources": [
                {
                    "id": "source:rh",
                    "title": "Manuel_RH_2025.pdf",
                    "full_text": RH_MANUEL,
                    "insights": [
                        {
                            "id": "insight:rh1",
                            "content": (
                                "Resume : teletravail limite a 3 jours par "
                                "semaine pour les salaries reguliers."
                            ),
                        }
                    ],
                },
                {
                    "id": "source:fin",
                    "title": "Rapport_2024.pdf",
                    "full_text": RAPPORT_FINANCE,
                },
            ]
        }
        answer = (
            "Les salaries reguliers sont autorises au teletravail pour un "
            "maximum de 3 jours par semaine."
        )
        out = attach_references(answer, docs)
        assert "Manuel_RH_2025.pdf" in out
        assert "Rapport_2024.pdf" not in out
