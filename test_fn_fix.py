# -*- coding: utf-8 -*-
"""Teste la fn::vector_search CORRIGEE (ORDER BY/LIMIT dans un SELECT externe)."""
import asyncio
from open_notebook.database.repository import repo_query
from open_notebook.utils.embedding import generate_embedding

FIXED_FN = """
REMOVE FUNCTION IF EXISTS fn::vector_search;
DEFINE FUNCTION IF NOT EXISTS fn::vector_search($query: array<float>, $match_count: int, $sources: bool, $show_notes: bool, $min_similarity: float) {
    let $source_embedding_search =
        IF $sources {(
            SELECT id, title, content, parent_id, similarity FROM (
                SELECT source.id as id, source.title as title, content,
                       source.id as parent_id,
                       (1 - vector::distance::knn()) as similarity
                FROM source_embedding WHERE embedding <|500|> $query
            ) WHERE similarity >= $min_similarity
        )} ELSE { [] };

    let $source_insight_search =
        IF $sources {(
            SELECT id, title, content, parent_id, similarity FROM (
                SELECT id, insight_type + ' - ' + (source.title OR '') as title, content,
                       source.id as parent_id,
                       (1 - vector::distance::knn()) as similarity
                FROM source_insight WHERE embedding <|500|> $query
            ) WHERE similarity >= $min_similarity
        )} ELSE { [] };

    let $note_content_search =
        IF $show_notes {(
            SELECT id, title, content, parent_id, similarity FROM (
                SELECT id, title, content, id as parent_id,
                       (1 - vector::distance::knn()) as similarity
                FROM note WHERE embedding <|500|> $query
            ) WHERE similarity >= $min_similarity
        )} ELSE { [] };

    let $all_results = array::union(
        array::union($source_embedding_search, $source_insight_search),
        $note_content_search
    );

    let $grouped = (SELECT id, parent_id, title,
                math::max(similarity) as similarity,
                array::flatten(content) as matches
            FROM $all_results WHERE id IS NOT NONE
            GROUP BY id, parent_id, title);

    RETURN (SELECT * FROM $grouped ORDER BY similarity DESC LIMIT $match_count);
};
"""


async def main():
    await repo_query(FIXED_FN)
    print("fn corrigee installee\n")
    for q in ["code projet ZEBRELUNE indexation", "prix du filet de bar",
              "procedure PL/SQL package"]:
        embed = await generate_embedding(q)
        rows = await repo_query(
            "SELECT * FROM fn::vector_search($q, 5, true, false, 0.2);", {"q": embed}
        )
        print(f"Requete: {q!r}")
        prev = 2.0
        sorted_ok = True
        for r in rows:
            s = r.get("similarity")
            if s > prev + 1e-9:
                sorted_ok = False
            prev = s
            print(f"   {r.get('parent_id')}  sim={s:.4f}")
        print(f"   -> trie par similarite DESC ? {'OUI' if sorted_ok else 'NON'}\n")


asyncio.run(main())
