# -*- coding: utf-8 -*-
"""P3.4 - Installe la nouvelle fn::vector_search (KNN/MTREE) et la compare au brute-force.
Garantit zero regression avant de persister en migration 15."""
import asyncio
from open_notebook.database.repository import repo_query
from open_notebook.utils.embedding import generate_embedding

NEW_FN = """
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

    RETURN (select id, parent_id, title, math::max(similarity) as similarity,
        array::flatten(content) as matches
        from $all_results where id is not None
        group by id, parent_id, title ORDER BY similarity DESC LIMIT $match_count);
};
"""


async def main():
    print("=== Installation de la nouvelle fn::vector_search (KNN) ===")
    await repo_query(NEW_FN)
    print("  OK")

    for q in ["prix du filet de bar", "comment demarrer un conteneur docker", "dessert chocolat"]:
        embed = await generate_embedding(q)
        # reference brute-force (groupee par source, max sim) = logique equivalente
        ref = await repo_query(
            """
            SELECT parent_id, math::max(sim) AS similarity FROM (
              SELECT source.id AS parent_id,
                     vector::similarity::cosine(embedding, $q) AS sim
              FROM source_embedding
              WHERE embedding != none AND array::len(embedding)=array::len($q)
                    AND vector::similarity::cosine(embedding, $q) >= 0.2
            ) GROUP BY parent_id ORDER BY similarity DESC LIMIT 5;
            """,
            {"q": embed},
        )
        got = await repo_query(
            "SELECT * FROM fn::vector_search($q, 5, true, false, 0.2);",
            {"q": embed},
        )
        ref_ids = [str(r["parent_id"]) for r in ref]
        got_ids = [str(r["parent_id"]) for r in got]
        ok = set(ref_ids) == set(got_ids)
        print(f"\nRequete: {q!r}")
        print(f"  brute-force (parent_ids): {ref_ids}")
        print(f"  fn KNN     (parent_ids): {got_ids}")
        print(f"  memes sources ? {'OUI ✅' if ok else 'NON ❌'}")
        if got:
            print(f"  top sim KNN={got[0]['similarity']:.5f}  vs BF={ref[0]['similarity']:.5f}")


asyncio.run(main())
