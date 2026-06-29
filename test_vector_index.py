# -*- coding: utf-8 -*-
"""P3.4 - Teste un index vectoriel MTREE + recherche KNN, et compare au brute-force.
Objectif : prouver que l'index donne EXACTEMENT les memes resultats (MTREE = exact)
tout en etant accelere, AVANT de persister en migration.
"""
import asyncio, time
from open_notebook.database.repository import repo_query
from open_notebook.utils.embedding import generate_embedding

EMB_FIELDS = [
    ("source_embedding", "idx_se_vec"),
    ("source_insight", "idx_si_vec"),
    ("note", "idx_note_vec"),
]


async def show_indexes(table):
    info = await repo_query(f"INFO FOR TABLE {table};")
    if isinstance(info, list):
        info = info[0] if info else {}
    return info.get("indexes", {}) if isinstance(info, dict) else {}


async def main():
    print("=== Index AVANT ===")
    for t, _ in EMB_FIELDS:
        print(f"  {t}:", list((await show_indexes(t)).keys()))

    # 1) Definir les index MTREE (DIMENSION 768, distance COSINE)
    print("\n=== Creation des index MTREE (DIMENSION 768 DIST COSINE) ===")
    for table, idxname in EMB_FIELDS:
        t0 = time.time()
        await repo_query(
            f"DEFINE INDEX IF NOT EXISTS {idxname} ON {table} "
            f"FIELDS embedding MTREE DIMENSION 768 DIST COSINE TYPE F32;"
        )
        print(f"  {idxname} sur {table}  ({time.time()-t0:.2f}s)")

    print("\n=== Index APRES ===")
    for t, _ in EMB_FIELDS:
        print(f"  {t}:", list((await show_indexes(t)).keys()))

    # 2) Embedding d'une requete reelle
    q = "prix du filet de bar"
    embed = await generate_embedding(q)
    print(f"\nRequete: {q!r}  (dim={len(embed)})")

    # 3) BRUTE-FORCE (verite terrain) : top-5 par cosine
    t0 = time.time()
    bf = await repo_query(
        """
        SELECT meta::id(id) AS rid, content,
               vector::similarity::cosine(embedding, $q) AS sim
        FROM source_embedding
        WHERE embedding != none AND array::len(embedding)=array::len($q)
        ORDER BY sim DESC LIMIT 5;
        """,
        {"q": embed},
    )
    bf_ms = (time.time()-t0)*1000

    # 4) KNN via index MTREE : top-5
    t0 = time.time()
    knn = await repo_query(
        """
        SELECT meta::id(id) AS rid, content,
               (1 - vector::distance::knn()) AS sim
        FROM source_embedding
        WHERE embedding <|5|> $q
        ORDER BY sim DESC;
        """,
        {"q": embed},
    )
    knn_ms = (time.time()-t0)*1000

    print(f"\n--- BRUTE-FORCE ({bf_ms:.1f} ms) ---")
    for r in bf: print(f"  {r['rid']}  sim={r['sim']:.5f}  {r['content'][:40]!r}")
    print(f"\n--- KNN MTREE ({knn_ms:.1f} ms) ---")
    for r in knn: print(f"  {r['rid']}  sim={r['sim']:.5f}  {r['content'][:40]!r}")

    bf_ids = [r["rid"] for r in bf]
    knn_ids = [r["rid"] for r in knn]
    same = bf_ids == knn_ids
    print(f"\nMemes IDs, meme ordre ? {'OUI ✅' if same else 'NON ❌'}")
    if same:
        maxdiff = max(abs(a["sim"]-b["sim"]) for a, b in zip(bf, knn))
        print(f"Ecart max de similarite : {maxdiff:.2e} (doit etre ~0)")


asyncio.run(main())
