# -*- coding: utf-8 -*-
"""Diagnostic : le nouvel embedding est-il bien indexe ET retrouve par KNN ?
Compare brute-force vs KNN pour la meme requete, et montre le rang du nouveau doc."""
import asyncio, sys
from open_notebook.database.repository import repo_query
from open_notebook.utils.embedding import generate_embedding

NEW = sys.argv[1] if len(sys.argv) > 1 else None  # source:xxxx
Q = "code projet ZEBRELUNE indexation"


async def main():
    embed = await generate_embedding(Q)

    # brute-force : TOUS les chunks, similarite, garder le rang du nouveau doc
    bf = await repo_query(
        """
        SELECT meta::id(source) AS sid,
               vector::similarity::cosine(embedding, $q) AS sim, content
        FROM source_embedding
        ORDER BY sim DESC;
        """,
        {"q": embed},
    )
    print("=== BRUTE-FORCE top 8 (par chunk) ===")
    for r in bf[:8]:
        tag = " <== NOUVEAU" if NEW and r["sid"] in NEW else ""
        print(f"  {r['sid']}  sim={r['sim']:.4f}  {r['content'][:35]!r}{tag}")

    # rang du nouveau doc en brute-force
    if NEW:
        nid = NEW.split(":")[1]
        rank = next((i for i, r in enumerate(bf) if r["sid"] == nid), None)
        print(f"\nRang brute-force du nouveau doc ({nid}) : "
              f"{rank+1 if rank is not None else 'ABSENT'} / {len(bf)} chunks")

    # KNN brut : le nouveau chunk est-il dans les candidats ?
    knn = await repo_query(
        """
        SELECT meta::id(source) AS sid, (1 - vector::distance::knn()) AS sim
        FROM source_embedding WHERE embedding <|500|> $q ORDER BY sim DESC;
        """,
        {"q": embed},
    )
    print(f"\n=== KNN : {len(knn)} candidats ramenes ===")
    if NEW:
        nid = NEW.split(":")[1]
        krank = next((i for i, r in enumerate(knn) if r["sid"] == nid), None)
        print(f"Nouveau doc present dans les candidats KNN ? "
              f"{'OUI rang '+str(krank+1) if krank is not None else 'NON'}")
    # coherence top: comparer les 8 premiers sids
    print("Top-8 brute-force == Top-8 KNN ?",
          [r["sid"] for r in bf[:8]] == [r["sid"] for r in knn[:8]])

    # appel de la VRAIE fonction
    fn = await repo_query(
        "SELECT * FROM fn::vector_search($q, 5, true, false, 0.2);", {"q": embed}
    )
    print("\n=== fn::vector_search (la fonction reelle) ===")
    for r in fn:
        pid = str(r.get("parent_id"))
        tag = " <== NOUVEAU" if NEW and NEW.split(":")[1] in pid else ""
        print(f"  {pid}  sim={r.get('similarity'):.4f}{tag}")
    print("Nouveau doc dans fn::vector_search ?",
          "OUI" if NEW and any(NEW.split(":")[1] in str(r.get("parent_id")) for r in fn) else "NON")


asyncio.run(main())
