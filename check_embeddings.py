# -*- coding: utf-8 -*-
"""P3.3 - Diagnostic vectorisation via la couche applicative (repo_query)."""
import asyncio
from open_notebook.database.repository import repo_query


async def main():
    total = await repo_query("SELECT count() AS n FROM source GROUP ALL;")
    nemb = await repo_query("SELECT count() AS n FROM source_embedding GROUP ALL;")
    dim = await repo_query("SELECT array::len(embedding) AS d FROM source_embedding LIMIT 1;")
    # sources sans embeddings mais AVEC texte
    missing = await repo_query(
        "SELECT id, string::len(full_text) AS chars FROM source "
        "WHERE id NOT IN (SELECT VALUE source FROM source_embedding) "
        "AND full_text != NONE AND full_text != '';"
    )
    print("Sources totales        :", total[0]["n"] if total else 0)
    print("Chunks source_embedding:", nemb[0]["n"] if nemb else 0)
    print("Dimension vecteurs     :", dim[0]["d"] if dim else "?")
    print("Sources SANS embeddings (avec texte) :", len(missing))
    for m in missing:
        print("   -", m["id"], "(", m.get("chars"), "chars )")


asyncio.run(main())
