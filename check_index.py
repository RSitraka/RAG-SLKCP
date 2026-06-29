# -*- coding: utf-8 -*-
"""P3.4 - Verifie l'etat post-migration : version, index MTREE, fonction KNN."""
import asyncio
from open_notebook.database.repository import repo_query


async def main():
    ver = await repo_query("SELECT version FROM _sbl_migrations ORDER BY version DESC LIMIT 1;")
    print("Version migration courante :", ver[0]["version"] if ver else "?")

    for t in ["source_embedding", "source_insight", "note"]:
        info = await repo_query(f"INFO FOR TABLE {t};")
        if isinstance(info, list):
            info = info[0] if info else {}
        idxs = info.get("indexes", {}) if isinstance(info, dict) else {}
        vec = {k: v for k, v in idxs.items() if "MTREE" in str(v)}
        print(f"\n{t} - index vectoriel :")
        for k, v in vec.items():
            print(f"   {k}: {v}")

    # definition de la fonction
    dbinfo = await repo_query("INFO FOR DB;")
    if isinstance(dbinfo, list):
        dbinfo = dbinfo[0] if dbinfo else {}
    funcs = dbinfo.get("functions", {}) if isinstance(dbinfo, dict) else {}
    fn = str(funcs.get("vector_search", ""))
    print("\nfn::vector_search utilise KNN ?", "OUI" if "knn" in fn.lower() or "<|" in fn else "NON (brute-force)")


asyncio.run(main())
