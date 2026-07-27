import asyncio
from open_notebook.graphs.ask import graph as ask_graph

async def main():
    for q in ["Quel est le montant de la prime annuelle de performance ?",
              "What is the notice period for a manager?"]:
        print("=== %s" % q)
        r = await ask_graph.ainvoke({"question": q}, config={"configurable": {}})
        print(r["final_answer"][:700])
        print()

asyncio.run(main())
