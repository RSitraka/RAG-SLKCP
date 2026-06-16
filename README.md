# RAG-SLKCP — mon NotebookLM local et gratuit

Ce dépôt installe et fait tourner **[Open Notebook](https://github.com/lfnovo/open-notebook)**
(une alternative open-source à Google NotebookLM) **100 % en local et gratuitement**,
en utilisant **Ollama + mistral** comme IA. Aucune clé API payante, aucune donnée envoyée
dans le cloud.

> Le code d'Open Notebook a été cloné tel quel dans le dossier dédié **[`open-notebook/`](open-notebook/)**.
> C'est la "méthode des auteurs" ; je n'ai fait que l'adapter pour un fonctionnement
> natif (sans Docker) avec ton Ollama déjà installé.

---

## 🚀 Démarrer / arrêter (le plus important)

Dans un terminal **WSL Ubuntu** :

```bash
cd ~/RG/open-notebook
./start_local.sh     # démarre TOUT (base + API + worker + interface)
./stop_local.sh      # arrête TOUT
```

Puis ouvre dans ton navigateur 👉 **http://localhost:3001**

| Service        | URL                            | Rôle                                    |
|----------------|--------------------------------|-----------------------------------------|
| Interface web  | http://localhost:3001          | Là où tu utilises l'app (notebooks)     |
| API + doc      | http://localhost:5055/docs     | Le cerveau (REST API, Swagger)          |
| Base SurrealDB | http://localhost:8000          | Stockage (documents + embeddings)       |

> Le port de l'interface est **3001** (et non 3000) car le port 3000 est déjà pris
> par ton autre projet `~/mine`.

---

## 🧩 Comment c'est construit (architecture)

Open Notebook = **3 briques** qui tournent en même temps, plus l'IA :

```
   Toi (navigateur)
        │
        ▼
┌───────────────────┐   HTTP
│  FRONTEND  :3001  │  Interface Next.js / React
│  (open-notebook/  │
│   frontend/)      │
└─────────┬─────────┘
        │ /api/* (proxy)
        ▼
┌───────────────────┐
│   API   :5055     │  FastAPI (Python). Orchestration RAG via LangGraph.
│ (open-notebook/   │  Découpe les docs, calcule les embeddings, fait
│  api/ + open_     │  la recherche vectorielle, parle à l'IA.
│  notebook/)       │
└────┬────────┬─────┘
   │        │
   │        └──────────────► OLLAMA :11434  (mistral + nomic-embed-text)
   │                          ↑ ton IA locale, gratuite
   ▼
┌───────────────────┐
│ SurrealDB :8000   │  Base de données + recherche vectorielle.
│ (données dans     │  Tes notebooks, sources, notes, embeddings.
│  surreal_data/)   │
└───────────────────┘

   WORKER (en arrière-plan) : traite les tâches longues (extraction de
   fichiers, calcul des embeddings, insights). Sans lui, une source
   importée resterait bloquée en "processing" pour toujours.
```

### Ce que fait chaque dossier de `open-notebook/`
- **`api/`** — l'API REST (FastAPI). Chaque fonctionnalité = un "router" + un "service".
- **`open_notebook/`** — le cœur métier : modèles de données, accès base, et surtout
  **`graphs/`** (les workflows RAG : `ask.py` = recherche+synthèse, `chat.py` = conversation,
  `source.py` = ingestion d'un document, `transformation.py` = résumés/insights).
- **`open_notebook/ai/`** — la couche IA multi-fournisseurs (ici : Ollama).
- **`open_notebook/database/migrations/`** — le schéma SurrealDB (appliqué tout seul au démarrage de l'API).
- **`frontend/`** — l'interface web (Next.js + React + Tailwind).
- **`prompts/`** — les gabarits de prompts (Jinja) envoyés à l'IA.
- **`docs/`** — la doc officielle complète (installation, concepts, guide utilisateur).

---

## 🤖 Configuration IA (déjà faite)

Tout est **déjà configuré et testé** :

| Usage                         | Modèle Ollama              |
|-------------------------------|----------------------------|
| Chat / Transformations / Tools / Grand contexte | `mistral:latest` |
| Embeddings (recherche vectorielle)              | `nomic-embed-text:latest` |
| (Vision, dispo si besoin)                        | `llava:latest` |

La connexion à Ollama passe par `OLLAMA_API_BASE=http://localhost:11434` (voir `open-notebook/.env`).
Pour changer un modèle plus tard : interface → **Settings → Models**.

> ⚠️ **Pas de podcasts / synthèse vocale** : ça nécessite un fournisseur TTS/STT
> (OpenAI, ElevenLabs…), qu'Ollama ne fournit pas. Tout le reste (notebooks,
> sources, notes, chat, recherche, RAG) marche en 100 % local.

---

## 🔍 Comment utiliser le RAG (important à comprendre)

1. **Créer un Notebook** (un projet de recherche).
2. **Ajouter des Sources** : PDF, page web, texte, vidéo… Le worker les découpe en
   morceaux et calcule leurs embeddings (ça prend quelques secondes/minutes).
3. Deux façons d'interroger tes documents :
   - **Ask / Search** → fait du **vrai RAG** : recherche vectorielle automatique dans
     toutes tes sources, puis réponse de mistral citée. **C'est ici que la magie opère.**
   - **Chat** → ne récupère PAS automatiquement : tu dois cocher les sources à mettre
     dans le contexte. Utile pour discuter de documents précis.

> 💡 Si mistral semble "ignorer" un gros document dans le chat : sa fenêtre de contexte
> est limitée. Augmente `num_ctx` (ex. 32768) dans le credential Ollama
> (Settings → API Keys). Ta RTX 3060 12 Go encaisse bien pour un modèle 7-8B.

---

## 🛠️ Détails techniques de l'installation (pour info)

Adaptations faites par rapport à la doc officielle (qui suppose Docker) :
- **SurrealDB** : lancé via le **binaire natif** `~/.surrealdb/surreal` (v2.6.5),
  données dans `open-notebook/surreal_data/` — *pas* de conteneur Docker.
- **`.env`** : `SURREAL_URL` pointe sur `localhost` (et non `surrealdb`), Ollama activé.
- **Node 20** (via nvm) forcé pour le frontend (le node système est en v18, trop vieux
  pour Next.js 16).
- Dépendances Python installées avec **`uv sync`**.

Les commandes manuelles équivalentes (4 terminaux) sont décrites dans
[`open-notebook/docs/1-INSTALLATION/from-source.md`](open-notebook/docs/1-INSTALLATION/from-source.md).
`start_local.sh` automatise tout ça.

---

## 📂 Que versionner ?

Le `.gitignore` exclut le lourd/généré (`.venv`, `node_modules`, `.next`,
`surreal_data`, `logs`, et le `.env` secret). Le **code source d'open-notebook reste
visible** dans le dépôt pour que tu puisses l'explorer et le modifier.
