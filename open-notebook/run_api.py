#!/usr/bin/env python3
"""
Startup script for Open Notebook API server.
"""

import os
import sys
from pathlib import Path

import uvicorn

# Add the current directory to Python path so imports work
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    # Default configuration
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "5055"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    # Keep-alive volontairement plus long que le pool de connexions du proxy.
    # Par défaut uvicorn ferme une connexion inactive au bout de 5 s. Le proxy
    # /api/* de Next.js garde des sockets en pool : s'il en réutilise une à
    # l'instant où uvicorn la ferme, Node reçoit un ECONNRESET ("socket hang
    # up") et renvoie 500 au navigateur — sans que la requête atteigne jamais
    # l'API, donc sans aucune trace dans logs/api.log. Garder le serveur plus
    # patient que le client supprime cette course.
    timeout_keep_alive = int(os.getenv("API_TIMEOUT_KEEP_ALIVE", "75"))

    print(f"Starting Open Notebook API server on {host}:{port}")
    print(f"Reload mode: {reload}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(current_dir)] if reload else None,
        timeout_keep_alive=timeout_keep_alive,
    )
