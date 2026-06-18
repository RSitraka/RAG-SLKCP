import { NextResponse } from 'next/server'

/**
 * Runtime Configuration Endpoint
 *
 * This endpoint provides server-side environment variables to the client at runtime.
 * This solves the NEXT_PUBLIC_* limitation where variables are baked into the build.
 *
 * Environment Variables:
 * - API_URL: Where the browser/client should make API requests (public/external URL)
 * - INTERNAL_API_URL: Where Next.js server-side should proxy API requests (internal URL)
 *   Default: http://localhost:5055 (used by Next.js rewrites in next.config.ts)
 *
 * Why two different variables?
 * - API_URL: Used by browser clients, can be https://your-domain.com or http://server-ip:5055
 * - INTERNAL_API_URL: Used by Next.js rewrites for server-side proxying, typically http://localhost:5055
 *
 * Auto-detection logic for API_URL:
 * 1. If API_URL env var is set, use it (explicit override)
 * 2. Otherwise, detect from incoming HTTP request headers (zero-config)
 * 3. Fallback to localhost:5055 if detection fails
 *
 * This allows the same Docker image to work in different deployment scenarios.
 */
export async function GET() {
  // Priority 1: Check if API_URL is explicitly set
  const envApiUrl = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL

  if (envApiUrl) {
    return NextResponse.json({
      apiUrl: envApiUrl,
    })
  }

  // Déploiement local/LAN (modifié) : on renvoie un chemin RELATIF (apiUrl vide).
  // Le navigateur appellera alors http://<host-utilisé>:3001/api/*, que Next.js
  // relaie côté serveur vers localhost:5055 (voir rewrites dans next.config.ts).
  // => fonctionne pour les utilisateurs distants SANS exposer le port 5055, et
  //    quelle que soit l'IP du PC. L'ancienne auto-détection "host:5055" cassait
  //    l'accès réseau car l'API n'écoute que sur 127.0.0.1.
  return NextResponse.json({
    apiUrl: '',
  })
}
