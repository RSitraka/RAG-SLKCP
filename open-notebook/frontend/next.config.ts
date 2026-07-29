import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for optimized Docker deployment
  output: "standalone",

  // Autorise le chargement des ressources de dev (/_next/*) quand l'app est
  // ouverte depuis le réseau local (sinon Next.js 16 bloque => écran blanc).
  // Ajoute ici les IP/hôtes depuis lesquels on accède au serveur de dev.
  // "192.168.1.*" couvre tout le LAN : l'accès survit à un changement d'IP DHCP.
  allowedDevOrigins: [
    "192.168.1.*",
    "192.168.1.113",
    "serveur-slkcp",
    "localhost",
    "127.0.0.1",
  ],

  // Experimental features
  // Type assertion needed: proxyClientMaxBodySize is valid in Next.js 15 but types lag behind
  experimental: {
    // Increase proxy body size limit for file uploads (default is 10MB)
    // This allows larger files to be uploaded through the /api/* rewrite proxy to FastAPI
    proxyClientMaxBodySize: '100mb',

    // Délai avant que le proxy /api/* abandonne la requête vers FastAPI.
    // Défaut Next.js : 30 s (`proxyTimeout || 30000` dans
    // server/lib/router-utils/proxy-request.js). Bien trop court pour le chat
    // et les transformations : un LLM local (Ollama) met couramment 1 à 2 min
    // sur un contexte réel. Passé ce délai le proxy coupe la connexion, le
    // navigateur reçoit un 500 et la requête n'apparaît JAMAIS dans
    // logs/api.log — l'API, elle, termine son travail dans le vide.
    // Aligné sur le timeout du client axios (600 s, cf. src/lib/api/client.ts).
    proxyTimeout: 600_000,
  } as NextConfig['experimental'],

  // API Rewrites: Proxy /api/* requests to FastAPI backend
  // This simplifies reverse proxy configuration - users only need to proxy to port 8502
  // Next.js handles internal routing to the API backend on port 5055
  async rewrites() {
    // INTERNAL_API_URL: Where Next.js server-side should proxy API requests
    // Default: http://localhost:5055 (single-container deployment)
    // Override for multi-container: INTERNAL_API_URL=http://api-service:5055
    const internalApiUrl = process.env.INTERNAL_API_URL || 'http://localhost:5055'

    console.log(`[Next.js Rewrites] Proxying /api/* to ${internalApiUrl}/api/*`)

    return [
      {
        source: '/api/:path*',
        destination: `${internalApiUrl}/api/:path*`,
      },
    ]
  },
};

export default nextConfig;
