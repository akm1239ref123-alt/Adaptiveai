# AdaptiveAI — Persona 20M

This repository is arranged for a Next.js frontend at the repository root and the Persona Python backend under `persona-backend/`.

## Frontend

Vercel should use the repository root. `package.json` is at the root, so no Root Directory override is required.

Set this environment variable on Vercel after the Persona backend has a public HTTPS URL:

`PERSONA_API_URL=https://YOUR-PERSONA-BACKEND.example.com`

## Backend

Deploy `persona-backend/` to a Python/Docker host. It requires the trained checkpoint `persona-backend/checkpoints/persona-20m.pt`.

The current project package includes the model architecture and starter dataset; it does **not** contain a trained 20M checkpoint yet. Training must finish before the backend can serve the trained model.

For the owner account, configure:
- `PERSONA_OWNER_EMAIL=akm1239ref123@gmail.com`
- `GOOGLE_CLIENT_ID=<your Google OAuth client ID>`
