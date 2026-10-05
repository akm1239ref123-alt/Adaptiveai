# Deployment

## 1. Vercel frontend

Import this GitHub repository into Vercel. Keep **Root Directory empty** because the Next.js app is at the repository root.

Set:
`PERSONA_API_URL=https://YOUR-BACKEND-URL`

## 2. Persona backend

Deploy `persona-backend/` using Docker/Python hosting. The backend starts with:

`uvicorn server:app --host 0.0.0.0 --port $PORT`

Required environment variables:
- `GOOGLE_CLIENT_ID`
- `PERSONA_OWNER_EMAIL` (defaults to `akm1239ref123@gmail.com`)

The backend needs `checkpoints/persona-20m.pt` before it can start.
