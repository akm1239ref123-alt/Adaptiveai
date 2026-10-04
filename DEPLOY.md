# Persona v2 deployment

## Backend
This package contains a FastAPI backend and a Dockerfile. Railway can deploy the Python service from a GitHub repository or Dockerfile; the service needs `GOOGLE_CLIENT_ID` only if Google One Tap owner verification is desired.

## Frontend
The Next.js app in `web/` is intentionally chat-only. It has no header, account panel, settings button, or extra controls. If `NEXT_PUBLIC_GOOGLE_CLIENT_ID` is configured, Google One Tap is invoked silently rather than rendering a sign-in button.

Set `PERSONA_API_URL` on the frontend to the public Persona backend URL.
