# Persona v2 — from-scratch learning AI

Persona is a small causal Transformer trained from random weights. It does **not** call Gemini, OpenAI, or Vercel AI Gateway.

## Current model
- 1,254,656 trainable parameters
- 6 Transformer layers, 128 hidden size, 4 attention heads
- 256-token context
- Byte-level tokenizer with 256 vocabulary entries
- Latest checkpoint has continued training beyond the first v2 run

## What's improved
- Byte-level tokenizer (256 tokens, handles arbitrary UTF-8 text)
- 4 Transformer blocks, 128 hidden size, 4 attention heads, 256-token context
- Top-k/top-p sampling and repetition penalty
- Checkpoint + optimizer state for continued training
- Feedback can be converted into future training data
- Google Sign-In can securely identify the user
- Owner mode is tied to the verified Google account `akm1239ref123@gmail.com`
- Owner mode has no daily prompt/token quota; each response is capped only by the server's 2048-token generation ceiling

## Important
"Unlimited" here means Persona does not count down a user quota. It does not mean infinite compute or an infinite single response. The server still needs CPU/GPU resources.

## Run

```bash
pip install -r requirements.txt
python train.py --steps 3000
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Google Sign-In

Create a Google OAuth Web Client ID and put the same client ID in:

- `web/.env.local` as `NEXT_PUBLIC_GOOGLE_CLIENT_ID`
- Persona server environment as `GOOGLE_CLIENT_ID`

Do **not** put a Google client secret in the frontend. The backend verifies the Google ID token before granting owner mode.

`PERSONA_OWNER_EMAIL` defaults to `akm1239ref123@gmail.com` and can be changed on the server.

## Continue learning

Collect owner feedback at `/feedback`, then retrain:

```bash
python train.py --resume --steps 3000
```

The model is intentionally tiny. The path to a genuinely capable model is to grow the dataset, tokenizer, parameter count, context length, compute, and evaluation system gradually.


## Web app
The web client is intentionally chat-only. It defaults to `http://127.0.0.1:8000` for local development, so the old `PERSONA_API_URL is not configured` error is gone locally. For a hosted web deployment, set `PERSONA_API_URL` to the public Persona FastAPI URL.

## Quick model check
```bash
python benchmark.py
```

The model is still a research-scale prototype. Parameter count is not a direct measure of intelligence: larger public models such as SmolLM-135M and TinyLlama-1.1B were trained on vastly more data and compute.
