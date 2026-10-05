from pathlib import Path
import json, os, time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import jwt
from jwt import PyJWKClient
from persona.model import PersonaGPT
from persona.tokenizer import ByteTokenizer

ROOT = Path(__file__).parent
CK_PATH = ROOT / 'checkpoints' / 'persona-20m.pt'
TOKENIZER_PATH = ROOT / 'checkpoints' / 'tokenizer-20m.json'
ck = torch.load(CK_PATH, map_location='cpu', weights_only=False)
tok = ByteTokenizer.load(TOKENIZER_PATH)
model = PersonaGPT(**ck['config']); model.load_state_dict(ck['model']); model.eval()

app = FastAPI(title='Persona', version='20M')
OWNER_EMAIL = os.getenv('PERSONA_OWNER_EMAIL', 'akm1239ref123@gmail.com').strip().lower()
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '').strip()
ANON_MAX_TOKENS = int(os.getenv('PERSONA_ANON_MAX_TOKENS', '192'))
OWNER_MAX_TOKENS = int(os.getenv('PERSONA_OWNER_MAX_TOKENS', '2048'))

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    context: str = Field(default='', max_length=12000)
    google_id_token: str | None = None
    max_new_tokens: int = Field(default=192, ge=1, le=2048)
    temperature: float = Field(default=0.75, ge=0.05, le=1.5)

class FeedbackRequest(BaseModel):
    prompt: str
    response: str
    rating: int = 1
    correction: str = ''
    google_id_token: str | None = None

def authenticate(token: str | None):
    if not token:
        return {'authenticated': False, 'owner': False, 'email': None}
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail='Google Sign-In is not configured on the Persona server.')
    try:
        jwks = PyJWKClient('https://www.googleapis.com/oauth2/v3/certs')
        signing_key = jwks.get_signing_key_from_jwt(token)
        info = jwt.decode(token, signing_key.key, algorithms=['RS256'], audience=GOOGLE_CLIENT_ID, issuer=['accounts.google.com', 'https://accounts.google.com'])
        email = str(info.get('email', '')).lower()
        verified = bool(info.get('email_verified'))
        if not verified:
            raise ValueError('Google email is not verified')
        return {'authenticated': True, 'owner': email == OWNER_EMAIL, 'email': email, 'sub': info.get('sub')}
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f'Invalid Google sign-in: {exc}')

@app.get('/health')
def health():
    return {
        'ok': True,
        'model': 'Persona 20M',
        'parameters': sum(p.numel() for p in model.parameters()),
        'owner_mode': 'Google verified',
        'owner_email': OWNER_EMAIL,
    }

@app.post('/chat')
def chat(req: ChatRequest):
    auth = authenticate(req.google_id_token)
    limit = OWNER_MAX_TOKENS if auth['owner'] else ANON_MAX_TOKENS
    requested = min(req.max_new_tokens, limit)
    # The owner has no daily prompt/token quota. A per-response ceiling remains a safety/resource limit.
    prompt = ('System: You are Persona, an AI trained from scratch. Be helpful, honest, concise, and learn from corrections.\n'
              + ('Memory/context:\n' + req.context[-3000:] + '\n' if req.context else '')
              + 'User: ' + req.message + '\nPersona: ')
    ids = tok.encode(prompt)
    ids = ids[-model.block_size:]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, requested, req.temperature, 50, 0.95)[0].tolist()
    text = tok.decode(out)
    answer = text[len(tok.decode(ids)):].split('\nUser:')[0].strip()
    return {
        'text': answer,
        'model': 'Persona 20M',
        'authenticated': auth['authenticated'],
        'owner': auth['owner'],
        'account': auth['email'],
        'unlimited_quota': auth['owner'],
        'max_new_tokens': requested,
    }

@app.post('/feedback')
def feedback(req: FeedbackRequest):
    auth = authenticate(req.google_id_token)
    if not auth['owner']:
        raise HTTPException(status_code=403, detail='Only the verified Persona owner can submit training feedback.')
    path = ROOT / 'data' / 'feedback.jsonl'; path.parent.mkdir(exist_ok=True)
    row = req.model_dump(exclude={'google_id_token'}) | {'timestamp': time.time()}
    with path.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {'saved': True, 'message': 'Feedback queued for the next Persona training run.'}
