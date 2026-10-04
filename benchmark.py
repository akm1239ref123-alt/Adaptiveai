from pathlib import Path
import torch
from persona.model import PersonaGPT
from persona.tokenizer import ByteTokenizer

ROOT = Path(__file__).parent
ck = torch.load(ROOT / "checkpoints/persona-v2.pt", map_location="cpu", weights_only=False)
tok = ByteTokenizer.load(ROOT / "checkpoints/tokenizer.json")
model = PersonaGPT(**ck["config"])
model.load_state_dict(ck["model"])
model.eval()

prompts = [
    "User: Hello!\nPersona: ",
    "User: What is 2 + 2?\nPersona: ",
    "User: What is Python?\nPersona: ",
    "User: How do I learn programming?\nPersona: ",
]
for prompt in prompts:
    ids = tok.encode(prompt)
    x = torch.tensor([ids[-model.block_size:]], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, 80, temperature=0.55, top_k=40, top_p=0.95)[0].tolist()
    full = tok.decode(out)
    answer = full[len(tok.decode(x[0].tolist())):].split("\nUser:")[0].strip()
    print(f"\nQ: {prompt.split('User: ',1)[1].splitlines()[0]}\nA: {answer}")
