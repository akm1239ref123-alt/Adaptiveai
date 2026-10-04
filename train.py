import argparse, json, time
from pathlib import Path
import torch
from persona.model import PersonaGPT
from persona.tokenizer import ByteTokenizer

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=1800)
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--block-size', type=int, default=256)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--n-embd', type=int, default=128)
p.add_argument('--n-head', type=int, default=4)
p.add_argument('--n-layer', type=int, default=6)
p.add_argument('--out', default='checkpoints/persona-v2.pt')
p.add_argument('--resume', action='store_true')
a = p.parse_args()

torch.manual_seed(1337)
root = Path(__file__).parent
text_path = root / 'data' / 'train_v2.txt'
feedback = root / 'data' / 'feedback.jsonl'
text = text_path.read_text(encoding='utf-8')
if feedback.exists():
    for line in feedback.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            if int(row.get('rating', 0)) > 0:
                text += f"\nUser: {row.get('prompt','')}\nPersona: {row.get('response','')}\n"
            if row.get('correction'):
                text += f"\nUser: {row.get('prompt','')}\nPersona: {row.get('correction','')}\n"
        except Exception:
            pass

tok = ByteTokenizer(); tok.save(root/'checkpoints'/'tokenizer.json')
ids = torch.tensor(tok.encode(text), dtype=torch.long)
cut = max(1, int(len(ids) * 0.9))
train_ids, val_ids = ids[:cut], ids[cut:]

def batch(data):
    source = train_ids if len(data) <= a.block_size + 1 else data
    ix = torch.randint(len(source) - a.block_size - 1, (a.batch_size,))
    x = torch.stack([source[i:i+a.block_size] for i in ix])
    y = torch.stack([source[i+1:i+a.block_size+1] for i in ix])
    return x, y

cfg = dict(vocab_size=256, block_size=a.block_size, n_embd=a.n_embd, n_head=a.n_head, n_layer=a.n_layer)
model = PersonaGPT(**cfg)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
warmup = max(20, a.steps // 20)
def lr_at(step):
    if step <= warmup:
        return a.lr * step / warmup
    progress = (step - warmup) / max(1, a.steps - warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item())
if a.resume and Path(a.out).exists():
    ck = torch.load(a.out, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    if 'optimizer' in ck:
        opt.load_state_dict(ck['optimizer'])
    print('resumed', a.out)

params = sum(x.numel() for x in model.parameters())
print(f'Persona from-scratch v1: {params:,} parameters; corpus={len(text):,} chars')
for step in range(1, a.steps + 1):
    model.train(); x, y = batch(train_ids); _, loss = model(x, y)
    lr = lr_at(step)
    for g in opt.param_groups: g['lr'] = lr
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step == 1 or step % 100 == 0:
        model.eval(); vx, vy = batch(val_ids if len(val_ids) > a.block_size + 1 else train_ids)
        with torch.no_grad(): _, vl = model(vx, vy)
        print(f'step {step:5d} train={loss.item():.4f} val={vl.item():.4f}')
        if step % 500 == 0:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg, 'step': step}, a.out)
            print('checkpoint saved', a.out)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg, 'step': a.steps}, a.out)
print('saved', a.out)
